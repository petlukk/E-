#!/usr/bin/env bash
# Cross-compile Ea programs to AArch64, link statically, SCP to Pi, execute via SSH.
# Prerequisites:
#   sudo apt install gcc-aarch64-linux-gnu
#   Pi reachable at $PI_HOST (default: peter@10.46.0.21)
set -euo pipefail

PI_HOST="${PI_HOST:-peter@10.46.0.21}"
PI_KEY="${PI_KEY:-$HOME/.ssh/id_ed25519_pi}"
SSH_OPTS="-i $PI_KEY -o ConnectTimeout=5"
EA="cargo run --features=llvm --"
CROSS_CC="aarch64-linux-gnu-gcc"
TRIPLE="aarch64-unknown-linux-gnu"
TMPDIR="$(mktemp -d)"
PASS=0
FAIL=0

cleanup() { rm -rf "$TMPDIR"; ssh $SSH_OPTS "$PI_HOST" "rm -rf /tmp/ea_arm_test" 2>/dev/null || true; }
trap cleanup EXIT

ssh $SSH_OPTS "$PI_HOST" "mkdir -p /tmp/ea_arm_test" 2>/dev/null

run_test() {
    local name="$1"
    local ea_source="$2"
    local expected="$3"
    local c_harness="${4:-}"

    local ea_file="$TMPDIR/${name}.ea"
    local obj_file="$TMPDIR/${name}.o"
    local bin_file="$TMPDIR/${name}"

    echo "$ea_source" > "$ea_file"

    # Cross-compile to AArch64 object file
    if ! $EA "$ea_file" --target-triple="$TRIPLE" 2>"$TMPDIR/${name}.err"; then
        echo "FAIL [$name]: compilation failed"
        cat "$TMPDIR/${name}.err"
        FAIL=$((FAIL + 1))
        return
    fi
    # Compiler writes {stem}.o in CWD
    mv "${name}.o" "$obj_file"

    # Link
    if [ -n "$c_harness" ]; then
        local c_file="$TMPDIR/${name}.c"
        echo "$c_harness" > "$c_file"
        if ! $CROSS_CC -static "$c_file" "$obj_file" -o "$bin_file" -lm 2>"$TMPDIR/${name}_link.err"; then
            echo "FAIL [$name]: linking failed"
            cat "$TMPDIR/${name}_link.err"
            FAIL=$((FAIL + 1))
            return
        fi
    else
        if ! $CROSS_CC -static "$obj_file" -o "$bin_file" -lm 2>"$TMPDIR/${name}_link.err"; then
            echo "FAIL [$name]: linking failed"
            cat "$TMPDIR/${name}_link.err"
            FAIL=$((FAIL + 1))
            return
        fi
    fi

    # SCP to Pi and execute
    scp -q $SSH_OPTS "$bin_file" "$PI_HOST:/tmp/ea_arm_test/${name}"
    local actual
    actual=$(ssh $SSH_OPTS "$PI_HOST" "/tmp/ea_arm_test/${name}" 2>&1) || true

    if [ "$actual" = "$expected" ]; then
        echo "PASS [$name]"
        PASS=$((PASS + 1))
    else
        echo "FAIL [$name]: expected '$expected', got '$actual'"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Ea ARM/NEON cross-compilation tests ==="
echo "Pi: $PI_HOST"
echo ""

# --- Test 1: Hello world ---
run_test "hello" \
    'func main() { println(42) }' \
    "42"

# --- Test 2: Arithmetic ---
run_test "arith" \
    'func main() {
    let a: i32 = 10
    let b: i32 = 32
    println(a + b)
}' \
    "42"

# --- Test 3: f32x4 SIMD ---
run_test "f32x4_add" \
    'export func vadd(a: *f32, b: *f32, out: *mut f32) {
    let va: f32x4 = load(a, 0)
    let vb: f32x4 = load(b, 0)
    let vc: f32x4 = va .+ vb
    store(out, 0, vc)
}' \
    "3.000000 5.000000 7.000000 9.000000" \
    '#include <stdio.h>
extern void vadd(const float*, const float*, float*);
int main() {
    float a[4] = {1,2,3,4};
    float b[4] = {2,3,4,5};
    float c[4];
    vadd(a, b, c);
    printf("%f %f %f %f\n", c[0], c[1], c[2], c[3]);
    return 0;
}'

# --- Test 4: i32x4 SIMD ---
run_test "i32x4_add" \
    'export func iadd(a: *i32, out: *mut i32) {
    let va: i32x4 = load(a, 0)
    let vb: i32x4 = splat(10)
    let vc: i32x4 = va .+ vb
    store(out, 0, vc)
}' \
    "11 12 13 14" \
    '#include <stdio.h>
extern void iadd(const int*, int*);
int main() {
    int a[4] = {1,2,3,4};
    int c[4];
    iadd(a, c);
    printf("%d %d %d %d\n", c[0], c[1], c[2], c[3]);
    return 0;
}'

# --- Test 5: FMA ---
run_test "fma" \
    'export func do_fma(a: *f32, b: *f32, c: *f32, out: *mut f32) {
    let va: f32x4 = load(a, 0)
    let vb: f32x4 = load(b, 0)
    let vc: f32x4 = load(c, 0)
    let vr: f32x4 = fma(va, vb, vc)
    store(out, 0, vr)
}' \
    "12.000000 27.000000 48.000000 75.000000" \
    '#include <stdio.h>
extern void do_fma(const float*, const float*, const float*, float*);
int main() {
    float a[4] = {1,2,3,4};
    float b[4] = {2,3,4,5};
    float c[4] = {10,21,36,55};
    float r[4];
    do_fma(a, b, c, r);
    printf("%f %f %f %f\n", r[0], r[1], r[2], r[3]);
    return 0;
}'

# --- Test 6: i8x16 ---
run_test "i8x16" \
    'export func sum_bytes(a: *i8) -> i8 {
    let v: i8x16 = load(a, 0)
    return reduce_add(v)
}' \
    "120" \
    '#include <stdio.h>
#include <stdint.h>
extern int8_t sum_bytes(const int8_t*);
int main() {
    int8_t a[16];
    for (int i = 0; i < 16; i++) a[i] = (int8_t)i;
    printf("%d\n", (int)sum_bytes(a));
    return 0;
}'

# --- Test 7: Structs ---
run_test "structs" \
    'struct Point { x: f32, y: f32 }
export func sum_xy(p: *Point) -> f32 {
    return p.x + p.y
}' \
    "7.000000" \
    '#include <stdio.h>
typedef struct { float x; float y; } Point;
extern float sum_xy(const Point*);
int main() {
    Point p = {3.0f, 4.0f};
    printf("%f\n", sum_xy(&p));
    return 0;
}'

# --- Test 8: foreach ---
run_test "foreach" \
    'func main() {
    let mut sum: i32 = 0
    foreach (i in 0..10) {
        sum = sum + i
    }
    println(sum)
}' \
    "45"

# --- Test 9: Control flow ---
run_test "control_flow" \
    'func main() {
    let mut x: i32 = 0
    while x < 5 {
        x = x + 1
    }
    println(x)
}' \
    "5"

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] || exit 1
