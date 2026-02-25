#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === Phase 8: Integer Types ===

    #[test]
    fn test_i8_literal() {
        assert_output(
            r#"
            func main() {
                let x: i8 = -1
                println(x)
            }
            "#,
            "-1",
        );
    }

    #[test]
    fn test_u8_literal() {
        assert_output(
            r#"
            func main() {
                let x: u8 = 255
                println(x)
            }
            "#,
            "255",
        );
    }

    #[test]
    fn test_i16_literal() {
        assert_output(
            r#"
            func main() {
                let x: i16 = -1000
                println(x)
            }
            "#,
            "-1000",
        );
    }

    #[test]
    fn test_u16_literal() {
        assert_output(
            r#"
            func main() {
                let x: u16 = 60000
                println(x)
            }
            "#,
            "60000",
        );
    }

    #[test]
    fn test_u8_wrapping_add() {
        assert_output(
            r#"
            func main() {
                let x: u8 = 255
                let y: u8 = x + 1
                println(y)
            }
            "#,
            "0",
        );
    }

    #[test]
    fn test_u8_comparison_unsigned() {
        assert_output(
            r#"
            func main() {
                let a: u8 = 200
                let b: u8 = 100
                if a > b {
                    println(1)
                } else {
                    println(0)
                }
            }
            "#,
            "1",
        );
    }

    // Verify u8 comparison is unsigned: 200 > 100 must be true,
    // but if signed (i8), 200 wraps to -56 which is < 100.
    #[test]
    fn test_u8_large_values_unsigned_ordering() {
        assert_output(
            r#"
            func main() {
                let a: u8 = 200
                let b: u8 = 50
                if b < a {
                    println(1)
                } else {
                    println(0)
                }
            }
            "#,
            "1",
        );
    }

    #[test]
    fn test_i8_pointer_c_interop() {
        assert_c_interop(
            r#"
            export func sum_i8(ptr: *i8, n: i32) -> i8 {
                let mut i: i32 = 0
                let mut acc: i8 = 0
                while i < n {
                    acc = acc + ptr[i]
                    i = i + 1
                }
                return acc
            }
            "#,
            r#"
            #include <stdio.h>
            extern signed char sum_i8(signed char* ptr, int n);
            int main() {
                signed char data[4] = {1, 2, 3, 4};
                printf("%d\n", (int)sum_i8(data, 4));
                return 0;
            }
            "#,
            "10",
        );
    }

    #[test]
    fn test_u8_pointer_c_interop() {
        assert_c_interop(
            r#"
            export func scale_bytes(ptr: *mut u8, n: i32, factor: u8) {
                let mut i: i32 = 0
                while i < n {
                    let v: u8 = ptr[i]
                    ptr[i] = v * factor
                    i = i + 1
                }
            }
            "#,
            r#"
            #include <stdio.h>
            extern void scale_bytes(unsigned char* ptr, int n, unsigned char factor);
            int main() {
                unsigned char data[4] = {10, 20, 30, 40};
                scale_bytes(data, 4, 2);
                for (int i = 0; i < 4; i++) printf("%d\n", (int)data[i]);
                return 0;
            }
            "#,
            "20\n40\n60\n80",
        );
    }

    // === Phase 8: SIMD byte vector types ===

    #[test]
    fn test_i8x16_literal_element() {
        assert_output(
            r#"
            func main() {
                let v: i8x16 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]i8x16
                let x: i8 = v[0]
                println(x)
                let y: i8 = v[15]
                println(y)
            }
            "#,
            "1\n16",
        );
    }

    #[test]
    fn test_i8x16_add_dot() {
        assert_output(
            r#"
            func main() {
                let a: i8x16 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]i8x16
                let b: i8x16 = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]i8x16
                let c: i8x16 = a .+ b
                let x: i8 = c[0]
                println(x)
                let y: i8 = c[15]
                println(y)
            }
            "#,
            "11\n26",
        );
    }

    #[test]
    fn test_i8x32_splat() {
        assert_output(
            r#"
            func main() {
                let s: i8 = 42
                let v: i8x32 = splat(s)
                let x: i8 = v[0]
                println(x)
                let y: i8 = v[31]
                println(y)
            }
            "#,
            "42\n42",
        );
    }

    #[test]
    fn test_u8x16_reduce_max() {
        assert_output(
            r#"
            func main() {
                let v: u8x16 = [10, 200, 30, 150, 5, 180, 20, 100, 15, 90, 25, 60, 50, 40, 70, 80]u8x16
                let m: u8 = reduce_max(v)
                println(m)
            }
            "#,
            "200",
        );
    }

    #[test]
    fn test_u8x16_load_store_c_interop() {
        assert_c_interop(
            r#"
            export func add_constant_u8x16(src: *u8, dst: *mut u8, n: i32, val: u8) {
                let mut i: i32 = 0
                let splat_val: u8x16 = splat(val)
                while i < n {
                    let chunk: u8x16 = load(src, i)
                    let result: u8x16 = chunk .+ splat_val
                    store(dst, i, result)
                    i = i + 16
                }
            }
            "#,
            r#"
            #include <stdio.h>
            #include <string.h>
            extern void add_constant_u8x16(unsigned char* src, unsigned char* dst, int n, unsigned char val);
            int main() {
                unsigned char src[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
                unsigned char dst[16] = {0};
                add_constant_u8x16(src, dst, 16, 10);
                for (int i = 0; i < 4; i++) printf("%d\n", (int)dst[i]);
                return 0;
            }
            "#,
            "11\n12\n13\n14",
        );
    }

    #[test]
    fn test_i8x32_c_interop() {
        assert_c_interop(
            r#"
            export func negate_i8x32(src: *i8, dst: *mut i8, n: i32) {
                let mut i: i32 = 0
                let zero: i8x32 = splat(0)
                while i < n {
                    let chunk: i8x32 = load(src, i)
                    let result: i8x32 = zero .- chunk
                    store(dst, i, result)
                    i = i + 32
                }
            }
            "#,
            r#"
            #include <stdio.h>
            extern void negate_i8x32(signed char* src, signed char* dst, int n);
            int main() {
                signed char src[32];
                signed char dst[32];
                for (int i = 0; i < 32; i++) src[i] = (signed char)(i + 1);
                negate_i8x32(src, dst, 32);
                printf("%d\n", (int)dst[0]);
                printf("%d\n", (int)dst[1]);
                printf("%d\n", (int)dst[31]);
                return 0;
            }
            "#,
            "-1\n-2\n-32",
        );
    }

    // === Phase 8: Widening / Narrowing ===

    #[test]
    fn test_widen_i8_f32x4() {
        assert_output(
            r#"
            func main() {
                let v: i8x16 = [10, 20, -30, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]i8x16
                let w: f32x4 = widen_i8_f32x4(v)
                println(w[0])
                println(w[1])
                println(w[2])
                println(w[3])
            }
            "#,
            "10\n20\n-30\n40",
        );
    }

    #[test]
    fn test_widen_u8_f32x4() {
        assert_output(
            r#"
            func main() {
                let v: u8x16 = [200, 100, 50, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]u8x16
                let w: f32x4 = widen_u8_f32x4(v)
                println(w[0])
                println(w[1])
                println(w[2])
                println(w[3])
            }
            "#,
            "200\n100\n50\n25",
        );
    }

    #[test]
    fn test_narrow_f32x4_i8() {
        assert_output(
            r#"
            func main() {
                let v: f32x4 = [10.0, 20.0, -5.0, 127.0]f32x4
                let n: i8x16 = narrow_f32x4_i8(v)
                let a: i8 = n[0]
                let b: i8 = n[1]
                let c: i8 = n[2]
                let d: i8 = n[3]
                println(a)
                println(b)
                println(c)
                println(d)
            }
            "#,
            "10\n20\n-5\n127",
        );
    }

    #[test]
    fn test_roundtrip_u8_f32_u8() {
        assert_c_interop(
            r#"
            export func normalize_first4(src: *u8, dst: *mut f32) {
                let chunk: u8x16 = load(src, 0)
                let floats: f32x4 = widen_u8_f32x4(chunk)
                let scale: f32x4 = splat(0.00392156862)
                let normalized: f32x4 = floats .* scale
                store(dst, 0, normalized)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <math.h>
            extern void normalize_first4(unsigned char* src, float* dst);
            int main() {
                unsigned char src[16] = {0, 128, 255, 64, 0};
                float dst[4] = {0.0f};
                normalize_first4(src, dst);
                printf("%d\n", (int)roundf(dst[0] * 255.0f));
                printf("%d\n", (int)roundf(dst[1] * 255.0f));
                printf("%d\n", (int)roundf(dst[2] * 255.0f));
                printf("%d\n", (int)roundf(dst[3] * 255.0f));
                return 0;
            }
            "#,
            "0\n128\n255\n64",
        );
    }

    // === Vector Bitwise Operations (.&, .|, .^) ===

    #[test]
    fn test_u8x16_and_dot() {
        assert_c_interop(
            r#"
export func test_and(a: *u8, b: *u8, out: *mut u8, len: i32) {
    let mut i: i32 = 0
    while i + 16 <= len {
        let va: u8x16 = load(a, i)
        let vb: u8x16 = load(b, i)
        store(out, i, va .& vb)
        i = i + 16
    }
}
"#,
            r#"
#include <stdio.h>
#include <stdint.h>

extern void test_and(const uint8_t*, const uint8_t*, uint8_t*, int);

int main() {
    uint8_t a[16] = {0xFF, 0x0F, 0xAA, 0x55, 1,2,3,4,5,6,7,8,9,10,11,12};
    uint8_t b[16] = {0x0F, 0xFF, 0x55, 0xAA, 255,254,253,252,251,250,249,248,247,246,245,244};
    uint8_t out[16] = {0};
    test_and(a, b, out, 16);
    printf("%d %d %d %d\n", out[0], out[1], out[2], out[3]);
    return 0;
}
"#,
            "15 15 0 0",
        );
    }

    #[test]
    fn test_u8x16_or_dot() {
        assert_c_interop(
            r#"
export func test_or(a: *u8, b: *u8, out: *mut u8, len: i32) {
    let mut i: i32 = 0
    while i + 16 <= len {
        let va: u8x16 = load(a, i)
        let vb: u8x16 = load(b, i)
        store(out, i, va .| vb)
        i = i + 16
    }
}
"#,
            r#"
#include <stdio.h>
#include <stdint.h>

extern void test_or(const uint8_t*, const uint8_t*, uint8_t*, int);

int main() {
    uint8_t a[16] = {0xF0, 0x0F, 0xAA, 0x00, 1,2,3,4,5,6,7,8,9,10,11,12};
    uint8_t b[16] = {0x0F, 0xF0, 0x55, 0xFF, 255,254,253,252,251,250,249,248,247,246,245,244};
    uint8_t out[16] = {0};
    test_or(a, b, out, 16);
    printf("%d %d %d %d\n", out[0], out[1], out[2], out[3]);
    return 0;
}
"#,
            "255 255 255 255",
        );
    }

    #[test]
    fn test_u8x16_xor_dot() {
        assert_c_interop(
            r#"
export func test_xor(a: *u8, b: *u8, out: *mut u8, len: i32) {
    let mut i: i32 = 0
    while i + 16 <= len {
        let va: u8x16 = load(a, i)
        let vb: u8x16 = load(b, i)
        store(out, i, va .^ vb)
        i = i + 16
    }
}
"#,
            r#"
#include <stdio.h>
#include <stdint.h>

extern void test_xor(const uint8_t*, const uint8_t*, uint8_t*, int);

int main() {
    uint8_t a[16] = {0xFF, 0xFF, 0xAA, 0x00, 1,2,3,4,5,6,7,8,9,10,11,12};
    uint8_t b[16] = {0xFF, 0x00, 0x55, 0xFF, 255,254,253,252,251,250,249,248,247,246,245,244};
    uint8_t out[16] = {0};
    test_xor(a, b, out, 16);
    printf("%d %d %d %d\n", out[0], out[1], out[2], out[3]);
    return 0;
}
"#,
            "0 255 255 255",
        );
    }

    #[test]
    fn test_i32x4_and_dot() {
        assert_c_interop(
            r#"
export func test_and_i32(a: *i32, b: *i32, out: *mut i32, len: i32) {
    let mut i: i32 = 0
    while i + 4 <= len {
        let va: i32x4 = load(a, i)
        let vb: i32x4 = load(b, i)
        store(out, i, va .& vb)
        i = i + 4
    }
}
"#,
            r#"
#include <stdio.h>
#include <stdint.h>

extern void test_and_i32(const int32_t*, const int32_t*, int32_t*, int);

int main() {
    int32_t a[4] = {0x0F0F0F0F, -1, 0, 0x12345678};
    int32_t b[4] = {(int32_t)0xF0F0F0F0, 0x7FFFFFFF, -1, 0x0F0F0F0F};
    int32_t out[4] = {0};
    test_and_i32(a, b, out, 4);
    printf("%d %d\n", out[0], out[1]);
    return 0;
}
"#,
            "0 2147483647",
        );
    }

    #[test]
    fn test_i16x8_or_dot() {
        assert_c_interop(
            r#"
export func test_or_i16(a: *i16, b: *i16, out: *mut i16, len: i32) {
    let mut i: i32 = 0
    while i + 8 <= len {
        let va: i16x8 = load(a, i)
        let vb: i16x8 = load(b, i)
        store(out, i, va .| vb)
        i = i + 8
    }
}
"#,
            r#"
#include <stdio.h>
#include <stdint.h>

extern void test_or_i16(const int16_t*, const int16_t*, int16_t*, int);

int main() {
    int16_t a[8] = {0x00F0, 0x0F00, 0, 0, 0, 0, 0, 0};
    int16_t b[8] = {0x000F, 0x00F0, 0, 0, 0, 0, 0, 0};
    int16_t out[8] = {0};
    test_or_i16(a, b, out, 8);
    printf("%d %d\n", out[0], out[1]);
    return 0;
}
"#,
            "255 4080",
        );
    }

    // === Hex and Binary Literals ===

    #[test]
    fn test_hex_literal() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 0xFF
                println(x)
            }
            "#,
            "255",
        );
    }

    #[test]
    fn test_binary_literal() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 0b11110000
                println(x)
            }
            "#,
            "240",
        );
    }

    #[test]
    fn test_hex_u8_splat() {
        assert_c_interop(
            r#"
export func mask_and(data: *u8, out: *mut u8, len: i32) {
    let mask: u8x16 = splat(0x0F)
    let mut i: i32 = 0
    while i + 16 <= len {
        let v: u8x16 = load(data, i)
        store(out, i, v .& mask)
        i = i + 16
    }
}
"#,
            r#"
#include <stdio.h>
#include <stdint.h>

extern void mask_and(const uint8_t*, uint8_t*, int);

int main() {
    uint8_t data[16] = {0xFF, 0xAB, 0x12, 0x00, 0,0,0,0,0,0,0,0,0,0,0,0};
    uint8_t out[16] = {0};
    mask_and(data, out, 16);
    printf("%d %d %d %d\n", out[0], out[1], out[2], out[3]);
    return 0;
}
"#,
            "15 11 2 0",
        );
    }

    #[test]
    fn test_negative_hex_literal() {
        assert_output(
            r#"
            func main() {
                let x: i32 = -0x01
                println(x)
            }
            "#,
            "-1",
        );
    }

    #[test]
    fn test_binary_bitmask() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 0b10101010
                println(x)
            }
            "#,
            "170",
        );
    }

    // === Unary negation ===

    #[test]
    fn test_negate_i32_variable() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 42
                let y: i32 = -x
                println(y)
            }
            "#,
            "-42",
        );
    }

    #[test]
    fn test_negate_f32_variable() {
        assert_output(
            r#"
            func main() {
                let x: f32 = 3.5
                let y: f32 = -x
                println(y)
            }
            "#,
            "-3.5",
        );
    }

    #[test]
    fn test_negate_f64_variable() {
        assert_output(
            r#"
            func main() {
                let x: f64 = 2.25
                let y: f64 = -x
                println(y)
            }
            "#,
            "-2.25",
        );
    }

    #[test]
    fn test_negate_in_expression() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 10
                let y: i32 = 5 + -x
                println(y)
            }
            "#,
            "-5",
        );
    }

    #[test]
    fn test_double_negate() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 7
                let y: i32 = - -x
                println(y)
            }
            "#,
            "7",
        );
    }

    // === sqrt / rsqrt ===

    #[test]
    fn test_sqrt_f32_scalar() {
        assert_output(
            r#"
            func main() {
                let x: f32 = 4.0
                let y: f32 = sqrt(x)
                println(y)
            }
            "#,
            "2",
        );
    }

    #[test]
    fn test_sqrt_f64_scalar() {
        assert_output(
            r#"
            func main() {
                let x: f64 = 9.0
                let y: f64 = sqrt(x)
                println(y)
            }
            "#,
            "3",
        );
    }

    #[test]
    fn test_sqrt_f32x4_vector() {
        assert_output(
            r#"
            func main() {
                let v: f32x4 = splat(16.0)
                let r: f32x4 = sqrt(v)
                println(r[0])
            }
            "#,
            "4",
        );
    }

    #[test]
    fn test_rsqrt_f32_scalar() {
        assert_output(
            r#"
            func main() {
                let x: f32 = 4.0
                let y: f32 = rsqrt(x)
                println(y)
            }
            "#,
            "0.5",
        );
    }

    #[test]
    fn test_rsqrt_f32x4_vector() {
        assert_output(
            r#"
            func main() {
                let v: f32x4 = splat(4.0)
                let r: f32x4 = rsqrt(v)
                println(r[0])
            }
            "#,
            "0.5",
        );
    }

    #[test]
    fn test_sqrt_in_magnitude() {
        assert_output(
            r#"
            func main() {
                let x: f32 = 3.0
                let y: f32 = 4.0
                let mag: f32 = sqrt(x * x + y * y)
                println(mag)
            }
            "#,
            "5",
        );
    }

    // === Type conversions ===

    #[test]
    fn test_to_f32_from_i32() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 42
                let y: f32 = to_f32(x)
                println(y)
            }
            "#,
            "42",
        );
    }

    #[test]
    fn test_to_f64_from_i32() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 100
                let y: f64 = to_f64(x)
                println(y)
            }
            "#,
            "100",
        );
    }

    #[test]
    fn test_to_i32_from_f32() {
        assert_output(
            r#"
            func main() {
                let x: f32 = 3.7
                let y: i32 = to_i32(x)
                println(y)
            }
            "#,
            "3",
        );
    }

    #[test]
    fn test_to_i32_from_f64() {
        assert_output(
            r#"
            func main() {
                let x: f64 = 9.9
                let y: i32 = to_i32(x)
                println(y)
            }
            "#,
            "9",
        );
    }

    #[test]
    fn test_to_f32_from_i64() {
        assert_output(
            r#"
            func main() {
                let x: i64 = 1000
                let y: f32 = to_f32(x)
                println(y)
            }
            "#,
            "1000",
        );
    }

    #[test]
    fn test_to_i64_from_f64() {
        assert_output(
            r#"
            func main() {
                let x: f64 = 123.456
                let y: i64 = to_i64(x)
                println(y)
            }
            "#,
            "123",
        );
    }

    #[test]
    fn test_to_f32_in_expression() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 50
                let width: i32 = 100
                let u: f32 = to_f32(x) / to_f32(width)
                println(u)
            }
            "#,
            "0.5",
        );
    }

    #[test]
    fn test_to_f64_from_f32() {
        assert_output(
            r#"
            func main() {
                let x: f32 = 2.5
                let y: f64 = to_f64(x)
                println(y)
            }
            "#,
            "2.5",
        );
    }

    #[test]
    fn test_to_f32_from_f64() {
        assert_output(
            r#"
            func main() {
                let x: f64 = 1.5
                let y: f32 = to_f32(x)
                println(y)
            }
            "#,
            "1.5",
        );
    }

    // === Combined: the spec example ===

    #[test]
    fn test_spec_magnitude_example() {
        // From EA_V2_SPECIFICATION.md Part 6:
        //   export func magnitude(v: *Vec2) -> f32 {
        //       return sqrt(v.x * v.x + v.y * v.y);
        //   }
        assert_c_interop(
            r#"
            struct Vec2 { x: f32, y: f32 }

            export func magnitude(v: *Vec2) -> f32 {
                return sqrt(v.x * v.x + v.y * v.y)
            }
            "#,
            r#"
                #include <stdio.h>
                typedef struct { float x; float y; } Vec2;
                extern float magnitude(const Vec2*);
                int main() {
                    Vec2 v = {3.0f, 4.0f};
                    printf("%g\n", magnitude(&v));
                    return 0;
                }
            "#,
            "5",
        );
    }

    #[test]
    fn test_combined_sqrt_cast_negate() {
        // Use all three features together
        assert_output(
            r#"
            func main() {
                let ix: i32 = 3
                let iy: i32 = -4
                let fx: f32 = to_f32(ix)
                let fy: f32 = to_f32(iy)
                let dist: f32 = sqrt(fx * fx + fy * fy)
                println(dist)
            }
            "#,
            "5",
        );
    }

    // === u32 / u64 types ===

    #[test]
    fn test_u32_basic() {
        assert_c_interop(
            r#"
            export func add_u32(a: u32, b: u32) -> u32 {
                return a + b
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern uint32_t add_u32(uint32_t, uint32_t);
            int main() {
                printf("%u\n", add_u32(3, 4));
                return 0;
            }
            "#,
            "7",
        );
    }

    #[test]
    fn test_u64_basic() {
        assert_c_interop(
            r#"
            export func add_u64(a: u64, b: u64) -> u64 {
                return a + b
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern uint64_t add_u64(uint64_t, uint64_t);
            int main() {
                printf("%lu\n", add_u64(100000, 200000));
                return 0;
            }
            "#,
            "300000",
        );
    }

    // === Cornell Box ===

    #[test]
    fn test_cornell_box_compiles_and_renders() {
        assert_c_interop(
            include_str!("../demo/cornell_box/cornell.ea"),
            r#"
            #include <stdio.h>
            extern void render(float*, int, int);
            int main() {
                float buf[16 * 16 * 3];
                render(buf, 16, 16);
                int center = (8 * 16 + 8) * 3;
                float r = buf[center];
                float g = buf[center + 1];
                float b = buf[center + 2];
                printf("%d\n", (r > 0.01f || g > 0.01f || b > 0.01f) ? 1 : 0);
                return 0;
            }
            "#,
            "1",
        );
    }

    // === Particle Life fused kernel ===

    #[test]
    fn test_particle_life_fused_step() {
        let result = compile_and_link_with_c(
            r#"
export func particle_life_step(
    px: *mut f32, py: *mut f32,
    vx: *mut f32, vy: *mut f32,
    types: *i32,
    matrix: *f32,
    n: i32, num_types: i32,
    r_max: f32, dt: f32, friction: f32, size: f32
) {
    let mut i: i32 = 0
    while i < n {
        let xi: f32 = px[i]
        let yi: f32 = py[i]
        let ti: i32 = types[i]
        let mut fx: f32 = 0.0
        let mut fy: f32 = 0.0

        let r_max2: f32 = r_max * r_max
        let mut j: i32 = 0
        while j < n {
            let dx: f32 = px[j] - xi
            let dy: f32 = py[j] - yi
            let dist2: f32 = dx * dx + dy * dy
            if dist2 > 0.0 {
                if dist2 < r_max2 {
                    let dist: f32 = sqrt(dist2)
                    let strength: f32 = matrix[ti * num_types + types[j]]
                    let force: f32 = strength * (1.0 - dist / r_max)
                    fx = fx + force * dx / dist
                    fy = fy + force * dy / dist
                }
            }
            j = j + 1
        }

        vx[i] = (vx[i] + fx * dt) * friction
        vy[i] = (vy[i] + fy * dt) * friction
        px[i] = px[i] + vx[i] * dt
        py[i] = py[i] + vy[i] * dt

        let cur_px: f32 = px[i]
        let cur_py: f32 = py[i]
        if cur_px < 0.0 { px[i] = cur_px + size }
        if cur_px >= size { px[i] = cur_px - size }
        if cur_py < 0.0 { py[i] = cur_py + size }
        if cur_py >= size { py[i] = cur_py - size }

        i = i + 1
    }
}
"#,
            r#"
#include <stdio.h>
#include <math.h>

extern void particle_life_step(
    float* px, float* py,
    float* vx, float* vy,
    int* types,
    float* matrix,
    int n, int num_types,
    float r_max, float dt, float friction, float size
);

int main() {
    float px[2] = {0.0f, 50.0f};
    float py[2] = {0.0f, 0.0f};
    float vx[2] = {0.0f, 0.0f};
    float vy[2] = {0.0f, 0.0f};
    int types[2] = {0, 0};
    float matrix[1] = {1.0f};

    particle_life_step(px, py, vx, vy, types, matrix,
                       2, 1, 100.0f, 1.0f, 0.5f, 1000.0f);

    /* Particle 0 expected:
       dx=50, dy=0, dist2=2500, r_max2=10000 -> in range
       dist=50, strength=1.0, force=1.0*(1-50/100)=0.5
       fx=0.5*50/50=0.5, fy=0
       vx=(0+0.5*1.0)*0.5=0.25, vy=(0+0)*0.5=0
       px=0+0.25*1.0=0.25, py=0+0*1.0=0 */
    int ok = 1;
    if (fabsf(px[0] - 0.25f) > 0.001f) ok = 0;
    if (fabsf(py[0] - 0.0f) > 0.001f) ok = 0;
    if (fabsf(vx[0] - 0.25f) > 0.001f) ok = 0;
    if (fabsf(vy[0] - 0.0f) > 0.001f) ok = 0;
    printf("%d\n", ok);
    return 0;
}
"#,
        );
        assert_eq!(result.stdout.trim(), "1");
    }

    // === Particle Life unfused kernels ===

    #[test]
    fn test_particle_life_unfused_step() {
        let result = compile_and_link_with_c(
            r#"
export func compute_forces(
    px: *f32, py: *f32,
    types: *i32,
    matrix: *f32,
    fx: *mut f32, fy: *mut f32,
    n: i32, num_types: i32,
    r_max: f32
) {
    let r_max2: f32 = r_max * r_max
    let mut i: i32 = 0
    while i < n {
        let xi: f32 = px[i]
        let yi: f32 = py[i]
        let ti: i32 = types[i]
        let mut sum_fx: f32 = 0.0
        let mut sum_fy: f32 = 0.0

        let mut j: i32 = 0
        while j < n {
            let dx: f32 = px[j] - xi
            let dy: f32 = py[j] - yi
            let dist2: f32 = dx * dx + dy * dy
            if dist2 > 0.0 {
                if dist2 < r_max2 {
                    let dist: f32 = sqrt(dist2)
                    let strength: f32 = matrix[ti * num_types + types[j]]
                    let force: f32 = strength * (1.0 - dist / r_max)
                    sum_fx = sum_fx + force * dx / dist
                    sum_fy = sum_fy + force * dy / dist
                }
            }
            j = j + 1
        }

        fx[i] = sum_fx
        fy[i] = sum_fy
        i = i + 1
    }
}

export func update_velocities(
    vx: *mut f32, vy: *mut f32,
    fx: *f32, fy: *f32,
    n: i32, dt: f32, friction: f32
) {
    let mut i: i32 = 0
    while i < n {
        vx[i] = (vx[i] + fx[i] * dt) * friction
        vy[i] = (vy[i] + fy[i] * dt) * friction
        i = i + 1
    }
}

export func update_positions(
    px: *mut f32, py: *mut f32,
    vx: *f32, vy: *f32,
    n: i32, dt: f32, size: f32
) {
    let mut i: i32 = 0
    while i < n {
        px[i] = px[i] + vx[i] * dt
        py[i] = py[i] + vy[i] * dt

        let cur_px: f32 = px[i]
        let cur_py: f32 = py[i]
        if cur_px < 0.0 { px[i] = cur_px + size }
        if cur_px >= size { px[i] = cur_px - size }
        if cur_py < 0.0 { py[i] = cur_py + size }
        if cur_py >= size { py[i] = cur_py - size }

        i = i + 1
    }
}
"#,
            r#"
#include <stdio.h>
#include <math.h>

extern void compute_forces(
    float* px, float* py,
    int* types,
    float* matrix,
    float* fx, float* fy,
    int n, int num_types,
    float r_max
);

extern void update_velocities(
    float* vx, float* vy,
    float* fx, float* fy,
    int n, float dt, float friction
);

extern void update_positions(
    float* px, float* py,
    float* vx, float* vy,
    int n, float dt, float size
);

int main() {
    float px[2] = {0.0f, 50.0f};
    float py[2] = {0.0f, 0.0f};
    float vx[2] = {0.0f, 0.0f};
    float vy[2] = {0.0f, 0.0f};
    int types[2] = {0, 0};
    float matrix[1] = {1.0f};
    float fx[2] = {0.0f, 0.0f};
    float fy[2] = {0.0f, 0.0f};

    compute_forces(px, py, types, matrix, fx, fy,
                   2, 1, 100.0f);
    update_velocities(vx, vy, fx, fy,
                      2, 1.0f, 0.5f);
    update_positions(px, py, vx, vy,
                     2, 1.0f, 1000.0f);

    /* Both particles see original positions (unfused):
       Particle 0: dx=50, dist=50, strength=1.0, force=0.5, fx=0.5
       Particle 1: dx=-50, dist=50, strength=1.0, force=0.5, fx=-0.5
       After update_velocities: vx[0]=0.25, vx[1]=-0.25
       After update_positions: px[0]=0.25, px[1]=49.75 */
    int ok = 1;
    if (fabsf(px[0] - 0.25f) > 0.001f) ok = 0;
    if (fabsf(px[1] - 49.75f) > 0.001f) ok = 0;
    if (fabsf(vx[0] - 0.25f) > 0.001f) ok = 0;
    if (fabsf(vx[1] - (-0.25f)) > 0.001f) ok = 0;
    if (fabsf(vy[0]) > 0.001f) ok = 0;
    if (fabsf(vy[1]) > 0.001f) ok = 0;
    if (fabsf(py[0]) > 0.001f) ok = 0;
    if (fabsf(py[1]) > 0.001f) ok = 0;

    if (ok) {
        printf("PASS\n");
    } else {
        printf("FAIL px[0]=%f px[1]=%f vx[0]=%f vx[1]=%f\n",
               px[0], px[1], vx[0], vx[1]);
    }
    return 0;
}
"#,
        );
        assert_eq!(result.stdout.trim(), "PASS");
    }
}
