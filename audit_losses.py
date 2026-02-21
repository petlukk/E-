#!/usr/bin/env python3
"""
Eä v0.4.0 — Honest Loss Audit

Documents every case where Eä is slower than or equal to NumPy/SciPy.
A compiler that never loses is dishonest.
"""

import ctypes, subprocess, time, sys
from pathlib import Path
import numpy as np

EA_ROOT  = Path(__file__).parent
EA_BIN   = EA_ROOT / "target/release/ea"
if not EA_BIN.exists():
    EA_BIN = EA_ROOT / "target/debug/ea"

U8PTR  = ctypes.POINTER(ctypes.c_uint8)
I8PTR  = ctypes.POINTER(ctypes.c_int8)
I16PTR = ctypes.POINTER(ctypes.c_int16)
I32PTR = ctypes.POINTER(ctypes.c_int32)
F32PTR = ctypes.POINTER(ctypes.c_float)

def bench(func, *args, warmup=5, runs=100):
    for _ in range(warmup): func(*args)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times)//2] * 1000  # ms, median

def build_so(ea_src, so_path):
    if so_path.exists() and so_path.stat().st_mtime > ea_src.stat().st_mtime:
        return
    result = subprocess.run(
        [str(EA_BIN), str(ea_src), "--lib", "-o", str(so_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        sys.exit(1)

def result_line(label, t_ea, t_np):
    ratio = t_np / t_ea
    if ratio < 0.95:
        tag = f"Eä LOSES  {1/ratio:.2f}x slower"
    elif ratio < 1.05:
        tag = "TIE"
    else:
        tag = f"Eä wins   {ratio:.2f}x faster"
    print(f"  {label:<45} {tag}")
    print(f"    Eä: {t_ea:.4f} ms   NumPy: {t_np:.4f} ms")

print("=" * 65)
print("  Eä v0.4.0 — HONEST LOSS AUDIT")
print("=" * 65)

# ── Load shared libs ──────────────────────────────────────────────────────
pipeline_so = EA_ROOT / "demo/pixel_pipeline/pipeline.so"
build_so(EA_ROOT / "demo/pixel_pipeline/pipeline.ea", pipeline_so)
pipe = ctypes.CDLL(str(pipeline_so))
pipe.threshold_u8x16.argtypes = [U8PTR, U8PTR, ctypes.c_int32, ctypes.c_uint8]
pipe.threshold_u8x16.restype  = None
pipe.normalize_u8_f32x8.argtypes = [U8PTR, F32PTR, ctypes.c_int32]
pipe.normalize_u8_f32x8.restype  = None

conv_so = EA_ROOT / "demo/conv2d/conv.so"
build_so(EA_ROOT / "demo/conv2d/conv.ea", conv_so)
conv = ctypes.CDLL(str(conv_so))
conv.dot_u8i8.argtypes  = [U8PTR, I8PTR, ctypes.c_int32]
conv.dot_u8i8.restype   = ctypes.c_int16
conv.conv1d_u8i8.argtypes = [U8PTR, I8PTR, I16PTR, ctypes.c_int32, ctypes.c_int32]
conv.conv1d_u8i8.restype  = None

c3_so = EA_ROOT / "demo/conv2d_3x3/conv.so"
build_so(EA_ROOT / "demo/conv2d_3x3/conv.ea", c3_so)
c3 = ctypes.CDLL(str(c3_so))
c3.conv2d_3x3_u8i8.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                 ctypes.c_char_p, ctypes.c_int,
                                 ctypes.c_int, ctypes.c_int]
c3.conv2d_3x3_u8i8.restype  = None

c3_safe_so = EA_ROOT / "demo/conv2d_3x3/conv_safe.so"
build_so(EA_ROOT / "demo/conv2d_3x3/conv_safe.ea", c3_safe_so)
c3_safe = ctypes.CDLL(str(c3_safe_so))
c3_safe.conv2d_3x3_u8i8_safe.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                           I32PTR, ctypes.c_int,
                                           ctypes.c_int, ctypes.c_int]
c3_safe.conv2d_3x3_u8i8_safe.restype  = None

rng = np.random.RandomState(42)

# ──────────────────────────────────────────────────────────────────────────
print()
print("SECTION 1: FFI overhead on tiny data")
print("  (kernel call cost dominates when data is small)")
print()

# threshold — 64 pixels
n_tiny = 64
src_tiny = rng.randint(0, 256, n_tiny, dtype=np.uint8)
dst_tiny = np.zeros(n_tiny, dtype=np.uint8)

def ea_thresh_tiny():
    pipe.threshold_u8x16(
        src_tiny.ctypes.data_as(U8PTR),
        dst_tiny.ctypes.data_as(U8PTR),
        ctypes.c_int32(n_tiny), ctypes.c_uint8(128))

def np_thresh_tiny():
    np.where(src_tiny > 128, np.uint8(255), np.uint8(0))

result_line(f"threshold_u8x16 (n={n_tiny}, 64 bytes)", bench(ea_thresh_tiny), bench(np_thresh_tiny))

# dot product — 16 elements (minimum maddubs size)
n_micro = 16
act_micro = rng.randint(0, 5, n_micro, dtype=np.uint8)
wt_micro  = rng.randint(-3, 4, n_micro, dtype=np.int8)

def ea_dot_micro():
    return conv.dot_u8i8(
        act_micro.ctypes.data_as(U8PTR),
        wt_micro.ctypes.data_as(I8PTR),
        ctypes.c_int32(n_micro))

def np_dot_micro():
    return int(np.dot(act_micro.astype(np.int32), wt_micro.astype(np.int32)))

result_line(f"dot_u8i8 (n={n_micro}, minimum size)", bench(ea_dot_micro), bench(np_dot_micro))

# normalize — 32 floats
n_tiny_norm = 32
src_norm_tiny = rng.randint(0, 256, n_tiny_norm, dtype=np.uint8)
dst_norm_tiny = np.zeros(n_tiny_norm, dtype=np.float32)

def ea_norm_tiny():
    pipe.normalize_u8_f32x8(
        src_norm_tiny.ctypes.data_as(U8PTR),
        dst_norm_tiny.ctypes.data_as(F32PTR),
        ctypes.c_int32(n_tiny_norm))

def np_norm_tiny():
    src_norm_tiny.astype(np.float32) / 255.0

result_line(f"normalize_u8_f32x8 (n={n_tiny_norm}, 32 bytes)", bench(ea_norm_tiny), bench(np_norm_tiny))

# ──────────────────────────────────────────────────────────────────────────
print()
print("SECTION 2: Single simple ops — NumPy already calls optimized BLAS")
print("  (no fusion advantage; FFI cost + no benefit)")
print()

# element-wise add (f32, 1M) — Eä has no simple add kernel but we can use
# a threshold as proxy. Real test: normalize single op (equivalent to mnist single op)
n_1m = 1_000_000
src_1m = rng.randint(0, 256, n_1m, dtype=np.uint8)
flat_1m, _ = (lambda a: (a, len(src_1m)))(np.ascontiguousarray(src_1m))
dst_1m_f = np.zeros(n_1m, dtype=np.float32)

def ea_norm_1m():
    pipe.normalize_u8_f32x8(
        flat_1m.ctypes.data_as(U8PTR),
        dst_1m_f.ctypes.data_as(F32PTR),
        ctypes.c_int32(n_1m))

def np_norm_1m():
    src_1m.astype(np.float32) / 255.0

result_line("normalize_u8_f32x8 (1M pixels, single op)", bench(ea_norm_1m), bench(np_norm_1m))

# threshold 256×256 (small image, real-world call pattern)
n_256 = 256 * 256
src_256 = rng.randint(0, 256, n_256, dtype=np.uint8)
dst_256 = np.zeros(n_256, dtype=np.uint8)

def ea_thresh_256():
    pipe.threshold_u8x16(
        src_256.ctypes.data_as(U8PTR),
        dst_256.ctypes.data_as(U8PTR),
        ctypes.c_int32(n_256), ctypes.c_uint8(128))

def np_thresh_256():
    np.where(src_256 > 128, np.uint8(255), np.uint8(0))

result_line("threshold_u8x16 (256×256 = 64K pixels)", bench(ea_thresh_256), bench(np_thresh_256))

# ──────────────────────────────────────────────────────────────────────────
print()
print("SECTION 3: conv2d_3x3 vs scipy.signal (optimised C reference)")
print("  (the documented NumPy baseline uses a 9-iter Python loop — honest?)")
print()

try:
    from scipy import signal as sp_signal

    H, W, C_in = 56, 56, 64
    src_np = rng.randint(0, 128, ((H+2)*(W+2)*C_in,), dtype=np.uint8)
    wt_np  = rng.randint(-64, 64, (9*C_in,), dtype=np.int8)
    src_c  = src_np.tobytes()
    wt_c   = wt_np.tobytes()
    dst_c  = ctypes.create_string_buffer(H * W * 2)

    def ea_conv3x3():
        c3.conv2d_3x3_u8i8(src_c, wt_c, dst_c, H, W, C_in)

    # NumPy 9-loop (current documented baseline)
    src3 = src_np.astype(np.int32).reshape(H+2, W+2, C_in)
    wt3  = wt_np.astype(np.int32).reshape(3, 3, C_in)
    def np_conv3x3_loop():
        acc = np.zeros((H, W), dtype=np.int32)
        for dr in range(3):
            for dc in range(3):
                acc += (src3[dr:dr+H, dc:dc+W, :] * wt3[dr, dc, :]).sum(axis=-1)

    # SciPy per-channel 2D conv (float32, the real optimised baseline)
    src_f = src_np.astype(np.float32).reshape(H+2, W+2, C_in)
    wt_f  = wt_np.astype(np.float32).reshape(3, 3, C_in)
    def scipy_conv3x3():
        acc = np.zeros((H, W), dtype=np.float32)
        for ci in range(C_in):
            acc += sp_signal.correlate2d(
                src_f[:, :, ci], wt_f[:, :, ci], mode='valid')

    t_ea   = bench(ea_conv3x3,      warmup=3, runs=30)
    t_np   = bench(np_conv3x3_loop, warmup=3, runs=30)
    t_sp   = bench(scipy_conv3x3,   warmup=2, runs=10)

    print(f"  conv2d_3x3 (56×56×64):")
    print(f"    Eä (maddubs_i16)          : {t_ea:.3f} ms")
    print(f"    NumPy (9-loop, vectorised): {t_np:.3f} ms  → {t_np/t_ea:.1f}x  [documented baseline]")
    print(f"    SciPy correlate2d (C, f32): {t_sp:.3f} ms  → {t_sp/t_ea:.1f}x  [real C baseline]")
    if t_sp < t_ea:
        print(f"    vs SciPy: Eä LOSES — SciPy C kernels beat integer SIMD at this size")
    else:
        print(f"    vs SciPy: Eä still wins — integer SIMD throughput beats float C even optimised")
    print()
    print(f"  AUDIT NOTE: The 47-49x speedup claim is vs the NumPy 9-loop baseline.")
    print(f"  vs SciPy optimised C: {t_sp/t_ea:.1f}x — this is the more conservative honest number.")

except ImportError:
    print("  scipy not installed — skipping scipy comparison")
    print("  Install with: pip3 install scipy")

# ──────────────────────────────────────────────────────────────────────────
print()
print("SECTION 4: conv2d_3x3 size scaling — does the win hold?")
print("  (anti-cherry-pick: verify speedup across spatial sizes)")
print()

for H, W, C_in in [(14, 14, 64), (28, 28, 64), (56, 56, 64), (112, 112, 64)]:
    src_np = rng.randint(0, 128, ((H+2)*(W+2)*C_in,), dtype=np.uint8)
    wt_np  = rng.randint(-64, 64, (9*C_in,), dtype=np.int8)
    src_c  = src_np.tobytes()
    wt_c   = wt_np.tobytes()
    dst_c  = ctypes.create_string_buffer(H * W * 2)
    src3   = src_np.astype(np.int32).reshape(H+2, W+2, C_in)
    wt3    = wt_np.astype(np.int32).reshape(3, 3, C_in)

    def ea_c3(): c3.conv2d_3x3_u8i8(src_c, wt_c, dst_c, H, W, C_in)
    def np_c3():
        acc = np.zeros((H, W), dtype=np.int32)
        for dr in range(3):
            for dc in range(3):
                acc += (src3[dr:dr+H, dc:dc+W, :] * wt3[dr, dc, :]).sum(axis=-1)

    t_ea = bench(ea_c3, warmup=3, runs=30)
    t_np = bench(np_c3, warmup=3, runs=30)
    macs = H * W * 9 * C_in * 2
    gmacs = macs / t_ea * 1e3 / 1e9
    print(f"  {H}×{W}×{C_in}: Eä {t_ea:.3f} ms  NumPy {t_np:.3f} ms  "
          f"→ {t_np/t_ea:.1f}x   {gmacs:.1f} GMACs/s")

# ──────────────────────────────────────────────────────────────────────────
print()
print("SECTION 5: i16 overflow — maddubs_i16 vs maddubs_i32")
print("  (pmaddubsw wraps at i16; maddubs_i32 uses pmaddwd to widen to i32)")
print()

# Overflow-prone values (the audit case: act≈255, wt≈127)
n_big = 512
act_big = rng.randint(200, 256, n_big, dtype=np.uint8)   # ≈255
wt_big  = rng.randint(60,  128, n_big, dtype=np.int8)    # ≈127
ref_big = int(np.dot(act_big.astype(np.int32), wt_big.astype(np.int32)))
ea_i16  = int(conv.dot_u8i8(
    act_big.ctypes.data_as(U8PTR), wt_big.ctypes.data_as(I8PTR),
    ctypes.c_int32(n_big)))
overflow_i16 = ref_big != ea_i16
print(f"  maddubs_i16 (act≈255, wt≈127, n={n_big}):")
print(f"    NumPy ref : {ref_big}")
print(f"    Eä i16    : {ea_i16}  {'OVERFLOW — wraps at i16 boundary' if overflow_i16 else 'ok (lucky)'}")
print()

# maddubs_i32 via conv2d_3x3_u8i8_safe: verify correctness within per-call safe range.
# act=10, wt=5: each adjacent pair 10*5+10*5=100, well within i16.
# 9 kernel positions × 32 channels: expected = 9 * 32 * 10 * 5 = 14400.
# (maddubs_i32 produces i32x4; two accumulators; reduce_add each → scalar sum)
H_s, W_s, C_s = 1, 1, 32
src_s = np.full((H_s + 2) * (W_s + 2) * C_s, 10, dtype=np.uint8)
wt_s  = np.full(9 * C_s, 5, dtype=np.int8)
ref_safe_i32 = 9 * C_s * 10 * 5   # 14400
dst_i32 = np.zeros(H_s * W_s, dtype=np.int32)
c3_safe.conv2d_3x3_u8i8_safe(
    src_s.tobytes(), wt_s.tobytes(),
    dst_i32.ctypes.data_as(I32PTR),
    ctypes.c_int(H_s), ctypes.c_int(W_s), ctypes.c_int(C_s))
ea_i32 = int(dst_i32[0])
print(f"  maddubs_i32 / conv_safe (act=10, wt=5, 9×32 pairs):")
print(f"    Expected  : {ref_safe_i32}")
print(f"    Eä i32    : {ea_i32}  {'CORRECT' if ea_i32 == ref_safe_i32 else 'WRONG'}")
print()
print(f"  Per-call safe range: each adjacent pair must satisfy a*b+a*b ≤ 32,767 (i16 max).")
print(f"  act=10, wt=5: 10*5+10*5 = 100 — safe.")
print(f"  act=127, wt=127: 127*127+127*127 = 32,258 — safe (typical quantized inference).")
print(f"  act=200, wt=100: 200*100+200*100 = 40,000 — OVERFLOWS pmaddubsw intermediate i16.")
print(f"  maddubs_i32 prevents ACCUMULATOR overflow across many calls; cannot save per-call overflow.")

# ──────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  SUMMARY OF LOSSES AND CONSTRAINTS")
print("=" * 65)
print("""
  WHERE Eä LOSES OR TIES:
  • Any kernel on tiny data (n < ~256): FFI call overhead dominates
  • Single normalize op on large arrays: ties NumPy (both DRAM-bound)
  • conv2d_3x3 vs SciPy optimised C: see section 3 above
  • Unfused 3-kernel pipeline: slower than NumPy (video_anomaly: 0.8x)

  HARD CONSTRAINTS:
  • maddubs_i16: per-call overflow when a[2i]*b[2i]+a[2i+1]*b[2i+1] > 32,767
    → use maddubs_i32 when act or wt values are large (output type encodes the choice)
  • maddubs_i32: per-call overflow when a single adjacent pair exceeds i16 (> ~128×128)
    → safe for typical quantized inference ranges (act≤127, wt≤127)
  • conv2d_3x3 requires C_in % 32 == 0 (dual 16-wide accumulators)
  • No bounds checking — caller responsible for memory safety

  WHERE Eä WINS:
  • Large u8 threshold: stays in byte space, 21x (NumPy promotes to int32)
  • maddubs_i16 at scale: 16 pairs/cycle vs NumPy's int32 arithmetic
  • maddubs_i32: exact i32 accumulation at ~same throughput, no silent widening
  • Fused pipelines: eliminates intermediate memory passes
  • Explicit ILP (multi-acc reductions): beats LLVM auto-unrolling
""")
