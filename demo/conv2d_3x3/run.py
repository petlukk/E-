"""
Phase D demo: 3x3 int8 conv2d vs numpy reference
Input: (H+2) x (W+2) x C_in padded uint8 image
Kernel: 3x3 x C_in int8 weights
Output: H x W int16 accumulation

C_in must be a multiple of 32 (dual-accumulator kernel).
"""
import ctypes, os, subprocess, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EA_SRC = os.path.join(HERE, "conv.ea")
SO_PATH = os.path.join(HERE, "conv.so")
EA_BIN = os.path.join(HERE, "../../target/release/ea")
if not os.path.exists(EA_BIN):
    EA_BIN = os.path.join(HERE, "../../target/debug/ea")

# --- compile ---
print("Compiling conv.ea ...", flush=True)
subprocess.run(
    [EA_BIN, EA_SRC, "--lib", "-o", SO_PATH],
    check=True, capture_output=True
)
lib = ctypes.CDLL(SO_PATH)
lib.conv2d_3x3_u8i8.argtypes = [
    ctypes.c_char_p,   # src  *u8
    ctypes.c_char_p,   # wt   *i8
    ctypes.c_char_p,   # dst  *mut i16
    ctypes.c_int,      # H
    ctypes.c_int,      # W
    ctypes.c_int,      # C_in
]
lib.conv2d_3x3_u8i8.restype = None

# --- problem size ---
H, W, C_in = 56, 56, 64   # typical first conv layer output spatial
assert C_in % 32 == 0, "C_in must be multiple of 32"

# padded input: (H+2) x (W+2) x C_in
src_np = np.random.randint(0, 128, ((H+2)*(W+2)*C_in,), dtype=np.uint8)
wt_np  = np.random.randint(-64, 64, (9*C_in,), dtype=np.int8)
dst_ea = np.zeros(H * W, dtype=np.int16)

src_c = src_np.tobytes()
wt_c  = wt_np.tobytes()
dst_c = ctypes.create_string_buffer(dst_ea.nbytes)

# --- numpy reference ---
def conv_numpy(src_flat, wt_flat, H, W, C_in):
    stride = (W + 2) * C_in
    src = src_flat.astype(np.int32).reshape((H+2)*(W+2), C_in)
    wt  = wt_flat.astype(np.int32).reshape(9, C_in)
    out = np.zeros((H, W), dtype=np.int32)
    for dr in range(3):
        for dc in range(3):
            # src rows for this (dr,dc) offset: rows [dr..dr+H], cols [dc..dc+W]
            src_patch = src[(np.arange(H)[:,None]+dr)*(W+2) + (np.arange(W)[None,:]+dc)]  # H×W×C
            # wait, simpler:
            rows = np.arange(H)
            cols = np.arange(W)
            # src_off = (row+dr)*(W+2)*C_in + (col+dc)*C_in
            # flatten manually
            pass
    # Simpler: reshape src to (H+2, W+2, C_in)
    src3 = src_flat.astype(np.int32).reshape(H+2, W+2, C_in)
    wt3  = wt_flat.astype(np.int32).reshape(3, 3, C_in)
    acc  = np.zeros((H, W), dtype=np.int32)
    for dr in range(3):
        for dc in range(3):
            acc += (src3[dr:dr+H, dc:dc+W, :] * wt3[dr, dc, :]).sum(axis=-1)
    return acc.astype(np.int16)

ref = conv_numpy(src_np, wt_np, H, W, C_in)

# --- correctness ---
lib.conv2d_3x3_u8i8(src_c, wt_c, dst_c, H, W, C_in)
dst_ea_arr = np.frombuffer(dst_c.raw, dtype=np.int16)
ref_flat = ref.flatten()
if np.array_equal(dst_ea_arr, ref_flat):
    print("Correctness: PASS ✓")
else:
    diff = np.abs(dst_ea_arr.astype(np.int32) - ref_flat.astype(np.int32))
    print(f"Correctness: FAIL — max diff {diff.max()}, mismatches {(diff>0).sum()}/{H*W}")
    print(f"  first few ea:  {dst_ea_arr[:8]}")
    print(f"  first few ref: {ref_flat[:8]}")

# --- benchmark ---
WARMUP, RUNS = 5, 50

for _ in range(WARMUP):
    lib.conv2d_3x3_u8i8(src_c, wt_c, dst_c, H, W, C_in)
ea_times = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    lib.conv2d_3x3_u8i8(src_c, wt_c, dst_c, H, W, C_in)
    ea_times.append(time.perf_counter() - t0)

for _ in range(WARMUP):
    conv_numpy(src_np, wt_np, H, W, C_in)
np_times = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    conv_numpy(src_np, wt_np, H, W, C_in)
    np_times.append(time.perf_counter() - t0)

ea_ms  = np.median(ea_times)  * 1000
np_ms  = np.median(np_times)  * 1000
ea_std = np.std(ea_times)     * 1000
np_std = np.std(np_times)     * 1000

print(f"\nEä  conv2d_3x3 ({H}×{W}×{C_in}): {ea_ms:.3f} ms ± {ea_std:.3f}")
print(f"NumPy reference              : {np_ms:.3f} ms ± {np_std:.3f}")
print(f"Speedup: {np_ms/ea_ms:.2f}x")
print(f"\nMACs: {H*W*9*C_in*2:,}")
mac_per_s = (H*W*9*C_in*2) / ea_ms * 1e3
print(f"Eä throughput: {mac_per_s/1e9:.2f} GMACs/s")
