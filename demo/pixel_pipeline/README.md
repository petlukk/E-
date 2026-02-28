# Pixel Pipeline — Eä Demo

Two kernels that every vision pipeline starts with, implemented in u8x16 SIMD.
No float conversion for the threshold path. Direct u8 → f32 widening for normalize.

## Kernels

**`threshold_u8x16`** — binary segmentation in uint8 space.
Loads 16 pixels per iteration, applies `select(chunk .> thresh, 255, 0)`.
No conversion to float. Operates on raw camera output directly.

**`normalize_u8_f32x8`** — load uint8, widen to float, scale by 1/255.
Uses `widen_u8_f32x4` to convert the lower and upper halves of a u8x16 chunk
to f32x4, then multiplies by 0.00392156863 (1/255). Produces 8 f32 values per iteration.

## Results

AMD Ryzen 7 1700 (Zen 1, AVX2). 16MP image (4096×4096). 100 runs, median time.

```
threshold_u8x16:
  NumPy   (np.where)     :  61.31 ms  ±1.94
  Eä                     :   2.96 ms  ±0.18   20.7x faster

normalize_u8_f32x8:
  NumPy   (astype float) :  49.26 ms  ±2.08
  Eä                     :  23.36 ms  ±1.14    2.1x faster
```

Correctness: verified byte-for-byte against NumPy reference for both kernels.

## The kernels

```
export func threshold_u8x16(src: *u8, dst: *mut u8, n: i32, thresh: u8) {
    let mut i: i32 = 0
    let t: u8x16 = splat(thresh)
    let ff: u8x16 = splat(255)
    let zero: u8x16 = splat(0)
    while i < n {
        let chunk: u8x16 = load(src, i)
        let result: u8x16 = select(chunk .> t, ff, zero)
        store(dst, i, result)
        i = i + 16
    }
}

export func normalize_u8_f32x8(src: *u8, dst: *mut f32, n: i32) {
    let mut i: i32 = 0
    let scale: f32x4 = splat(0.00392156863)
    while i < n {
        let chunk: u8x16 = load(src, i)
        let lo: f32x4 = widen_u8_f32x4(chunk)
        store(dst, i, lo .* scale)
        let shifted: u8x16 = shuffle(chunk, [4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15])
        let hi: f32x4 = widen_u8_f32x4(shifted)
        store(dst, i + 4, hi .* scale)
        i = i + 8
    }
}
```

## Why is threshold so fast?

The threshold kernel stays entirely in u8 space — no widening, no float math.
`select(chunk .> t, ff, zero)` compiles to `vpcmpgtb` + `vpand` — 2 instructions
for 16 pixels. NumPy converts to float internally for comparison, adding 4× the memory traffic.

## How to run

```bash
# Build compiler (once)
cargo build --features=llvm --release

# Run demo
python3 demo/pixel_pipeline/run.py
```

## How it works

The script generates a synthetic 16MP image, compiles `pipeline.ea` to `pipeline.so`,
and calls `threshold_u8x16` and `normalize_u8_f32x8` via ctypes. Results verified
against NumPy before timing.

```bash
ea pipeline.ea --lib   # → pipeline.so
```

```python
lib = ctypes.CDLL("./pipeline.so")
lib.threshold_u8x16(src_ptr, dst_ptr, n, ctypes.c_uint8(128))
lib.normalize_u8_f32x8(src_ptr, dst_ptr, n)
```
