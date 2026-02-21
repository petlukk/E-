# Conv2d (dot/1d) — Eä Demo

Quantized integer SIMD building blocks using `maddubs` (SSSE3 `pmaddubsw`).
Unsigned activations × signed weights → i16 accumulation.

## Kernels

**`dot_u8i8`** — dot product of n uint8 activations × n int8 weights → i16 scalar.
Uses `maddubs` to compute 16 pairwise multiply-adds per instruction.
`n` must be a multiple of 16.

**`conv1d_u8i8`** — 1-D convolution: `output[j] = dot(src[j..j+k], weights[0..k])`.
Kernel length `k` must be a multiple of 16. Output type: i16.

## Results

AMD Ryzen 7 1700 (Zen 1, SSSE3/AVX2). 100 runs, median time.

```
dot_u8i8 (n=1024):
  NumPy   (np.dot, float32)  :  ~12 µs
  Eä      (maddubs)          :  ~2 µs    5.9x faster

conv1d_u8i8 (n=4096, k=16):
  NumPy   (np.convolve)      :  ~180 µs
  Eä      (maddubs)          :  ~60 µs   3.0x faster
```

Correctness: verified against NumPy integer reference for both kernels.

## The kernels

```
export func dot_u8i8(act: *u8, wt: *i8, n: i32) -> i16 {
    let mut acc0: i16x8 = splat(0)
    let mut i: i32 = 0
    while i < n {
        let a0: u8x16 = load(act, i)
        let b0: i8x16 = load(wt, i)
        acc0 = acc0 .+ maddubs_i16(a0, b0)
        i = i + 16
    }
    return reduce_add(acc0)
}

export func conv1d_u8i8(src: *u8, wt: *i8, dst: *mut i16, n: i32, k: i32) {
    let mut j: i32 = 0
    while j < n {
        let mut acc: i16x8 = splat(0)
        let mut ki: i32 = 0
        while ki < k {
            let a: u8x16 = load(src, j + ki)
            let b: i8x16 = load(wt, ki)
            acc = acc .+ maddubs_i16(a, b)
            ki = ki + 16
        }
        dst[j] = reduce_add(acc)
        j = j + 1
    }
}
```

## What is maddubs?

`maddubs_i16(u8x16, i8x16) -> i16x8` maps to the SSSE3 instruction `pmaddubsw`:

```
result[lane] = clamp(a[2*lane] * b[2*lane] + a[2*lane+1] * b[2*lane+1], -32768, 32767)
```

16 unsigned × signed multiply-adds in one instruction. This is the innermost
operation of TFLite and ONNX Runtime int8 GEMM kernels. The saturation on
addition is hardware-implemented — no overhead.

## Overflow note

The i16 accumulator can hold ±32767. With large n and large weight values
(up to 127 × 255 × 2 per lane per iteration), accumulator overflow is possible.
Keep n × max_weight × max_activation within i16 range, or use partial sums
with i32 expansion in the caller.

## How to run

```bash
# Build compiler (once)
cargo build --features=llvm --release

# Run demo
python3 demo/conv2d/run.py
```
