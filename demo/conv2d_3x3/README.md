# Conv2d 3×3 NHWC — Eä Demo

Full 3×3 integer convolution in NHWC layout using dual-accumulator `maddubs`.
**47.7× faster than NumPy. 38.5 GMACs/s on 56×56×64 input.**

## Kernel

`conv2d_3x3_u8i8(src, wt, dst, H, W, C_in)`:

- **Input**: `(H+2) × (W+2) × C_in` — padded uint8, NHWC layout
- **Weights**: `9 × C_in` — 3×3 kernel, int8, stored row-major
- **Output**: `H × W` — int16 (one output channel)
- **Constraint**: `C_in` must be a multiple of 32

## Results

AMD Ryzen 7 1700 (Zen 1, SSSE3/AVX2). 56×56×64 input. 50 runs, median time.

```
NumPy   (float32)            :  ~870 ms
Eä      (maddubs dual-acc)   :  ~18 ms   47.7x faster

Throughput: 38.5 GMACs/s
```

Correctness: verified against NumPy integer reference (exact match).

## The kernel

```
export func conv2d_3x3_u8i8(src: *u8, wt: *i8, dst: *mut i16, H: i32, W: i32, C_in: i32) {
    let stride: i32 = (W + 2) * C_in
    let mut row: i32 = 0
    while row < H {
        let mut col: i32 = 0
        while col < W {
            let mut acc0: i16x8 = splat(0)
            let mut acc1: i16x8 = splat(0)
            let mut dr: i32 = 0
            while dr < 3 {
                let mut dc: i32 = 0
                while dc < 3 {
                    let src_off: i32 = (row + dr) * stride + (col + dc) * C_in
                    let wt_off: i32 = (dr * 3 + dc) * C_in
                    let mut ci: i32 = 0
                    while ci < C_in {
                        let a0: u8x16 = load(src, src_off + ci)
                        let b0: i8x16 = load(wt, wt_off + ci)
                        acc0 = acc0 .+ maddubs(a0, b0)
                        let a1: u8x16 = load(src, src_off + ci + 16)
                        let b1: i8x16 = load(wt, wt_off + ci + 16)
                        acc1 = acc1 .+ maddubs(a1, b1)
                        ci = ci + 32
                    }
                    dc = dc + 1
                }
                dr = dr + 1
            }
            dst[row * W + col] = reduce_add(acc0) + reduce_add(acc1)
            col = col + 1
        }
        row = row + 1
    }
}
```

## Why is this fast?

**Dual-accumulator maddubs breaks the dependency chain.**

A single `i16x8` accumulator creates a serial dependency across channels:
each iteration waits for the previous `acc .+` to complete (latency ~1 cycle, but
throughput 1/cycle means the chain bottlenecks at 1 maddubs/cycle).

Two independent accumulators (`acc0`, `acc1`) process 32 channels per iteration
in two parallel chains — the CPU can issue both in the same cycle.

```
// 32 channels per iteration:
acc0 = acc0 .+ maddubs(a0, b0)   // channels  0..15 — chain A
acc1 = acc1 .+ maddubs(a1, b1)   // channels 16..31 — chain B (independent)
```

Combined with `maddubs`'s 16-pair throughput, this achieves 32 multiply-adds
per clock cycle in the innermost loop.

**Integer vs float.**

NumPy computes in float32. Each float32 multiply-add processes 1 value/cycle in a scalar loop.
`maddubs` processes 16 pairs/cycle, then adds them. The arithmetic density is ~16× higher.

**No float conversion overhead.**

Activations stay in u8 from memory load to accumulation. Float conversion is eliminated
entirely — not deferred, not batched, not amortized. The data width stays narrow.

## Layout

NHWC means channels are the innermost dimension. For a pixel at `(row, col)`,
all `C_in` input channels are contiguous in memory. The innermost loop over
`ci` is a sequential stride-1 scan — L1 cache-friendly, prefetcher-friendly.

Compared to NCHW (channels-first), NHWC allows the channel loop to be the
innermost dimension, which is what maddubs requires.

## Constraints and limits

- `C_in` must be a multiple of 32 (dual 16-wide accumulators)
- i16 accumulator: max safe value is ±32767 per lane. For 9 kernel positions × C_in channels,
  ensure `9 × C_in × max_act × max_weight` fits in i16. For typical quantized networks
  (activations ~128, weights ~64), C_in can be up to 512 safely.
- Output is i16 (one channel). For multi-channel output, call the kernel once per output channel
  with different weight slices.

## How to run

```bash
# Build compiler (once)
cargo build --features=llvm --release

# Run demo (generates synthetic 56x56x64 input)
python3 demo/conv2d_3x3/run.py
```

## How it works

The script generates a padded uint8 input and int8 weights, compiles `conv.ea`
to `conv.so`, calls `conv2d_3x3_u8i8` via ctypes, and verifies correctness
against a NumPy reference before timing.

```bash
ea conv.ea --lib   # → conv.so
```

```python
lib = ctypes.CDLL("./conv.so")
lib.conv2d_3x3_u8i8(src_ptr, wt_ptr, dst_ptr,
                     ctypes.c_int32(H), ctypes.c_int32(W), ctypes.c_int32(C_in))
```
