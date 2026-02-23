# scikit-image Pipeline Fusion — Ea Demo

This demo fuses a multi-stage edge detection pipeline into a single Ea kernel.
It compares three implementations:

- **NumPy** — idiomatic array operations (blur → sobel → threshold → dilate)
- **Ea unfused** — 4 separate Ea kernel calls, same memory passes as NumPy
- **Ea fused** — 1 fused kernel (blur+sobel+threshold) + dilation

The unfused Ea version isolates the variable: same compiler, same SIMD,
same data — only the kernel boundary changes.

## Results

768x512 Kodak benchmark image. 50 runs, median.

```
NumPy (4 stages)     :   9.75 ms
Ea unfused (4 calls) :   1.57 ms   (6.2x vs NumPy)
Ea fused (2 calls)   :   1.65 ms   (5.9x vs NumPy)
```

Memory: NumPy 20.9 MB → Ea fused 3.0 MB (**7x reduction**).
Correctness: 99.92% pixel match (0.08% differ at threshold boundary from FP rounding).

### Fusion speedup grows with image size

At 768x512 the image fits in L3 cache. Intermediates are served from cache
even without fusion. As image size grows and intermediates spill to DRAM,
fusion eliminates those misses:

```
       Size      Pixels   Unfused   Fused    Fusion speedup
  768x512       393,216    1.83 ms   1.81 ms   1.02x
 1920x1080    2,073,600   11.20 ms  10.23 ms   1.10x
 3840x2160    8,294,400   73.61 ms  55.49 ms   1.33x
 4096x4096   16,777,216  149.30 ms 114.65 ms   1.30x
```

This is cache-theory confirmation: fusion's value is proportional to
the cost of the memory traffic it eliminates.

## What actually happened (the real story)

### Attempt 1: Naive fusion — SLOWER than unfused

The first fused kernel computed Gaussian blur and Sobel as written in
textbooks: for each output pixel, compute 8 Gaussian-blurred neighbor
values, then apply Sobel to those 8 values.

```
Ea unfused (4 calls) :  1.64 ms
Ea fused (naive)     :  2.25 ms   ← 0.7x — SLOWER
```

Why? The naive fusion computed 8 redundant 3x3 Gaussian blurs per 4 output
pixels — ~120 FP ops and 25 loads. The unfused path did ~18 loads and ~30 ops
across blur + sobel + threshold. The fusion *increased* compute faster than it
*removed* memory traffic. And on a 768x512 image, the intermediates fit in
L1/L2 cache anyway (stencils have high spatial locality — adjacent pixels
share 6 of 9 neighbors).

### Attempt 2: Algebraic reformulation — fusion wins

The fix was not a language change or a compiler change. It was mathematics.

The Gaussian [1,2,1; 2,4,2; 1,2,1]/16 composed with Sobel factors into
separable 5x5 convolutions:

- **Gx:** horizontal diffs `(p[row,4]-p[row,0]) + 2*(p[row,3]-p[row,1])`
  weighted by vertical Gaussian `[1, 4, 6, 4, 1] / 16`
- **Gy:** vertical diffs `(p[4,col]-p[0,col]) + 2*(p[3,col]-p[1,col])`
  weighted by horizontal Gaussian `[1, 4, 6, 4, 1] / 16`

This reduces to ~50 ops and 24 loads per 4 pixels. The center pixel (p22)
has zero weight in both kernels and is skipped entirely.

```
Ea unfused (4 calls) :  1.57 ms
Ea fused (optimized) :  1.65 ms   ← on par at 768x512
                                     1.33x faster at 3840x2160
```

### The insight

**Fusion does not make bad kernels fast. Fusion amplifies good kernel design.**

The language didn't change. LLVM didn't change. SIMD width didn't change.
The compute formulation changed. This is the same lesson that:

- XLA learns from fusion failures
- TVM learns from schedule search
- Halide separates as "algorithm vs schedule"
- CUDA programmers learn after their first fused kernel is slower

Fusion works when: `compute_added ≤ memory_traffic_eliminated`

The naive kernel violated this. The optimized kernel satisfies it.

## The fused kernel

```
// Gx: horizontal diffs weighted by vertical Gaussian [1,4,6,4,1]/16
let d0: f32x4 = (p04 .- p00) .+ (p03 .- p01) .* vtwo
let d1: f32x4 = (p14 .- p10) .+ (p13 .- p11) .* vtwo
let d2: f32x4 = (p24 .- p20) .+ (p23 .- p21) .* vtwo
let d3: f32x4 = (p34 .- p30) .+ (p33 .- p31) .* vtwo
let d4: f32x4 = (p44 .- p40) .+ (p43 .- p41) .* vtwo
let gx: f32x4 = (d0 .+ d4 .+ (d1 .+ d3) .* vfour .+ d2 .* vsix) .* inv16

// Gy: vertical diffs weighted by horizontal Gaussian [1,4,6,4,1]/16
let c0: f32x4 = (p40 .- p00) .+ (p30 .- p10) .* vtwo
let c1: f32x4 = (p41 .- p01) .+ (p31 .- p11) .* vtwo
let c2: f32x4 = (p42 .- p02) .+ (p32 .- p12) .* vtwo
let c3: f32x4 = (p43 .- p03) .+ (p33 .- p13) .* vtwo
let c4: f32x4 = (p44 .- p04) .+ (p34 .- p14) .* vtwo
let gy: f32x4 = (c0 .+ c4 .+ (c1 .+ c3) .* vfour .+ c2 .* vsix) .* inv16

// |Gx| + |Gy| → threshold → binary output
let mag: f32x4 = abs_gx .+ abs_gy
store(out, r2 + x, select(mag .> vthresh, vone, vzero))
```

24 loads. ~50 ops. Zero intermediate arrays. One memory pass.

## The unfused kernels

Three separate kernels in `pipeline_unfused.ea`:

- `gaussian_blur_3x3` — 3x3 stencil, writes blurred image to RAM
- `sobel_magnitude` — 3x3 stencil, reads blur output, writes edges to RAM
- `threshold_f32x8` — streaming, reads edges, writes binary mask to RAM

Plus `dilate_3x3` in `dilation.ea` (shared by both paths).

## What this demonstrates

Ea is a language for expressing cache-optimal compute kernels.

| | NumPy | Ea unfused | Ea fused |
|---|---|---|---|
| Kernel calls | 4 Python | 4 ctypes | 2 ctypes |
| Memory passes | 4 | 4 | 2 |
| Intermediate arrays | 3 | 3 | 0 |
| Peak memory | 20.9 MB | ~20 MB | 3.0 MB |

The 6x speedup over NumPy comes from explicit SIMD compute boundaries.
The fusion benefit comes from eliminating memory traffic — and grows
with data size as intermediates leave cache.

Both require the kernel to be formulated correctly.
The language gives you the tools. The algebra is yours.

## How to run

```bash
# Build compiler (once)
cargo build --features=llvm --release

# Build kernels
bash demo/skimage_fusion/build.sh

# Run demo
python3 demo/skimage_fusion/run.py
```

Requires: Python 3, NumPy, Pillow (for image loading/saving).
Downloads Kodak benchmark image on first run.
