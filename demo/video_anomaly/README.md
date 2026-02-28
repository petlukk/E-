# Video Frame Anomaly Detection — Ea Demo

This demo compares four implementations of frame-difference anomaly detection
using real video data from the OpenCV sample dataset (pedestrian surveillance).
On first run, it auto-downloads the video and extracts two frames with motion.
Falls back to synthetic frames if OpenCV is not installed.

- **NumPy** — idiomatic Python, array operations
- **OpenCV** — industry-standard optimized C++
- **Ea (3 kernels)** — three composable SIMD kernels (`anomaly.ea`)
- **Ea fused (1 kernel)** — single fused pipeline (`anomaly.ea`)

All four produce identical anomaly counts.

## Results

768x576 frames (OpenCV vtest.avi, pedestrian surveillance). 50 runs, median.

```
NumPy              :  1.51 ms  ±0.29
OpenCV (C++)       :  1.26 ms  ±0.18
Ea (3 kernels)     :  1.58 ms  ±0.30
Ea foreach (3 kern):  1.90 ms  ±0.22
Ea fused (1 kernel):  0.13 ms  ±0.03
```

The 3-kernel Ea is **slower** than NumPy. The fused Ea is **11.5x faster** than NumPy, **9.5x faster** than OpenCV.
Same language. Same compiler. Same data. Only the kernel boundary changed.

## The kernel

This is the entire Ea implementation. Three composable kernels — nothing is hidden.

```
// Video frame anomaly detection.
// Three composable kernels: diff, threshold, count.
// Called in sequence from Python — kernels compute, caller orchestrates.

// Per-pixel absolute difference: out[i] = |a[i] - b[i]|
export func frame_diff_f32x8(a: *restrict f32, b: *restrict f32, out: *mut f32, len: i32) {
    let vzero: f32x8 = splat(0.0)
    let mut i: i32 = 0
    while i + 8 <= len {
        let va: f32x8 = load(a, i)
        let vb: f32x8 = load(b, i)
        let diff: f32x8 = va .- vb
        let abs_diff: f32x8 = select(diff .< vzero, vzero .- diff, diff)
        store(out, i, abs_diff)
        i = i + 8
    }
    while i < len {
        let d: f32 = a[i] - b[i]
        if d < 0.0 {
            out[i] = 0.0 - d
        } else {
            out[i] = d
        }
        i = i + 1
    }
}

// Branchless threshold: out[i] = (data[i] > thresh) ? 1.0 : 0.0
export func threshold_f32x8(data: *restrict f32, out: *mut f32, len: i32, thresh: f32) {
    let vthresh: f32x8 = splat(thresh)
    let vone: f32x8 = splat(1.0)
    let vzero: f32x8 = splat(0.0)
    let mut i: i32 = 0
    while i + 8 <= len {
        let v: f32x8 = load(data, i)
        store(out, i, select(v .> vthresh, vone, vzero))
        i = i + 8
    }
    while i < len {
        if data[i] > thresh {
            out[i] = 1.0
        } else {
            out[i] = 0.0
        }
        i = i + 1
    }
}

// Sum reduction with multi-accumulator ILP.
// Returns total (count of 1.0s when applied to thresholded output).
export func sum_f32x8(data: *restrict f32, len: i32) -> f32 {
    let mut acc0: f32x8 = splat(0.0)
    let mut acc1: f32x8 = splat(0.0)
    let mut i: i32 = 0
    while i + 16 <= len {
        let v0: f32x8 = load(data, i)
        let v1: f32x8 = load(data, i + 8)
        acc0 = acc0 .+ v0
        acc1 = acc1 .+ v1
        i = i + 16
    }
    let mut total: f32 = reduce_add(acc0 .+ acc1)
    while i < len {
        total = total + data[i]
        i = i + 1
    }
    return total
}
```

## The fused kernel

The same pipeline as a single kernel. No intermediate arrays. All in register.

```
export func anomaly_count_fused(a: *restrict f32, b: *restrict f32, len: i32, thresh: f32) -> f32 {
    let vzero: f32x8 = splat(0.0)
    let vone: f32x8 = splat(1.0)
    let vthresh: f32x8 = splat(thresh)
    let mut acc0: f32x8 = splat(0.0)
    let mut acc1: f32x8 = splat(0.0)
    let mut i: i32 = 0
    while i + 16 <= len {
        let va0: f32x8 = load(a, i)
        let vb0: f32x8 = load(b, i)
        let diff0: f32x8 = va0 .- vb0
        let abs0: f32x8 = select(diff0 .< vzero, vzero .- diff0, diff0)
        acc0 = acc0 .+ select(abs0 .> vthresh, vone, vzero)

        let va1: f32x8 = load(a, i + 8)
        let vb1: f32x8 = load(b, i + 8)
        let diff1: f32x8 = va1 .- vb1
        let abs1: f32x8 = select(diff1 .< vzero, vzero .- diff1, diff1)
        acc1 = acc1 .+ select(abs1 .> vthresh, vone, vzero)

        i = i + 16
    }
    let mut total: f32 = reduce_add(acc0 .+ acc1)
    // ... scalar tail
    return total
}
```

## What this demonstrates

**Kernel fusion turns a loss into a 12x win.**

Three distinct kernel patterns, each under 20 lines:

- **Streaming** (`frame_diff_f32x8`) — per-element absolute difference, pure throughput
- **Branchless** (`threshold_f32x8`) — SIMD select instead of branches, no mispredictions
- **Reduction** (`sum_f32x8`) — multi-accumulator sum with `reduce_add`, maximizes ILP

The kernels are composed from Python. Each does one thing. The caller decides
the pipeline order and threshold value.

## How to run

```bash
# Build compiler (once)
cargo build --features=llvm --release

# Run demo
python demo/video_anomaly/run.py
```

## How it works

On first run, the script downloads the OpenCV `vtest.avi` sample video and extracts
frames 0 and 30 (which have pedestrian motion between them) as grayscale PNGs.
If OpenCV is not installed, it falls back to synthetic frames. Ea does the compute.
The three kernels compile to a single `.so` and are called via `ctypes` —
no runtime, no framework, no bindings library.

```bash
ea anomaly.ea --lib   # -> anomaly.so
```

```python
lib = ctypes.CDLL("./anomaly.so")
lib.frame_diff_f32x8(frame_a, frame_b, diff_buf, n)
lib.threshold_f32x8(diff_buf, mask_buf, n, threshold)
anomaly_count = lib.sum_f32x8(mask_buf, n)
```

The pipeline: compute per-pixel differences, threshold to a binary mask,
sum the mask to get the anomaly pixel count.
