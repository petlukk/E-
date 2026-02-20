# Video Frame Anomaly Detection — Ea Demo

This demo compares three implementations of frame-difference anomaly detection
using real video data from the OpenCV sample dataset (pedestrian surveillance).
On first run, it auto-downloads the video and extracts two frames with motion.
Falls back to synthetic frames if OpenCV is not installed.

- **NumPy** — idiomatic Python, array operations
- **OpenCV** — industry-standard optimized C++
- **Ea** — three composable SIMD kernels, compiled to shared library

All three produce identical anomaly counts.

## Results

Results vary by machine. Run `python run.py` to measure on yours.

```
NumPy          : ~4.2 ms
OpenCV (C++)   : ~1.5 ms
Ea (anomaly.so): ~0.6 ms
```

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

## What this demonstrates

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
