# MNIST Preprocessing — Ea Demo

Normalizes and preprocesses 60,000 real MNIST handwritten digit images.
On first run, auto-downloads MNIST training data from Google Cloud.
Falls back to synthetic data if download fails.

Two benchmarks that tell the full story:

1. **Single operation** (normalize only) — Ea loses. Memory-bound.
2. **Full pipeline** (normalize + standardize + clip, fused) — Ea wins. Fusion.

## Results

AMD Ryzen 7 1700 (Zen 1, AVX2). 60,000 images, 47M pixels. 50 runs, median.

### Part 1: Single Operation (normalize only)

```
NumPy (x / 255.0)  :   75 ms
Ea (1 kernel)       :   81 ms
```

Ea vs NumPy: **0.9x (slower)**.

A single element-wise multiply on 47M pixels is memory-bound. Both implementations
hit the same DRAM bandwidth wall. This is the expected result.

### Part 2: Full Pipeline (normalize + standardize + clip)

```
NumPy (multi-pass)  :  336 ms   (3-4 memory passes)
Ea fused (1 pass)   :  129 ms   (1 memory pass)
```

Ea fused vs NumPy: **2.6x faster**.

NumPy does:
```python
x = x / 255.0            # pass 1: read + write 180 MB
x = (x - mean) / std     # pass 2-3: read + write 180 MB twice
x = np.clip(x, 0, 1)     # pass 4: read + write 180 MB
```

Ea does:
```
load → scale → subtract → multiply → clamp → store   (one pass)
```

Same principle as the video anomaly fusion: **if data leaves registers, you
probably ended a kernel too early.**

## The kernels

### Single operation (`preprocess.ea`)

```
export func normalize_f32x8(input: *restrict f32, out: *mut f32, len: i32, scale: f32) {
    let vscale: f32x8 = splat(scale)
    let mut i: i32 = 0
    while i + 8 <= len {
        let v: f32x8 = load(input, i)
        store(out, i, v .* vscale)
        i = i + 8
    }
    // ... scalar tail
}
```

### Fused pipeline (`preprocess.ea`)

```
export func preprocess_fused(
    input: *restrict f32, out: *mut f32, len: i32,
    scale: f32, mean: f32, inv_std: f32
) {
    let vscale: f32x8 = splat(scale)
    let vmean: f32x8 = splat(mean)
    let vinv_std: f32x8 = splat(inv_std)
    let vzero: f32x8 = splat(0.0)
    let vone: f32x8 = splat(1.0)
    let mut i: i32 = 0
    while i + 8 <= len {
        let v: f32x8 = load(input, i)
        let norm: f32x8 = v .* vscale
        let centered: f32x8 = norm .- vmean
        let scaled: f32x8 = centered .* vinv_std
        let clamped_lo: f32x8 = select(scaled .< vzero, vzero, scaled)
        let clamped: f32x8 = select(clamped_lo .> vone, vone, clamped_lo)
        store(out, i, clamped)
        i = i + 8
    }
    // ... scalar tail
}
```

Five operations, one memory pass. All intermediate values live in registers.

## How to run

```bash
# Build compiler (once)
cargo build --features=llvm --release

# Run demo (auto-downloads MNIST on first run)
python demo/mnist_normalize/run.py
```

## Why this demo matters

It shows both sides of the same coin:

| Workload | Memory passes | Ea wins? | Why |
|----------|:---:|:---:|---|
| Single op | 1 vs 1 | No | Both memory-bound |
| Full pipeline | 1 vs 3-4 | **Yes** | Fusion eliminates traffic |

Performance comes from expressing the computation boundary correctly.
Not from a better compiler.
