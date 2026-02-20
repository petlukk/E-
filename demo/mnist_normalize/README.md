# MNIST Normalization — Ea Demo

Normalizes 60,000 real MNIST handwritten digit images from [0, 255] to [0.0, 1.0].
On first run, auto-downloads MNIST training data from Google Cloud.
Falls back to synthetic data if download fails.

- **NumPy** — `data / 255.0`
- **Ea** — streaming SIMD kernel, one pass

Both produce identical output (within floating-point tolerance).

## Results

AMD Ryzen 7 1700 (Zen 1, AVX2). 60,000 images, 47M pixels. 50 runs, median.

```
NumPy              :  115 ms   (1.6 GB/s)
Ea (normalize.so)  :  126 ms   (1.5 GB/s)
```

Ea vs NumPy: **0.9x (slower)**.

## Why Ea doesn't win here

This is intentional. A single element-wise multiply on 47 million pixels is
**memory-bound**. Both implementations hit the same DRAM bandwidth wall (~1.5 GB/s).
The CPU spends most of its time waiting for RAM, not computing.

This is the streaming pattern from [COMPUTE_PATTERNS.md](../../COMPUTE_PATTERNS.md):
when the operation is trivially simple and the data is large, all implementations
converge to the same bandwidth limit. ctypes FFI overhead makes Ea slightly slower.

**Ea wins when:**
- Operations have dependency structure (reductions with multi-accumulator ILP)
- Access patterns benefit from explicit SIMD (stencils)
- Multiple operations fuse into one pass (fused pipelines)
- Memory model matters (streaming datasets with O(1) extra memory)

A kernel language that claims to win on every workload is lying.

## The kernel

```
export func normalize_f32x8(input: *restrict f32, out: *mut f32, len: i32, scale: f32) {
    let vscale: f32x8 = splat(scale)
    let mut i: i32 = 0
    while i + 8 <= len {
        let v: f32x8 = load(input, i)
        store(out, i, v .* vscale)
        i = i + 8
    }
    while i < len {
        out[i] = input[i] * scale
        i = i + 1
    }
}
```

Six lines of compute. Called with `scale = 1.0 / 255.0`.

## How to run

```bash
# Build compiler (once)
cargo build --features=llvm --release

# Run demo (auto-downloads MNIST on first run)
python demo/mnist_normalize/run.py
```

## How it works

Python downloads MNIST, loads 60,000 images as a flat f32 array (47M elements),
and calls the Ea kernel via ctypes. The kernel does a single streaming pass:
multiply every pixel by 1/255. No allocation inside the kernel.

```python
lib = ctypes.CDLL("./normalize.so")
lib.normalize_f32x8(input_ptr, output_ptr, n_pixels, ctypes.c_float(1.0 / 255.0))
```
