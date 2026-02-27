# Integration Examples

**You don't adopt Eä. You embed it.**

These examples show how to call Eä SIMD kernels from existing build systems and
language ecosystems. Each example has a simple kernel, the glue code, a build
mechanism, and a README. The kernels are intentionally simple — the point is the
integration pattern, not kernel complexity.

## Prerequisites

The `ea` compiler must be on PATH. Build from source:

```bash
cargo build --features=llvm --release
export PATH="$PWD/target/release:$PATH"
```

## Examples

### 1. [Python setuptools](python-extension/) — `pip install .`

Package an Eä kernel as a pip-installable Python module. `setup.py` compiles the
kernel at install time; `__init__.py` loads it via ctypes and exposes a numpy-friendly API.

```bash
cd python-extension
pip install .
python example.py
```

**Kernel:** `scale(data, out, n, factor)` — f32x8 with masked tail.

---

### 2. [CMake](cmake/) — `add_ea_kernel()`

Reusable CMake module that compiles `.ea` files and exposes them as linkable
targets with generated C headers.

```bash
cd cmake
cmake -B build && cmake --build build && ./build/app
```

**Kernel:** `fma_kernel(a, b, c, out, n)` — f32x4 fused multiply-add.

---

### 3. [Rust build.rs](rust-crate/) — `cargo run`

Compile and link an Eä kernel at build time using Cargo's build script. The
kernel is statically linked and called via `extern "C"` FFI.

```bash
cd rust-crate
cargo run
```

**Kernel:** `dot_product(a, b, n) -> f32` — f32x4 with reduce_add.

---

### 4. [PyTorch custom op](pytorch-custom-op/) — `torch.autograd.Function`

Wrap an Eä kernel as a PyTorch custom operation. Compiles on first import,
extracts tensor data pointers, calls the kernel. Forward-only — Eä kernels are
raw SIMD compute, not differentiable ops.

```bash
cd pytorch-custom-op
pip install torch  # if not installed
python example.py
```

**Kernel:** `fma(a, b, c, out, n)` — f32x8 fused multiply-add.

---

### 5. [FFmpeg filter](ffmpeg-filter/) — libav* decode + Eä compute

Standalone video processing program that uses FFmpeg for decode/format and an Eä
kernel for per-scanline compute. Not an in-tree AVFilter — this is the realistic
embed pattern.

```bash
cd ffmpeg-filter
apt install libavformat-dev libavcodec-dev libswscale-dev libavutil-dev  # if needed
./build.sh
./ea_threshold input.mp4 output.raw 128
```

**Kernel:** `threshold_u8(src, dst, n, thresh)` — u8x16 binary threshold.

## The pattern

Every integration follows the same structure:

1. **Compile** — run `ea kernel.ea` (produces `.o`) or `ea kernel.ea --lib` (produces `.so`)
2. **Link or load** — static link the `.o`, or load the `.so` via ctypes/dlopen
3. **Call** — C ABI function call, no wrappers needed

The Eä compiler produces standard object files and shared libraries. Any language
that can call C can call Eä kernels.
