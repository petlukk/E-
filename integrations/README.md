# Integration Examples

For Python, Rust, C++, PyTorch, and CMake — use `ea bind`:

```bash
ea kernel.ea --lib
ea bind kernel.ea --python --rust --cpp --pytorch --cmake
```

See the main [README](../README.md#ea-bind) for details.

## Manual integration

For ecosystems that `ea bind` doesn't cover, Eä kernels are standard C ABI shared
libraries. Any language that can call C can call Eä kernels.

### [FFmpeg filter](ffmpeg-filter/) — libav* decode + Eä compute

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
