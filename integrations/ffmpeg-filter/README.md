# FFmpeg filter — Eä kernel for video processing

This example shows how to embed an Eä SIMD kernel into a program that
uses FFmpeg's libav* libraries for video decode. The Eä kernel does the
per-pixel compute; FFmpeg handles container format, codec, and color
conversion.

This is **not** an in-tree AVFilter (that would require patching FFmpeg
source). Instead, it's a standalone program — the realistic embed
pattern for using Eä kernels in media pipelines.

## How it works

1. `kernel.ea` defines `threshold_u8()` — a u8x16 SIMD binary threshold
2. `build.sh` compiles the kernel with `ea kernel.ea --header` (produces
   `.o` and `.h`), then compiles `filter.c` and links everything
3. `filter.c` uses libavformat/libavcodec to decode video, sws_scale to
   convert to grayscale, and calls the Eä kernel per scanline

## Prerequisites

- `ea` compiler on PATH
- C compiler (gcc/clang)
- FFmpeg development libraries:
  ```bash
  # Debian/Ubuntu
  apt install libavformat-dev libavcodec-dev libswscale-dev libavutil-dev pkg-config
  ```

## Usage

```bash
./build.sh
./ea_threshold input.mp4 output.raw 128
ffplay -f rawvideo -pix_fmt gray -video_size 1920x1080 output.raw
```

## Files

| File | Purpose |
|------|---------|
| `kernel.ea` | Binary threshold kernel (u8x16, scalar tail) |
| `filter.c` | FFmpeg decode + Eä kernel + raw output |
| `build.sh` | One-step build script |
