#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Compiling kernel.ea..."
ea kernel.ea --header

echo "Building ea_threshold..."
cc -o ea_threshold filter.c kernel.o \
    $(pkg-config --cflags --libs libavformat libavcodec libswscale libavutil) -lm

echo "Done. Usage:"
echo "  ./ea_threshold input.mp4 output.raw 128"
echo "  ffplay -f rawvideo -pix_fmt gray -video_size WxH output.raw"
