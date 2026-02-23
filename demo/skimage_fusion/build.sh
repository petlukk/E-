#!/bin/bash
# Build the Eä skimage fusion kernels to shared libraries.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EA_ROOT="$SCRIPT_DIR/../.."

for kernel in pipeline_fused pipeline_unfused dilation; do
    echo "Compiling ${kernel}.ea → ${kernel}.so"
    (cd "$EA_ROOT" && cargo run --features=llvm --release -- "$SCRIPT_DIR/${kernel}.ea" --lib)
    mv "$EA_ROOT/${kernel}.so" "$SCRIPT_DIR/${kernel}.so"
done
echo "Done: all kernels built"
