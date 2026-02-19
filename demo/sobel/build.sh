#!/bin/bash
# Build the Eä Sobel kernel to a shared library.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EA_ROOT="$SCRIPT_DIR/../.."

echo "Compiling sobel.ea → sobel.so"
(cd "$EA_ROOT" && cargo run --features=llvm --release -- "$SCRIPT_DIR/sobel.ea" --lib)
mv "$EA_ROOT/sobel.so" "$SCRIPT_DIR/sobel.so"
echo "Done: $SCRIPT_DIR/sobel.so"
