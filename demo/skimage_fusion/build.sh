#!/bin/bash
# Build the Ea skimage fusion pipeline to a shared library.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EA_ROOT="$SCRIPT_DIR/../.."

echo "Compiling pipeline.ea -> pipeline.so"
(cd "$EA_ROOT" && cargo run --features=llvm --release -- "$SCRIPT_DIR/pipeline.ea" --lib)
mv "$EA_ROOT/pipeline.so" "$SCRIPT_DIR/pipeline.so"
echo "Done: $SCRIPT_DIR/pipeline.so"

echo "Kernel analysis:"
(cd "$EA_ROOT" && cargo run --features=llvm --release -- inspect "$SCRIPT_DIR/pipeline.ea")
