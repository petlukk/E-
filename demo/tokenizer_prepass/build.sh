#!/bin/bash
# Build the Ea tokenizer prepass kernel to a shared library.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EA_ROOT="$SCRIPT_DIR/../.."

echo "Compiling prepass.ea -> prepass.so"
(cd "$EA_ROOT" && cargo run --features=llvm --release -- "$SCRIPT_DIR/prepass.ea" --lib)
mv "$EA_ROOT/prepass.so" "$SCRIPT_DIR/prepass.so"
echo "Done: $SCRIPT_DIR/prepass.so"

echo "Kernel analysis:"
(cd "$EA_ROOT" && cargo run --features=llvm --release -- inspect "$SCRIPT_DIR/prepass.ea")
