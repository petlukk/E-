#!/bin/bash
# Build the Eä stacking kernel to a shared library.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EA_ROOT="$SCRIPT_DIR/../.."

echo "Compiling stack.ea → stack.so"
(cd "$EA_ROOT" && cargo run --features=llvm --release -- "$SCRIPT_DIR/stack.ea" --lib)
mv "$EA_ROOT/stack.so" "$SCRIPT_DIR/stack.so"
echo "Done: $SCRIPT_DIR/stack.so"
