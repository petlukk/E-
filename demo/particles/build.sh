#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EA_ROOT="$SCRIPT_DIR/../.."

echo "Compiling particle.ea -> particle.so"
(cd "$EA_ROOT" && cargo run --features=llvm --release -- "$SCRIPT_DIR/particle.ea" --lib)
mv "$EA_ROOT/particle.so" "$SCRIPT_DIR/particle.so"
echo "Done"

echo "Kernel analysis:"
(cd "$EA_ROOT" && cargo run --features=llvm --release -- inspect "$SCRIPT_DIR/particle.ea")
