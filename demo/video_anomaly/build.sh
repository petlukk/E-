#!/bin/bash
# Build the Eä anomaly detection kernel to a shared library.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EA_ROOT="$SCRIPT_DIR/../.."

echo "Compiling anomaly.ea → anomaly.so"
(cd "$EA_ROOT" && cargo run --features=llvm --release -- "$SCRIPT_DIR/anomaly.ea" --lib)
mv "$EA_ROOT/anomaly.so" "$SCRIPT_DIR/anomaly.so"
echo "Done: $SCRIPT_DIR/anomaly.so"
