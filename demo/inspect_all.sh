#!/bin/bash
# Run ea inspect on every demo and example kernel.
# Quick sanity check that all kernels parse and analyze correctly.
set -euo pipefail

EA="$(dirname "$0")/../target/release/ea"
if [ ! -x "$EA" ]; then
    echo "ea binary not found at $EA — run: cargo build --release"
    exit 1
fi

for ea in demo/*/*.ea demo/*/kernels/*.ea examples/*.ea; do
    [ -f "$ea" ] || continue
    echo "--- $ea ---"
    "$EA" inspect "$ea"
    echo
done
