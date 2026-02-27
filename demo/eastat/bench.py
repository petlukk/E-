#!/usr/bin/env python3
"""
Honest benchmark: Eastat (Ea kernels) vs pandas.

In-process timing with phase breakdowns showing where each tool spends time.
Separates pandas percentile cost (requires sorting) from comparable work
(count, mean, std, min, max) that both tools compute.

Usage:
    python bench.py [test_file.csv]
"""

import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def bench_eastat(filepath):
    """Benchmark eastat in-process with phase breakdown."""
    # Import locally to get clean timing
    from eastat import process

    # Warmup
    process(filepath)

    # Timed run
    _, _, _, _, timings = process(filepath)

    return timings


def bench_pandas(filepath):
    """Benchmark pandas with phase breakdown isolating percentile cost."""
    try:
        import pandas as pd
    except ImportError:
        print("  pandas not installed, skipping")
        return None

    # Warmup
    _ = pd.read_csv(filepath).describe()

    # Phase 1: read_csv
    t0 = time.perf_counter()
    df = pd.read_csv(filepath)
    t_read = time.perf_counter() - t0

    # Phase 2: .describe() with percentiles (full — default 25/50/75%)
    t0 = time.perf_counter()
    _ = df.describe()
    t_describe_full = time.perf_counter() - t0

    # Phase 3: .describe() without percentiles (count, mean, std, min, max only)
    t0 = time.perf_counter()
    _ = df.describe(percentiles=[])
    t_describe_no_pct = time.perf_counter() - t0

    return {
        'read_csv': t_read,
        'describe_full': t_describe_full,
        'describe_no_pct': t_describe_no_pct,
        'percentile_cost': t_describe_full - t_describe_no_pct,
        'total_full': t_read + t_describe_full,
        'total_no_pct': t_read + t_describe_no_pct,
    }


def main():
    if len(sys.argv) > 1:
        filepath = Path(sys.argv[1])
    else:
        candidates = [
            SCRIPT_DIR / "test_1000000.csv",
            SCRIPT_DIR / "test_10k.csv",
        ]
        filepath = None
        for c in candidates:
            if c.exists():
                filepath = c
                break
        if filepath is None:
            print("No test file found. Run generate_test.py first.")
            print("  python generate_test.py --rows=1000000")
            sys.exit(1)

    file_size = filepath.stat().st_size
    size_mb = file_size / (1024**2)
    print(f"File: {filepath.name} ({size_mb:.1f} MB)")
    print("=" * 64)

    # --- Eastat ---
    ea = bench_eastat(filepath)

    print("\neastat breakdown:")
    print(f"  scan  (structural extraction):  {ea.get('scan', 0)*1000:7.1f} ms")
    print(f"  layout (row/delim index):       {ea.get('layout', 0)*1000:7.1f} ms")
    print(f"  stats  (parse + reduce):        {ea.get('stats', 0)*1000:7.1f} ms")
    ea_total = ea.get('total', 0)
    print(f"  total:                          {ea_total*1000:7.1f} ms")

    # --- pandas ---
    pd_timings = bench_pandas(filepath)

    if pd_timings:
        print(f"\npandas breakdown:")
        print(f"  read_csv (parse -> DataFrame): {pd_timings['read_csv']*1000:7.1f} ms")
        print(f"  .describe() full:              {pd_timings['describe_full']*1000:7.1f} ms")
        print(f"    of which percentiles alone:  {pd_timings['percentile_cost']*1000:7.1f} ms  <- work eastat doesn't do")
        print(f"  total (full):                  {pd_timings['total_full']*1000:7.1f} ms")
        print(f"  total (no percentiles):        {pd_timings['total_no_pct']*1000:7.1f} ms")

    # --- Comparison ---
    print("\n" + "=" * 64)

    if pd_timings and ea_total > 0:
        ratio_no_pct = pd_timings['total_no_pct'] / ea_total
        ratio_full = pd_timings['total_full'] / ea_total

        print(f"\nComparable work (count/mean/std/min/max):")
        print(f"  eastat vs pandas (no percentiles):  {ratio_no_pct:.1f}x faster")
        print(f"\nFull pandas (including 25/50/75% percentiles):")
        print(f"  eastat vs pandas (full .describe): {ratio_full:.1f}x faster")

        print(f"\nNote: eastat computes count, mean, std, min, max (f32 SIMD).")
        print(f"      pandas additionally computes 25%, 50%, 75% (requires sorting).")
        print(f"      The '{ratio_no_pct:.1f}x' number compares equivalent work.")


if __name__ == '__main__':
    main()
