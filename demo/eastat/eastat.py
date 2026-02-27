#!/usr/bin/env python3
"""
eastat -- CSV column statistics powered by Ea kernels

Zero manual FFI. All kernel calls via auto-generated bindings from `ea bind --python`.

Usage:
    python eastat.py data.csv
    python eastat.py -d '\\t' data.tsv
    python eastat.py --json data.csv
    python eastat.py -c 1,3,5 data.csv
"""

import argparse
import math
import mmap
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import csv_parse
import csv_stats


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process(filepath, delimiter=',', has_header=True, selected_columns=None):
    """Full CSV analysis pipeline. Returns (results, headers, n_rows, col_count, timings)."""
    timings = {}
    t_total = time.perf_counter()

    # 1. Memory-map the file
    t0 = time.perf_counter()
    file_size = filepath.stat().st_size
    if file_size == 0:
        print("Error: empty file", file=sys.stderr)
        sys.exit(1)

    fd = os.open(str(filepath), os.O_RDONLY)
    try:
        mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    finally:
        os.close(fd)

    text = np.frombuffer(mm, dtype=np.uint8)
    n = len(text)
    timings['mmap'] = time.perf_counter() - t0

    # 2. Structural scan — extract delimiter and LF positions
    t0 = time.perf_counter()
    delim_byte = ord(delimiter)
    delim_pos = np.empty(n // 4 + 256, dtype=np.int32)
    lf_pos = np.empty(n // 20 + 256, dtype=np.int32)
    counts = np.zeros(3, dtype=np.int32)

    csv_parse.extract_positions_quoted(text, delim_byte, delim_pos, lf_pos, counts)

    n_delims = int(counts[0])
    n_lfs = int(counts[1])
    header_dc = int(counts[2])
    delim_pos = delim_pos[:n_delims]
    lf_pos = lf_pos[:n_lfs]
    timings['scan'] = time.perf_counter() - t0

    # 3. Build layout — row arrays + delimiter index
    t0 = time.perf_counter()
    col_count = header_dc + 1

    # Extract header names
    if has_header:
        first_nl = int(lf_pos[0]) if n_lfs > 0 else n
        hdr = bytes(text[:first_nl])
        if hdr.endswith(b'\r'):
            hdr = hdr[:-1]
        headers = hdr.decode('utf-8', errors='replace').split(delimiter)
        header_end = int(lf_pos[0]) if n_lfs > 0 else -1
    else:
        headers = [f"col_{i}" for i in range(col_count)]
        header_end = -1

    # Build row start/end arrays via kernel
    row_starts = np.empty(n_lfs + 2, dtype=np.int32)
    row_ends = np.empty(n_lfs + 2, dtype=np.int32)
    n_rows_out = np.zeros(1, dtype=np.int32)

    csv_parse.build_row_arrays(
        lf_pos, n_lfs, header_end, n,
        row_starts, row_ends, n_rows_out
    )

    data_rows = int(n_rows_out[0])
    row_starts = row_starts[:data_rows]
    row_ends = row_ends[:data_rows]

    if data_rows == 0:
        try:
            mm.close()
        except BufferError:
            pass
        return [], headers, 0, col_count, timings

    # Data delimiters (skip header's delimiters)
    data_delim_pos = delim_pos[header_dc:] if has_header else delim_pos
    n_data_delims = len(data_delim_pos)

    # Build per-row delimiter index
    delims_per_row = np.zeros(data_rows, dtype=np.int32)
    row_delim_offset = np.zeros(data_rows, dtype=np.int32)

    if n_data_delims > 0:
        csv_parse.build_row_delim_index(
            data_delim_pos, n_data_delims,
            row_ends, data_rows,
            delims_per_row, row_delim_offset
        )

    timings['layout'] = time.perf_counter() - t0

    # 4. Per-column statistics
    t0 = time.perf_counter()
    cols = selected_columns if selected_columns else list(range(col_count))
    results = []

    # Shared buffers (reused across columns)
    fs_buf = np.empty(data_rows, dtype=np.int32)
    fe_buf = np.empty(data_rows, dtype=np.int32)
    val_buf = np.empty(data_rows, dtype=np.float32)
    cnt_buf = np.zeros(1, dtype=np.int32)

    for ci in cols:
        if ci >= col_count:
            continue
        col_name = headers[ci] if ci < len(headers) else f"col_{ci}"

        # Field boundaries
        csv_parse.compute_field_bounds(
            ci, col_count, data_rows,
            row_starts, row_ends, data_delim_pos,
            row_delim_offset, delims_per_row,
            fs_buf, fe_buf
        )

        # Try numeric parse
        cnt_buf[0] = 0
        csv_parse.batch_atof(text, fs_buf, fe_buf, val_buf, cnt_buf)
        count = int(cnt_buf[0])

        if count >= data_rows * 0.5:
            stats = _numeric_stats(val_buf, count, data_rows)
            # Detect integer vs float
            mn, mx = stats['min'], stats['max']
            if mn == int(mn) and mx == int(mx) and abs(mx) < 1e7:
                col_type = 'integer'
            else:
                col_type = 'float'
            stats['type'] = col_type
        else:
            stats = _string_stats(fs_buf, fe_buf, data_rows)
            stats['type'] = 'string'

        stats['name'] = col_name
        stats['col_index'] = ci
        results.append(stats)

    timings['stats'] = time.perf_counter() - t0
    timings['total'] = time.perf_counter() - t_total

    del text
    try:
        mm.close()
    except BufferError:
        pass
    return results, headers, data_rows, col_count, timings


def _numeric_stats(val_buf, count, total_rows):
    """Compute numeric column stats from parsed values."""
    values = val_buf[:count]
    nulls = total_rows - count

    if count == 0:
        return {'rows': total_rows, 'nulls': nulls, 'count': 0,
                'min': None, 'max': None, 'mean': None, 'stddev': None, 'sum': None}

    if count >= 16:
        out_sum = np.zeros(1, dtype=np.float32)
        out_min = np.zeros(1, dtype=np.float32)
        out_max = np.zeros(1, dtype=np.float32)
        out_sumsq = np.zeros(1, dtype=np.float32)

        csv_stats.f32_column_stats(values, out_sum, out_min, out_max, out_sumsq)

        total_sum = float(out_sum[0])
        total_min = float(out_min[0])
        total_max = float(out_max[0])
        total_sumsq = float(out_sumsq[0])
    else:
        arr = values.astype(np.float64)
        total_sum = float(np.sum(arr))
        total_min = float(np.min(arr))
        total_max = float(np.max(arr))
        total_sumsq = float(np.sum(arr ** 2))

    mean = total_sum / count
    variance = max(0.0, total_sumsq / count - mean * mean)
    stddev = math.sqrt(variance)

    return {
        'rows': total_rows, 'nulls': nulls, 'count': count,
        'min': total_min, 'max': total_max,
        'mean': mean, 'stddev': stddev, 'sum': total_sum,
    }


def _string_stats(fs_buf, fe_buf, total_rows):
    """Compute string column stats from field boundaries."""
    out_min = np.zeros(1, dtype=np.int32)
    out_max = np.zeros(1, dtype=np.int32)
    out_total = np.zeros(1, dtype=np.int32)
    out_null = np.zeros(1, dtype=np.int32)

    csv_parse.field_length_stats(fs_buf, fe_buf, out_min, out_max, out_total, out_null)

    null_count = int(out_null[0])
    valid = total_rows - null_count
    total_len = int(out_total[0])

    return {
        'rows': total_rows, 'nulls': null_count, 'count': valid,
        'min_length': int(out_min[0]),
        'max_length': int(out_max[0]),
        'mean_length': total_len / valid if valid > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_number(n):
    """Format number with commas."""
    if n is None:
        return "N/A"
    if isinstance(n, float):
        if n == int(n) and abs(n) < 1e15:
            return f"{int(n):,}"
        if abs(n) >= 1000:
            return f"{n:,.2f}"
        if abs(n) < 0.01:
            return f"{n:.6g}"
        return f"{n:.4g}"
    return f"{n:,}"


def print_table(results, headers, row_count, col_count, timings, file_size):
    """Print formatted statistics table."""
    for r in results:
        col_idx = r['col_index']
        col_name = r['name']
        col_type = r['type'].capitalize()

        print(f"\n  {col_idx + 1}. \"{col_name}\" ({col_type})")
        print(f"        Rows: {format_number(r['rows']):>14}    Nulls: {format_number(r['nulls'])}")

        if r['type'] in ('integer', 'float'):
            if r.get('min') is not None:
                print(f"        Min: {format_number(r['min']):>15}    Max: {format_number(r['max'])}")
                print(f"        Mean: {format_number(r['mean']):>14}    StdDev: {format_number(r['stddev'])}")
        else:
            print(f"        Min Length: {format_number(r.get('min_length', 0)):>10}    Max Length: {format_number(r.get('max_length', 0))}")

    elapsed = timings.get('total', 0)
    throughput = file_size / elapsed / (1024**3) if elapsed > 0 else 0
    print(f"\nScanned {format_number(row_count)} rows x {col_count} columns in {elapsed:.2f}s ({throughput:.2f} GB/s)")

    print(f"\n  Timings:")
    for phase in ['mmap', 'scan', 'layout', 'stats']:
        t = timings.get(phase, 0)
        if t > 0.0001:
            print(f"    {phase:>12}: {t*1000:8.1f} ms")


def print_json(results, headers, row_count, col_count, timings, file_size):
    """Print statistics as JSON."""
    elapsed = timings.get('total', 0)
    output = {
        'file_size': file_size,
        'row_count': row_count,
        'col_count': col_count,
        'elapsed_seconds': round(elapsed, 4),
        'throughput_gbps': round(file_size / elapsed / (1024**3), 2) if elapsed > 0 else 0,
        'columns': []
    }
    for r in results:
        col = {
            'index': r['col_index'],
            'name': r['name'],
            'type': r['type'],
            'rows': r['rows'],
            'nulls': r['nulls'],
        }
        if r['type'] in ('integer', 'float'):
            col.update({
                'min': r.get('min'), 'max': r.get('max'),
                'mean': r.get('mean'), 'stddev': r.get('stddev'),
            })
        else:
            col.update({
                'min_length': r.get('min_length', 0),
                'max_length': r.get('max_length', 0),
            })
        output['columns'].append(col)

    print(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog='eastat',
        description='CSV column statistics powered by Ea kernels'
    )
    parser.add_argument('file', help='CSV file to analyze')
    parser.add_argument('--delimiter', '-d', default=',',
                        help='Field delimiter (default: comma)')
    parser.add_argument('--json', '-j', action='store_true', dest='json_output',
                        help='Output as JSON')
    parser.add_argument('--columns', '-c', default=None,
                        help='Comma-separated column indices (0-based)')
    parser.add_argument('--no-header', action='store_true',
                        help='File has no header row')

    args = parser.parse_args()

    delimiter = args.delimiter
    if delimiter == '\\t':
        delimiter = '\t'

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    file_size = filepath.stat().st_size
    selected_columns = None
    if args.columns:
        selected_columns = [int(c.strip()) for c in args.columns.split(',')]

    results, headers, row_count, col_count, timings = process(
        filepath=filepath,
        delimiter=delimiter,
        has_header=not args.no_header,
        selected_columns=selected_columns,
    )

    if args.json_output:
        print_json(results, headers, row_count, col_count, timings, file_size)
    else:
        print_table(results, headers, row_count, col_count, timings, file_size)


if __name__ == '__main__':
    main()
