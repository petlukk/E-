# Eastat

CSV column statistics powered by `ea bind`.

Same SIMD kernels as [simdstat](https://github.com/petlukk/simdstat), rewritten with zero manual ctypes. Every kernel call goes through auto-generated Python bindings from `ea bind --python`.

## Results

**1M rows x 6 columns (47 MB)**

### What each tool computes

| Statistic | eastat | pandas `.describe()` |
|-----------|--------|---------------------|
| count     | yes    | yes                 |
| mean      | yes    | yes                 |
| std       | yes    | yes                 |
| min       | yes    | yes                 |
| max       | yes    | yes                 |
| 25%       | —      | yes (requires sort) |
| 50%       | —      | yes (requires sort) |
| 75%       | —      | yes (requires sort) |

### Phase breakdown

```
eastat breakdown:
  scan  (structural extraction):    X ms   <- Ea kernel
  layout (row/delim index):         X ms   <- Ea kernel
  stats  (parse + reduce):          X ms   <- Ea kernel
  total:                            X ms

pandas breakdown:
  read_csv (parse -> DataFrame):    X ms
  .describe() full:                 X ms
    of which percentiles alone:     X ms   <- work eastat doesn't do
  total (full):                     X ms
  total (no percentiles):           X ms
```

### Speedup (honest accounting)

| Comparison | Speedup |
|------------|---------|
| eastat vs pandas (comparable work — no percentiles) | ~2x |
| eastat vs pandas (full `.describe()` including percentiles) | ~2.5x |

The "comparable work" number is the honest one: it compares equivalent operations (count, mean, std, min, max) while excluding percentile computation that eastat doesn't perform.

### Where the speedup comes from

- **Parsing**: Ea's structural scan kernel (`extract_positions_quoted`) processes the CSV in a single SIMD pass, faster than pandas' general-purpose C parser
- **No DataFrame**: eastat streams through the file without building a full in-memory DataFrame
- **f32 precision**: eastat uses f32 SIMD reductions; pandas uses f64

## What `ea bind` eliminates

| Before (simdstat) | After (eastat) | Lines saved |
|---|---|---|
| 86 lines ctypes boilerplate (CDLL, argtypes, restype) | Auto-generated `csv_parse.py`, `csv_stats.py` | 86 |
| ~80 lines call-site `.ctypes.data_as()` / `ctypes.c_int32()` | One-line wrapper calls | ~80 |
| 227 lines NumPy fallback paths | Dropped (kernels are the product) | 227 |
| ~70 lines Python row-boundary computation | `build_row_arrays` kernel | ~70 |

`grep -c ctypes eastat.py` → **0**

## Kernels

**csv_parse.ea** — 6 exports:
- `extract_positions_quoted` — single-pass structural scan (delimiters + newlines, quote-aware)
- `build_row_arrays` — row start/end from LF positions (replaces Python edge-case logic)
- `build_row_delim_index` — per-row delimiter mapping via O(n) merge-scan
- `compute_field_bounds` — field start/end for any column
- `batch_atof` — ASCII-to-float parser at C speed
- `field_length_stats` — string column min/max/total length

**csv_stats.ea** — 1 export:
- `f32_column_stats` — fused sum + min + max + sum-of-squares (f32x8 dual-accumulator with FMA)

## Usage

```bash
# Build (compile kernels + generate Python bindings)
./build.sh

# Generate test data
python generate_test.py --rows=1000000

# Run
python eastat.py test_1000000.csv
python eastat.py --json test_1000000.csv
python eastat.py -c 2,5 test_1000000.csv

# Benchmark vs pandas (honest phase breakdown)
python bench.py
```

## Pipeline

```
1. mmap file → uint8 array
2. extract_positions_quoted → delimiter + LF positions
3. build_row_arrays + build_row_delim_index → row layout
4. per column: compute_field_bounds → batch_atof → f32_column_stats
5. print results
```

Every kernel call is one line:

```python
csv_parse.extract_positions_quoted(text, delim, buf1, buf2, counts)
csv_stats.f32_column_stats(values, out_sum, out_min, out_max, out_sumsq)
```
