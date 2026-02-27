#!/usr/bin/env python3
"""
Generate test CSV files for Eastat benchmarking.

Usage:
    python generate_test.py                  # 10k rows (default)
    python generate_test.py --rows=1000000   # 1M rows
    python generate_test.py --size=100MB     # target file size
"""

import argparse
import os
import random
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Columns: id (int), name (string), price (float), quantity (int), category (string), score (float)
NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Heidi",
    "Ivan", "Judy", "Karl", "Laura", "Mallory", "Niaj", "Oscar", "Peggy",
    "Quentin", "Ruth", "Sybil", "Trent", "Ursula", "Victor", "Wendy",
    "Xander", "Yvonne", "Zelda",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
]

CATEGORIES = [
    "Electronics", "Clothing", "Food", "Books", "Sports", "Home", "Garden",
    "Automotive", "Health", "Beauty", "Toys", "Office", "Music", "Movies",
]

NULL_RATE = 0.001


def generate_row(row_id):
    """Generate a single CSV row."""
    name = random.choice(NAMES) + " " + random.choice(LAST_NAMES)
    price = round(random.uniform(0.01, 99999.99), 2)
    quantity = random.randint(0, 10000)
    category = random.choice(CATEGORIES)
    score = round(random.uniform(0.0, 100.0), 4)

    if random.random() < NULL_RATE:
        name = ""
    if random.random() < NULL_RATE:
        price_str = ""
    else:
        price_str = f"{price}"
    if random.random() < NULL_RATE:
        quantity_str = ""
    else:
        quantity_str = str(quantity)
    if random.random() < NULL_RATE:
        score_str = ""
    else:
        score_str = f"{score}"

    # Occasionally add quoted fields with commas
    if random.random() < 0.01:
        name = f'"Last, First: {name}"'

    return f"{row_id},{name},{price_str},{quantity_str},{category},{score_str}\n"


def estimate_row_size():
    """Estimate average row size in bytes."""
    sizes = [len(generate_row(i).encode()) for i in range(1000)]
    return sum(sizes) / len(sizes)


def generate_csv(output_path, n_rows, target_size=None):
    """Generate a CSV file with the specified number of rows."""
    header = "id,name,price,quantity,category,score\n"

    if target_size:
        avg_row = estimate_row_size()
        n_rows = int((target_size - len(header)) / avg_row)
        print(f"Target size: {target_size / (1024**2):.1f} MB, estimated rows: {n_rows:,}")

    print(f"Generating {n_rows:,} rows to {output_path}...")
    t0 = time.perf_counter()

    with open(output_path, 'w', buffering=1024*1024) as f:
        f.write(header)
        for i in range(1, n_rows + 1):
            f.write(generate_row(i))
            if i % 1_000_000 == 0:
                elapsed = time.perf_counter() - t0
                rate = i / elapsed
                print(f"  {i:>12,} rows ({rate:,.0f} rows/s)", end='\r')

    elapsed = time.perf_counter() - t0
    file_size = os.path.getsize(output_path)
    print(f"\nDone: {n_rows:,} rows, {file_size / (1024**2):.1f} MB in {elapsed:.1f}s")
    return output_path


def parse_size(s):
    """Parse size string like '100MB' or '1GB'."""
    s = s.upper().strip()
    multipliers = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            return int(float(s[:-len(suffix)]) * mult)
    return int(s)


def main():
    parser = argparse.ArgumentParser(description='Generate test CSV files')
    parser.add_argument('--rows', type=int, default=None, help='Number of rows')
    parser.add_argument('--size', type=str, default=None, help='Target file size (e.g., 100MB)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)

    if args.size:
        target = parse_size(args.size)
        output = args.output or str(SCRIPT_DIR / f"test_{args.size.lower()}.csv")
        generate_csv(output, 0, target_size=target)
    elif args.rows:
        output = args.output or str(SCRIPT_DIR / f"test_{args.rows}.csv")
        generate_csv(output, args.rows)
    else:
        output = args.output or str(SCRIPT_DIR / "test_10k.csv")
        generate_csv(output, 10_000)


if __name__ == '__main__':
    main()
