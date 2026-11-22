"""Combine QSS and TSL metrics into one tidy CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    files = [args.results / "baseline_metrics.csv"] + [
        args.results / f"qss_{name}.csv" for name in ("ETTh1", "ETTm1", "weather", "traffic")
    ]
    rows = []
    for file in files:
        if not file.exists():
            continue
        with file.open(newline="", encoding="utf-8") as f:
            rows.extend(dict(row) for row in csv.DictReader(f))
    if not rows:
        raise SystemExit("No metric CSVs found; run the model evaluations first.")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    out = args.results / "final_metrics.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} metric rows to {out.resolve()}")


if __name__ == "__main__":
    main()
