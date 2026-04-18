#!/usr/bin/env python3
"""Benchmark ECGtizer extraction performance.

Measures wall-clock time for each pipeline stage across DPI values and
extraction methods. Results are printed as tables and optionally saved
as a PNG bar chart.

Usage
-----
    python -m scripts.benchmark
    python -m scripts.benchmark --dpi 200 300 500 --methods lazy full fragmented
    python -m scripts.benchmark --save output/benchmark.png
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from ecgtizer.PDF2XML import (
    convert_PDF2image,
    check_noise_type,
    text_extraction,
    tracks_extraction,
    lead_extraction,
    lead_cutting,
)

SAMPLE_PDF = "data/PTB-XL/PDF/00009_hr.pdf"


def _time(func, *args, **kwargs):
    """Run *func* and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - t0


def benchmark_pipeline(pdf_path: str, dpi: int, method: str) -> dict[str, float]:
    """Run the full pipeline once and return per-stage timings (seconds)."""
    timings = {}

    # Stage 1: PDF -> image
    (pages, n_pages, ok), timings["pdf_to_image"] = _time(
        convert_PDF2image, pdf_path, dpi
    )
    if not ok:
        raise RuntimeError(f"PDF conversion failed for {pdf_path}")
    image = np.array(pages[0])

    # Stage 2: noise / format detection
    (TYPE, NOISE), timings["noise_detect"] = _time(
        check_noise_type, image, dpi, False
    )

    # Stage 3: text masking
    image_clean, timings["text_mask"] = _time(
        text_extraction, image, 0, dpi, NOISE, TYPE, False
    )

    # Stage 4: track segmentation
    (dic_tracks_img, _, _), timings["track_segment"] = _time(
        tracks_extraction, image, TYPE, dpi, "", NOISE=NOISE, DEBUG=False
    )

    # Stage 5: waveform extraction
    (dic_extracted, dic_bin, _), timings["extraction"] = _time(
        lead_extraction, dic_tracks_img, method, TYPE, NOISE, False
    )

    # Stage 6: lead cutting / calibration
    _, timings["lead_cut"] = _time(
        lead_cutting, dic_extracted, dpi, TYPE, "", 0, NOISE, False
    )

    timings["total"] = sum(timings.values())
    return timings


def run_benchmarks(pdf_path: str, dpi_values: list[int], methods: list[str], repeats: int = 1):
    """Run benchmarks and return a list of result dicts."""
    results = []
    for dpi in dpi_values:
        for method in methods:
            all_timings = []
            for _ in range(repeats):
                t = benchmark_pipeline(pdf_path, dpi, method)
                all_timings.append(t)
            # Average over repeats
            avg = {}
            for key in all_timings[0]:
                avg[key] = np.mean([t[key] for t in all_timings])
            results.append({"dpi": dpi, "method": method, **avg})
    return results


def print_results(results: list[dict]):
    """Pretty-print benchmark results as a table."""
    stages = ["pdf_to_image", "noise_detect", "text_mask", "track_segment", "extraction", "lead_cut", "total"]
    headers = ["DPI", "Method"] + [s.replace("_", " ").title() for s in stages]

    # Calculate column widths
    widths = [max(len(h), 8) for h in headers]
    for r in results:
        widths[0] = max(widths[0], len(str(r["dpi"])))
        widths[1] = max(widths[1], len(r["method"]))
        for i, s in enumerate(stages):
            widths[i + 2] = max(widths[i + 2], len(f"{r[s]:.3f}s"))

    # Header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "-+-".join("-" * w for w in widths)
    print(f"\n{header_line}")
    print(sep)

    # Rows
    for r in results:
        row = [str(r["dpi"]).ljust(widths[0]), r["method"].ljust(widths[1])]
        for i, s in enumerate(stages):
            val = f"{r[s]:.3f}s"
            row.append(val.ljust(widths[i + 2]))
        print(" | ".join(row))
    print()


def save_chart(results: list[dict], path: str):
    """Save a grouped bar chart of total times."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted(set(r["method"] for r in results))
    dpis = sorted(set(r["dpi"] for r in results))

    x = np.arange(len(dpis))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        totals = [next(r["total"] for r in results if r["dpi"] == d and r["method"] == method) for d in dpis]
        ax.bar(x + i * width, totals, width, label=method)

    ax.set_xlabel("DPI")
    ax.set_ylabel("Total time (s)")
    ax.set_title("ECGtizer Pipeline Benchmark")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([str(d) for d in dpis])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark ECGtizer pipeline")
    parser.add_argument("--pdf", default=SAMPLE_PDF, help="PDF file to benchmark")
    parser.add_argument("--dpi", nargs="+", type=int, default=[200, 300, 500], help="DPI values to test")
    parser.add_argument("--methods", nargs="+", default=["lazy", "full", "fragmented"], help="Extraction methods")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per configuration")
    parser.add_argument("--save", default=None, help="Save bar chart to this path (e.g. output/benchmark.png)")
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(f"Error: {args.pdf} not found")
        return

    print(f"Benchmarking: {args.pdf}")
    print(f"DPI values: {args.dpi}")
    print(f"Methods: {args.methods}")
    print(f"Repeats: {args.repeats}")

    results = run_benchmarks(args.pdf, args.dpi, args.methods, args.repeats)
    print_results(results)

    if args.save:
        save_chart(results, args.save)


if __name__ == "__main__":
    main()
