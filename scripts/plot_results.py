#!/usr/bin/env python3
"""Plot throughput and accuracy vs path count from mc_pricer CSV output.

Usage:
    python scripts/plot_results.py results/out.csv [more.csv ...] [--out results/throughput.png]

If the same (impl, paths) appears more than once (CSV is append-only), the
most recent row wins.
"""

import argparse
import csv
import math
from collections import defaultdict

import matplotlib.pyplot as plt

# Fixed order and colors so an implementation keeps its color across plots,
# even if some are missing from a given CSV. Markers are a second encoding so
# lines stay distinguishable without color (colorblind readers, print).
IMPLS = ["cpu_single", "cpu_fast", "cpu_omp", "gpu_naive", "gpu_opt"]
COLORS = {
    "cpu_single": "#2a78d6",
    "cpu_fast": "#eb6834",
    "cpu_omp": "#1baf7a",
    "gpu_naive": "#eda100",
    "gpu_opt": "#e87ba4",
}
MARKERS = {"cpu_single": "o", "cpu_fast": "s", "cpu_omp": "^", "gpu_naive": "D", "gpu_opt": "v"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"


def load(paths):
    latest = {}
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                latest[(row["impl"], int(row["paths"]))] = row
    series = defaultdict(list)
    for (impl, n), row in latest.items():
        series[impl].append((n, float(row["paths_per_sec"]), float(row["error_in_se"])))
    for pts in series.values():
        pts.sort()
    return series


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel("paths", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_xscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(which="both", colors=MUTED, labelcolor=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="results/throughput.png")
    args = ap.parse_args()

    series = load(args.csv)
    if not series:
        raise SystemExit("no rows found")

    fig, (ax_tp, ax_err) = plt.subplots(1, 2, figsize=(12, 4.5))

    for impl in IMPLS + sorted(set(series) - set(IMPLS)):
        if impl not in series:
            continue
        n, pps, err_se = zip(*series[impl])
        kw = dict(color=COLORS.get(impl, MUTED), marker=MARKERS.get(impl, "o"),
                  markersize=6, linewidth=2, label=impl)
        ax_tp.plot(n, pps, **kw)
        ax_err.plot(n, err_se, **kw)

    style(ax_tp, "Throughput", "paths / second")
    ax_tp.set_yscale("log")
    ax_tp.legend(frameon=False, fontsize=8, labelcolor=INK)

    style(ax_err, "Error vs Black-Scholes (in std errors)", "(MC - BS) / SE")
    # A correct pricer should land inside +/-3 SE almost always; drifting
    # outside as n grows means bias, not noise.
    for y in (-3, 3):
        ax_err.axhline(y, color=MUTED, linestyle="--", linewidth=1)
    ax_err.axhline(0, color=GRID, linewidth=1)
    lim = max(4.0, max(abs(e) for pts in series.values() for _, _, e in pts if math.isfinite(e)) + 0.5)
    ax_err.set_ylim(-lim, lim)
    ax_err.legend(frameon=False, fontsize=8, labelcolor=INK)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
