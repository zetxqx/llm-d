#!/usr/bin/env python3
"""Plot the cost-aware routing benchmark results.

Usage: python analysis/plot_results.py [results_dir]

Reads results/<arm>/rate<frac>_rep<rep>.json (bench aggregates) and the
matching .meta files (offered rate, taint flag), and writes two figures into
<results_dir>/figures/:

  1_p99_latency.png    — p99 latency vs offered rate (the money plot)
  2_mean_latency.png   — mean latency vs offered rate

Multiple reps: the line is the median over untainted reps.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- palette (reference dataviz palette, light mode) -------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# Arms = categorical identity, fixed slot order (blue, green); marker shape is
# the secondary (non-color) encoding.
ARM_COLOR = {"baseline": "#2a78d6", "cost-aware": "#128a5f"}
ARM_MARKER = {"baseline": "o", "cost-aware": "s"}
ARM_LABEL = {
    "baseline": "baseline (k8s Service)",
    "cost-aware": "diffusion-cost-aware (llm-d EPP)",
}

FNAME_RE = re.compile(r"rate([0-9.]+)_rep(\d+)\.json$")


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)


def load(results_dir: Path):
    """-> {arm: {offered_rate: [(rep, metrics_dict, tainted)]}}"""
    data = defaultdict(lambda: defaultdict(list))
    for arm_dir in sorted(p for p in results_dir.iterdir() if p.is_dir() and p.name in ARM_COLOR):
        arm = arm_dir.name
        for jf in sorted(arm_dir.glob("rate*_rep*.json")):
            m = FNAME_RE.search(jf.name)
            if not m:
                continue
            frac, rep = float(m.group(1)), int(m.group(2))
            metrics = json.loads(jf.read_text())
            rate, tainted = frac, False
            meta = jf.with_suffix(".meta")
            if meta.exists():
                head = meta.read_text()
                rm = re.search(r"\brate=([0-9.]+)", head)
                if rm:
                    rate = float(rm.group(1))
                tainted = "tainted=true" in head
            data[arm][rate].append((rep, metrics, tainted))
    return data


def series(data, arm, key):
    """-> rates, median, lo, hi over untainted reps (falls back to all reps)."""
    rates, med, lo, hi = [], [], [], []
    for rate in sorted(data[arm]):
        vals = [m[key] for _, m, t in data[arm][rate] if not t and key in m]
        if not vals:
            vals = [m[key] for _, m, t in data[arm][rate] if key in m]
        if not vals:
            continue
        rates.append(rate)
        med.append(float(np.median(vals)))
        lo.append(min(vals))
        hi.append(max(vals))
    return np.array(rates), np.array(med), np.array(lo), np.array(hi)


def new_fig(title, subtitle):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    fig.suptitle(title, x=0.06, ha="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_title(subtitle, loc="left", fontsize=9.5, color=INK_2, pad=10)
    return fig, ax


def _plot_latency(data, outdir: Path, key, fname, title, subtitle, ylabel):
    fig, ax = new_fig(title, subtitle)
    for arm in [a for a in ("baseline", "cost-aware") if a in data]:
        rates, med, _, _ = series(data, arm, key)
        if not len(rates):
            continue
        ax.plot(rates, med, color=ARM_COLOR[arm], linewidth=2,
                marker=ARM_MARKER[arm], markersize=7, label=ARM_LABEL[arm])
    ax.set_xlabel("offered rate (req/s)", color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outdir / fname, facecolor=SURFACE)
    plt.close(fig)


def pool_desc(results_dir: Path) -> str:
    """Pool descriptor for subtitles, derived from the queue-depth CSVs."""
    for csv in sorted(results_dir.glob("*/rate*_rep*.csv")):
        pods = {ln.split(",")[1] for ln in csv.read_text().splitlines()[1:] if ln}
        if pods:
            return f"{len(pods)}×H100"
    return "H100 pool"


def plot_p99(data, outdir: Path, pool: str = "H100 pool"):
    _plot_latency(
        data, outdir, "latency_p99", "1_p99_latency.png",
        "Cost-aware routing cuts tail latency",
        f"p99 request latency vs offered rate — mixed-resolution t2i, {pool}, batch=1",
        "p99 latency (s)",
    )


def plot_mean(data, outdir: Path, pool: str = "H100 pool"):
    _plot_latency(
        data, outdir, "latency_mean", "2_mean_latency.png",
        "Mean latency under mixed-resolution load",
        f"Mean request latency vs offered rate — mixed-resolution t2i, {pool}, batch=1",
        "mean latency (s)",
    )


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "results"
    data = load(results_dir)
    if not data:
        sys.exit(f"no results found under {results_dir}")
    outdir = results_dir / "figures"
    outdir.mkdir(exist_ok=True)
    pool = pool_desc(results_dir)
    plot_p99(data, outdir, pool)
    plot_mean(data, outdir, pool)
    print(f"figures written to {outdir}/")
    for arm in data:
        rates = sorted(data[arm])
        print(f"  arm {arm}: {len(rates)} rate points, "
              f"{sum(len(v) for v in data[arm].values())} runs")


if __name__ == "__main__":
    main()
