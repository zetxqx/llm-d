#!/usr/bin/env python3
"""Plot the cost-aware routing benchmark results.

Usage: python analysis/plot_results.py [results_dir]

Reads results/<arm>/rate<frac>_rep<rep>.json (bench aggregates), the matching
.meta files (offered rate, taint flag) and .csv files (per-pod queue-depth
timeseries), and writes three figures into <results_dir>/figures/:

  1_p99_latency.png    — p99 latency vs offered rate, log-y (the money plot)
  2_mean_latency.png   — mean latency vs offered rate
  3_queue_depth.png    — per-pod queue depth timeseries at the highest common
                         rate, one panel per arm

Multiple reps: solid line = median, shaded band = min-max.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# --- palette (reference dataviz palette, light mode) -------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# Arms = categorical identity, fixed slot order (blue, aqua).
ARM_COLOR = {"baseline": "#2a78d6", "cost-aware": "#1baf7a"}
ARM_LABEL = {
    "baseline": "baseline (k8s Service)",
    "cost-aware": "diffusion-cost-aware (llm-d EPP)",
}
# Pods in the small multiples = one hue, stepped shades (blue 250/450/650).
POD_SHADES = ["#86b6ef", "#2a78d6", "#104281"]

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


def direct_labels(ax, ends):
    """Label each line at its end point, nudging collisions apart vertically.

    ends: [(x, y, arm)] in data coords; y-collisions are resolved in axes
    fraction space so it works on both linear and log axes.
    """
    to_frac = ax.transAxes.inverted().transform
    to_disp = ax.transData.transform
    labeled = sorted(ends, key=lambda e: to_frac(to_disp((e[0], e[1])))[1])
    prev = None
    for x, y, arm in labeled:
        fy = to_frac(to_disp((x, y)))[1]
        if prev is not None and fy - prev < 0.05:
            fy = prev + 0.05
        prev = fy
        ax.annotate(
            ARM_LABEL[arm].split(" (")[0],
            (x, y),
            xytext=(1.01, fy),
            textcoords=ax.transAxes,
            color=ARM_COLOR[arm],
            fontsize=9,
            fontweight="bold",
            va="center",
        )


def new_fig(title, subtitle):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    fig.suptitle(title, x=0.06, ha="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_title(subtitle, loc="left", fontsize=9.5, color=INK_2, pad=10)
    return fig, ax


def _plot_latency(data, outdir: Path, key, fname, title, subtitle, ylabel, log=False):
    fig, ax = new_fig(title, subtitle)
    if log:
        ax.set_yscale("log")
    ends = []
    for arm in [a for a in ("baseline", "cost-aware") if a in data]:
        rates, med, lo, hi = series(data, arm, key)
        if not len(rates):
            continue
        ax.fill_between(rates, lo, hi, color=ARM_COLOR[arm], alpha=0.12, linewidth=0)
        ax.plot(rates, med, color=ARM_COLOR[arm], linewidth=2, marker="o",
                markersize=5.5, label=ARM_LABEL[arm])
        ends.append((rates[-1], med[-1], arm))
    direct_labels(ax, ends)
    ax.set_xlabel("offered rate (req/s)", color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    if not log:
        ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")
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
        f"p99 request latency vs offered rate (log scale) — mixed-resolution t2i, {pool}, batch=1",
        "p99 latency (s)", log=True,
    )


def plot_mean(data, outdir: Path, pool: str = "H100 pool"):
    _plot_latency(
        data, outdir, "latency_mean", "2_mean_latency.png",
        "Mean latency under mixed-resolution load",
        f"Mean request latency vs offered rate — mixed-resolution t2i, {pool}, batch=1",
        "mean latency (s)",
    )


def plot_queues(data, results_dir: Path, outdir: Path):
    arms = [a for a in ("baseline", "cost-aware") if a in data]
    if not arms:
        return
    common = set.intersection(*(set(data[a]) for a in arms))
    if not common:
        return
    rate = max(common)
    fig, axes = plt.subplots(len(arms), 1, figsize=(7.2, 1.9 * len(arms) + 1.2),
                             dpi=160, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    fig.suptitle("Per-pod backlog: does routing keep the pods evenly busy?",
                 x=0.06, ha="left", fontsize=13, fontweight="bold", color=INK)
    for ax, arm in zip(axes, arms):
        style_axes(ax)
        rep, _, _ = sorted(data[arm][rate])[0]
        # Recover the file-name fraction for this offered rate.
        csvs = [c for c in (results_dir / arm).glob(f"rate*_rep{rep}.csv")
                if _meta_rate(c.with_suffix(".meta")) == rate]
        if not csvs:
            ax.set_visible(False)
            continue
        rows = [ln.split(",") for ln in csvs[0].read_text().splitlines()[1:] if ln]
        pods = sorted({r[1] for r in rows})
        t0 = min(float(r[0]) for r in rows)
        for i, pod in enumerate(pods[: len(POD_SHADES)]):
            pts = [(float(r[0]) - t0, float(r[2]) + float(r[3])) for r in rows if r[1] == pod]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=POD_SHADES[i], linewidth=2, label=f"pod {i}")
        ax.set_ylabel("in-flight + queued", color=INK_2, fontsize=8.5)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f"{ARM_LABEL[arm]}  ·  {rate:g} req/s", loc="left",
                     fontsize=9.5, color=INK_2)
        if ax is axes[0]:
            ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")
    axes[-1].set_xlabel("time since sweep start (s)", color=INK_2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outdir / "3_queue_depth.png", facecolor=SURFACE)
    plt.close(fig)


def _meta_rate(meta: Path):
    if not meta.exists():
        return None
    m = re.search(r"\brate=([0-9.]+)", meta.read_text())
    return float(m.group(1)) if m else None


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
    plot_queues(data, results_dir, outdir)
    print(f"figures written to {outdir}/")
    for arm in data:
        rates = sorted(data[arm])
        print(f"  arm {arm}: {len(rates)} rate points, "
              f"{sum(len(v) for v in data[arm].values())} runs")


if __name__ == "__main__":
    main()
