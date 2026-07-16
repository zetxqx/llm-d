#!/usr/bin/env python3
"""Plot the cost-aware routing benchmark results.

Usage: python analysis/plot_results.py [results_dir]

Reads results/<arm>/rate<frac>_rep<rep>.json (bench aggregates) and the
matching .meta files (offered rate, taint flag), and writes two figures into
<results_dir>/figures/:

  1_p99_latency.png    — p99 latency vs offered rate
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

ARMS = ("baseline", "cost-aware")
ARM_LABEL = {
    "baseline": "baseline (k8s Service)",
    "cost-aware": "diffusion-cost-aware (llm-d EPP)",
}

FNAME_RE = re.compile(r"rate([0-9.]+)_rep(\d+)\.json$")


def load(results_dir: Path):
    """-> {arm: {offered_rate: [(rep, metrics_dict, tainted)]}}"""
    data = defaultdict(lambda: defaultdict(list))
    for arm_dir in sorted(p for p in results_dir.iterdir() if p.is_dir() and p.name in ARMS):
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


def _plot_latency(data, outdir: Path, key, fname, title, ylabel):
    fig, ax = plt.subplots()
    for arm in [a for a in ARMS if a in data]:
        rates, med, _, _ = series(data, arm, key)
        if not len(rates):
            continue
        ax.plot(rates, med, label=ARM_LABEL[arm])
    ax.set_title(title)
    ax.set_xlabel("offered rate (req/s)")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.savefig(outdir / fname)
    plt.close(fig)


def pool_desc(results_dir: Path) -> str:
    """Pool descriptor for report text, derived from the queue-depth CSVs."""
    for csv in sorted(results_dir.glob("*/rate*_rep*.csv")):
        pods = {ln.split(",")[1] for ln in csv.read_text().splitlines()[1:] if ln}
        if pods:
            return f"{len(pods)}×H100"
    return "H100 pool"


def plot_p99(data, outdir: Path, pool: str = "H100 pool"):
    _plot_latency(data, outdir, "latency_p99", "1_p99_latency.png",
                  "p99 latency vs offered rate", "p99 latency (s)")


def plot_mean(data, outdir: Path, pool: str = "H100 pool"):
    _plot_latency(data, outdir, "latency_mean", "2_mean_latency.png",
                  "mean latency vs offered rate", "mean latency (s)")


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
