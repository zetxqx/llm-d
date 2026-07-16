#!/usr/bin/env python3
"""Generate the benchmark report: figures + results/REPORT.md.

Usage: python analysis/generate_report.py [results_dir]

Renders the two figures (via plot_results.py) and writes a self-contained
markdown report with:
  - run configuration (calibration, capacity)
  - headline latency comparison (p99/mean A vs B per rate, with reductions)
  - per-arm results table (offered rate, p99/p95/mean latency, throughput)
  - embedded figures and notes on tainted points (spot preemption)
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import plot_results  # noqa: E402  (same-directory module)


def fmt(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def latency_comparison(data):
    """Markdown table of p99/mean per rate for both arms, with reductions."""
    if not {"baseline", "cost-aware"} <= set(data):
        return None
    a99 = dict(zip(*plot_results.series(data, "baseline", "latency_p99")[:2]))
    b99 = dict(zip(*plot_results.series(data, "cost-aware", "latency_p99")[:2]))
    amean = dict(zip(*plot_results.series(data, "baseline", "latency_mean")[:2]))
    bmean = dict(zip(*plot_results.series(data, "cost-aware", "latency_mean")[:2]))
    common = sorted(set(a99) & set(b99))
    if not common:
        return None
    lines = [
        "| offered rate (req/s) | p99 baseline → cost-aware (s) | p99 reduction | mean baseline → cost-aware (s) | mean reduction |",
        "|---|---|---|---|---|",
    ]
    for r in common:
        lines.append(
            f"| {r:g} | {fmt(a99[r], 1)} → {fmt(b99[r], 1)} | **{(1 - b99[r] / a99[r]):.0%}** "
            f"| {fmt(amean[r], 1)} → {fmt(bmean[r], 1)} | {(1 - bmean[r] / amean[r]):.0%} |"
        )
    return "\n".join(lines)


def arm_table(data, arm):
    lines = [
        "| offered rate (req/s) | reps | p99 latency s (median) | p95 latency s | mean latency s | throughput req/s | failed | tainted |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rate in sorted(data[arm]):
        runs = data[arm][rate]
        clean = [(m, t) for _, m, t in runs if not t] or [(m, t) for _, m, t in runs]
        med = lambda key: plot_results.np.median([m[key] for m, _ in clean if key in m])  # noqa: E731
        failed = sum(m.get("failed_requests", 0) for m, _ in clean)
        tainted = sum(1 for _, _, t in runs if t)
        lines.append(
            f"| {rate:g} | {len(runs)} | {fmt(med('latency_p99'), 1)} "
            f"| {fmt(med('latency_p95'), 1)} | {fmt(med('latency_mean'), 1)} "
            f"| {fmt(med('throughput_qps'), 3)} | {failed or ''} | {tainted or ''} |"
        )
    return "\n".join(lines)


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "results"
    data = plot_results.load(results_dir)
    if not data:
        sys.exit(f"no results found under {results_dir}")

    # Figures first (also validates the data files).
    outdir = results_dir / "figures"
    outdir.mkdir(exist_ok=True)
    pool = plot_results.pool_desc(results_dir)
    plot_results.plot_p99(data, outdir, pool)
    plot_results.plot_mean(data, outdir, pool)

    # Calibration lives at the results/ root (shared across runs); a run dir
    # sits one level below it.
    calib = {}
    for cand in (results_dir / "calibration.json", results_dir.parent / "calibration.json"):
        if cand.exists():
            calib = json.loads(cand.read_text())
            break

    # Run metadata (written by run_all.sh).
    workload = "dataset_c.json"
    run_info = results_dir / "run.info"
    info_line = run_info.read_text().strip() if run_info.exists() else ""
    for tok in info_line.split():
        if tok.startswith("workload="):
            workload = tok.split("=", 1)[1]

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = [
        "# Cost-aware routing benchmark — report",
        "",
        f"Generated {stamp}. Workload: **{workload}** (mixed-resolution t2i, "
        f"see `workloads/`). Pool: {pool} running `Qwen/Qwen-Image`, batch=1."
        + (f" Run: `{info_line}`." if info_line else ""),
        "",
        "| arm | routing policy |",
        "|---|---|",
    ]
    for arm in sorted(data):
        md.append(f"| {arm} | {plot_results.ARM_LABEL.get(arm, arm)} |")

    if calib:
        md += [
            "",
            "## Calibration",
            "",
            "| bucket | measured service time (s) |",
            "|---|---|",
        ]
        for bucket, s in calib.get("service_time_s", {}).items():
            md.append(f"| {bucket}² | {fmt(s)} |")
        md += [
            "",
            f"Mixed mean service time S_mix = **{fmt(calib.get('S_mix_s', 0))} s** → "
            f"{calib.get('replicas', 'N')}-pod capacity ≈ "
            f"**{fmt(calib.get('capacity_rps', 0), 3)} req/s**.",
        ]

    comparison = latency_comparison(data)
    if comparison:
        md += [
            "",
            "## Headline: latency at the same offered load",
            "",
            "Median over repetitions; both arms replay the identical request",
            "sequence and arrival timeline, so rows are directly comparable.",
            "",
            comparison,
        ]

    md += [
        "",
        "![p99 latency vs offered rate](figures/1_p99_latency.png)",
        "",
        "![mean latency vs offered rate](figures/2_mean_latency.png)",
        "",
    ]

    for arm in sorted(data):
        md += [f"## {plot_results.ARM_LABEL.get(arm, arm)}", "", arm_table(data, arm), ""]

    md += [
        "",
        "## Notes",
        "",
        "- Median over repetitions; points marked tainted (pod churn mid-run, "
        "spot preemption) are excluded from medians when clean reps exist.",
        "- The request mix and Poisson arrival timelines are identical across "
        "arms (seeded), so rows at the same rate are directly comparable.",
        "- Raw data: `<arm>/rate<r>_rep<n>.json` (bench aggregates), `.csv` "
        "(per-pod queue-depth timeseries), `.meta` (pod set, taint flag).",
    ]

    report = results_dir / "REPORT.md"
    report.write_text("\n".join(md) + "\n")
    print(f"wrote {report}")
    print(f"figures in {outdir}/")


if __name__ == "__main__":
    main()
