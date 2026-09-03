#!/usr/bin/env python3
"""Fleet capacity sweep through the EPP router, with close_session enabled.

Runs the weka trace replay at increasing session concurrency (default 16/32/48)
against the WHOLE 4-pod fleet via the EPP service (binding-based session
affinity + close routing, image session-control-2003-65a81ad9). Per stage:

  1. rollout-restart the EPP (fresh binding table) and flush all sglang pods
  2. one in-cluster inference-perf Job, base_url = EPP :80, close on,
     idle_cap 30s (+ matching max_wait_ms via build_config)
  3. collect reports, the envoy access log slice (per-request pod
     attribution -> binding stability + close delivery accuracy), and
     per-pod release_session counts

Outputs under /tmp/router-sweep/c<N>/. Analysis is offline; this script only
prints the same per-stage summary as the A/B driver.

Usage:
  python3 run_router_capacity_sweep.py                  # 16,32,48 x 3600s
  python3 run_router_capacity_sweep.py --concurrencies 16,32 --timeout-sec 1800
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import run_close_session_bench as base

WORK = Path("/tmp/router-sweep")
SWEEP_SAMPLE_BYTES = 250_000_000  # ~160 traces; the 60MB A/B slice tops out at 39

NAMESPACE = base.NAMESPACE
EPP_DEPLOY = "session-control-epp"


def prepare_sweep_traces() -> list:
    """250MB slice, own cache (base.prepare_traces would reuse the 60MB one)."""
    WORK.mkdir(parents=True, exist_ok=True)
    sample = WORK / "traces_sample.jsonl"
    if not sample.exists():
        print(f"downloading first {SWEEP_SAMPLE_BYTES // 1_000_000}MB of traces.jsonl ...")
        subprocess.check_call(["curl", "-sL", "-r", f"0-{SWEEP_SAMPLE_BYTES}",
                               base.SAMPLE_URL, "-o", str(sample)])
    meta = []
    for i, line in enumerate(sample.read_bytes().split(b"\n")[:-1]):
        t = json.loads(line)
        parents = [r for r in t["requests"] if r["type"] != "subagent"]
        sas = sum(1 for r in t["requests"] if r["type"] == "subagent")
        out = sum(r["out"] for r in parents)
        meta.append((i, t["id"][:10], len(parents), sas, out, ""))
    print(f"sample holds {len(meta)} complete traces")
    return meta


def epp_ip() -> str:
    return base.kubectl("get", "service", EPP_DEPLOY, "-o", "jsonpath={.spec.clusterIP}")


def reset_fleet(pods: list):
    base.kubectl("rollout", "restart", f"deployment/{EPP_DEPLOY}")
    base.kubectl("rollout", "status", f"deployment/{EPP_DEPLOY}", "--timeout=3m")
    for p in pods:
        print(f"flush {p}:", base.sglang_exec(p, "/flush_cache")[:80])


def sglang_pods() -> list:
    out = base.kubectl("get", "pods", "-l", "llm-d.ai/engine-type=sglang",
                       "-o", "jsonpath={range .items[?(@.status.phase=='Running')]}{.metadata.name}{\"\\n\"}{end}")
    return [p for p in out.splitlines() if p]


def run_stage(c: int, timeout_s: int, idle_cap: float, meta: list, url: str, pods: list):
    d = WORK / f"c{c}"
    if (d / "reports" / "per_request_lifecycle_metrics.json").exists():
        print(f"=== c={c}: reports exist, skipping (delete {d} to rerun) ===")
        return
    import shutil
    shutil.rmtree(d, ignore_errors=True)
    (d / "reports").mkdir(parents=True)

    sel = base.select_traces(meta, "ab", c)
    if len(sel) < c:
        sys.exit(f"only {len(sel)} eligible traces for c={c}")
    indices = [m[0] for m in sel]
    print(f"\n=== stage c={c}: {timeout_s}s, traces {len(sel)} ===")
    reset_fleet(pods)
    stage_t0 = time.time()

    cfg = base.build_config({"trace_directory": "/work/traces"}, len(sel), timeout_s,
                            close=True, report_dir="/work/reports",
                            idle_cap=idle_cap, base_url=url)
    # the in-cluster fetch needs the bigger slice
    old_bytes = base.SAMPLE_BYTES
    base.SAMPLE_BYTES = SWEEP_SAMPLE_BYTES
    name, cm, job = base.render_job(f"sweep-c{c}", cfg, indices)
    base.SAMPLE_BYTES = old_bytes

    base.kubectl("delete", "job", name, "--ignore-not-found")
    base.kubectl("delete", "configmap", name, "--ignore-not-found")
    for manifest in (cm, job):
        p = subprocess.run(["kubectl", "apply", "-f", "-"], input=json.dumps(manifest),
                           text=True, capture_output=True)
        if p.returncode != 0:
            raise RuntimeError(f"apply failed: {p.stderr}")

    bench_pod, deadline = "", time.time() + timeout_s + 2700
    while time.time() < deadline:
        time.sleep(60)
        if not bench_pod:
            bench_pod = base.kubectl("get", "pod", "-l", f"job-name={name}",
                                     "-o", "jsonpath={.items[0].metadata.name}", check=False)
            continue
        phase = base.kubectl("get", "pod", bench_pod, "-o", "jsonpath={.status.phase}", check=False)
        if phase == "Failed":
            print(base.kubectl("logs", bench_pod, "--tail=30", check=False))
            raise RuntimeError(f"stage c={c} job pod failed")
        r = subprocess.run(["kubectl", "-n", NAMESPACE, "exec", bench_pod, "-c", "bench",
                            "--", "test", "-f", "/work/reports/DONE"], capture_output=True)
        print(f"  {time.strftime('%H:%M:%S')} c={c} phase={phase} done={'yes' if r.returncode == 0 else 'no'}")
        if r.returncode == 0:
            break
    else:
        raise RuntimeError(f"stage c={c} did not finish before deadline")

    subprocess.run(["kubectl", "-n", NAMESPACE, "cp", f"{bench_pod}:/work/reports",
                    str(d / "reports"), "-c", "bench"], check=True)
    base.kubectl("delete", "job", name, "--wait=false")
    base.kubectl("delete", "configmap", name)

    # evidence for binding stability + close delivery accuracy
    since = f"{int(time.time() - stage_t0) + 120}s"
    epp_pod = base.kubectl("get", "pod", "-l", f"llm-d-router-gateway={EPP_DEPLOY}",
                           "-o", "jsonpath={.items[0].metadata.name}")
    (d / "envoy_access.log").write_text(
        base.kubectl("logs", epp_pod, "-c", "envoy-proxy", f"--since={since}", check=False))
    releases = {}
    for p in pods:
        out = base.kubectl("logs", p, "-c", "modelserver", f"--since={since}", check=False)
        lines = [ln for ln in out.splitlines() if "release_session" in ln]
        releases[p] = {"total": len(lines),
                       "noop": sum(1 for ln in lines if "indexed 0 component leaves" in ln)}
        (d / f"metrics_after_{p}.txt").write_text(base.sglang_exec(p, "/metrics"))
    (d / "close_delivery.json").write_text(json.dumps(releases, indent=1))
    print("release_session per pod:", json.dumps(releases))
    base.summarize(f"c={c}", base.read_reports(d / "reports"))


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrencies", default="16,32,48,64")
    ap.add_argument("--timeout-sec", type=int, default=3600)
    ap.add_argument("--idle-cap-sec", type=float, default=30.0)
    args = ap.parse_args()

    cs = [int(x) for x in args.concurrencies.split(",")]
    meta = prepare_sweep_traces()
    pods = sglang_pods()
    if len(pods) < 4:
        print(f"WARNING: only {len(pods)} sglang pods Running")
    url = f"http://{epp_ip()}:80"
    print(f"EPP: {url}; pods: {pods}")

    for c in cs:
        run_stage(c, args.timeout_sec, args.idle_cap_sec, meta, url, pods)

    print("\n=== sweep done ===")
    for c in cs:
        r = base.read_reports(WORK / f"c{c}" / "reports")
        base.summarize(f"c={c}", r)
    print(f"artifacts under {WORK}/c<N>/")


if __name__ == "__main__":
    main()
