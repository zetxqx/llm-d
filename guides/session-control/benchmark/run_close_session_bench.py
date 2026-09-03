#!/usr/bin/env python3
"""Close-session A/B benchmark driver for the weka trace replay workload.

Runs against ONE sglang pod (kubectl port-forward), so the /close_session call
always lands on the pod that owns the session. Three phases:

  smoke  - (1) direct API probe: open a radix session, verify cached_tokens on
           turn 2, close it, verify /close_session returns 200;
           (2) a tiny inference-perf run (2 shortest traces) that must finish
           with successful requests and zero failures.
  ab     - two identical truncated runs (same traces, same seed, same timeout),
           arm "close" sends /close_session after each subagent stream's final
           response, arm "noclose" never closes. Cache is flushed between arms.
  both   - smoke first, then ab (default).

Traces come from a local sample of semianalysisai/cc-traces-weka-with-subagents
(first 60MB slice, 39 complete traces) so no 640MB download is needed.

Usage:
  python3 run_close_session_bench.py --mode smoke
  python3 run_close_session_bench.py --mode ab --sessions 12 --timeout-sec 2400
"""

import argparse
import glob
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

NAMESPACE = "llm-d-session-control"
MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
LOCAL_PORT = 8000
BASE = f"http://localhost:{LOCAL_PORT}"
SAMPLE_URL = (
    "https://huggingface.co/datasets/semianalysisai/"
    "cc-traces-weka-with-subagents-060826-256k/resolve/main/traces.jsonl"
)
SAMPLE_BYTES = 60_000_000
INFERENCE_PERF_DIR = Path.home() / "projects/sessioncontrol/inference-perf"
WORK = Path("/tmp/close-session-bench")


def sh(cmd: list, **kw) -> str:
    return subprocess.check_output(cmd, text=True, **kw).strip()


def http(method: str, url: str, body: dict = None, timeout: int = 300) -> tuple:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read() or b"{}"
            try:
                return r.status, json.loads(raw)
            except json.JSONDecodeError:  # e.g. /flush_cache returns plain text
                return r.status, {"raw": raw.decode("utf-8", errors="ignore")[:200]}
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:200]}


# ---------------------------------------------------------------- trace prep
def prepare_traces() -> list:
    """Download the sample slice if needed, split into per-trace JSON files.
    Returns [(index, id, n_parent_turns, n_subagents, total_out_tokens)]."""
    sample = WORK / "traces_sample.jsonl"
    if not sample.exists():
        cached = Path("/tmp/weka_sample.bin")
        if cached.exists():
            shutil.copy(cached, sample)
        else:
            print(f"downloading first {SAMPLE_BYTES // 1_000_000}MB of traces.jsonl ...")
            subprocess.check_call(["curl", "-sL", "-r", f"0-{SAMPLE_BYTES}",
                                   SAMPLE_URL, "-o", str(sample)])
    tdir = WORK / "traces"
    tdir.mkdir(parents=True, exist_ok=True)
    meta = []
    lines = sample.read_bytes().split(b"\n")[:-1]  # drop the truncated tail line
    for i, line in enumerate(lines):
        t = json.loads(line)
        parents = [r for r in t["requests"] if r["type"] != "subagent"]
        sas = sum(1 for r in t["requests"] if r["type"] == "subagent")
        out = sum(r["out"] for r in parents)
        f = tdir / f"{i:03d}_{t['id'][:10]}.json"
        if not f.exists():
            f.write_bytes(line)
        meta.append((i, t["id"][:10], len(parents), sas, out, str(f)))
    print(f"prepared {len(meta)} traces in {tdir}")
    return meta


def select_traces(meta: list, phase: str, n: int) -> list:
    if phase == "smoke":
        # 2 cheapest traces: few turns and little decode work
        return sorted(meta, key=lambda m: m[4] / 60 + m[2] * 3)[:2]
    # ab: subagent-rich first (that is where close fires), skip the >400-turn
    # monsters whose critical path alone exceeds any reasonable timeout
    ok = [m for m in meta if m[2] <= 400]
    rich = sorted([m for m in ok if m[3] > 0], key=lambda m: -m[3])
    rest = sorted([m for m in ok if m[3] == 0], key=lambda m: abs(m[2] - 80))
    return (rich + rest)[:n]


# -------------------------------------------------------------- k8s plumbing
_pf_proc = None


def ready_pods() -> list:
    out = sh(["kubectl", "get", "pods", "-n", NAMESPACE,
              "-l", "llm-d.ai/engine-type=sglang",
              "-o", 'jsonpath={range .items[*]}{.metadata.name}{" "}{.status.conditions[?(@.type=="Ready")].status}{"\\n"}{end}'])
    pods = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == "True":
            pods.append(parts[0])
    return pods


def pick_pod(explicit: str) -> str:
    if explicit:
        return explicit
    pods = ready_pods()
    if not pods:
        sys.exit("no Ready sglang pod found")
    return pods[0]


def port_forward(pod: str):
    global _pf_proc
    _pf_proc = subprocess.Popen(
        ["kubectl", "port-forward", "-n", NAMESPACE, f"pod/{pod}",
         f"{LOCAL_PORT}:8000"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        try:
            with socket.create_connection(("localhost", LOCAL_PORT), timeout=1):
                return
        except OSError:
            time.sleep(1)
    sys.exit("port-forward did not become ready")


def preflight(pod: str):
    st, info = http("GET", f"{BASE}/get_server_info")
    assert st == 200, f"/get_server_info -> {st}"
    assert info.get("enable_session_radix_cache") is True, "session radix cache is OFF on this pod"
    print(f"preflight ok: pod={pod} sglang={info.get('version')} "
          f"kv_pool={info.get('max_total_num_tokens')} session_radix=on")


def flush_cache():
    st, _ = http("POST", f"{BASE}/flush_cache", {})
    print(f"flush_cache -> HTTP {st}")


# ------------------------------------------------------------------- smoke 1
def smoke_api():
    sid = f"smoke-{int(time.time())}"
    msgs = [{"role": "user", "content": "Remember the magic word: pineapple. " * 40
             + "Write one short sentence."}]
    body = {"model": MODEL, "session_id": sid, "max_tokens": 20, "messages": msgs}
    st, r1 = http("POST", f"{BASE}/v1/chat/completions", body)
    assert st == 200, f"turn1 -> {st}: {r1}"
    msgs = msgs + [{"role": "assistant", "content": r1["choices"][0]["message"]["content"]},
                   {"role": "user", "content": "What was the magic word?"}]
    st, r2 = http("POST", f"{BASE}/v1/chat/completions",
                  {"model": MODEL, "session_id": sid, "max_tokens": 20, "messages": msgs})
    assert st == 200, f"turn2 -> {st}"
    cached = (r2.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    assert cached and cached > 0, f"turn2 cached_tokens={cached}, prefix cache did not hit"
    st, rc = http("POST", f"{BASE}/close_session", {"session_id": sid})
    assert st == 200, f"/close_session -> {st}: {rc}"
    print(f"smoke_api ok: turn2 cached_tokens={cached}, close_session HTTP 200")


# ----------------------------------------------------- inference-perf runner
def build_config(traces, n: int, timeout_s: int, close: bool, report_dir: str,
                 idle_cap: float, base_url: str) -> dict:
    """traces: {"trace_files": [...]} or {"trace_directory": "..."}"""
    cfg = {
        "load": {
            "type": "trace_session_replay",
            "stages": [{"concurrent_sessions": n, "num_sessions": n, "timeout": timeout_s}],
            # all events of a dispatched session are enqueued at once and each
            # waiting event holds a worker slot; rule of thumb is
            # worker_max_concurrency >= sessions * avg_events_per_session
            # (12 sessions of 100-300 events -> ~1500-3000 slots needed)
            "num_workers": 4,
            "worker_max_concurrency": 1024,
            "request_timeout": 1800,
        },
        "api": {
            "type": "chat",
            "streaming": True,
            "session_id_header_key": "x-session-id",
            "session_id_body_field": "session_id",
        },
        "server": {"type": "sglang", "model_name": MODEL, "base_url": base_url, "ignore_eos": True},
        "tokenizer": {"pretrained_model_name_or_path": MODEL},
        "data": {
            "type": "weka_trace_replay",
            "weka_trace_replay": {
                **traces,
                "use_static_model": True,
                "static_model_name": MODEL,
                "default_block_size": 64,
                "skip_invalid_files": True,
                "trace_idle_gap_cap_seconds": idle_cap,
                # runtime wait clamp must not undercut the idle cap, or the
                # effective gap stays at the 15s default regardless of idle_cap
                "max_wait_ms": int(max(15.0, idle_cap) * 1000),
                "subagent_separate_session_id": True,
                "close_subagent_sessions": True,
            },
        },
        "report": {"request_lifecycle": {
            "summary": True, "per_stage": True, "per_request": True,
            # issue #522 fix (per_request_fields, in this build): drop the raw
            # request/response text at the source instead of post-processing -
            # keeps the per-request report MBs instead of GBs and report
            # generation fast. computed_metrics adds per-request TTFT directly.
            "per_request_fields": {"request": False, "response": False,
                                   "response_chunks": False, "info": True,
                                   "computed_metrics": True},
            "use_server_output_tokens": True,
        }},
        "storage": {"local_storage": {"path": report_dir}},
    }
    if close:
        cfg["api"]["session_final_header_key"] = "x-session-final"
        cfg["api"]["session_close_path"] = "/close_session"
    return cfg


def write_config(path: Path, trace_files: list, n: int, timeout_s: int,
                 close: bool, report_dir: Path, idle_cap: float) -> None:
    cfg = build_config({"trace_files": trace_files}, n, timeout_s, close,
                       str(report_dir), idle_cap, BASE)
    path.write_text(json.dumps(cfg, indent=2))  # JSON is valid YAML


def run_inference_perf(cfg: Path, log: Path) -> int:
    py = INFERENCE_PERF_DIR / ".venv/bin/python"
    with open(log, "w") as lf:
        p = subprocess.Popen([str(py), "inference_perf/main.py", "-c", str(cfg)],
                             cwd=INFERENCE_PERF_DIR, stdout=lf, stderr=subprocess.STDOUT)
        while p.poll() is None:
            time.sleep(20)
            print(f"  ... running ({time.strftime('%H:%M:%S')}), log: {log}")
        return p.returncode


def read_reports(report_dir: Path) -> dict:
    """Parse per_request_lifecycle_metrics.json. Splits parent vs subagent
    requests (graph_event_id contains '_sa_' for subagent streams).
    TTFT = first streamed chunk time - request start (perf_counter clock)."""
    out = {"requests": 0, "failures": 0,
           "parent": {"ttft": [], "hit": []}, "sub": {"ttft": [], "hit": []}}
    f = report_dir / "per_request_lifecycle_metrics.json"
    if not f.exists():
        # kubectl cp may nest the remote dir one level deeper
        f = report_dir / "reports" / "per_request_lifecycle_metrics.json"
    if not f.exists():
        return out
    for e in json.loads(f.read_text()):
        out["requests"] += 1
        if e.get("error"):
            out["failures"] += 1
            continue
        info = e.get("info") or {}
        bucket = out["sub"] if "_sa_" in (info.get("graph_event_id") or "") else out["parent"]
        rm = info.get("response_metrics") or {}
        cm = e.get("computed_metrics") or {}
        if isinstance(cm.get("time_to_first_token"), (int, float)):
            bucket["ttft"].append(cm["time_to_first_token"])
        else:
            chunks = rm.get("chunk_times") or []
            if chunks and e.get("start_time"):
                bucket["ttft"].append(chunks[0] - e["start_time"])
        su = rm.get("server_usage") or {}
        det = su.get("prompt_tokens_details") or {}
        if det.get("cached_tokens") is not None and su.get("prompt_tokens"):
            bucket["hit"].append(det["cached_tokens"] / su["prompt_tokens"])
    return out


def summarize(tag: str, r: dict):
    def pct(v, p):
        return sorted(v)[min(len(v) - 1, int(len(v) * p / 100))] if v else float("nan")
    print(f"[{tag}] requests={r['requests']} failures={r['failures']}")
    for role in ("parent", "sub"):
        ttft, hit = r[role]["ttft"], r[role]["hit"]
        if not ttft and not hit:
            continue
        line = f"  {role:6s} n={max(len(ttft), len(hit))}"
        if ttft:
            line += f" ttft p50={pct(ttft,50):.2f}s p90={pct(ttft,90):.2f}s mean={statistics.mean(ttft):.2f}s"
        if hit:
            line += f" cache_hit mean={statistics.mean(hit):.3f} p10={pct(hit,10):.3f}"
        print(line)


# ------------------------------------------------- in-cluster A/B execution
BENCH_IMAGE = ("us-central1-docker.pkg.dev/bobzetian-gke-dev/bobinference/"
               "inference-perf:session-control-0471bcf")

# initContainer: fetch the sample slice and keep only the selected trace lines
FETCH_PY = """
import os, urllib.request
url = os.environ["SAMPLE_URL"]; n = int(os.environ["SAMPLE_BYTES"])
keep = set(int(x) for x in os.environ["TRACE_INDICES"].split(","))
req = urllib.request.Request(url, headers={"Range": "bytes=0-%d" % n})
data = urllib.request.urlopen(req, timeout=600).read()
os.makedirs("/work/traces", exist_ok=True)
kept = 0
for i, line in enumerate(data.split(b"\\n")[:-1]):
    if i in keep:
        open("/work/traces/%03d.json" % i, "wb").write(line)
        kept += 1
print("fetched", kept, "traces")
"""

# post-run: strip the multi-GB request/response text so kubectl cp stays small
COMPACT_PY = """
import json
p = "/work/reports/per_request_lifecycle_metrics.json"
try:
    d = json.load(open(p))
except FileNotFoundError:
    print("no per_request report"); raise SystemExit
for e in d:
    e.pop("request", None); e.pop("response", None)
    info = e.get("info") or {}
    info.pop("output_text", None); info.pop("output_message", None)
    rm = info.get("response_metrics") or {}
    rm["chunk_times"] = (rm.get("chunk_times") or [])[:1]
    rm.pop("response_chunks", None); rm.pop("output_token_times", None)
json.dump(d, open(p, "w"))
print("compacted", len(d), "entries")
"""


def kubectl(*args, check=True, capture=True) -> str:
    cmd = ["kubectl", "-n", NAMESPACE, *args]
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True)
    else:
        r = subprocess.run(cmd)
    if check and r.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {getattr(r, 'stderr', '')}")
    return (r.stdout or "").strip() if capture else ""


def sglang_exec(pod: str, path: str) -> str:
    return kubectl("exec", pod, "--", "curl", "-s", "-X", "POST" if path != "/metrics" else "GET",
                   f"localhost:8000{path}")


def render_job(arm: str, cfg: dict, indices: list) -> tuple:
    name = f"closebench-{arm}"
    cm = {
        "apiVersion": "v1", "kind": "ConfigMap",
        "metadata": {"name": name, "namespace": NAMESPACE},
        "data": {"config.yaml": json.dumps(cfg, indent=1),
                 "fetch.py": FETCH_PY, "compact.py": COMPACT_PY},
    }
    job = {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": name, "namespace": NAMESPACE},
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": cfg["load"]["stages"][0]["timeout"] + 10800,
            "template": {"spec": {
                "restartPolicy": "Never",
                "initContainers": [{
                    "name": "fetch-traces",
                    "image": "python:3.12-alpine",
                    "command": ["python3", "/config/fetch.py"],
                    "env": [
                        {"name": "SAMPLE_URL", "value": SAMPLE_URL},
                        {"name": "SAMPLE_BYTES", "value": str(SAMPLE_BYTES)},
                        {"name": "TRACE_INDICES", "value": ",".join(str(i) for i in indices)},
                    ],
                    "volumeMounts": [{"name": "work", "mountPath": "/work"},
                                     {"name": "config", "mountPath": "/config"}],
                }],
                "containers": [{
                    "name": "bench",
                    "image": BENCH_IMAGE,
                    # run the benchmark, shrink the report, then idle so the
                    # local driver can `kubectl cp` the results out - no GCS
                    # or extra IAM needed, everything rides the kubectl creds
                    "command": ["sh", "-c",
                                "python inference_perf/main.py -c /config/config.yaml; "
                                "echo $? > /work/reports/EXITCODE; "
                                "python /config/compact.py; "
                                "touch /work/reports/DONE; sleep 14400"],
                    "env": [{"name": "HF_TOKEN", "valueFrom": {"secretKeyRef": {
                        "name": "llm-d-hf-token", "key": "HF_TOKEN", "optional": True}}}],
                    "resources": {"requests": {"cpu": "4", "memory": "16Gi"},
                                  "limits": {"memory": "24Gi"}},
                    "volumeMounts": [{"name": "work", "mountPath": "/work"},
                                     {"name": "config", "mountPath": "/config"}],
                }],
                "volumes": [{"name": "work", "emptyDir": {}},
                            {"name": "config", "configMap": {"name": name}}],
            }},
        },
    }
    return name, cm, job


def _launch_arm(arm: str, target_pod: str, sel: list, indices: list,
                timeout_s: int, idle_cap: float) -> dict:
    d = WORK / f"ab-{arm}"
    shutil.rmtree(d, ignore_errors=True)
    (d / "reports").mkdir(parents=True)
    pod_ip = kubectl("get", "pod", target_pod, "-o", "jsonpath={.status.podIP}")
    print(f"=== arm {arm} -> pod {target_pod} ({pod_ip}) ===")
    print(f"  flush_cache: {sglang_exec(target_pod, '/flush_cache')[:120]}")
    (d / "metrics_before.txt").write_text(sglang_exec(target_pod, "/metrics"))

    cfg = build_config({"trace_directory": "/work/traces"}, len(sel), timeout_s,
                       close=(arm == "close"), report_dir="/work/reports",
                       idle_cap=idle_cap, base_url=f"http://{pod_ip}:8000")
    name, cm, job = render_job(arm, cfg, indices)
    kubectl("delete", "job", name, "--ignore-not-found")
    kubectl("delete", "configmap", name, "--ignore-not-found")
    for manifest in (cm, job):
        p = subprocess.run(["kubectl", "apply", "-f", "-"], input=json.dumps(manifest),
                           text=True, capture_output=True)
        if p.returncode != 0:
            raise RuntimeError(f"apply failed: {p.stderr}")
    print(f"  job {name} created")
    return {"arm": arm, "name": name, "dir": d, "target_pod": target_pod,
            "bench_pod": "", "done": False,
            "deadline": time.time() + timeout_s + 1800}


def _poll_arm(st: dict) -> None:
    """One poll step; sets st['done'] when the arm's DONE marker appears."""
    if st["done"]:
        return
    if time.time() > st["deadline"]:
        raise RuntimeError(f"arm {st['arm']} did not finish before deadline")
    if not st["bench_pod"]:
        st["bench_pod"] = kubectl("get", "pod", "-l", f"job-name={st['name']}",
                                  "-o", "jsonpath={.items[0].metadata.name}", check=False)
        if st["bench_pod"]:
            print(f"  [{st['arm']}] job pod: {st['bench_pod']}")
        return
    phase = kubectl("get", "pod", st["bench_pod"], "-o", "jsonpath={.status.phase}", check=False)
    if phase == "Failed":
        print(kubectl("logs", st["bench_pod"], "--tail=30", check=False))
        raise RuntimeError(f"arm {st['arm']} job pod failed")
    r = subprocess.run(["kubectl", "-n", NAMESPACE, "exec", st["bench_pod"], "-c", "bench",
                        "--", "test", "-f", "/work/reports/DONE"], capture_output=True)
    print(f"  {time.strftime('%H:%M:%S')} [{st['arm']}] phase={phase} done={'yes' if r.returncode == 0 else 'no'}")
    st["done"] = r.returncode == 0


def _collect_arm(st: dict) -> None:
    d = st["dir"]
    (d / "metrics_after.txt").write_text(sglang_exec(st["target_pod"], "/metrics"))
    # capture the bench log (close_session warnings, datagen timing, errors)
    (d / "run.log").write_text(kubectl("logs", st["bench_pod"], "-c", "bench", check=False))
    for attempt in range(3):  # apiserver tunnel can reset mid-stream; retry
        r = subprocess.run(["kubectl", "-n", NAMESPACE, "cp", f"{st['bench_pod']}:/work/reports",
                            str(d / "reports"), "-c", "bench"], capture_output=True, text=True)
        if r.returncode == 0:
            break
        print(f"  [{st['arm']}] kubectl cp failed (attempt {attempt + 1}): {r.stderr.strip()[:150]}")
        time.sleep(10)
    else:
        raise RuntimeError(f"arm {st['arm']}: kubectl cp failed 3x; job kept alive for manual copy")
    kubectl("delete", "job", st["name"], "--wait=false")
    kubectl("delete", "configmap", st["name"])

    exitcode_f = d / "reports" / "EXITCODE"
    exitcode = exitcode_f.read_text().strip() if exitcode_f.exists() else "?"
    r = read_reports(d / "reports")
    print(f"arm {st['arm']}: inference-perf exit={exitcode} "
          f"(timeout-truncated stages exit non-zero by design; judge by request counts)")
    if r["requests"] == 0:
        raise RuntimeError(f"arm {st['arm']} produced ZERO requests - see {d}/run.log")
    summarize(st["arm"], r)


def phase_ab_cluster(meta, sessions: int, timeout_s: int, idle_cap: float,
                     pods_by_arm: dict):
    """pods_by_arm: {'close': pod, 'noclose': pod}. Different pods -> both arms
    run CONCURRENTLY (independent KV pools, halves wall time). Same pod ->
    strictly sequential; two arms must never share a pool at the same time."""
    sel = select_traces(meta, "ab", sessions)
    indices = [m[0] for m in sel]
    print(f"ab traces ({len(sel)}):", [(m[1], f"{m[2]}t", f"{m[3]}sa") for m in sel])

    parallel = pods_by_arm["close"] != pods_by_arm["noclose"]
    if parallel:
        states = [_launch_arm(arm, pods_by_arm[arm], sel, indices, timeout_s, idle_cap)
                  for arm in ("close", "noclose")]
        while not all(s["done"] for s in states):
            time.sleep(30)
            for s in states:
                _poll_arm(s)
        for s in states:
            _collect_arm(s)
    else:
        print("single target pod: running arms sequentially (shared KV pool must not overlap)")
        for arm in ("close", "noclose"):
            st = _launch_arm(arm, pods_by_arm[arm], sel, indices, timeout_s, idle_cap)
            while not st["done"]:
                time.sleep(30)
                _poll_arm(st)
            _collect_arm(st)

    print("\n=== A/B done ===")
    for arm in ("close", "noclose"):
        summarize(arm, read_reports(WORK / f"ab-{arm}" / "reports"))
    print(f"reports under {WORK}/ab-close and {WORK}/ab-noclose")


# ------------------------------------------------------------------- phases
def phase_smoke(meta):
    smoke_api()
    sel = select_traces(meta, "smoke", 2)
    print("smoke traces:", [(m[1], f"{m[2]}turns", f"{m[4]}out") for m in sel])
    d = WORK / "smoke"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    cfg = d / "config.yaml"
    write_config(cfg, [m[5] for m in sel], n=2, timeout_s=900, close=True,
                 report_dir=d / "reports", idle_cap=1.0)
    rc = run_inference_perf(cfg, d / "run.log")
    r = read_reports(d / "reports")
    summarize("smoke", r)
    ok = rc == 0 and r["requests"] > 0 and r["failures"] == 0
    print("SMOKE " + ("PASS" if ok else f"FAIL (exit={rc}) - see {d}/run.log"))
    if not ok:
        sys.exit(1)


def phase_ab(meta, sessions: int, timeout_s: int, idle_cap: float):
    sel = select_traces(meta, "ab", sessions)
    print(f"ab traces ({len(sel)}):", [(m[1], f"{m[2]}t", f"{m[3]}sa") for m in sel])
    results = {}
    for arm in ("close", "noclose"):
        d = WORK / f"ab-{arm}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        flush_cache()
        time.sleep(3)
        cfg = d / "config.yaml"
        write_config(cfg, [m[5] for m in sel], n=len(sel), timeout_s=timeout_s,
                     close=(arm == "close"), report_dir=d / "reports", idle_cap=idle_cap)
        print(f"=== arm {arm}: up to {timeout_s}s ===")
        rc = run_inference_perf(cfg, d / "run.log")
        print(f"arm {arm} exit={rc} (timeout-truncated runs may exit non-zero; judging by request counts)")
        results[arm] = read_reports(d / "reports")
        summarize(arm, results[arm])
    print("\n=== A/B done. Reports: ===")
    print(f"  close:   {WORK}/ab-close/reports")
    print(f"  noclose: {WORK}/ab-noclose/reports")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "ab", "ab-cluster", "both"], default="smoke",
                    help="ab runs locally via port-forward; ab-cluster runs inference-perf "
                         "as an in-cluster Job (pod-to-pod traffic, kubectl cp for reports)")
    ap.add_argument("--pod", default="", help="sglang pod name (default: first Ready)")
    ap.add_argument("--sessions", type=int, default=12)
    ap.add_argument("--timeout-sec", type=int, default=2400)
    ap.add_argument("--idle-cap-sec", type=float, default=1.0)
    ap.add_argument("--serial", action="store_true",
                    help="ab-cluster: force both arms onto one pod sequentially "
                         "(default: two Ready pods run the arms concurrently)")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    meta = prepare_traces()
    pod = pick_pod(args.pod)

    if args.mode == "ab-cluster":
        pods = ready_pods()
        if args.pod or args.serial or len(pods) < 2:
            pods_by_arm = {"close": pod, "noclose": pod}
        else:
            pods_by_arm = {"close": pods[0], "noclose": pods[1]}
        # control-plane calls go through kubectl exec; no port-forward needed
        for p in sorted(set(pods_by_arm.values())):
            info = json.loads(sh(["kubectl", "-n", NAMESPACE, "exec", p, "--",
                                  "curl", "-s", "localhost:8000/get_server_info"]))
            assert info.get("enable_session_radix_cache") is True, f"session radix OFF on {p}"
            print(f"preflight ok: pod={p} sglang={info.get('version')} "
                  f"kv_pool={info.get('max_total_num_tokens')} session_radix=on")
        phase_ab_cluster(meta, args.sessions, args.timeout_sec, args.idle_cap_sec, pods_by_arm)
        return

    port_forward(pod)
    try:
        preflight(pod)
        if args.mode in ("smoke", "both"):
            phase_smoke(meta)
        if args.mode in ("ab", "both"):
            phase_ab(meta, args.sessions, args.timeout_sec, args.idle_cap_sec)
    finally:
        if _pf_proc:
            _pf_proc.terminate()


if __name__ == "__main__":
    main()
