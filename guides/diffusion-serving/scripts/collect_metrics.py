"""Per-pod queue-depth poller (runs as the metrics-poller container in the
bench Job; stdlib only — no pip install needed).

Every INTERVAL_S seconds, GET http://<ip>:8000/metrics for each IP in the
POD_IPS env var (comma-separated) and record the
vllm_omni:num_requests_running / vllm_omni:num_requests_waiting gauges.
Exits when DONE_FILE appears (dropped by the bench container), then prints the
CSV between marker lines for run_arm.sh to harvest from the container logs.
"""

import os
import re
import sys
import time
import urllib.request

GAUGE_RE = re.compile(
    r'^vllm_omni:num_requests_(running|waiting)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$',
    re.MULTILINE,
)


def scrape(ip: str, timeout: float = 3.0) -> dict[str, float]:
    body = urllib.request.urlopen(f"http://{ip}:8000/metrics", timeout=timeout).read().decode()
    values: dict[str, float] = {}
    for kind, value in GAUGE_RE.findall(body):
        # Sum across label sets (per-stage/replica labels may be present).
        values[kind] = values.get(kind, 0.0) + float(value)
    return values


def main() -> None:
    pod_ips = [ip for ip in os.environ.get("POD_IPS", "").split(",") if ip]
    done_file = os.environ.get("DONE_FILE", "/results/done")
    interval = float(os.environ.get("INTERVAL_S", "2"))
    if not pod_ips:
        print("collect_metrics: POD_IPS empty, exiting", file=sys.stderr)
        return

    rows = ["timestamp,pod_ip,running,waiting"]
    while not os.path.exists(done_file):
        now = time.time()
        for ip in pod_ips:
            try:
                v = scrape(ip)
                rows.append(f"{now:.1f},{ip},{v.get('running', 0):g},{v.get('waiting', 0):g}")
            except Exception as exc:  # noqa: BLE001 - pod may be restarting; keep polling
                print(f"collect_metrics: {ip}: {exc}", file=sys.stderr)
        time.sleep(interval)

    print("=====METRICS_CSV=====")
    print("\n".join(rows))
    print("=====END=====")


if __name__ == "__main__":
    main()
