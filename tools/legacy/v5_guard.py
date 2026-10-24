#!/usr/bin/env python3
"""Historical watchdog; retained for audit, not an active training entry point.

Its blind resumes inherited exhausted AMP counters. Use the current documented
experiment workflow after diagnosing the failure; do not launch this watchdog.

Keeps a GPU2 training job alive until its run.json reaches status=complete
with stop_reason=step_budget (the target step count). A stop_reason of
time_budget is a normal mid-course wall-clock stop and triggers another resume
toward max_steps rather than exit; abnormal stops (running/failed/interrupted
with the process gone) relaunch via `docker exec -d` (detached, owned by the
container init) using the same out_dir's last.pt, bounded by --max-restarts.
The job itself is detached from this process, so killing the watchdog never
kills the training.

Run the watchdog detached from any login/tool session, e.g.:
    setsid nohup python tools/legacy/v5_guard.py --container seeread-v5-run-20261020b \
        --config configs/v5/reading_paper_joint_short.yaml \
        --out-dir results/v5/reading/R4_paper_joint_short \
        > /tmp/v5_guard.log 2>&1 &
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HOST_ROOT = Path("/home/gyf/seeread_project")


def read_report(rel_out_dir):
    path = HOST_ROOT / rel_out_dir / "run.json"
    try:
        with path.open("r") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def find_running_pid(container):
    proc = subprocess.run(
        ["docker", "exec", container, "ps", "-eo", "pid=,args="],
        capture_output=True, text=True,
    )
    for line in proc.stdout.splitlines():
        if "v5_reading.py" not in line or "v5_guard" in line:
            continue
        if "--mode" not in line and "train" not in line:
            continue
        return line.split(None, 1)[0]
    return None


def launch(container, config, out_dir, max_seconds, max_steps):
    resume = f"{out_dir}/last.pt"
    # --resume must name this out_dir's last.pt (enforced inside v5_reading.py)
    argv = [
        "docker", "exec", "-d", "-w", "/workspace", container, "sh", "-c",
        "python tools/v5_reading.py"
        f" --config {config}"
        " --mode train"
        " --device cuda:0"
        f" --resume {resume}"
        f" --max-seconds {max_seconds}"
        f" --max-steps {max_steps}"
        f" > {out_dir}/train.stdout.log 2>&1",
    ]
    return subprocess.run(argv, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--container", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-seconds", type=int, default=7200)
    ap.add_argument("--max-steps", type=int, default=50740)
    ap.add_argument("--max-restarts", type=int, default=5)
    ap.add_argument("--poll-seconds", type=int, default=30)
    args = ap.parse_args()

    restarts = 0  # abnormal restarts only (crashes/interrupts), not time_budget continuation
    print(f"[guard] watching {args.out_dir} in {args.container} "
          f"(max_seconds={args.max_seconds}, max_steps={args.max_steps})",
          flush=True)
    while True:
        report = read_report(args.out_dir)
        status = report.get("status") if report else None
        reason = report.get("stop_reason") if report else None

        # True completion: target step budget reached.
        if status == "complete" and reason == "step_budget":
            step = (report.get("state") or {}).get("global_step")
            print(f"[guard] step_budget reached (step={step}). exiting.", flush=True)
            return 0

        pid = find_running_pid(args.container)
        if pid is not None:
            time.sleep(args.poll_seconds)
            continue

        # Process not running.
        if report is None:
            print("[guard] no run.json yet; waiting before first relaunch.", flush=True)
            time.sleep(args.poll_seconds)
            continue

        if status == "complete" and reason == "time_budget":
            # Wall-clock budget exhausted but target steps not yet reached:
            # continue toward max_steps without counting against restart quota.
            step = (report.get("state") or {}).get("global_step")
            print(f"[guard] time_budget hit (step={step}); continuing toward "
                  f"max_steps={args.max_steps}.", flush=True)
        elif status in ("running", "failed", "interrupted"):
            if restarts >= args.max_restarts:
                print(f"[guard] hit max_restarts={args.max_restarts}; giving up.",
                      flush=True)
                return 3
            restarts += 1
            print(f"[guard] process gone (status={status}); relaunch #{restarts} "
                  f"from last.pt", flush=True)
        else:
            print(f"[guard] unexpected status {status!r}; not relaunching. exiting.",
                  flush=True)
            return 2

        res = launch(args.container, args.config, args.out_dir,
                     args.max_seconds, args.max_steps)
        if res.returncode != 0:
            print(f"[guard] relaunch failed: {res.stderr.strip()}", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
