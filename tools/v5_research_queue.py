"""Run a fixed V5 experiment list serially on one authorized GPU.

No shell commands, automatic test-driven tuning, container updates or shared
service restarts. An external GPU process makes this queue wait or yield its
own job; it never signals anybody else's process.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.v5_job_runner import matches_record, process_identity, read_record, stop_job


def command(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def write(path, state):
    state['updated_utc'] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def gpu_processes(gpu=2):
    value = command(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    return {int(line.strip()) for line in value.splitlines() if line.strip().isdigit()}


def host_container_pids(container):
    value = command(['docker', 'top', container, '-eo', 'pid'])
    return {int(line.strip()) for line in value.splitlines() if line.strip().isdigit()}


def verify_container(container, gpu=2):
    if gpu not in (0, 1, 2, 3):
        raise ValueError('Only this server physical GPU0–3 are authorized')
    info = json.loads(command(['docker', 'inspect', container]))[0]
    limits = info['HostConfig']
    requests = limits.get('DeviceRequests') or []
    if len(requests) != 1 or requests[0].get('DeviceIDs') != [str(gpu)]:
        raise ValueError(f'Queue container must expose physical GPU{gpu} only')
    if not (0 < limits['Memory'] <= 24 * 1024**3 and limits['MemorySwap'] == limits['Memory'] and
            0 < limits['NanoCpus'] <= 8 * 10**9 and 0 < limits['PidsLimit'] <= 512):
        raise ValueError('Container resource caps differ from the agreed 24GiB/8CPU/512PID limits')
    if not info['State']['Running']:
        raise ValueError('Queue container must already be running')
    return info


def container_record_path(host_path, info):
    host_path = Path(host_path).resolve()
    candidates = []
    for mount in info.get('Mounts', []):
        if not mount.get('RW'):
            continue
        try:
            relative = host_path.relative_to(Path(mount['Source']).resolve())
        except (ValueError, KeyError):
            continue
        candidates.append((len(Path(mount['Source']).parts), Path(mount['Destination']) / relative))
    if not candidates:
        raise ValueError('Job ownership record must be on a writable host/container shared mount')
    return str(max(candidates, key=lambda item: item[0])[1])


def owned_host_pid(container, record_path, token):
    try:
        record = read_record(record_path, token)
    except (FileNotFoundError, json.JSONDecodeError):
        return None  # The runner has not finished publishing its identity yet.
    matches = []
    for pid in host_container_pids(container):
        try:
            identity = process_identity(pid)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if matches_record(identity, record, host=True):
            matches.append(pid)
    if len(matches) > 1:
        raise ValueError('Ambiguous owned process mapping; no process may be signaled')
    return matches[0] if matches else None


def stop_own_job(container, record_path, token, *, host_record_path=None):
    # The runner opens a pidfd first, then verifies start ticks, argv and its
    # unique environment token before signaling the pinned process descriptor.
    if host_record_path is not None:
        # Explicit host adapter: Conda Python in this container lacks pidfd.
        # Pin the verified host PID, then recheck namespace/start/argv/token.
        owned = owned_host_pid(container, host_record_path, token)
        if owned is None:
            return dict(signaled=False, reason='already_exited')
        return stop_job(host_record_path, token, host_pid=owned)
    value = command(['docker', 'exec', container, 'python', 'tools/v5_job_runner.py',
                     'stop', '--record', str(record_path), '--token', token])
    return json.loads(value)


def development_gate(gate):
    def load_metric(path):
        report = json.loads(Path(path).read_text())
        rows = list(report.get('validations', []))
        if report.get('result'):
            rows.append(report['result'])
        scores = [r['raw']['cer'] for r in rows if r.get('complete') and r.get('raw', {}).get('cer') is not None]
        return report, min(scores) if scores else None
    candidate, score = load_metric(gate['candidate_report'])
    reference, baseline = load_metric(gate['reference_report'])
    if candidate['manifests']['dev']['sha256'] != reference['manifests']['dev']['sha256']:
        raise ValueError('Development gate requires identical dev manifests')
    improvement = baseline - score if baseline is not None and score is not None else None
    passed = improvement is not None and improvement >= float(gate.get('minimum_cer_improvement', .01))
    return dict(passed=passed, candidate_best_dev_raw_cer=score, reference_dev_raw_cer=baseline,
                improvement=improvement, criterion='complete dev raw CER only; no official test metrics')


def run(plan_path, state_path):
    plan = json.loads(plan_path.read_text())
    container = plan['container']
    allocator = plan.get('cuda_allocator_config')
    if allocator not in (None, 'expandable_segments:True'):
        raise ValueError('Only the validated expandable_segments allocator override is supported')
    gpu = plan.get('gpu')
    verify_container(container, gpu=gpu)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(state_path) + '.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.exists():
            raise FileExistsError('Queue state already exists; preserve it and create a new explicit plan')
        state = dict(plan=str(plan_path), container=container, gpu=gpu, status='waiting', jobs=[])
        write(state_path, state)
        waiting = plan.get('wait_for_report')
        while waiting:
            path = Path(waiting)
            if path.exists():
                report = json.loads(path.read_text())
                if report.get('status') == 'complete':
                    break
                if report.get('status') in ('failed', 'interrupted'):
                    state.update(status='failed_prerequisite', prerequisite=waiting)
                    write(state_path, state)
                    return 1
            time.sleep(10)
        for job_index, job in enumerate(plan['jobs']):
            argv = job['argv']
            if len(argv) < 2 or argv[0] != 'python' or not argv[1].startswith('tools/v5_'):
                raise ValueError('Queue accepts explicit V5 python entry points only')
            if job.get('test_real') or 'test_real' in ' '.join(argv):
                raise ValueError('ESTR60 remains locked outside this development queue')
            current = dict(id=job['id'], argv=argv, status='waiting_for_gpu')
            state['jobs'].append(current)
            if job.get('development_gate'):
                current['development_gate'] = development_gate(job['development_gate'])
                if not current['development_gate']['passed']:
                    current['status'] = 'skipped_by_development_gate'
                    write(state_path, state)
                    continue
            write(state_path, state)
            while gpu_processes(gpu):
                time.sleep(10)
            info = verify_container(container, gpu=gpu)
            log_path = Path(job['log'])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            token = uuid.uuid4().hex
            record_path = state_path.parent / f'{state_path.stem}.job{job_index:03d}.{token}.owner.json'
            if record_path.exists():
                raise FileExistsError(f'Ownership record already exists: {record_path}')
            child_record = container_record_path(record_path, info)
            wrapper = ['python', 'tools/v5_job_runner.py', 'start', '--record', child_record,
                       '--token', token, '--', *argv]
            started = time.monotonic()
            with open(log_path, 'x') as log:
                launch = ['docker', 'exec']
                if allocator:
                    launch += ['-e', 'PYTORCH_CUDA_ALLOC_CONF=' + allocator]
                process = subprocess.Popen([*launch, container, *wrapper], stdout=log, stderr=subprocess.STDOUT)
                current.update(status='running', docker_exec_pid=process.pid, log=str(log_path),
                               owner_record=str(record_path.resolve()), container_owner_record=child_record,
                               owner_token=token)
                state['status'] = 'running'
                write(state_path, state)
                stopped = None
                while process.poll() is None:
                    time.sleep(10)
                    if process.poll() is not None:
                        break
                    try:
                        owned = owned_host_pid(container, record_path, token)
                    except (OSError, ValueError) as error:
                        current.update(status='ownership_unverified', error=str(error))
                        state['status'] = 'needs_attention'
                        write(state_path, state)
                        return 1  # Fail safe: do not signal any process.
                    current['owned_host_pid'] = owned
                    external = gpu_processes(gpu) - ({owned} if owned is not None else set())
                    elapsed = time.monotonic() - started
                    if external or elapsed > float(job.get('timeout_seconds', 8400)):
                        stopped = 'external_gpu_process' if external else 'timeout'
                        current.update(stop_reason=stopped, external_pids=sorted(external))
                        try:
                            current['owned_job_signal'] = stop_own_job(container, child_record, token,
                                                                     host_record_path=record_path)
                        except (OSError, ValueError, subprocess.CalledProcessError) as error:
                            current.update(status='ownership_unverified', error=str(error))
                            state['status'] = 'needs_attention'
                            write(state_path, state)
                            return 1  # No command-line or PID-only fallback.
                        write(state_path, state)
                        # Give this job time to save a healthy checkpoint. Do not
                        # force-kill processes or reset the device on a shared host.
                        try:
                            process.wait(timeout=120)
                        except subprocess.TimeoutExpired:
                            state.update(status='needs_attention')
                            write(state_path, state)
                            return 1
                        break
                code = process.wait()
            signal_result = current.get('owned_job_signal', {})
            if (stopped == 'external_gpu_process' and code == 0
                    and signal_result.get('signaled') is False
                    and signal_result.get('reason') == 'already_exited'):
                # NVML may briefly retain a PID after the owned child exits,
                # before docker exec reaps it. No signal was sent; verify the
                # report below and let the next job wait for GPU availability.
                current['exit_race_reconciled'] = True
                current.pop('stop_reason', None)
                stopped = None
            current.update(status='complete' if code == 0 and stopped is None else 'stopped',
                           returncode=code, elapsed_seconds=time.monotonic() - started)
            if code or stopped:
                state['status'] = 'needs_attention'
                write(state_path, state)
                return 1
            report_path = job.get('report')
            if report_path:
                report = json.loads(Path(report_path).read_text())
                terminal = report
                for key in job.get('report_status_path', ['status']):
                    terminal = terminal.get(key) if isinstance(terminal, dict) else None
                current['report_status'] = terminal
                if terminal not in job.get('acceptable_report_statuses', ['complete']):
                    current['status'] = 'report_incomplete'
                    state['status'] = 'needs_attention'
                    write(state_path, state)
                    return 1
            write(state_path, state)
        state['status'] = 'complete'
        write(state_path, state)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.plan, args.state))
