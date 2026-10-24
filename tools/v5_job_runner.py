"""Start one token-owned V5 process, or interrupt that exact Linux process.

The process records its identity before exec (which preserves PID/start time).
Stopping requires pidfd support and checks identity again after pidfd_open;
there is deliberately no PID-only or command-line-only signaling fallback.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import stat

TOKEN_ENV = 'SEEREAD_V5_JOB_TOKEN'
FORMAT = 'seeread_v5_owned_process_v1'


def process_identity(pid, proc_root=Path('/proc')):
    path = Path(proc_root) / str(pid)
    raw = (path / 'stat').read_text()
    # comm may contain spaces or parentheses. Fields after its final ')' start
    # at field 3; field 22 is the process start time in clock ticks.
    fields = raw[raw.rfind(')') + 1:].split()
    start_ticks = int(fields[19])
    argv = [os.fsdecode(value) for value in (path / 'cmdline').read_bytes().split(b'\0') if value]
    environment = {}
    for item in (path / 'environ').read_bytes().split(b'\0'):
        if b'=' in item:
            key, value = item.split(b'=', 1)
            environment[os.fsdecode(key)] = os.fsdecode(value)
    namespaces = []
    for line in (path / 'status').read_text().splitlines():
        if line.startswith('NSpid:'):
            namespaces = [int(value) for value in line.split()[1:]]
            break
    return dict(pid=int(pid), start_ticks=start_ticks, argv=argv,
                token=environment.get(TOKEN_ENV), namespace_pids=namespaces)


def read_record(path, token):
    if not hasattr(os, 'O_NOFOLLOW'):
        raise RuntimeError('Secure ownership records require O_NOFOLLOW')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'r') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError('Ownership record must be a regular file with mode 0600')
        record = json.load(stream)
    if record.get('format') != FORMAT or not token or record.get('token') != token:
        raise ValueError('Ownership record token/format mismatch')
    if (not isinstance(record.get('container_pid'), int) or record['container_pid'] <= 1 or
            not isinstance(record.get('start_ticks'), int) or record['start_ticks'] <= 0 or
            not isinstance(record.get('argv'), list) or not record['argv'] or
            not all(isinstance(value, str) and value for value in record['argv'])):
        raise ValueError('Invalid ownership identity')
    return record


def matches_record(identity, record, host=False):
    if host:
        namespace_ids = identity.get('namespace_pids', [])
        correct_pid = bool(namespace_ids) and namespace_ids[-1] == record['container_pid']
    else:
        correct_pid = identity.get('pid') == record['container_pid']
    return (correct_pid and identity.get('start_ticks') == record['start_ticks'] and
            identity.get('argv') == record['argv'] and identity.get('token') == record['token'])


def start_job(record_path, token, argv):
    if not token or not argv:
        raise ValueError('A unique token and explicit argv are required')
    pid = os.getpid()
    if pid <= 1:
        raise ValueError('An owned job must not replace or signal container init')
    identity = process_identity(pid)
    record = dict(format=FORMAT, token=token, container_pid=pid,
                  start_ticks=identity['start_ticks'], argv=argv)
    descriptor = os.open(record_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(record, stream)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    environment = os.environ.copy()
    environment[TOKEN_ENV] = token
    os.execvpe(argv[0], argv, environment)


def stop_job(record_path, token, *, host_pid=None):
    record = read_record(record_path, token)
    if not callable(getattr(os, 'pidfd_open', None)) or not callable(getattr(signal, 'pidfd_send_signal', None)):
        raise RuntimeError('pidfd support is required; refusing an unsafe PID-only signal')
    try:
        descriptor = os.pidfd_open(record['container_pid'] if host_pid is None else host_pid, 0)
    except ProcessLookupError:
        return dict(signaled=False, reason='already_exited', container_pid=record['container_pid'])
    try:
        try:
            identity = process_identity(record['container_pid'] if host_pid is None else host_pid)
        except FileNotFoundError:
            return dict(signaled=False, reason='already_exited', container_pid=record['container_pid'])
        if not matches_record(identity, record, host=host_pid is not None):
            raise ValueError('Process identity changed; no signal sent')
        try:
            signal.pidfd_send_signal(descriptor, signal.SIGINT, None, 0)
        except ProcessLookupError:
            return dict(signaled=False, reason='already_exited', container_pid=record['container_pid'])
    finally:
        os.close(descriptor)
    return dict(signaled=True, signal='SIGINT', container_pid=record['container_pid'],
                start_ticks=record['start_ticks'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('start', 'stop'))
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--token', required=True)
    args, argv = parser.parse_known_args()
    if argv[:1] == ['--']:
        argv = argv[1:]
    if args.mode == 'start':
        start_job(args.record, args.token, argv)
    else:
        if argv:
            parser.error('stop accepts no child command')
        print(json.dumps(stop_job(args.record, args.token)), flush=True)


if __name__ == '__main__':
    main()
