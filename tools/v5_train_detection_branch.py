"""Train an independent target after the shared full pretraining finishes.

The parent pipeline owns SynthText/MLT outputs. This entry point only reads
those outputs and trains the target through the unchanged stage trainer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seeread.config import load_config
from tools.v5_detection_pilot import plain, sha256
from tools.v5_train_detection import execute, preflight_pipeline


def wait_for_predecessor(report_path, upstream_state):
    while True:
        if report_path.exists():
            report = json.loads(report_path.read_text())
            status = report['state']['status']
            if status == 'stage_complete':
                return
            if status in ('error_abort', 'resource_abort') or status.startswith('budget_stop'):
                raise RuntimeError(f'Pretraining stopped before completion: {report_path}: {status}')
        if upstream_state.exists():
            status = json.loads(upstream_state.read_text())['status']
            if status in ('needs_attention', 'failed_prerequisite', 'complete'):
                raise RuntimeError(f'Upstream queue ended without complete pretraining: {status}')
        time.sleep(10)


def run(pipeline, upstream_state):
    descriptor = plain(load_config(pipeline))
    configs = [plain(load_config(pipeline.parent / value)) for value in descriptor['stages']]
    preflight_pipeline(configs)
    predecessor_report = Path(configs[-2]['train']['out_dir']) / 'status.json'
    print(json.dumps(dict(event='waiting_for_shared_pretraining',
                          report=str(predecessor_report), upstream_queue=str(upstream_state),
                          pipeline=str(pipeline), pipeline_sha256=sha256(pipeline),
                          launcher_sha256=sha256(__file__),
                          target=configs[-1]['train']['out_dir'])), flush=True)
    wait_for_predecessor(predecessor_report, upstream_state)
    # execute/inherit_weights independently require a healthy, compatible,
    # completed MLT checkpoint. No parent stage is executed or overwritten.
    state = execute(configs[-1], reduced=False)
    if state['status'] != 'stage_complete':
        raise RuntimeError(f'Target did not finish: {state["status"]}')
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', type=Path, required=True)
    parser.add_argument('--upstream-state', type=Path, required=True)
    args = parser.parse_args()
    run(args.pipeline, args.upstream_state)
