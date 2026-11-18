"""Evaluate captured late-epoch LATeD checkpoints serially on CPU."""
import json
from pathlib import Path
import subprocess
import time

runs = [
    ('height_official_train', 'results/v5/detection/LATeD_height005_official_train1000_20261027', 'results/v5/runtime/lated_height_scale005_20261027', 'results/v5/queue_logs/20261027_two_tasks_height005_gpu2_state.json'),
    ('new_structure_official_train', 'results/v5/detection/LATeD_final_scm_official_train1000_20261027', 'results/v5/runtime/seeing_final_scm_20261027', 'results/v5/queue_logs/20261027_two_tasks_new_structure_gpu0_state.json'),
]
terminal_since = {}
remaining = {(name, epoch) for name, _, _, _ in runs for epoch in (248, 249, 250)}
while remaining:
    for name, directory, runtime, state_path in runs:
        for epoch in (248, 249, 250):
            key = (name, epoch)
            if key not in remaining:
                continue
            checkpoint = Path(directory) / f'test_candidate_epoch_{epoch:03d}.pt'
            if checkpoint.exists():
                output = f'results/v5/diagnostics/lated_test500_{name}_epoch{epoch}_20261027'
                command = ['docker', 'exec', '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'OMP_NUM_THREADS=2',
                           '-e', 'CUDA_VISIBLE_DEVICES=', 'seeread-v5-run-20261020b', 'python',
                           'tools/v5_eval_lated_test.py', '--runtime', runtime, '--checkpoint', str(checkpoint), '--output', output]
                print(f'Starting {name} epoch {epoch}', flush=True)
                code = subprocess.run(command).returncode
                print(f'Finished {name} epoch {epoch}: return {code}', flush=True)
                remaining.remove(key)
            elif Path(state_path).exists():
                state = json.loads(Path(state_path).read_text()).get('status')
                if state in ('complete', 'needs_attention', 'failed_prerequisite'):
                    terminal_since.setdefault(name, time.monotonic())
                    if time.monotonic() - terminal_since[name] >= 60:
                        print(f'Missing {checkpoint}; queue ended with {state}', flush=True)
                        remaining.remove(key)
    if remaining:
        time.sleep(15)
