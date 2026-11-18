"""Preserve late-epoch checkpoints while two LATeD runs continue."""
import json
import os
from pathlib import Path
import time

runs = [
    (Path('results/v5/detection/LATeD_height005_official_train1000_20261027'), Path('results/v5/queue_logs/20261027_two_tasks_height005_gpu2_state.json')),
    (Path('results/v5/detection/LATeD_final_scm_official_train1000_20261027'), Path('results/v5/queue_logs/20261027_two_tasks_new_structure_gpu0_state.json')),
]
offsets = {str(path): 0 for path, _ in runs}
done = set()
while len(done) < len(runs):
    for directory, state_path in runs:
        name = str(directory)
        if name in done:
            continue
        events = directory / 'events.jsonl'
        if events.exists():
            with events.open() as stream:
                stream.seek(offsets[name])
                while True:
                    start = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if not line.endswith('\n'):
                        stream.seek(start)
                        break
                    event = json.loads(line)
                    epoch = event.get('epoch')
                    if event.get('event') == 'development' and event.get('complete') and epoch in (248, 249, 250):
                        target = directory / f'test_candidate_epoch_{epoch:03d}.pt'
                        if not target.exists():
                            os.link(directory / 'last.pt', target)
                            print(f'preserved {target} at step {event["step"]}', flush=True)
                offsets[name] = stream.tell()
        if state_path.exists():
            state = json.loads(state_path.read_text()).get('status')
            if state in ('complete', 'needs_attention', 'failed_prerequisite'):
                done.add(name)
                print(f'capture finished {directory}: {state}', flush=True)
    if len(done) < len(runs):
        time.sleep(5)
