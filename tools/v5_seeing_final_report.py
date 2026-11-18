"""Summarize the final Seeing runs on the fixed LATeD development split."""
import argparse
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/v5/detection'
LOGS = ROOT / 'results/v5/queue_logs'
REPORT = ROOT / 'results/v5/diagnostics/seeing_final_comparison_20261027.json'
RUNS = {
    'existing_balanced_100': DATA / 'LATeD_balanced_ce_20261026',
    'height_candidate_100': DATA / 'LATeD_height_scale005_20261027',
    'new_structure': DATA / 'LATeD_final_scm_20261027',
}
QUEUES = {
    'height_candidate_100': LOGS / '20261027_final_height_gpu2_state.json',
    'new_structure': LOGS / '20261027_final_scm_gpu0_state.json',
}


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def development(directory):
    provenance = read_json(directory / 'provenance.json')
    path = directory / 'events.jsonl'
    if not provenance or not path.exists():
        return None
    rows = []
    with path.open() as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # The trainer may be writing its current line.
            if event.get('event') == 'development' and event.get('complete'):
                rows.append(event)
    if not rows:
        return None
    best = max(rows, key=lambda row: row['metrics']['det_F'])
    return {
        'dev_manifest_sha256': provenance['data']['dev']['manifest_sha256'],
        'best_epoch': best['epoch'],
        'latest_evaluated_epoch': max(row['epoch'] for row in rows),
        'precision': best['metrics']['det_P'],
        'recall': best['metrics']['det_R'],
        'f1': best['metrics']['det_F'],
        'records': best['records'],
        'protocol': best['protocol'],
    }


def build_report():
    runs = {name: development(path) for name, path in RUNS.items()}
    queues = {name: (read_json(path) or {}).get('status', 'not_started') for name, path in QUEUES.items()}
    baseline = runs['existing_balanced_100']
    comparisons = {}
    for name in QUEUES:
        candidate = runs[name]
        if baseline and candidate and baseline['dev_manifest_sha256'] == candidate['dev_manifest_sha256']:
            comparisons[name] = {
                'same_development_manifest': True,
                'f1_difference_percentage_points': 100 * (candidate['f1'] - baseline['f1']),
                'note': 'Best development scores; check evaluated epochs before interpreting differences.',
            }
        elif candidate:
            comparisons[name] = {'same_development_manifest': False, 'note': 'Scores not compared: manifest differs.'}
    return {
        'updated_utc': datetime.now(timezone.utc).isoformat(),
        'queues': queues,
        'development_runs': runs,
        'development_comparisons': comparisons,
        'paper_lated_test_percent': {'precision': 82.6, 'recall': 57.0, 'f1': 67.1},
        'acceptance_test_percent_minimum': {'precision': 79.6, 'recall': 54.0, 'f1': 64.1},
        'note': 'Development scores are not paper test scores. Official acceptance remains pending locked model and official test evaluation.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        report = build_report()
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        temp = REPORT.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        os.replace(temp, REPORT)
        if not args.watch or all(status in ('complete', 'needs_attention', 'failed_prerequisite')
                                 for status in report['queues'].values()):
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
