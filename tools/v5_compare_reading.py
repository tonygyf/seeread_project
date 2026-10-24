"""Compare saved V5 predictions on identical dev words; never run a model."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seeread.v5.reading import metrics, normalize
from seeread.metrics.spotting import edit_distance


def local_path(value):
    path = Path(value)
    if not path.exists() and path.is_relative_to('/workspace'):
        path = ROOT / path.relative_to('/workspace')
    return path


def load_result(path, step=None):
    path = Path(path)
    report = json.loads(path.read_text())
    if report.get('format') != 'seeread_v5_reading_v1':
        raise ValueError('Expected a V5 reading report')
    if step is None:
        result = report.get('result')
    else:
        choices = [r for r in report['validations'] if r['global_step'] == step and r['complete']]
        result = choices[-1] if choices else None
    if not result or not result['complete']:
        raise ValueError('Choose a complete dev evaluation, with --step for a training run')
    rows = [json.loads(line) for line in local_path(result['records']).read_text().splitlines()]
    if len(rows) != result['samples_requested']:
        raise ValueError('Prediction count differs from the complete report')
    if any(r.get('split') != 'dev' for r in rows):
        raise ValueError('Comparison is restricted to dev predictions')
    return report, rows


def compare(a, b, iterations=2000, seed=0):
    if iterations < 1:
        raise ValueError('iterations must be positive')
    aa, bb = {r['id']: r for r in a}, {r['id']: r for r in b}
    if len(aa) != len(a) or len(bb) != len(b) or aa.keys() != bb.keys() or not aa:
        raise ValueError('Predictions must contain the same unique, nonempty dev IDs')
    for key, row in aa.items():
        if any(row[field] != bb[key][field] for field in ('text', 'image', 'source', 'source_group', 'split')):
            raise ValueError(f'Different reference or image for {key}')
    rng = random.Random(seed)
    result = dict(samples=len(a), bootstrap_seed=seed, bootstrap_iterations=iterations,
                  bootstrap_unit='manifest source_group; original scene isolation not guaranteed',
                  difference='B minus A; negative CER difference favors B', policies={})
    for policy in ('raw', 'lower_alnum'):
        groups = defaultdict(lambda: [0, 0, 0])
        for key, row in aa.items():
            ref = normalize(row['text'], policy)
            group = groups[row['source_group']]
            group[0] += edit_distance(normalize(row['prediction'], policy), ref)
            group[1] += edit_distance(normalize(bb[key]['prediction'], policy), ref)
            group[2] += len(ref)
        values = list(groups.values())
        boot = []
        for _ in range(iterations):
            e0 = e1 = chars = 0
            for _ in values:
                g = values[rng.randrange(len(values))]
                e0 += g[0]; e1 += g[1]; chars += g[2]
            if chars:
                boot.append((e1 - e0) / chars)
        boot.sort()
        ma, mb = metrics(a, policy), metrics(b, policy)
        result['policies'][policy] = dict(A=ma, B=mb,
            cer_difference=mb['cer'] - ma['cer'] if ma['cer'] is not None else None,
            paired_bootstrap_95_percentile_interval=[boot[int(.025 * (len(boot) - 1))],
                                                       boot[int(.975 * (len(boot) - 1))]] if boot else None,
            sources={source: dict(A=metrics([r for r in a if r['source'] == source], policy),
                                  B=metrics([r for r in b if r['source'] == source], policy))
                     for source in sorted({r['source'] for r in a})})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-a', required=True)
    p.add_argument('--run-b', required=True)
    p.add_argument('--step-a', type=int)
    p.add_argument('--step-b', type=int)
    p.add_argument('--out', required=True)
    p.add_argument('--iterations', type=int, default=2000)
    args = p.parse_args()
    ra, a = load_result(args.run_a, args.step_a)
    rb, b = load_result(args.run_b, args.step_b)
    if ra['manifests']['dev']['sha256'] != rb['manifests']['dev']['sha256']:
        raise ValueError('Dev manifest hashes differ')
    result = compare(a, b, args.iterations)
    result.update(run_a=args.run_a, run_b=args.run_b, step_a=args.step_a, step_b=args.step_b,
                  dev_manifest_sha256=ra['manifests']['dev']['sha256'])
    with open(args.out, 'x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({p: v['cer_difference'] for p, v in result['policies'].items()}))


if __name__ == '__main__':
    main()
