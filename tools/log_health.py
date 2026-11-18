"""Compare logged detection metrics around a resumed checkpoint."""
import argparse
import json
import statistics
from pathlib import Path


def updates(path):
    with Path(path).open() as stream:
        return [row for line in stream if (row := json.loads(line)).get('event') == 'update']


def summary(rows):
    return {
        'count': len(rows),
        'steps': [rows[0]['step'], rows[-1]['step']] if rows else [],
        'medians': {
            'loss': statistics.median(row['loss'] for row in rows),
            'scm_sem': statistics.median(row['losses']['scm_sem'] for row in rows),
            'gradient_norm_pre_clip': statistics.median(row['gradient_norm'] for row in rows),
        } if rows else {},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True)
    parser.add_argument('--after', required=True)
    parser.add_argument('--step', required=True, type=int)
    parser.add_argument('--windows', type=int, default=20)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    before = [row for row in updates(args.before) if row['step'] <= args.step][-args.windows:]
    after = [row for row in updates(args.after) if row['step'] > args.step][:args.windows]
    result = dict(before=summary(before), after=summary(after),
                  complete=len(before) == len(after) == args.windows,
                  interpretation='Logged updates; pre-clipping gradient norm; batches differ.')
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
