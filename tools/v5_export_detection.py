#!/usr/bin/env python3
"""Export saved predictions for explicit pinned author protocols; never runs evaluation."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seeread.v5.evaluation_export import SOURCES, export_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--predictions', required=True, help='Complete V5 per-image prediction JSONL')
    parser.add_argument('--output', required=True, help='New output directory')
    parser.add_argument('--protocol', action='append', required=True, choices=tuple(SOURCES))
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--gt-root', required=True, help='Original TT MAT directory or CTW list-path resolution root')
    parser.add_argument('--ctw-imageset', help='Original CTW test.txt, unmodified author order')
    parser.add_argument('--ctw-annotations', help='Original CTW test_label_curve.txt, same order')
    parser.add_argument('--quantization', choices=('none', 'truncate', 'nearest_even'), default='none',
                        help='TT requires an explicit integer policy; CTW preserves float coordinates')
    args = parser.parse_args()
    audit = export_predictions(args.manifest, args.predictions, args.output,
                               protocols=args.protocol, data_root=args.data_root, gt_root=args.gt_root,
                               ctw_imageset=args.ctw_imageset, ctw_annotations=args.ctw_annotations,
                               quantization=args.quantization)
    print(json.dumps(dict(output=args.output, images=audit['manifest']['images'],
                         protocols=list(audit['protocols']), evaluation_executed=False)))


if __name__ == '__main__':
    main()
