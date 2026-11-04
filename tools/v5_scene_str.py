"""Isolated LATeD STR data preparation and coverage-aware scoring; never trains."""
import argparse
import cv2
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seeread.v5.lated_annotations import labelled_rows, read_jsonl, verify_package, write_jsonl, render_pages
from seeread.v5.reading import WordCropDataset, check_disjoint, sha256
from seeread.v5.scene_str import audit_splits, export_reviewed, score_catalog, quality_hint, pair_training


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit')
    a.add_argument('--root', type=Path, required=True)
    a.add_argument('--manifests', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    e = sub.add_parser('export')
    e.add_argument('--package', type=Path, required=True)
    e.add_argument('--sidecar', type=Path, required=True)
    e.add_argument('--output', type=Path, required=True)
    e.add_argument('--method', choices=('axis_aligned', 'boundary_when_supported'), required=True)
    e.add_argument('--pilot', action='store_true')
    q = sub.add_parser('review-quality')
    q.add_argument('--package', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    pair = sub.add_parser('pair-train')
    pair.add_argument('--a', type=Path, required=True)
    pair.add_argument('--b', type=Path, required=True)
    pair.add_argument('--output', type=Path, required=True)
    c = sub.add_parser('compare')
    for name in ('catalog-a', 'predictions-a', 'catalog-b', 'predictions-b'):
        c.add_argument('--' + name, type=Path, required=True)
    c.add_argument('--output', type=Path, required=True)
    s = sub.add_parser('score')
    s.add_argument('--catalog', type=Path, required=True)
    s.add_argument('--predictions', type=Path, required=True)
    s.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.command == 'review-quality':
        _, tasks = verify_package(args.package)
        hints = []
        for task in tasks:
            image = cv2.imread(str(args.package / task['crop']))
            if image is None:
                raise FileNotFoundError(task['crop'])
            hints.append(dict(id=task['id'], **quality_hint(image)))
        if args.output.resolve() != (args.package / 'quality_hints.jsonl').resolve():
            raise ValueError('Review hints must be a sidecar within the selected package')
        write_jsonl(args.output, hints)
        render_pages(args.package)
        print(json.dumps(dict(instances=len(hints), extreme_candidates=sum(h['needs_extreme_review'] for h in hints),
                              automatic_exclusions=0)))
        return
    if args.command == 'audit':
        splits = {s: read_jsonl(args.manifests / f'latedspot_{suffix}.jsonl')
                  for s, suffix in [('train', 'train'), ('dev', 'val'), ('test', 'test')]}
        result = audit_splits(args.root, splits)
        if [result[s]['images'] for s in ('train', 'dev', 'test')] != [900, 100, 500]:
            raise ValueError('Expected the existing 900/100/500 image split')
    elif args.command == 'pair-train':
        sources = [read_jsonl(p / 'train.jsonl') for p in (args.a, args.b)]
        arms = pair_training(*sources)
        for name, parent, rows in zip(('a', 'b'), (args.a, args.b), arms):
            out = args.output / name
            out.mkdir(parents=True)
            write_jsonl(out / 'train.jsonl', rows)
            for filename in ('dev.jsonl', 'dev.catalog.jsonl'):
                (out / filename).write_bytes((parent / filename).read_bytes())
            protocol = json.loads((parent / 'protocol.json').read_text())
            protocol.update(parent_protocol_sha256=sha256(parent / 'protocol.json'),
                            paired_training_ids=len(rows), parent_training_count=len(sources[0 if name == 'a' else 1]),
                            manifests={s: sha256(out / f'{s}.jsonl') for s in ('train', 'dev')})
            (out / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
        print(json.dumps(dict(paired_train=len(arms[0]), original_train=[len(s) for s in sources])))
        return
    elif args.command == 'compare':
        from tools.v5_compare_reading import compare
        arms = []
        for catalog_path, prediction_path in ((args.catalog_a, args.predictions_a), (args.catalog_b, args.predictions_b)):
            catalog, predictions = read_jsonl(catalog_path), read_jsonl(prediction_path)
            score_catalog(catalog, predictions)  # Reject incomplete successful-crop predictions.
            by_id = {r['id']: r['prediction'] for r in predictions}
            arms.append([dict(id=r['id'], text=r['text'], prediction='' if r['crop_error'] else by_id[r['id']],
                              image=r['source_image'], source_group=r['source_image'],
                              source='LATeD_human_reviewed_GT', split=r['split'])
                         for r in catalog if r['eligible']])
        if any(r['split'] != 'dev' for arm in arms for r in arm):
            raise ValueError('Paired method selection only accepts dev')
        result = compare(*arms)
        result['bootstrap_unit'] = 'original image; scene-near-duplicates not independently verified'
    elif args.command == 'score':
        result = score_catalog(read_jsonl(args.catalog), read_jsonl(args.predictions))
    else:
        protocol, _ = verify_package(args.package)
        root = Path(protocol['data_root'])
        sources = {s: labelled_rows(args.package, args.sidecar, s, pilot=args.pilot) for s in ('train', 'dev')}
        audit_splits(root, sources)
        image_dir = root / 'derived/v5/scene_str' / args.output.name
        if image_dir.exists():
            raise FileExistsError(image_dir)
        args.output.mkdir(parents=True)
        counts = {}
        for split, rows in sources.items():
            words, catalog = export_reviewed(rows, root, image_dir, split, args.method)
            write_jsonl(args.output / f'{split}.jsonl', words)
            write_jsonl(args.output / f'{split}.catalog.jsonl', catalog)
            counts[split] = dict(eligible=sum(r['eligible'] for r in catalog), exported=len(words), all_instances=len(catalog))
        datasets = [WordCropDataset(args.output / f'{s}.jsonl', root, s) for s in ('train', 'dev')]
        check_disjoint(*datasets)
        result = dict(status='ready', mode='pilot' if args.pilot else 'reviewed_package', counts=counts,
                      root=str(root), method=args.method, package_sha256=sha256(args.package / 'protocol.json'),
                      sidecar_sha256=sha256(args.sidecar), filtering='none; human readability defines transcription eligibility',
                      selection='all eligible dev references, including failed crops as empty predictions',
                      test='not exported; separate locked final protocol required',
                      manifests={s: sha256(args.output / f'{s}.jsonl') for s in sources})
        (args.output / 'protocol.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
        print(json.dumps(result, ensure_ascii=False)); return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
