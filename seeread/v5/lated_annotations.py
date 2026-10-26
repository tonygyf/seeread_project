"""Independent, human-reviewed LATeD transcription overlays; frozen GT stays intact."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unicodedata

from seeread.v5.reading import sha256


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))


def select_training(rows, root, count=200, seed=0):
    """Choose whole content groups; duplicate images cannot straddle a selection."""
    groups = {}
    for row in rows:
        digest = sha256(Path(root) / row['image'])
        groups.setdefault(digest, []).append(row)
    ordered = sorted(groups, key=lambda h: hashlib.sha256(f'{seed}:{h}'.encode()).hexdigest())
    selected = []
    for digest in ordered:
        if len(selected) >= count:
            break
        selected.extend(groups[digest])
    if len(selected) != count:
        raise ValueError('Requested image count crosses a duplicate-content group; choose a whole-group count')
    return selected


def latin_text(text):
    return any(c.isalnum() for c in text) and all(not c.isalpha() or 'LATIN' in unicodedata.name(c, '') for c in text)


def render_pages(output):
    output = Path(output)
    items = read_jsonl(output / 'tasks.jsonl')
    drafts = {r['id']: r for r in read_jsonl(output / 'drafts.jsonl')} if (output / 'drafts.jsonl').exists() else {}
    items = [{**it, 'draft': drafts.get(it['id'], {}).get('text', '')} for it in items]
    template = (Path(__file__).resolve().parents[2] / 'tools/data/lated_review_template.html').read_text()
    for name, selected in [('review', items), ('pilot', [i for i in items if i['pilot']])]:
        payload = json.dumps(selected, ensure_ascii=False).replace('<', '\\u003c')
        page = template.replace('__ITEMS__', payload).replace('__PROTOCOL__', json.dumps(sha256(output / 'protocol.json')))
        (output / f'{name}.html').write_text(page)


def verify_package(package):
    package = Path(package)
    protocol = json.loads((package / 'protocol.json').read_text())
    if protocol['schema'] != 'lated_transcription_v1':
        raise ValueError('Unsupported transcription package')
    if sha256(package / 'tasks.jsonl') != protocol['tasks_sha256']:
        raise ValueError('Task identities/geometry changed after package preparation')
    for source in protocol['sources'].values():
        if sha256(source['path']) != source['sha256']:
            raise ValueError('Frozen source annotation changed')
    tasks = read_jsonl(package / 'tasks.jsonl')
    images = {row['image']: row['image_sha256'] for row in tasks}
    for image, digest in images.items():
        if sha256(Path(protocol['data_root']) / image) != digest:
            raise ValueError('Source image bytes changed after annotation preparation')
    return protocol, tasks


def import_reviews(package, export_path, output):
    protocol, tasks = verify_package(package)
    exported = json.loads(Path(export_path).read_text())
    binding = sha256(Path(package) / 'protocol.json')
    if exported.get('protocol_sha256') != binding or not exported.get('reviewer', '').strip():
        raise ValueError('Review export requires the exact package identity and a reviewer name')
    expected = {row['id']: row for row in tasks}
    output = Path(output)
    previous = json.loads(output.read_text()) if output.exists() else {'protocol_sha256': binding, 'annotations': []}
    if previous['protocol_sha256'] != binding:
        raise ValueError('Existing sidecar belongs to another package')
    merged = {row['id']: row for row in previous['annotations']}
    seen = set()
    for row in exported['annotations']:
        identity = row['id']
        if identity not in expected or identity in seen:
            raise ValueError('Unknown or duplicate instance identity')
        seen.add(identity)
        if row['status'] == 'pending':
            merged.pop(identity, None)
            continue
        if row['status'] not in ('reviewed', 'unreadable') or row['language'] not in ('latin', 'non_latin', 'mixed', 'unknown'):
            raise ValueError('Invalid review status/language')
        text = unicodedata.normalize('NFC', row.get('text', '').strip())
        if row['status'] == 'reviewed' and row['language'] == 'unknown':
            raise ValueError('Reviewed instances need an explicit language category')
        if row['status'] == 'reviewed' and row['language'] == 'latin' and not latin_text(text):
            raise ValueError('Readable Latin instances need a nonempty Latin transcription')
        merged[identity] = dict(id=identity, text=text, language=row['language'], status=row['status'],
                                reviewer=exported['reviewer'].strip(), split=expected[identity]['split'],
                                image=expected[identity]['image'], image_sha256=expected[identity]['image_sha256'])
    result = dict(schema='lated_transcription_sidecar_v1', protocol_sha256=binding,
                  annotations=[merged[k] for k in expected if k in merged],
                  total_instances=len(expected), reviewed_instances=len(merged))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(output)
    return result


def labelled_rows(package, sidecar, split, require_complete=True, pilot=False):
    protocol, tasks = verify_package(package)
    reviewed = json.loads(Path(sidecar).read_text())
    if reviewed.get('protocol_sha256') != sha256(Path(package) / 'protocol.json'):
        raise ValueError('Sidecar is not bound to this package')
    if split not in ('train', 'dev'):
        raise ValueError('MVP transcription/evaluation only accepts train/dev')
    entries = reviewed['annotations']
    annotations = {row['id']: row for row in entries}
    if len(annotations) != len(entries):
        raise ValueError('Duplicate sidecar instance')
    task_ids = {row['id'] for row in tasks}
    if reviewed.get('schema') != 'lated_transcription_sidecar_v1' or set(annotations) - task_ids:
        raise ValueError('Invalid transcription sidecar or unknown instance')
    if any(a.get('status') not in ('reviewed', 'unreadable') or not a.get('reviewer') for a in entries):
        raise ValueError('Sidecar contains unconfirmed annotations')
    selected = {row['image'] for row in tasks if row['split'] == split and (not pilot or row.get('pilot'))}
    source = read_jsonl(protocol['sources'][split]['path'])
    rows = [copy.deepcopy(row) for row in source if row['image'] in selected]
    for row in rows:
        row['split'] = split
        for instance in row['instances']:
            annotation = annotations.get(instance['id'])
            eligible = False
            if annotation:
                if annotation['image'] != row['image'] or annotation['split'] != split:
                    raise ValueError('Sidecar instance has incompatible image/split')
                eligible = annotation['status'] == 'reviewed' and annotation['language'] == 'latin'
                if eligible and not latin_text(annotation['text']):
                    raise ValueError('Invalid Latin sidecar transcription')
            elif require_complete and not instance.get('ignore'):
                raise ValueError(f'Human review incomplete: {instance["id"]}')
            instance['text'] = annotation['text'] if eligible else ''
            instance['ocr_eligible'] = eligible and not instance.get('ignore', False)
            instance['transcription_status'] = annotation['status'] if annotation else 'pending'
            instance['lang'] = annotation['language'] if annotation else 'unknown'
    return rows
