"""Merge human-reviewed transcriptions into a real-benchmark annotation file.

Input:  lated_test_transcriptions.json  ({id: {t, f}} exported from data/lated_trans/review.html)
Output: data/annotations/latedspot_test_text.jsonl  (non-ignored instances get real text)
"""
import sys, json

exp_path = sys.argv[1] if len(sys.argv) > 1 else 'lated_test_transcriptions.json'
exp = json.load(open(exp_path, encoding='utf-8'))
items = {r['id']: r for r in map(json.loads, open('data/lated_trans/prefill.jsonl', encoding='utf-8'))}
orig2fn = {r.get('orig') or r['id']: r['id'] for r in items.values()}

rows = [json.loads(l) for l in open('data/annotations/latedspot_test.jsonl', encoding='utf-8')]
filled = flagged = unknown = 0
out = []
for r in rows:
    for ins in r['instances']:
        if ins.get('ignore'):
            continue
        fn = orig2fn.get(ins.get('id'))
        e = exp.get(fn) if fn else None
        if e is None:
            unknown += 1
            continue
        t = e.get('t', '').strip()
        if e.get('f') or t == '':
            flagged += 1
            continue
        ins['text'] = t
        filled += 1
    out.append(json.dumps(r, ensure_ascii=False) + '\n')

open('data/annotations/latedspot_test_text.jsonl', 'w', encoding='utf-8').writelines(out)
total = sum(not i.get('ignore') for r in rows for i in r['instances'])
print(f'filled {filled} / {total} non-ignored; skipped(flagged/empty) {flagged}; missing-from-export {unknown}')
print(f'-> data/annotations/latedspot_test_text.jsonl  (e2e-capable real dark benchmark once filled >=90%)')
