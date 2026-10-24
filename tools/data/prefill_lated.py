"""Semi-automatic transcription scaffold for LATeD test (500 imgs / ~4.8k instances).

Pipeline: GT polygons (no detection needed) -> crops from the BRIGHT counterpart for the human,
recognizer pre-fill for the machine (bright: raw rec, dark: enh+rec), native-res crop saved for review.
Outputs (data/lated_trans/): crops/<id>.png + prefill.jsonl + review.html (self-contained, file:// OK).
Human reviews review.html in a browser, exports lated_test_transcriptions.json,
then tools/data/finalize_lated.py merges it into data/annotations/latedspot_test_text.jsonl.
"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, cv2, torch
from seeread.config import load_config
from seeread.models.recognizer import build_recognizer
from seeread.models.seeread import SeeRead
from seeread.data.geometry import centerline_and_height
from seeread.models.rectify import rectify_strip
from seeread.metrics.spotting import norm_text

ROOT, ANN = 'data', 'data/annotations/latedspot_test.jsonl'
OUT, CKPT, HW = 'data/lated_trans', 'results/archive/legacy/s4_rec_v4/best.pt', (64, 256)

cfg = load_config('configs/legacy/s4_rec_v4.yaml', {'train': {'amp': False}})
dev = 'cuda'
m = SeeRead(build_recognizer(cfg.rec), cfg).to(dev).eval()
sd = torch.load(CKPT, map_location=dev)
for k in ('det', 'enh', 'rec'):
    getattr(m, k).load_state_dict(sd[k])

rows = [json.loads(l) for l in open(ANN)]
os.makedirs(f'{OUT}/crops', exist_ok=True)


def read(p):
    im = cv2.imread(os.path.join(ROOT, p), cv2.IMREAD_COLOR)
    assert im is not None, p
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


items, crops, ctxs, kind = [], [], [], []   # kind: 'b' bright-raw, 'd' dark-enh
used = set()
for n, r in enumerate(rows):
    stem = r['meta']['id']
    assert '/image/' in r['image'], r['image']
    dark, bright = read(r['image']), read(r['image'].replace('/image/', '/image_bright/'))
    d512 = cv2.resize(dark, (cfg.data.size, cfg.data.size))
    x = torch.from_numpy(d512).permute(2, 0, 1).float()[None].to(dev) / 255
    with torch.no_grad():
        o = m.det(x)
        ctx_img = m.det.illum_context(x, o['c5'])[0]
    for k, ins in enumerate(r['instances']):
        if ins.get('ignore'):
            continue
        fn = re.sub(r'[^A-Za-z0-9_.-]', '_', ins.get('id') or f'{stem}_{k}')
        assert fn not in used, fn
        used.add(fn)
        poly = np.asarray(ins['poly'], np.float32)
        c, hh = centerline_and_height(poly, 7)
        tiny = bool(hh.mean() < 4)
        if not tiny:
            cen = torch.from_numpy(c.astype(np.float32)).to(dev)
            hg = torch.from_numpy(hh.astype(np.float32)).to(dev)
            db = torch.from_numpy(bright).permute(2, 0, 1).float()[None].to(dev) / 255
            dd = torch.from_numpy(dark).permute(2, 0, 1).float()[None].to(dev) / 255
            crops.append(rectify_strip(db, cen, hg, HW).squeeze(0).cpu()); kind.append('b')
            crops.append(rectify_strip(dd, cen, hg, HW).squeeze(0).cpu()); kind.append('d')
            ctxs.append(ctx_img.cpu()); ctxs.append(ctx_img.cpu())
        # native-res crop for the human (bright), upscaled if small
        x0, y0 = np.floor(poly.min(0) - 8).astype(int); x1, y1 = np.ceil(poly.max(0) + 8).astype(int)
        crop = bright[max(0, y0):y1, max(0, x0):x1]
        if crop.size == 0:
            crop = bright[:32, :32]; tiny = True
        ch, cw = crop.shape[:2]
        if ch < 24:
            f = max(1, 24 // ch)
            crop = cv2.resize(crop, (cw * f, ch * f), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f'{OUT}/crops/{fn}.jpg', cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        items.append(dict(id=fn, orig=ins.get('id'), img=stem, lang=ins.get('lang', '?'), tiny=tiny))
    if (n + 1) % 100 == 0:
        print(f'{n + 1}/{len(rows)} images, {len(items)} instances', flush=True)

print(f'{len(items)} instances, predicting...', flush=True)

pb, pd = {}, {}
with torch.no_grad():
    for i in range(0, len(crops), 256):
        chunk = crops[i:i + 256]
        xb = torch.stack(chunk).to(dev)
        zb = torch.stack(ctxs[i:i + 256]).to(dev)
        y, _ = m.enh(xb, zb)
        outs_b = m.rec.predict(xb)
        outs_d = m.rec.predict(y)
        for j in range(len(chunk)):
            gi = i + j
            it = items[gi // 2]
            (pb if kind[gi] == 'b' else pd)[it['id']] = (outs_b if kind[gi] == 'b' else outs_d)[j]

agree = sum(1 for it in items if not it['tiny'] and norm_text(pb[it['id']]) == norm_text(pd[it['id']]))
valid = sum(1 for it in items if not it['tiny'])
with open(f'{OUT}/prefill.jsonl', 'w') as f:
    for it in items:
        it = dict(it, pred_b=pb.get(it['id'], ''), pred_d=pd.get(it['id'], ''))
        f.write(json.dumps(it, ensure_ascii=False) + '\n')

tmpl = open('tools/data/lated_review_template.html', encoding='utf-8').read()
html = tmpl.replace('__ITEMS__', json.dumps(items, ensure_ascii=False))
open(f'{OUT}/review.html', 'w', encoding='utf-8').write(html)
print(f'DONE: {valid} valid + {len(items) - valid} tiny; bright/dark pred agreement {agree}/{valid} '
      f'({100 * agree / max(1, valid):.1f}%)', flush=True)
print(f'Open {OUT}/review.html in a browser; export -> tools/data/finalize_lated.py', flush=True)
