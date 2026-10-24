"""Generate recognition crops for the standalone TrOCR recognizer (baseline recipe)."""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, cv2, torch
from shapely.geometry import Polygon

from seeread.config import load_config
from seeread.data.geometry import centerline_and_height
from seeread.data.lowlight_synth import darken
from seeread.models.detector import SeeReadDetector
from seeread.models.rectify import rectify_strip
from seeread.models.tsr import tsr_decode
from seeread.metrics.spotting import norm_text, iou

ROOT = 'data'
ANN_SETS = {'totaltext': 'data/annotations/totaltext_train_main.jsonl',
            'icdar15': 'data/annotations/icdar2015_train.jsonl'}
DET_CKPT = 'results/archive/legacy/v2/s3_joint_v2/best.pt'
OUT = 'results/archive/legacy/det_crops'
SEED = 1234
MAX_DET = 100
MIN_H = 4.0
IOU_THR = 0.5


def _poly(p):
    q = Polygon(np.asarray(p, np.float64).reshape(-1, 2))
    return q if q.is_valid else q.buffer(0)


def read_rgb(p):
    im = cv2.imread(os.path.join(ROOT, p), cv2.IMREAD_COLOR)
    assert im is not None, p
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def to_canvas(img_u8, s, nh, nw, size):
    img = cv2.resize(img_u8, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:min(nh, size), :min(nw, size)] = img[:size, :size]
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', nargs='*', default=list(ANN_SETS), choices=list(ANN_SETS))
    ap.add_argument('--limit', type=int, default=0, help='max images per set (0 = all)')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--det-ckpt', default=DET_CKPT)
    ap.add_argument('--no-det', action='store_true', help='skip detector-in-the-loop crops (GT only)')
    a = ap.parse_args()

    cfg = load_config('configs/base.yaml', {'data': {}, 'train': {'amp': False}})
    size, stride = cfg.data.size, 4
    hw = tuple(cfg.model.crop_hw)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    crops_dir = os.path.join(a.out, 'crops')
    os.makedirs(crops_dir, exist_ok=True)

    det = SeeReadDetector(ch=cfg.model.det_ch, pretrained=False, ctx_dim=cfg.model.ctx_dim).to(dev).eval()
    sd = torch.load(a.det_ckpt, map_location=dev)
    det.load_state_dict(sd['det'])
    del sd
    print(f'loaded det={a.det_ckpt} crop_hw={hw} size={size}', flush=True)

    cv2.setNumThreads(1)

    def det_forward(x):
        with torch.no_grad():
            out = det(x)
            tp = torch.sigmoid(out['text'][:, 0]).cpu().numpy()
            cp = torch.sigmoid(out['center'][:, 0]).cpu().numpy()
            geo = torch.cat([out['geo'][:, :1].clamp(min=0), out['geo'][:, 1:]], 1).cpu().numpy()
        dets = tsr_decode(tp[0], cp[0], geo[0], stride, 0.5, 0.5)[:MAX_DET]
        for d in dets:
            if len(d['centers']) < 2:
                d['centers'] = np.concatenate([d['centers'] - [2, 0], d['centers'] + [2, 0]])
                d['heights'] = np.repeat(d['heights'], 2)
        return dets

    def to_tensor(u8):
        return torch.from_numpy(np.ascontiguousarray(u8)).permute(2, 0, 1).float()[None].to(dev) / 255.0

    man = []
    n = dict(gt=0, det=0, matched=0)

    def emit(x_t, centers, heights, text, cond, source, set_name, row, img_id):
        C = torch.from_numpy(np.asarray(centers, np.float32)).to(dev)
        Hh = torch.from_numpy(np.asarray(heights, np.float32)).to(dev)
        strip = rectify_strip(x_t, C, Hh, hw)[0]
        u8 = (strip.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        rel = f'crops/{cond}_{source}_{n[source]:06d}.png'
        cv2.imwrite(os.path.join(a.out, rel), u8[..., ::-1])
        man.append(dict(crop=rel, text=text, cond=cond, source=source,
                        set=set_name, img=row['image'], id=img_id))
        n[source] += 1

    img_idx = 0
    total_imgs = sum((a.limit if a.limit else len(open(ANN_SETS[s], encoding='utf-8').readlines()))
                     for s in a.sets)
    for set_name in a.sets:
        rows = [json.loads(l) for l in open(ANN_SETS[set_name], encoding='utf-8') if l.strip()]
        if a.limit:
            rows = rows[:a.limit]
        for ri, row in enumerate(rows):
            img = read_rgb(row['image'])
            H, W = img.shape[:2]
            s = size / max(H, W)
            nh, nw = int(round(H * s)), int(round(W * s))
            lit_u8 = to_canvas(img, s, nh, nw, size)
            rng = np.random.default_rng((SEED, img_idx))
            dark_raw, meta = darken(img, rng, mode=cfg.data.synth_mode)
            dark_u8 = to_canvas(dark_raw, s, nh, nw, size)
            img_id = row['meta'].get('id') or os.path.basename(row['image'])

            gts = []
            for gi, ins in enumerate(row['instances']):
                poly = np.asarray(ins['poly'], np.float32) * s
                text = ins.get('text', '')
                gts.append(dict(poly=poly, gp=_poly(poly), text=text,
                                ignore=bool(ins.get('ignore')) or text in ('###', ''), gi=gi))

            for g in gts:
                if g['ignore'] or norm_text(g['text']) == '':
                    continue
                c, hh = centerline_and_height(g['poly'], k=7)
                if hh.mean() < MIN_H:
                    continue
                emit(to_tensor(lit_u8), c, hh, g['text'], 'lit', 'gt', set_name, row, img_id)
                emit(to_tensor(dark_u8), c, hh, g['text'], 'dark', 'gt', set_name, row, img_id)

            if a.no_det:
                img_idx += 1
                continue

            for cond, u8 in (('lit', lit_u8), ('dark', dark_u8)):
                x_t = to_tensor(u8)
                dets = det_forward(x_t)
                order = sorted(range(len(dets)), key=lambda i: -dets[i]['score'])
                used, matched = set(), []
                for i in order:
                    pp = _poly(dets[i]['poly'])
                    best, bj = 0.0, -1
                    for j, g in enumerate(gts):
                        if j in used:
                            continue
                        v = iou(pp, g['gp'])
                        if v > best:
                            best, bj = v, j
                    if best >= IOU_THR and bj >= 0:
                        used.add(bj)
                        if not gts[bj]['ignore'] and norm_text(gts[bj]['text']) != '':
                            matched.append((bj, i))
                for bj, di in matched:
                    emit(x_t, dets[di]['centers'], dets[di]['heights'],
                         gts[bj]['text'], cond, 'det', set_name, row, img_id)
                n['matched'] += len(matched)

            img_idx += 1
            if img_idx % 100 == 0:
                print(f'[{img_idx}/{total_imgs}] gt={n["gt"]} det={n["det"]} matched_pairs={n["matched"]}', flush=True)

    mp = os.path.join(a.out, 'manifest.jsonl')
    with open(mp, 'w', encoding='utf-8') as f:
        for line in man:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    print(f'done: {len(man)} crops -> {mp} '
          f'(gt={n["gt"]} det={n["det"]}, matched det/gt pairs={n["matched"]})', flush=True)


if __name__ == '__main__':
    main()
