"""Standalone TrOCR LoRA recognizer trainer (baseline recipe, enhancer = identity).

Consumes the crop manifest written by tools/legacy/gen_det_crops.py and trains ONLY the TrOCR recognizer
(no SeeRead wrapper, no enhancer, no detection loss). The detector is loaded frozen ONLY for
end-to-end validation. best.pt is selected by the e2e F1 on the bright val set (detector + recognizer).

Recognition-only diagnostics (GT-crop word accuracy, lit + dark) are logged every validation but do
not drive checkpoint selection.
"""
import sys, os, json, argparse, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, cv2, torch
from torch.utils.data import Dataset, DataLoader
from shapely.geometry import Polygon

from seeread.config import load_config
from seeread.data.geometry import centerline_and_height
from seeread.data.lowlight_synth import darken
from seeread.models.detector import SeeReadDetector
from seeread.models.recognizer import TrOCRRecognizer
from seeread.models.rectify import rectify_strip
from seeread.models.tsr import tsr_decode
from seeread.metrics.spotting import SpottingEvaluator, norm_text, iou

DET_CKPT = 'results/archive/legacy/v2/s3_joint_v2/best.pt'
MANIFEST = 'results/archive/legacy/det_crops/manifest.jsonl'
CROP_ROOT = 'results/archive/legacy/det_crops'
VAL_ANN = 'data/annotations/totaltext_val.jsonl'
OUT_DIR = 'results/archive/legacy/rec_standalone'
MAX_DET = 100
REC_K = 7
SEED = 1234


def _poly(p):
    q = Polygon(np.asarray(p, np.float64).reshape(-1, 2))
    return q if q.is_valid else q.buffer(0)


def read_rgb(root, p):
    im = cv2.imread(os.path.join(root, p), cv2.IMREAD_COLOR)
    assert im is not None, p
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def to_canvas(img_u8, s, nh, nw, size):
    img = cv2.resize(img_u8, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:min(nh, size), :min(nw, size)] = img[:size, :size]
    return canvas


class CropDataset(Dataset):
    def __init__(self, manifest, root):
        self.items = [json.loads(l) for l in open(manifest, encoding='utf-8') if l.strip()]
        self.root = root

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = cv2.imread(os.path.join(self.root, it['crop']), cv2.IMREAD_COLOR)
        assert img is not None, it['crop']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return t, it['text']


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def det_forward(det, x, stride, dev):
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


def rectify(x_t, centers, heights, hw, dev):
    C = torch.from_numpy(np.asarray(centers, np.float32)).to(dev)
    Hh = torch.from_numpy(np.asarray(heights, np.float32)).to(dev)
    return rectify_strip(x_t, C, Hh, hw)


@torch.no_grad()
def evaluate(det, rec, ann, size, stride, hw, dev, seed):
    """Returns dict(e2e_F, rec_acc_lit, rec_acc_dark) on the bright val set."""
    rows = [json.loads(l) for l in open(ann, encoding='utf-8') if l.strip()]
    det.eval(); rec.eval()
    ev = SpottingEvaluator()
    acc = {'lit': [0, 0], 'dark': [0, 0]}   # [correct, total]

    for i, row in enumerate(rows):
        img = read_rgb('data', row['image'])
        H, W = img.shape[:2]
        s = size / max(H, W)
        nh, nw = int(round(H * s)), int(round(W * s))
        lit_u8 = to_canvas(img, s, nh, nw, size)
        rng = np.random.default_rng((seed, i))
        dark_raw, _ = darken(img, rng, mode='physical')
        dark_u8 = to_canvas(dark_raw, s, nh, nw, size)

        # --- e2e: det on lit canvas -> rec on predicted strips -> match vs GT (original px)
        x_t = torch.from_numpy(np.ascontiguousarray(lit_u8)).permute(2, 0, 1).float()[None].to(dev) / 255.0
        dets = det_forward(det, x_t, stride, dev)
        preds = []
        for d in dets:
            strip = rectify(x_t, d['centers'], d['heights'], hw, dev)
            txt = rec.predict(strip)[0]
            preds.append(dict(poly=np.asarray(d['poly']) / s, text=txt, score=d['score']))
        ev.add(preds, row['instances'], image_bin=row['meta'].get('illum_bin'))

        # --- rec acc on GT crops (lit + dark)
        for cond, u8 in (('lit', lit_u8), ('dark', dark_u8)):
            x_t = torch.from_numpy(np.ascontiguousarray(u8)).permute(2, 0, 1).float()[None].to(dev) / 255.0
            for ins in row['instances']:
                if bool(ins.get('ignore')) or ins.get('text', '') in ('###', ''):
                    continue
                poly = np.asarray(ins['poly'], np.float32) * s
                c, hh = centerline_and_height(poly, k=REC_K)
                strip = rectify(x_t, c, hh, hw, dev)
                txt = rec.predict(strip)[0]
                acc[cond][1] += 1
                acc[cond][0] += int(norm_text(txt) == norm_text(ins['text']))

    sm = ev.summary()['all']
    return dict(e2e_F=sm['e2e_F'],
                rec_acc_lit=acc['lit'][0] / max(acc['lit'][1], 1),
                rec_acc_dark=acc['dark'][0] / max(acc['dark'][1], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=MANIFEST)
    ap.add_argument('--crop-root', default=CROP_ROOT)
    ap.add_argument('--val-ann', default=VAL_ANN)
    ap.add_argument('--det-ckpt', default=DET_CKPT)
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--name', default='microsoft/trocr-base-printed')
    ap.add_argument('--lora-r', type=int, default=16)
    ap.add_argument('--lora-alpha', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--amp', type=int, default=1)
    ap.add_argument('--val-every', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--init-rec', default=None, help='checkpoint with a "rec" key (e.g. s4_rec_v4/best.pt)')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    cv2.setNumThreads(1)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = load_config('configs/base.yaml', {'data': {}, 'train': {'amp': bool(a.amp)}})
    size, stride = cfg.data.size, 4
    hw = tuple(cfg.model.crop_hw)

    rec = TrOCRRecognizer(name=a.name, lora_r=a.lora_r, lora_alpha=a.lora_alpha, freeze=False).to(dev)
    if a.init_rec:
        sd = torch.load(a.init_rec, map_location=dev)
        rec.load_state_dict(sd['rec'] if 'rec' in sd else sd)
        print(f'init rec from {a.init_rec}', flush=True)

    det = SeeReadDetector(ch=cfg.model.det_ch, pretrained=False, ctx_dim=cfg.model.ctx_dim).to(dev).eval()
    sd_d = torch.load(a.det_ckpt, map_location=dev)
    det.load_state_dict(sd_d['det'])
    for p in det.parameters():
        p.requires_grad = False
    del sd_d

    ds = CropDataset(a.manifest, a.crop_root)
    dl = DataLoader(ds, a.batch_size, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True)
    params = [p for p in rec.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(dl))
    scaler = torch.amp.GradScaler(enabled=bool(a.amp) and dev == 'cuda')
    os.makedirs(a.out_dir, exist_ok=True)
    print(f'train crops={len(ds)} batches/epoch={len(dl)} trainable={sum(p.numel() for p in params)}', flush=True)

    def save(path):
        torch.save(dict(rec=rec.state_dict(), cfg=dict(name=a.name, lora_r=a.lora_r, lora_alpha=a.lora_alpha)), path)

    best_f1 = -1.0
    for ep in range(a.epochs):
        rec.train()
        for it, (crops, texts) in enumerate(dl):
            crops = crops.to(dev)
            with torch.autocast(dev, enabled=bool(a.amp)):
                loss = rec.forward_loss(crops, texts)
            if loss is None or not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            if it % 20 == 0:
                print(f'ep{ep} it{it} loss {loss.item():.4f}', flush=True)
        save(os.path.join(a.out_dir, 'last.pt'))
        if (ep + 1) % a.val_every == 0:
            m = evaluate(det, rec, a.val_ann, size, stride, hw, dev, SEED)
            print(f'[val] ep{ep} e2e_F={m["e2e_F"]:.4f} rec_acc lit={m["rec_acc_lit"]:.4f} '
                  f'dark={m["rec_acc_dark"]:.4f}', flush=True)
            if m['e2e_F'] > best_f1:
                best_f1 = m['e2e_F']
                save(os.path.join(a.out_dir, 'best.pt'))
                print(f'[val] ep{ep} new best e2e_F={best_f1:.4f} -> best.pt', flush=True)
    print(f'done, best e2e_F={best_f1:.4f}', flush=True)


if __name__ == '__main__':
    main()
