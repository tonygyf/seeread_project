"""Unified dataset for SeeRead.

Annotation format (one JSON per line, see docs/02_dataset_protocol.md):
{"image": "images/0001.jpg", "lit": "images_lit/0001.jpg" | null,
 "instances": [{"poly": [[x,y],...], "text": "STOP", "lang": "en", "ignore": false}],
 "meta": {"illum_y": 0.05, "illum_bin": "dark", "scene": "street", "illum_kind": "real|synth|lit"}}
`illum_kind == "lit"` means a well-lit source image that will be darkened ONLINE (train only).
"""
from __future__ import annotations
import json, os
import numpy as np, cv2, torch
from torch.utils.data import Dataset
from .geometry import make_det_targets, centerline_and_height
from .lowlight_synth import darken

cv2.setNumThreads(1)          # cap OpenCV TBB pools (each DataLoader worker inherits this); keeps PID/thread budget bounded


def _read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class SeeReadDataset(Dataset):
    def __init__(self, root, ann_file, train=True, size=640, synth_mode="physical",
                 max_instances=16, rec_k=7, seed=0, min_text_h=4):
        self.root, self.train, self.size = root, train, size
        self.synth_mode, self.max_inst, self.rec_k, self.min_h = synth_mode, max_instances, rec_k, min_text_h
        with open(ann_file, encoding="utf-8") as f:
            self.items = [json.loads(l) for l in f if l.strip()]
        self.seed = seed

    def __len__(self):
        return len(self.items)

    # ---- geometry -------------------------------------------------------------------------------
    def _params(self, h, w, rng):
        s = self.size / max(h, w)
        if self.train:
            s *= rng.uniform(0.7, 1.3)
        nh, nw = int(round(h * s)), int(round(w * s))
        if self.train:
            ox = int(rng.integers(min(0, self.size - nw), max(0, self.size - nw) + 1))
            oy = int(rng.integers(min(0, self.size - nh), max(0, self.size - nh) + 1))
        else:
            ox = oy = 0
        return s, nh, nw, ox, oy

    def _apply(self, img, p):
        s, nh, nw, ox, oy = p
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.size, self.size, 3), np.uint8)
        x0, y0, x1, y1 = max(ox, 0), max(oy, 0), min(ox + nw, self.size), min(oy + nh, self.size)
        if x1 > x0 and y1 > y0:
            canvas[y0:y1, x0:x1] = img[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        return canvas

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, int(torch.initial_seed() % 2**31)) if self.train else (self.seed, i))
        it = self.items[i]
        img = _read_rgb(os.path.join(self.root, it["image"]))
        meta = dict(it.get("meta", {}))
        lit = None
        if meta.get("illum_kind") == "lit" and self.train:   # online darkening (train only); eval keeps the lit image as-is
            lit = img
            img, m = darken(lit, rng, mode=self.synth_mode)
            meta.update(m)
        elif it.get("lit"):                          # real paired capture
            lit = _read_rgb(os.path.join(self.root, it["lit"]))
        H, W = img.shape[:2]
        p = self._params(H, W, rng)
        s, nh, nw, ox, oy = p
        img_t = self._apply(img, p)
        lit_t = self._apply(cv2.resize(lit, (W, H)), p) if lit is not None else None
        inst_det, inst_rec = [], []
        for ins in it["instances"]:
            poly = np.asarray(ins["poly"], np.float32) * s + np.array([ox, oy], np.float32)
            x0, y0 = poly.min(0); x1, y1 = poly.max(0)
            if x1 < 0 or y1 < 0 or x0 > self.size or y0 > self.size:
                continue
            inside = x0 >= 0 and y0 >= 0 and x1 <= self.size and y1 <= self.size
            # Det geometry is learned for in-canvas text, including textless ("###") GT like LATeD/CTW.
            # Explicit ignore=true (protocol don't-care, e.g. TotalText '#') opts out of det AND rec;
            # "###"/"" with ignore=false is a positive det target but excluded from rec (P0-13/P0-14 split).
            inst_det.append(dict(poly=poly, ignore=(not inside) or bool(ins.get("ignore"))))
            textless = bool(ins.get("ignore")) or ins.get("text", "") in ("###", "")
            if not textless:
                c, hh = centerline_and_height(poly, self.rec_k)
                if hh.mean() >= self.min_h:
                    inst_rec.append(dict(centers=c.astype(np.float32), heights=hh.astype(np.float32), text=ins["text"]))
        tg = make_det_targets(self.size, self.size, inst_det)
        out = dict(image=torch.from_numpy(img_t).permute(2, 0, 1).float() / 255.0,
                   targets={k: torch.from_numpy(v) for k, v in tg.items()},
                   instances=inst_rec[: self.max_inst], meta=meta, index=i,
                   scale=s, offset=(ox, oy), orig_size=(H, W),
                   gt_orig=it["instances"])
        if lit_t is not None:
            out["lit"] = torch.from_numpy(lit_t).permute(2, 0, 1).float() / 255.0
        return out


def collate(batch):
    out = dict(image=torch.stack([b["image"] for b in batch]),
               targets={k: torch.stack([b["targets"][k] for b in batch]) for k in batch[0]["targets"]},
               instances=[b["instances"] for b in batch], meta=[b["meta"] for b in batch],
               scale=[b["scale"] for b in batch], offset=[b["offset"] for b in batch],
               orig_size=[b["orig_size"] for b in batch], gt_orig=[b["gt_orig"] for b in batch],
               index=[b["index"] for b in batch])
    lit = [b.get("lit") for b in batch]
    if any(v is not None for v in lit):          # mixed batch: pad unpaired slots; lit_mask keeps CIC to paired samples
        out["lit"] = torch.stack([v if v is not None else torch.zeros_like(bb["image"]) for v, bb in zip(lit, batch)])
        out["lit_mask"] = torch.tensor([v is not None for v in lit])
    return out
