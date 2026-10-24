"""Usage: python tools/legacy/eval.py --config configs/legacy/seeread_joint.yaml --ckpt results/x/last.pt --ann data/annotations/latedspot_test.jsonl --out results/x/test.json
Reports det/E2E P-R-F, CER, and illumination-stratified recall (bins from per-instance GT 'bin' or image meta)."""
import argparse, json, os, sys
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from seeread.config import load_config
from seeread.data.datasets import SeeReadDataset, collate
from seeread.metrics.spotting import SpottingEvaluator
from seeread.models.recognizer import build_recognizer
from seeread.models.seeread import SeeRead
from tools.legacy.train import parse_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ann", required=True); ap.add_argument("--out", required=True); ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--set", nargs="*")
    a = ap.parse_args(); cfg = load_config(a.config, parse_set(a.set))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SeeRead(build_recognizer(cfg.rec), cfg).to(dev)
    sd = torch.load(a.ckpt, map_location=dev)
    model.det.load_state_dict(sd["det"]); model.enh.load_state_dict(sd["enh"]); model.rec.load_state_dict(sd["rec"])
    ds = SeeReadDataset(cfg.data.root, a.ann, train=False, size=cfg.data.size)
    dl = DataLoader(ds, a.bs, shuffle=False, num_workers=cfg.data.workers, collate_fn=collate)
    ev = SpottingEvaluator()
    for batch in dl:
        res = model.infer(batch["image"].to(dev))
        for b, preds in enumerate(res):
            s, (ox, oy) = batch["scale"][b], batch["offset"][b]
            for p in preds:                                    # back to original image coordinates
                p["poly"] = (np.asarray(p["poly"]) - [ox, oy]) / s
            ev.add(preds, batch["gt_orig"][b], image_bin=batch["meta"][b].get("illum_bin"))
    out = ev.summary(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2); print(json.dumps(out["all"], indent=2))


if __name__ == "__main__":
    main()
