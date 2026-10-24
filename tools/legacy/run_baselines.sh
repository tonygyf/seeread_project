#!/usr/bin/env bash
# Reproducible experiment matrix (fill in the detector/recognizer commands of YOUR third-party baselines).
set -e
TEST=data/annotations/latedspot_test.jsonl
# B0  oracle polygons + recognizer (isolates recognition) -> implement tools/eval_oracle.py using GT polys + seeread.metrics
# B1  cascade: {identity,gamma,clahe,ssr,zerodce,sci,ruas,srdace,gdp} -> detector -> recognizer (pretrained / fine-tuned)
# B2  end-to-end spotters fine-tuned on dark data: DeepSolo / TESTR / ESTextSpotter (official repos), evaluate with seeread.metrics.spotting
# Ours + ablations (3 seeds each)
for seed in 0 1 2; do
  for cfg in seeread_joint ablation_no_ctx ablation_no_cic ablation_no_scm; do
    python tools/legacy/train.py --config configs/legacy/$cfg.yaml --set train.seed=$seed train.out_dir=results/${cfg}_s$seed
    python tools/legacy/eval.py  --config configs/legacy/$cfg.yaml --ckpt results/${cfg}_s$seed/last.pt --ann $TEST --out results/${cfg}_s$seed/test.json
  done
done
