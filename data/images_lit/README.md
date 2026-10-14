# data/images_lit/ — 亮图/配对亮帧目录占位（待放置）

JSONL 中 `lit` 字段的根目录（相对 `data/`）。用途：
1. `meta.illum_kind=lit` → 训练时在线暗化，暗/亮天然配对供 CIC 损失（`seeread/losses.py`）；
2. 真实配对亮帧（`lit` 字段非空、`illum_kind=real`）→ CIC 与上限分析。

## 应放置的内容

| 内容 | 来源 | 用途 |
|---|---|---|
| 亮源图（SynthText / MLT17） | `raw/` 对应目录复制或链接 | 预训练在线暗化 + CIC |
| 亮源图（CTW1500 / Total-Text 原图） | `raw/` | `make_synth_dark.py` 输出 JSONL 的 `lit` 指向 |
| 手电/补光亮帧（ESTR-v2） | 三脚架自采，光源不入画面（docs/02 §3） | 真实暗/亮配对 → CIC |

## 注意

- 配对帧先过 `python tools/align_pairs.py` QC：ECC 偏移 >2px 的剔除或重配准；标注只在亮帧上做一次，多边形复用到暗帧。
- 该工具目前仅做 QC 输出 shift/FLAG，不做真正的几何配准（docs/03 P2-08），非平移错位需人工复核。
- 隐私：入库前人脸/车牌模糊（docs/02 §3.5）。
- 混合配对/非配对样本同 batch 时 CIC 会整体丢失（docs/03 P0-09，`collate` 要求 batch 内全部样本含 `lit`）。
