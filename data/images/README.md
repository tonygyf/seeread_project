# data/images/ — 暗图目录占位（待放置）

JSONL 中 `image` 字段的根目录：`seeread/data/datasets.py` 按 `os.path.join(data.root, "images/...")` 读图，`data.root=configs/base.yaml` 的 `data/`。

## 应放置的内容

| 内容 | 来源 | 命名 |
|---|---|---|
| LATeD-Spot 真实暗图 | `raw/lated/` 解包后保持官方相对路径（如 `images/0001.jpg`） | 官方原名 |
| ESTR-v2 自采夜景暗帧 | 后期采集（协议 docs/02 §3：三脚架、记录 EXIF） | `<scene_id>_<frame>.jpg` |
| 离线合成暗图 | `tools/make_synth_dark.py` 输出到 `data/dark_<name>/`，其 JSONL 的 `image` 指向该处 | `<idx>_<level>.png` |

## 注意

- 亮源图**不要**放本目录：`illum_kind=lit` 的条目无论训练/评测都会被 `datasets.py` 在线暗化（docs/03 P0-12），亮图原样评测需用其他加载方式。
- `meta.illum_kind=synth` 的离线合成图带固定种子，不得二次暗化或重新生成。
