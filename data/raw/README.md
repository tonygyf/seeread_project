# data/raw/ — 原始数据集下载占位（尚未下载，计划在服务器上获取）

官方原样数据放这里，只读不改；协议化 JSONL 转换结果写入 `data/annotations/`。
JSONL 协议见 `docs/02_dataset_protocol.md`，研究设计见 `docs/01_research_design.md` §4。

## 下载清单（子目录名为约定名）

| 子目录 | 数据集 | 规模 | 获取方式 | 说明 |
|---|---|---|---|---|
| `lated/` | LATeD（Seeing 论文） | 1,500 图（train 1000 / test 500），line-level 多边形 | 论文官方发布；"code and dataset will be released"，**先向老师确认可用性与署名许可**（docs/01 §0.3） | **无转写**，LATeD-Spot 需补转写（13,923 实例，docs/03 P0-02） |
| `synthtext/` | SynthText | ~80 万实例 | 官方发布 | 检测器预训练；`illum_kind=lit` 在线暗化，自带 CIC 亮配对 |
| `mlt17/` | MLT17 | 官方 train | 官方（需注册） | 同上 |
| `ctw1500/` | CTW1500 | test 200 图 | 官方 GitHub | 亮图基线评测 + Dark-CTW1500 合成源 |
| `totaltext/` | Total-Text | test 300 图 | 官方 GitHub | 同上 |
| `lstr/` | LSTR（Reading in the Dark） | 11,273 train / 6,588 test 裁剪 | 论文 HuggingFace 发布 | 识别器训练与对齐实验 |
| `estr/` | ESTR（Reading 原版） | 60 裁剪 | 论文 HuggingFace 发布 | 仅识别测试 |

## 注意

- ESTR-v2 为**自采数据集**（非下载），图像与配对亮帧分别放 `data/images/` 与 `data/images_lit/`，采集规范见 `docs/02_dataset_protocol.md` §3。
- 各数据集 → 协议 JSONL 的转换器目前缺失（docs/03 P0-11 / docs/01 §6 "需要你补"），落地后命令统一记录在 `data/annotations/README.md`。
- 下载后核对许可：公开数据集引用原文；自采数据发布前做人脸/车牌模糊（docs/02 §3.5）。
