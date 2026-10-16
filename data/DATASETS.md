# 数据集清单（位置 / 状态 / 来源）

> 维护日期：2026-10-14。目录规范：`data/raw/<数据集名>/`，转换后的冻结 jsonl → `data/annotations/`，平铺图像 → `data/images/`、`data/images_lit/`（协议见 docs/02_dataset_protocol.md）。
> 状态含义：`✓ 完成` / `⏳ 下载中` / `✗ 阻塞（需人工）`。

| 数据集 | 存储位置 | 获取方式 | 来源 | 状态 | 大小/规模 |
|---|---|---|---|---|---|
| LATeD | `data/raw/LATeD/` | 本地复制（免下载） | `/home/gyf/DarkText-private/data/LATeD` | ✓ 完成 | 9.0GB；image/image_bright/label 各 1500、test 500，train/test list md5 校验一致 |
| SynthText | `data/raw/SynthText/SynthText/`（zip 40,717,671,794B 已校验，解压 78GB） | 容器下载 | HF `Spatial1ntelligence/synthtext-recovered-mirror`（官方 SynthText.zip 镜像） | ✓ 完成 | 858,750 图 / 7,266,866 词实例；已转 `annotations/synthtext_train.jsonl`（1.1GB，词框-文本错配 0），lit 源训练时在线暗化，不做 illum_stats |
| MLT17 | `data/raw/MLT17/`（空） | — | 官方 `rrc.cvc.uab.es`（需注册）；HF 五轮搜索无完整镜像 | ✗ 阻塞 | ICDAR2017 MLT 全量 ~12GB、10 语种；det 预训练按备注②仅用 SynthText |
| CTW1500 | `data/raw/CTW1500/ctw1500/` | 容器下载 | HF `HB16888/CTW1500`（mmocr 重打包，含 imgs+标注） | ✓ 完成 | test 500 图 / 3075 实例（无转写，检测-only），`annotations/ctw1500_test.jsonl`；派生暗图 `data/dark_ctw1500_test/` 1500 张三档 |
| Total-Text | `data/raw/TotalText/` | 容器下载 | HF `yunusserhat/Total-Text-Dataset`（Images 1555 + groundtruth_pixel/textregion + txt_format 多边形标注） | ✓ 完成 | train 1254 图/10,530 实例（官方缺 img61）、test 300/2543（328 don't-care）；seed0 切 1154 train+100 val（`totaltext_splits.json`），派生暗图 `data/dark_totaltext_test/` 900 张三档 |
| ICDAR2015 | `data/raw/ICDAR2015/` | 容器下载 | HF `dlxjj/ICDAR2015`（官方 4 个 zip：train/test 图像 + GT） | ⏳ 下载中 | ~700MB |
| ~~ESTR~~ 实为 LSTR 合成 | `data/raw/ESTR/` | 容器下载 | HF `lumimusta/Low-light_Scene_Text_Dataset`。**2026-10-15 核实：非 Reading 原版 ESTR**。构成 = SynthText 词裁剪亮图 + 合成暗化（train 11,273 / test 6,588，如 word_1="Genaxis Theatre" 即 SynthText 样本）+ 仅 60 张真实暗拍（`low_light_test_real/`）。与 Reading 论文 LSTR 合成规模一致 | ✓ 已核伪 | 大集均合成；真实仅 60 裁剪 |
| ESTR-v2 | （自采） | 人工采集 | docs/02 §3 配对采集规范 | N/A 自采 | 目标 300–500 场景 |
| Dark-CTW1500 / Dark-TotalText / Dark-SynthText / Dark-MLT17 | （派生） | `tools/make_synth_dark.py` / 在线暗化 | 由上述原始数据生成，非下载 | N/A 派生 | 固定种子三档 |

## 备注

1. **MLT17 阻塞原因**：官方下载需在 rrc.cvc.uab.es 注册账号；HF/Kaggle 公开镜像经多轮检索未找到完整版（仅有过滤子集 `m-biriuchinskii/ICDAR2017-filtered-*`，不符合 Dark-MLT17 预训练全量需求）。替代方案：① 注册后手动下载放入 `data/raw/MLT17/`；② 先用 SynthText 单独做检测预训练（docs/01 §5 三阶段中 stage-det 的数据可裁剪）。
2. **CTW1500 格式差异**：所用镜像为 mmocr 重打包（非官方 Drive 原始 zip），转换器写 `ctw1500 → jsonl` 时按 mmocr instances 格式解析。
3. 下载统一经 `docker compose run --rm seeread`（seeread:latest 容器）完成，满足"全部容器化下载"要求；LATeD 为本地复制特例。
