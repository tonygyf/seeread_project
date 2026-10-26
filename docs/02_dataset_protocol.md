# 02 数据与评测协议

本文维护通用格式和数据位置；v5 的固定划分、论文对应关系及进度统一见 [04](04_v5_alignment.md)。

## 1. JSONL 与 ignore

场景标注一行一张图，路径相对 `data.root`：

```json
{"image":"images/0001.jpg","lit":"images_lit/0001.jpg","instances":[{"poly":[[10,20],[90,20],[90,40],[10,40]],"text":"STOP","lang":"en","ignore":false,"bin":"dark"}],"meta":{"illum_kind":"real","scene":"street-001"}}
```

| 字段 | 含义 |
|---|---|
| `poly` | 原图 XY 多边形，按轮廓顺序保存；保留源顶点与坐标，不将任意形状默默替换为四边形 |
| `text` | 原始转写；`###` 或空字符串表示无可用转写 |
| `ignore=false` 且无转写 | 仍可用于检测监督和检测计分，不计算识别或 spotting |
| `ignore=true` | 保留源 don't-care；从检测有效监督与分母中排除，预测过滤依具体官方协议 |
| `lit` | 可选的同场景配对亮图；不是识别真值 |
| `meta.illum_kind` | `real` 真实低光、`synth` 固定合成、`lit` 亮光源图 |

词图识别有独立 manifest，由 `tools/v5_prepare_reading.py` 生成；其字段、配对摘要和来源由对应 provenance 记录，不强行转换成场景 spotting 标注。

旧 `SeeReadDataset` 仅在训练时对标为 `lit` 的源图在线暗化，val/test 保留原图。v5 Seeing 使用独立数据集，按明确配置训练增强，不沿用这项隐式暗化。离线合成数据不能重复暗化。

## 2. 数据位置与处理入口

| 数据 | 原始位置 | 用途或处理入口 |
|---|---|---|
| LATeD | `data/raw/LATeD/` | 真实低光检测；源多边形没有转写，不能直接评 spotting |
| SynthText | `data/raw/SynthText/SynthText/` | 检测预训练；完整 manifest 在 `data/annotations/synthtext_train.jsonl` |
| MLT17 | `data/raw/MLT17/` | 11个ZIP已校验并解压；`tools/v5_prepare_detection.py --prepare-mlt17`建立独立清单，细节见04 |
| CTW1500 | `data/raw/CTW1500/ctw1500/` | 常光检测；现有镜像为 MMOCR 重打包，须保留源 ignore |
| Total-Text | `data/raw/TotalText/` | 多边形与转写，可做组合 spotting 开发 |
| ICDAR2015 | `data/raw/ICDAR2015/` | 已有图像和标注归档；额外用于训练须登记为配方变更 |
| 作者 LSTR / ESTR60 | `data/raw/ESTR/` | 历史目录名；LSTR 是合成低光词图，ESTR60 是真实低光词图 |
| 派生暗图 | `data/dark_totaltext_test/`、`data/dark_ctw1500_test/` 等 | 历史固定种子暗化；使用时必须绑定具体 manifest 与协议 |

转换标注用 `tools/convert_annotations.py`；当前冻结协议用 `tools/v5_prepare_reading.py`、`tools/v5_prepare_detection.py`。图像不要求平铺进 `images/`，以 manifest 的实际相对路径为准。

v5融合开发集固定为 `data/annotations/v5/fusion/totaltext_dev_{bright,dark}.jsonl`，同100张原dev图、修正标注821实例/94ignore；暗图位于 `data/derived/v5/fusion/totaltext_dev_dark/`。`tools/data/prepare_v5_fusion_dev.py` 固定physical/dark与seed1234，协议和逐图SHA见同目录 `totaltext_dev_protocol.json`；历史 `dark_totaltext_test` 是官方test的暗化，不作为开发集。

光照分析入口为 `tools/data/illum_stats.py`。既有阈值 `0.04/0.08/0.15` 是未校准的默认分档，不能直接当作真实低光标准；改边界或合成强度须在评测冻结前完成。

## 3. 配对采集

ESTR-v2 是旧设计中的自采方案，尚不是已交付数据，也不同于作者 ESTR60。后续需要时：固定相机、曝光与对焦，保存暗/补光帧及设备参数，按场景去重。`tools/data/align_pairs.py` 只做 ECC 质量检查，不自动修正标注或完成几何配准；亮帧多边形复用到暗帧前须核验对齐。测试场景及其配对亮帧均不进入训练。

## 4. 转写与评测

人工转写保留原始字符，自动 OCR 只作草稿；双人校对，不一致项复核。无法辨认时明确标记，源数据已有 ignore 不得因缺转写而清除。

LATeD首批固定200张训练＋100张开发，先审核20张试标图。`tools/data/prefill_lated.py`生成原图上下文、行裁剪及独立任务包；`finalize_lated.py`导入带复核人与包摘要的JSON到sidecar，保留稳定实例ID、图像摘要、语言和审核状态，原检测GT不改。`pending`不是真值；重新导入pending会撤销旧确认。原`lang=en`为占位值，必须重新判定。首个MVP只计人工确认的拉丁字符行识别，检测保留全部文字；匹配到非拉丁/不可辨认GT的预测从OCR精度分母排除，未匹配预测仍计OCR假阳性。正式500张test不进入首批标注与开发。

| 任务 | 指标 | 解释边界 |
|---|---|---|
| 检测 | P / R / F1 | 不要求转写；benchmark 必须使用锁定的官方匹配协议 |
| GT 词图/裁剪识别 | CER、词准确率 | 隔离识别器；不能等同预测检测框上的识别 |
| 匹配预测框识别 | CER、词准确率及匹配样本数 | 检测器变化会改变样本总体，不能直接推断识别器退化 |
| 场景 spotting | 位置匹配且文本正确的 P / R / F1 | 必须同时有多边形和可靠转写 |

大小写、标点、空格、字符单位和 CER 聚合方式按实验提前固定，不能事后挑选归一化方式。内部 `SpottingEvaluator` 用于开发诊断，不自动等价于 CTW / Total-Text 作者评估器。

## 5. 划分与冻结

- train 用于更新，dev 用于选择 checkpoint 与超参，test 仅在候选及协议锁定后验收。
- 固定 seed、图像列表、manifest 和来源摘要；原图、暗化图、配对图、重复图按同组隔离。
- 记录场景隔离与内容去重的实际范围；无法恢复场景来源时不宣称完整场景隔离。
- 发现冻结数据问题时另立候选版本，保留原协议，不在运行中覆盖。
- 官方测试与 ESTR60 不用于挑后续模块；真实 LATeD 转写未完成时不报告其识别/spotting 改善。
