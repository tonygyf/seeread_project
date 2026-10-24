# 03 历史实验与结论

本文合并旧设计、问题清单、部署记录和 S1–S4 审查，保留影响后续决策的证据。当前 v5 统一见 [04](04_v5_alignment.md)。旧权重与原始结果仍保留在 `results/`。

## 1. 历史阶段

| 阶段 | 已有工作 | 后续使用方式 |
|---|---|---|
| S1 / S1.5 | SynthText 检测预训练及目标数据适配，检测损失与混合数据结果有响应 | 说明链路可学习；缺完整论文预训练与协议对齐，不作正式复现 |
| S2 | CRNN 识别训练 | 作为固定输入归因参照 |
| S3 v2/v3 | 检测、增强、CRNN 与 CIC/几何课程的联合实验 | 属项目扩展；v3 的“TrOCR+LoRA”旧描述已纠正，实际为 CRNN+CTC |
| S4 v4 | 换用 TrOCR，诊断预测裁剪问题 | 证明几何接口会显著影响 spotting；不作为公共起点的正式基线 |
| DarkText 移植 | VGG FPN 检测推理与兼容权重加载 | 本地源码缺 SCM，不能当成 Seeing 的 ResNet50+DSF 完整作者实现 |

## 2. 已确认的修复

| 问题 | 已落实行为 | 位置 |
|---|---|---|
| 无转写被误当全图无文本 | `###` 与显式 ignore 分开；无转写仍学检测，源 ignore 保留 | `datasets.py`、`convert_annotations.py`、`metrics/spotting.py` |
| 训练/评测暗化混用 | 旧加载器 val/test 保留亮图；离线暗图不再二次暗化 | `seeread/data/datasets.py` |
| 冻结参数后 BN 仍更新 | 冻结模块保持 eval；joint 冻结识别器也覆盖，保留到输入的梯度 | `tools/legacy/train.py`、`models/recognizer.py` |
| 部分裁切词配完整标签 | 识别只收完整落在画布内的词，检测按其 ignore 规则处理 | `seeread/data/datasets.py` |
| rec 阶段误用预测高度 | 高度课程只在 joint 推进 | `seeread/models/seeread.py` |
| OOV/空损失与混合配对 | 跳过空编码及无有效监督更新；配对样本用 mask 处理，不整批丢弃 CIC | `recognizer.py`、`datasets.py`、`tools/legacy/train.py` |
| AMP 跳步被计作更新 | scheduler、样本位置与课程仅随有效更新推进 | 旧训练器与后续独立训练器 |
| CLI checkpoint 路径仅写入报告 | 实际 loader 使用指定权重 | `tools/legacy/eval_darktext_det.py` |
| 配置路径及损失归属不一致 | `_base_` 相对配置文件解析；SCM 计入检测损失权重 | `seeread/config.py`、`models/seeread.py` |

这些修复有对应回归。历史记录中“待做”不自动代表当前仍未实现，当前能力以源码、测试和运行记录为准。

## 3. 决定性实验

| 对照 | 结果 | 支持的判断 |
|---|---|---|
| S3-v3 frozen 与 S2 状态比较 | 识别参数相同，四层 BN 的 batch 计数均增加60000 | 冻结 BN 缺陷确定，但单独修复不足以解释全部低分 |
| 相同679词的 GT crop、恒等增强 | S3-v3 FT best/last 亮词准确率24.74%/36.52% | 不能把预测框上的识别下降直接解释为识别器遗忘 |
| S4 同一匹配词 GT / 预测裁剪 | 亮词80.46%→14.50%，暗词60.44%→8.66% | 检测/裁剪接口是重要瓶颈；不与 CRNN 记录混算 |
| 同一旧 S4 best、固定 v2 detector、100图/726GT，只换中心线路径 | 亮 spotting F1 5.47%→14.47%；固定暗化2.92%→19.21% | 几何修复有明确收益；19.21%是合成暗光历史结果 |
| 旧短程 S4 续训 | 最佳暗 spotting F1约19.38% | 起始权重与上行不同，不能把差值当净训练收益 |
| 固定检测器、CRNN起点，两臂各100更新 | 增强+识别+CIC与恒等+独立识别均未展示可靠 E2E 收益 | 短程、单seed、完整配方对比，不能判定某个模块普遍无效 |
| 旧检测诊断 | train/test F1约28.45%/28.29%；提高分辨率后约33% | 存在低召回和虚警，未排除欠训练、监督或后处理问题 |

证据入口：`results/archive/diagnostics/s3_audit/`、`results/archive/diagnostics/attribution/d1_geometry.json`、`results/archive/diagnostics/diag_det/`、`results/archive/diagnostics/v5_preflight/s4_old_best_paired_v2.json`。不同记录的检测器、GT集合和光照种子不同，不能拼成同一学习曲线。

## 4. 已纠正的判断与保留事项

| 早期说法 | 合并后的结论 |
|---|---|
| “S3 总会把识别器练坏” | 固定 GT 输入的证据不支持统一归因；需分开检测、裁剪、增强和识别 |
| “检测约28%是架构硬天花板” | train≈test不等于架构上限，仍需核验配方与实现 |
| “IAE/CSR/CIC已经证明无效，应全部删除” | 旧实验预算和控制不足；正式复现先不用自研模块，后续受控消融再决定 |
| “权重 shape 可加载就证明移植完全正确” | 只能证明参数结构兼容，不能排除前向、坐标和预处理差异 |
| “LSTR是合成，所以作者ESTR也是伪真实” | LSTR合成词图与ESTR60真实词图是不同子集 |
| “无转写意味着 ignore=false” | 无转写与源 don't-care 独立，必须保留明确的源 ignore |

旧 CRNN 实例数截断、CTC 时间步、竖排方向、亮度阈值、几何课程与官方评测差异，均应按具体实验检查；早期清单不作为要求当前批量添加功能的待办。LATeD 转写、自采 ESTR-v2 和外部 spotting 基线是未完成计划，不作为已交付成果。

历史部署采用过不同镜像、资源限额及源码挂载方式；实跑曾为 PyTorch 2.7.1+cu128，不能只凭 Dockerfile 的基础镜像声明重建历史环境。当前运行约束见 [04](04_v5_alignment.md)。

## 5. 原文档去向

| 原文档 | 归并位置 |
|---|---|
| `01_research_design.md` 的早期设想 | 01 当前研究顺序；本篇阶段与判断纠偏 |
| `02_dataset_protocol.md`、`data/DATASETS.md` | 02 通用协议与数据位置；04 当前冻结划分 |
| `03_issue_fix_alignment.md`、`05_code_alignment.md` | 本篇已确认修复与保留事项 |
| `04_deployment_changelog.md` | 本篇历史部署结论；04 当前资源约束 |
| `07_review_and_replan.md`、`08_proposed_architecture.md`、`09_s3_audit_and_paper_alignment.md` | 本篇历史证据与判断纠偏 |
| `10_v5_design.md`、`v5_reproduction_targets.md`、`v5_seeing_alignment.md`、`v5_detection_evaluation.md` | 04 的数据、实现、指标与验收表 |

整理前原文及相关导航完整压缩在 [单个归档包](../results/archive/documents/docs_before_consolidation_20261021_184006.tar.gz)。归档保留原目录名，含详细旧问题编号、逐文件哈希和历史配置说明；日常只维护本目录四篇文档。训练源码中保留的旧文档注释可按上表定位，避免因改注释破坏运行中的源码指纹。
