# 01 项目设计与目录导航

SeeRead 研究低光场景文本 spotting：检测整图文字，裁剪后增强与识别，最终输出位置和转写。当前主线是独立复现两篇论文，再验证组合收益。

## 1. 阅读顺序

| 文档 | 内容 |
|---|---|
| [02 数据与评测协议](02_dataset_protocol.md) | 数据位置、JSONL、ignore、配对与评测口径 |
| [03 历史实验与结论](03_history_and_findings.md) | S1–S4 的有效证据、已修复问题与已纠正判断 |
| [04 v5 论文对齐与实验状态](04_v5_alignment.md) | 当前唯一的 v5 状态文档：划分、论文目标、实际结果和缺项 |

## 2. 研究顺序

1. **Reading 识别基线**：公开 TrOCR 与 Self-DACE 起点，核验原论文的数据、损失、训练和解码。
2. **Seeing 检测基线**：公开骨干起点，按 SynthText → MLT → 目标集训练，独立验收检测。
3. **固定接口组合**：保持识别器、裁剪和匹配规则一致，比较固定检测器与 Seeing 检测器。
4. **单因素消融**：逐项检验增强、识别适配、裁剪与连接设计。
5. **项目方法**：有受控收益后再增加自研模块、删除冗余模块。

组合主目标是暗光开发集 spotting F1，同时监测同图亮光退化；固定 GT crop CER、检测 F1 和预测框识别用于归因。真实低光 spotting 结论需要人工复核的场景转写。

## 3. 仓库结构

```text
.agents/                 项目 skills 与调用说明
configs/
  v5/                    当前独立复现与训练阶段配置
  legacy/                历史 S1–S4 与旧消融配置
  base.yaml              历史链路共用默认值
seeread/
  v5/                    当前 Reading / Seeing 实现
  models/                历史 SeeRead 模型与兼容识别器
  data/                  公用标注、几何和暗化工具
  metrics/               内部检测、识别与 spotting 指标
  baselines/             既有增强基线注册
tools/                   训练、准备数据、诊断与导出入口
  v5_*.py                当前实验入口
  data/                  暗化、亮度、配对和转写工具
  legacy/                旧训练、评测、诊断与退役 watchdog
  convert_annotations.py 通用标注转换；冻结协议引用
  convert_new/           历史转换器；冻结协议仍绑定其原路径
tests/
  v5/                    当前模型、数据、训练与队列回归
  legacy/                历史链路回归
  test_smoke.py          Docker 默认 CPU 冒烟入口
experiments/autoresearch/  实验执行规则、队列计划、锁与结果索引
data/                    原始数据、冻结标注与图像
results/
  v5/                    当前实验、队列、锁定测试与源码快照
  archive/legacy/        v1–v4、旧识别训练与裁剪数据
  archive/diagnostics/   历史归因、审查、已停止队列与工程探针
  archive/documents/     整理前文档与代码压缩备份
  logs/                  历史独立启动日志
docs/                    本目录四篇主文档
paper/                   论文草稿与参考文献
```

## 4. 使用约定

- 当前开发入口为 `tools/v5_*.py`；旧工具和配置用于历史复核，不用旧 S3/S4 权重初始化正式 v5。
- 配方以配置文件为准，数据身份以 provenance 为准，实验结论以完整验证、锁定测试和逐样本记录为准。
- 训练中的源码、配置与冻结数据保持不变；改配方使用新目录，exact resume 必须满足原指纹。
- `paper/main.tex` 中的 `XX.X` 和 TODO 是占位，不能作为实测结果；方法贡献随实验成立后再确定。
- Karpathy/autoresearch 提供固定目标、预算和 keep/discard 循环；实际训练使用本项目入口。

## 5. 历史路径迁移

2026-10-21整理将旧入口、配置、测试及未运行的结果目录归类，迁移表见 [layout_20261021.tsv](../results/archive/layout_20261021.tsv)。日常代码和文档已使用新位置；历史checkpoint、JSON与旧日志内部保存的原路径不改写，可按迁移表查找。所有权重保留，目录同盘重命名，没有重新压缩大结果或创建兼容软链接。

`seeread/`的既有module、当前`tools/v5_*.py`、`configs/v5/`、正式锁、当前队列与正式实验输出保持原位和内容，保证当前运行、后续resume与论文验收指纹一致。旧批量脚本已归入legacy，不作为当前实验计划。
