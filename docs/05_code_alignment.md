# 代码层对齐修复记录（docs/03 → 实施分册）

> 日期：2026-10-14。本文记录依据 [docs/03_issue_fix_alignment.md](03_issue_fix_alignment.md) 实施**并经用户确认**的 9 项"确认级代码 bug"最小修复；部署层变更见 docs/04，本文只覆盖训练/模型代码层。原则：最小 diff，保留原结构与命名；每项附一个"此前会失败、修复后通过"的回归用例（全部扩展进 [tests/test_smoke.py](../tests/test_smoke.py)，未新建测试文件）。
>
> **验证环境**：`seeread:latest`（pytorch 2.7.1-cu128 基础镜像），容器内执行 `python tests/test_smoke.py`，**12/12 PASS**（原 5 项 + 新增 7 项，其中 P1-01/P1-04 合并为一个用例）。镜像已重建，内置源码与宿主机一致。

## 1. 已修复（9 项）

### P0-05 `rec` 阶段固定 GT 高度，几何课程仅在 `joint` 推进
- **位置**：[seeread/models/seeread.py:46](../seeread/models/seeread.py#L46)
- **变更**：高度课程条件由 `random.random() < pred_geom_prob` 改为 `stage == "joint" and random.random() < pred_geom_prob`。`rec` 阶段无论 `pred_geom_prob` 取值一律使用 GT 高度；`det` 阶段不建 crop，行为不变。
- **回归用例**：`test_p005_pred_heights_only_in_joint`——以 `sample_map` 调用计数作为"使用预测高度"的观测量：`stage=rec`、`pred_geom_prob=1.0` 时调用 0 次；`stage=joint` 时 >0。此前 rec 阶段同样会被调用，断言失败。
- **docs/03 验收对照**：GT/预测高度使用率与书面阶段设定一致（以调用计数衡量）✓。

### P0-08 空编码（OOV）跳过样本并计数告警，禁止伪类别标签
- **位置**：[seeread/models/recognizer.py:56-69](../seeread/models/recognizer.py#L56-L69)、[seeread/models/seeread.py:60-62](../seeread/models/seeread.py#L60-L62)
- **变更**：`CRNNRecognizer.forward_loss` 先编码后过滤：保留非空编码的实例，全部为空时返回 `None`；`n_skip` 累计跳过文本数，每越过 100 的倍数打印一次告警。`forward_train` 收到 `None` 时不添加 `rec` 损失。原 `or [1]`（空编码 → 类别 1 = 字符 "0"）删除。**未**构建全量字符表——等真实转写冻结后按仅训练集建表（见第 3 节）。
- **回归用例**：`test_p008_empty_encoding_skipped_not_class1`——`encode("停止")==[]`；混合 batch `["停止","stop"]` 正常产生损失且 `n_skip==1`；全 OOV batch 返回 `None` 且 `n_skip==3`。此前全 OOV 会返回针对类别 1 的损失，`is None` 断言失败。
- **验收对照**：纯中文不会生成 "0" 标签 ✓；OOV 计数可通过 `n_skip` 与告警日志复核 ✓；逐语种字符级 OOV 统计与字符表冻结属 P0-02/P1-08 数据层工作，未在本项范围。

### P0-09 混合配对/非配对 batch 不再整体丢失 CIC
- **位置**：[seeread/data/datasets.py:106-110](../seeread/data/datasets.py#L106-L110)、[seeread/models/seeread.py:42-68](../seeread/models/seeread.py#L42-L68)、[tools/train.py:68-74](../tools/train.py#L68-L74)
- **变更**：`collate` 只要 batch 内存在配对样本即输出 `lit`（未配对槽位补零图占位）与 `lit_mask`（逐样本布尔配对标志）；`forward_train` 只为 `lit_mask` 为真的样本提取亮图 crop，并记录配对 crop 的索引 `idx_paired`，CIC 损失仅在这些 crop 的增强特征与对应亮图特征间计算（未配对样本贡献为 0，不伪造亮图）。训练日志每 20 步输出 `pair=x/y`，每个 epoch 末输出 `ncic/nb`（应用 CIC 的 batch 比例）。全配对 batch 的数值行为与修复前完全一致。
- **回归用例**：`test_p009_mixed_batch_cic_applies_to_paired_only`——[有配对, 无配对] batch：`lit_mask==[True, False]`；patch `cic_loss` 捕获特征 batch 大小 `==1`（仅配对样本的 1 个实例）且 `"cic" in L`。此前 `collate` 用 `all()` 判定，一个无配对样本使全 batch 丢 `lit`，无 CIC，断言失败。
- **验收对照**：混合 batch 有配对产生 CIC、无配对不参与 ✓；配对样本数与 CIC 有效 batch 比例有日志 ✓（逐实例计数未单列，登记第 4 节）。

### P0-11 合成暗图按暗化后像素重算实例分档
- **位置**：[tools/make_synth_dark.py:20-29](../tools/make_synth_dark.py#L20-L29)
- **变更**：对每个合成等级，用暗化输出图的实例外接框均值亮度按 `illum_bin` 重算实例级 `bin`（与 `tools/illum_stats.py` 同一 bbox 规则）；源亮图实例 `bin`（若存在）移入独立字段 `bin_source`，不再参与评测优先级。
- **回归用例**：`test_p011_synth_dark_recomputes_instance_bin`——源实例带 `bin="lit"` 跑全脚本：每行 `bin_source=="lit"`、逐行以输出像素复核 `bin` 一致、extreme 行 `bin != "lit"`。此前 `instances` 原样复制，extreme 行仍是 `bin="lit"` 且像素复核必然失败。
- **验收对照**：输出实例分档依据新图重新计算、抽样像素复核一致 ✓；源档位独立字段 ✓。

### P0-12 亮图在 val/test 原样加载，不在线暗化
- **位置**：[seeread/data/datasets.py:64](../seeread/data/datasets.py#L64)
- **变更**：`illum_kind == "lit"` 的在线暗化加 `self.train` 条件；`train=False` 时亮图原样加载（无 `lit` 配对键，评测不需要）。训练行为不变。离线合成集（`illum_kind="synth"`）本就不触发该分支。
- **回归用例**：`test_p012_lit_image_untouched_at_eval`——eval 样本像素与"原图按同一 resize 规则手工构造的参考张量"逐位相等且无 `lit` 键；train 样本含 `lit` 且均值显著降低。此前 eval 也被暗化，逐位相等断言失败。
- **验收对照**：亮图 val/test 加载与原图一致 ✓；训练同源在线合成 ✓；离线 synth 不二次暗化（未改动）✓。

### P1-01 空损失 batch 跳过更新并计数，调度只按真实步数推进
- **位置**：[tools/train.py:58-62](../tools/train.py#L58-L62)、[tools/train.py:75-76](../tools/train.py#L75-L76)
- **变更**：`forward_train` 返回空 dict 时（如 `rec` 阶段整 batch 无有效实例）打印 `SKIP empty-loss batch`、`n_skip` 计数并 `continue`——不执行 backward/optimizer/scheduler，`step` 不前进（cosine 调度与 `pred_geom_prob` 课程均只按真实更新计）。训练结束输出总 skip 统计。
- **回归用例**：`test_p101_p104_rec_stage_skip_and_frozen_bn`（子进程部分）——`stage=rec`、4 行全 `###` 标注：此前 `sum({})` 得 int 0，`backward()` 崩溃非零退出；现 returncode 0、`SKIP` ×2、日志含 "2 batches skipped"、`last.pt` 正常保存。
- **验收对照**：空有效文本 batch 不崩溃 ✓；步数与 LR 调度只按真实更新计 ✓；日志有 skip 统计 ✓。

### P1-04 参数冻结模块每 epoch 保持 `eval()`，BN running stats 真正冻结
- **位置**：[tools/train.py:23-28](../tools/train.py#L23-L28)、[tools/train.py:53](../tools/train.py#L53)
- **变更**：新增 `set_frozen_eval(model, stage)`（`rec` → detector eval；`det` → enhancer+recognizer eval），每个 epoch `model.train()` 之后调用。`requires_grad=False` 只挡梯度，不挡 BN running stats；此前 `rec` 阶段检测器 BN 统计量会随前向漂移。
- **回归用例**：`test_p101_p104_rec_stage_skip_and_frozen_bn`（进程内部分）——`set_frozen_eval(m, "rec")` 后 `det.training` 为 False、enh/rec 仍为 True；`rec` 阶段前向后检测器 BN `running_mean/running_var` 逐位不变。此前无此机制，BN 统计量必然变化。
- **验收对照**：rec 前后检测器 BN 缓冲区变化与设定一致（不变）✓。

### P2-03 `_base_` 以 config 文件所在目录解析，支持绝对路径
- **位置**：[seeread/config.py:16-23](../seeread/config.py#L16-L23)
- **变更**：相对 `_base_` 一律按当前 config 文件所在目录（`os.path.dirname(os.path.abspath(path))`）解析；绝对 `_base_` 直接使用；删除原 `path.count("/")` 分支与内联 `__import__("os")`。
- **回归用例**：`test_p203_base_resolved_relative_to_config_dir`——从不同 cwd 以绝对路径加载、裸文件名加载、绝对 `_base_` 三种方式合并结果一致。
- **验收对照与诚实说明**：docs/03 指出原实现在"config 路径无 `/`（典型为 Windows 反斜杠路径）"时按 cwd 误解析；在 POSIX 正斜杠调用下旧实现与新实现等价（docs/03 亦注明"正斜杠调用通常不触发"）。本项属稳健性硬化，用例同时固化了验收要求的不变式。

### P2-06 SCM 的 L_sr/L_ss 归入 λ_det，与 docs/01 §3 损失公式一致
- **位置**：[seeread/models/seeread.py:38](../seeread/models/seeread.py#L38)
- **变更**：`det_sr`/`det_ss` 权重由写死的 `1.0` 改为 `w.det`，即 `L = λ_det·(L_seg + L_H + L_dir + L_sr + L_ss) + ...`。默认 `w.det=1.0` 数值行为不变。`losses.py` 内 `w_scm=(1,1)` 内部权重保持；L_sr 未用 ignore 权重掩码属 docs/03 P2-06 后半句的待决设计，登记第 3 节。
- **回归用例**：`test_p206_scm_sr_ss_scaled_by_det_weight`——固定种子下 `w.det=1.0` 与 `2.0` 两次前向：所有 `det_*` 项比值为 2.0、`rec`/`enh` 为 1.0。此前 `det_sr`/`det_ss` 比值为 1.0，断言失败。
- **验收对照**：非 1 的 `w.det` 人工用例与书面定义对得上 ✓。

### P0-13 无转写实例（"###"）仍参与检测几何学习，don't-care 仅作用于 rec 与评测
- **位置**：[seeread/data/datasets.py:81-89](../seeread/data/datasets.py#L81-L89)
- **发现**：LATeD 训练探针（det 阶段，1 epoch）暴露 `det_h=0.000、det_dir=0.000` 恒零；进一步用真实样本核查，24/24 样本 `text/center/height` 目标图全空——因 LATeD 官方标注仅有多边形无转写，转换器按协议标 `text="###"/ignore=true`，而数据集把 don't-care 实例同时从检测正样本中剔除，900 张图退化为"全图无文本"的空训练。
- **变更**：拆分 det/rec 的 don't-care 语义——检测几何目标由"是否完整落在画布内"决定（`ignore=not inside`），协议 don't-care（`"###"`/`ignore=true`）仅排除 rec crop 采集；评测侧维持 docs/02 §1 的吸收语义（不计召回、命中不罚）。LATeD 多边形是确定性文本（基准 GT），检测器必须学会检出；TotalText 的 `#` 区域同理变为"检出但不转写"的正样本，符合 spotting 惯例。
- **回归用例**：`test_p013_hash_text_still_learns_det_geometry`——含 `text="stop"` 与 `text="###"` 各一实例的样本：`text/center/height` 目标非零（"###" 是检测正样本）、rec 实例仅 1 个且为 "stop"。
- **验收对照**：修复后 det 探针 `det_h≈0.26、det_dir≈0.16` 非零且随训练下降 ✓；5 epoch 收敛 4.02→2.22 无异常 ✓。
- **2026-10-14 晚修订（P0-14，见下）**：本条中"评测侧维持吸收语义"与"TotalText `#` 作检测正样本"两点已被三层对齐方案取代：评测侧按 ignore 标志拆分，训练侧 TotalText `#`（ignore=true）不再进入检测目标。本条的训练侧核心结论（LATeD `###` 必须学检测几何）保持有效。

### P0-14 don't-care 语义三层对齐：det 几何 ignore 与"无转写"彻底拆分（2026-10-14 正式训练前）
- **发现**：正式训练前的探针 ckpt 全矩阵基线评测暴露 LATeD/CTW 测试 `n_gt=0`、所有检测指标恒 0——这两个基准只发布多边形、转换器统一写 `text="###", ignore=true`，评测器 `dont = ignore or text∈{###,""}` 把全部 GT 吸收，主基准失去分母。P0-13 只在**训练侧**拆了 det/rec 语义，评测侧与标注侧仍把"无转写的确定文本"与"协议 don't-care"混为一谈；且 P0-13 让训练侧无视了显式 ignore（TotalText 官方 `#` 也被当成检测正样本，偏离 TotalText 官方协议）。
- **决策**（grilling 会话，用户拍板"三层全对齐"）：
  1. **标注**：LATeD（train 8201 / val 914 / test 4808 实例）与 CTW1500（test 3075 + 离线暗图 ann 9225 实例）翻转为 `ignore=false, text="###"`——多边形是检测正样本，仅无转写；TotalText 官方 `#` 保持 `ignore=true` 不动。转换源码 [tools/convert_annotations.py](../tools/convert_annotations.py) 同步修改，旧文件翻转前备份于 `/tmp/annotations_backup_pre_p014`。
  2. **评测器** [seeread/metrics/spotting.py](../seeread/metrics/spotting.py)：`ignore=true` 才吸收匹配预测（不入 n_pred/n_gt）；`###`/空文本但 `ignore=false` 计入检测分母与 tp_det、计入漏检，但不进 e2e（新增 `n_gt_e2e`/`n_match_e2e` 计数，`e2e_recall`/`e2e_P`/`frac_misread` 全部以"有转写 GT"为分母，纯无转写基准 e2e 字段为 0 且不解读）。
  3. **训练侧** [seeread/data/datasets.py:87](../seeread/data/datasets.py#L87)：检测目标改为 `ignore=(not inside) or bool(ins.get("ignore"))`——显式 don't-care 不做检测正样本；`###`+`ignore=false` 仍是检测正样本、不进 rec。对 S1（SynthText 全 false）与 S1.5（LATeD 翻转后 false）无数值影响；S3 起 TotalText `#` 不再进入检测目标，回归官方协议。
- **回归用例**：`test_p014_eval_semantics_split`（无转写命中→det_F=1/e2e=0；显式 ignore→吸收；混合批分母正确）、`test_p014_ignore_true_opt_out_of_det_training`（ignore=true 无检测目标，###+false 有检测目标无 rec crop）。
- **验收对照**：修复后重跑探针基线，LATeD test `n_gt=4808`、CTW test `n_gt=3075`、dark-CTW `n_gt=9225`，指标可计算 ✓。

### 训练增强：val 加载、峰值 best.pt、eval --set（2026-10-14，正式训练前）
- [tools/train.py](../tools/train.py)：新增 `train.val_every`（默认 0=关闭）。开启后按 N epoch 在 `data.val_ann` 上跑完整 infer+`SpottingEvaluator`，按亮图 val 的 `e2e_F` 存 `best.pt`（峰值 epoch 权重），每 epoch 仍存 `last.pt`。原 P0-03 登记项由此部分关闭（best 选择已实现；早停未做）。
- TotalText 官方 train 1254 图按 seed=0 随机切 100 val / 1154 train（官方 300 test 不接触），冻结于 `data/annotations/totaltext_splits.json`，产物 `totaltext_train_main.jsonl`/`totaltext_val.jsonl`。
- [tools/eval.py](../tools/eval.py)：新增 `--set`（复用 train.py 的 parse_set），用于覆盖 `data.root`（离线暗图 ann 自带目录前缀，实际保留默认 root=data 即可）。
- 回归用例：`test_val_hook_saves_best_pt`（2 epoch 迷你子进程任务，2 次 val、保存 best.pt/last.pt）。
- 字符表冻结：默认 36 字符（0-9+a-z）对 SynthText 可编码率 99.59%、TotalText train 99.97%/test 100%、零非 ASCII；丢弃字符仅标点/空格，与评测 `norm_text` 一致。CTW1500 无转写，仅参与检测评测。

## 2. 验证结果

- 命令：`docker compose run --rm -v $PWD:/workspace --entrypoint bash seeread -c "CUDA_VISIBLE_DEVICES= python tests/test_smoke.py"`（挂载最新源码，CPU）。
- 结果：**17/17 PASS**（2026-10-14，含 P0-14 与 val 钩子）——原基线 5 项 + P0-05/08/09/11/12、P1-01/04、P2-03/06 共 7 项 + P0-13 + val 钩子 + P0-14 两项。
- 新增测试含两个子进程用例（make_synth_dark.py、train.py 端到端），在 CPU 上运行（测试内显式 `CUDA_VISIBLE_DEVICES=""`），不占用 GPU。
- 备注：正式训练/评测容器均以 `-v $PWD:/workspace` 挂载运行，镜像内 COPY 源码待本轮全部收尾后统一重建。

## 3. 仅登记、未修改（与 grilling 会话确认范围一致）

| 编号 | 一句话理由 |
|---|---|
| P0-01/02 | 数据与转写交付缺口：需先冻结 split 与真实转写，非代码修复 |
| P0-03 | 验证集进训练：功能新增（val loader/best 选择/早停），超出最小修复范围 |
| P0-04 | 三阶段实验编排：脚本策略问题，需实验计划先行 |
| P0-06/07 | 测试集隔离与 ESTR 协议冲突：协议决策，需改写文档与脚本策略 |
| P0-10 | GT/预测中心线分布落差：需 val 量化实验，属研究设计 |
| P1-02/03 | 实例截断 16、CTC 时步限制：先要真实数据分布统计再定方案 |
| P1-05 | resume 空实现：需完整断点协议（opt/sched/scaler/RNG），单独立项 |
| P1-06 | 随机性管理：需先定义复现边界（同环境/跨 worker/断点） |
| P1-07 | ignore 区匹配规则：必须先对照所选官方 benchmark 协议，不能凭直觉改 |
| P1-08 | 转写/训练/评测三套规范化：等字符表与转写冻结 |
| P1-09/10 | 亮度阈值校准、实例亮度定义：需真实数据统计与预注册 |
| P1-11/12/13 | baseline 矩阵、消融设计、GPU/真实路径覆盖：实验执行层 |
| P2-01/02/04/05 | warmup、rec_k、竖排方向、TSR 差异：需消融/可视化证据，docs/03 明确警告勿直接改 |
| P2-06（后半） | L_sr 是否屏蔽 don't-care：待决设计，需实验 |
| P2-07/08/09/10 | checkpoint 元数据、配对几何配准、多尺度/速度报告、TrOCR 验证：可追溯性与验证缺口 |

## 4. 遗留小项（本阶段实现中知悉、暂缓）

- P0-09 验收中的"逐实例配对计数"未在日志单列（现有 `pair=x/y` 为样本级、`ncic/nb` 为 batch 级）。
- P0-08 的字符级/语种级 OOV 统计依赖字符表冻结（P0-02），当前只有 `n_skip` 计数与告警。
- `train.py` 的 `p_pred` 在非 joint 阶段仍被计算（模型侧已忽略），未删以保持最小 diff。
