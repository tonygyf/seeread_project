# 04 v5 论文对齐与实验状态

更新：2026-10-25 15:59（北京时间）。**CTW目标训练已完成250轮/22500更新，内部dev best F1 55.0292%（epoch217），末轮52.8061%；GPU3已恢复Read裁剪域两轮训练。GPU2 LATeD约87/250轮，按原预算继续。用户选择优先检验Read多轮：现有两轮及全部连接对照完成后，保持同学习率/数据/模块接到5轮，满足开发门槛才到10轮；独立Read和See基线保留。官方评测与终测后置。**

## 1. 识别：完整测试结果与论文差距

以下为固定官方LSTR test6588词图、同raw CER定义；每个模型按完整dev所选best锁定后评一次，未使用测试选择新模块。ESTR60继续封存。

| 基线 | 论文CER% | 本地test CER% | 差值/百分点 | test词准确率% | 完整dev best CER% |
|---|---:|---:|---:|---:|---:|
| 冻结TrOCR，R0str | 53.16 | **53.5778** | **+0.4178** | 37.8719 | 53.7597 |
| LoRA TrOCR，R2str | 52.84 | **49.4897** | **−3.3503** | 40.2095 | 39.9654，step37400 |
| Self-DACE+LoRA，R4str | 50.02 | **49.5191** | **−0.5009** | 40.3157 | 39.7753，step36200 |

数字接近目标：冻结与联合均在3个百分点内，LoRA低于对应论文值3.3503个百分点。两训练支均50740更新/20轮，公共STR起点、seed0、batch4和train10146/dev1127相同；完整train11273 refit尚未做。上述是数值验收结果，实现的登记假设仍见§3。来源：[Reading Table2/3](https://arxiv.org/html/2604.23685v1)。

联合增强的增量收益尚未确立：同预算dev CER比纯LoRA低0.1901个百分点，按source_group配对bootstrap的95%区间为[−1.1740,+0.7784]个百分点；区间跨0。test比纯LoRA高0.0294个百分点，仅多10次字符编辑；不根据这次test选择删模块。[开发比较](../results/v5/diagnostics/R2str_R4str_equal_budget_dev.json)。

### 同一R4权重的开发集移除诊断

固定同一R4 best的TrOCR、增强权重和1127 dev；只改变推理路径，不重训、不运行test。full结果逐样本预测与原step36200验证记录逐字节一致。

| 推理路径 | dev CER% | 相对full/百分点 | 结论 |
|---|---:|---:|---|
| 原light+denoise→TrOCR | 39.7753 | 0 | 固定对照 |
| 去denoise，保留light | 40.3457 | +0.5704 | 当前权重下变差；配对95%区间[+0.0349,+1.1546] |
| 绕过整个增强器→同TrOCR | 39.8963 | +0.1210 | 差异小；配对95%区间[−0.3807,+0.6400] |

**本轮不删除模块。** 单删denoise有损，整个增强器的增量收益较小；这是推理移除诊断，不能代替独立重训消融。结果为[移除记录](../results/v5/reading/R4str_enhancer_removal_dev/run.json)与[配对比较](../results/v5/reading/R4str_enhancer_removal_dev/paired_comparison.json)。旧printed/R4b结果保留作历史，不覆盖。

## 2. 检测F1、训练进度与MLT

以下均为SynthText预训练开发结果，无MLT和目标训练。CTW/LATeD各完整dev100张，640×640、同冻结decoder，内部score-greedy IoU≥.5；不能作为正式test或spotting F1。检查点不同的成绩分开列出。

| 检查点 | 开发集 | Precision% | Recall% | F1% |
|---|---|---:|---:|---:|
| 健康step85600，修复前固定快照 | LATeD | **43.4505** | **14.8796** | **22.1679** |
| 同step85600 | CTW1500 | **19.9747** | **22.4113** | **21.1230** |
| 修复后step85875，第1完整epoch | CTW1500 | **17.9518** | **21.1348** | **19.4137** |

[step85600 LATeD](../results/v5/diagnostics/detection_stage_20261023/lated_dev/run.json) / [CTW](../results/v5/diagnostics/detection_stage_20261023/ctw_dev/run.json)；首轮验证见新SynthText目录的`dev_epoch_0001_attempt_001.jsonl`。此前step57200的LATeD/CTW F1为23.5880/20.0712%；阶段波动不能证明持续提升，也不能据此判断MLT效果。当前Recall仍低，先完成预登记训练，再评估目标阶段。

| 任务 | 当前进度 | 后续 |
|---|---|---|
| SynthText | 已完成2轮、171750更新 | 健康末权重已用于MLT |
| MLT17 | **已完成100轮、72000更新**；train7200/dev1795 | LATeD和CTW共享同一完整父权重 |
| LATeD / GPU2 | 原链41轮后溢出；修复后健康3900已恢复，当前超过48轮；原最佳内部dev F1 **51.3750%**（epoch38） | `LATeD_geometry_fast_20261025`，batch10、无短程上限，续满250轮 |
| CTW / GPU3 | **完成250/250轮、22500更新**；内部dev best F1 **55.0292%**（217轮），末轮52.8061% | 已交接GPU3给Read两轮与连接对照，后接5→10轮受控探查 |
| Read / connector | R2/R4独立训练和LSTR验收完成；新增完整MLT、LATeD阶段连接诊断 | 固定增强器、仅LoRA做裁剪域适配；不覆盖独立R4 |

MLT末轮dev内部P/R/F1为54.5029/40.0639/46.1811%，最佳F1为51.9490%（epoch82）。它已完整用于两个目标检测分支；不同检测checkpoint的域迁移结果见§4，不把跨阶段差值称为MLT因果收益。MLT未加入独立Reading训练。

双卡沿用每卡独立容器和串行任务。GPU3已完成原CTW训练并恢复Read；GPU2已恢复LATeD长程并优先续满250轮；Read短程及复核完成，追加长程中断前的健康last保留，由GPU3在CTW后接续。共享预训练只读，未加入DDP；正式评测仍后置。

Seeing论文test F1目标LATeD/CTW/TotalText为67.1/86.2/88.5%，见[Table2/3](https://arxiv.org/html/2404.08965v3)。当前阶段dev F1不替代这三项正式成绩。

### 数值修复与本轮代码补齐

此前center/text分类头FP16溢出已分别改为FP32；本次step85672的首个非有限值位于最后一层PM普通卷积：输入有限，FP16卷积输出Inf，继而BN输出NaN；同批全FP32正常。三层PM共用普通Conv/BN分支现均以FP32计算，归一化后转回AMP dtype；网络结构、损失、几何头及其他AMP不变。坏批修复后loss1.58709，全FP32对照1.58633；未跳过样本。[坏批重放](../results/v5/diagnostics/synthtext_step85672_pm_green.json)。

从健康step85600显式迁移，逐项核验model/optimizer/scheduler/scaler/RNG/sampler/loader状态完全保留；父检查点和旧结果保持。新闭包恢复后已重放原坏批、完成首轮验证，健康检查点全部参数有限且源码摘要吻合。[迁移及恢复核验](../results/v5/detection/SynthText_pm_fp32_20261023/numeric_migration.json)。这属于数值实现迁移，不称跨源码exact resume。

相关检测回归26项及2个subtests通过；GT导出与数据准备14项通过，两路代码复核无遗留发现。pilot元数据中“MLT不可用”的过时说明已修正为“该短程pilot不执行预训练”。本轮未修改锁定Reading闭包、数据协议或增加自研模块。

### 2026-10-24：DSConv精度与恢复

最新中断位于step153502，训练已达SynthText约89%；原坏批逐层重放确认`pm.2.snake_v.operator.dsc_conv_y`有限输入经FP16卷积输出Inf，BF16/FP32正常。修正只将DSConv反变形卷积及GroupNorm设为FP32，保留官方offset/坐标/插值算子和参数路径；训练时重算无随机、无BN的DSConv中间激活，保持batch10和原显存限制。未改变网络结构、数据、损失和优化器。真实坏批完整更新loss1.35816、梯度范数6.59766，全参数有限，峰值allocated20.53GiB/reserved23.10GiB。[坏批诊断](../results/v5/diagnostics/synthtext_failed153502_20261024.json) / [反向通过](../results/v5/diagnostics/synthtext_failed153502_backward_green_20261024.json)。

从健康153400而非异常快照显式迁移，model/optimizer/scheduler/scaler/RNG/sampler/loader/state八项与父记录逐项核验一致；旧结果不覆盖。新主链`SynthText_dsconv_fp32_20261024→MLT17_dsconv_fp32→LATeD_dsconv_fp32`已恢复并通过原故障位置，GPU3在完整MLT后训练`CTW1500_dsconv_fp32`。[迁移核验](../results/v5/detection/SynthText_dsconv_fp32_20261024/numeric_migration.json)。DSConv及训练恢复回归19项、连接回归19项通过；Reading源码未改，Seeing活动源码已锁定。

## 3. 数据和实现对齐

**当前主线必需数据已齐，本轮不再增加数据集下载。** LSTR/ESTR、SynthText、MLT17、LATeD、CTW1500和TotalText均已在本地，足够继续识别、检测、合成暗光spotting与模块消融。MSRA-TD500属于Seeing论文的额外常光检测基准，不纳入本轮验收，也不作为继续实验的前置条件。

下面的“完整train refit未做”表示用已有训练数据重新训练，不是缺文件：LSTR已有11273张，当前10146训练/1127开发；LATeD官方train已有1000张，当前900训练/100开发。CTW的问题是已有标注版本与官方评测核对，不是缺少整套训练图像。LATeD文字转写属于对现有图像补标注，待方法验证后再做；当前检测不受影响。

| 环节 | 当前事实 | 尚需完成 |
|---|---|---|
| LSTR / ESTR | 作者固定发布：train11273/test6588；开发隔离10146/1127；ESTR60封存 | 完整train refit尚未做；作者实际TrOCR checkpoint未公开核实 |
| TrOCR | 官方 `microsoft/trocr-base-str@26c151f56a7983e2cfcf05d55dc14f2f5f70ce64`，权重1.336GB整SHA匹配官方LFS；训练/生成prefix均2 | 该模型在IC13/IC15/IIIT5K/SVT训练集微调；不能视为作者实际checkpoint身份已确认 |
| SRDACE | 固定Self-DACE old_version stage2公共light/denoise起点；LoRA r16/alpha32；外层损失L_LLE+1e5 L_edge+100 L_OCR | resize后增强、内部LLE系数/Sobel/LoRA targets属于登记重实现假设 |
| RLLIE | 论文IBL/PRT识别增强分支 | **本地尚未实现**；当前R4为Self-DACE，不把位置实验称为RLLIE消融 |
| Seeing结构 | ResNet50 ImageNetV2、256ch、DSF/官方DSConv pro、SCM/TSR；算子前向与梯度核验通过 | SCM布局、注意力轴、双轴融合及几何是重实现假设；完整作者代码未取得 |
| Seeing测试尺寸 | 论文§5.1：LATeD/CTW640×640，TotalText640×1200，MSRA640×960；本次LATeD/CTW开发尺寸对应 | 正式TotalText验收另设1200宽协议；当前640方图融合开发不冒充正式TotalText基线 |
| Seeing配方 | SynthText2→MLT100→目标250，Adam1e-4/batch10/每100轮×0.1；补齐scale/crop/color/noise/flip/rotate | 增强概率、scale[.75,1.5]、stretch/仿射顺序及目标250时长为登记解释 |
| MLT17 | 全11ZIP校验、9000图全解码/GT配对；train7200/dev1795；5验证副本隔离、4单帧GIF保留源并转无损PNG | 已完成100/100轮；压缩包校验后已安全清理，解压数据保留 |
| SynthText / LATeD | SynthText858750路径非空、清单摘要匹配；LATeD900/100/500，无转写 | LATeD全train1000 refit未做；暂无真实暗光spotting真值 |
| CTW1500 | MMOCR train1000/7703实例保留坐标；冻结900/100。镜像test3075实例/387ignore；另存作者检测候选test3068实例 | 训练/测试图像已齐、可继续实验；两源67图测试几何不同，待锁定标注版本与官方评测；不要求另下载整套训练图 |
| Total-Text | 新train-only候选1255图/10579实例/1294ignore；按作者第6列orientation判定ignore，修复6处旧候选错误；10个单点ignore单列 | 候选未采用，冻结训练/测试及现融合均未改；当前融合100图无真实ignore差异，1个单点过滤已登记 |
| MSRA-TD500 | 未下载；当前方案不使用 | 可选扩展基准，含自身训练/测试划分；仅覆盖论文该列时需要，本轮不下载 |
| 官方检测评测 | 已缓存固定版本CTW/TotalText作者评测源并核SHA；GT导出允许逐坐标相同的作者退化轮廓，预测仍严格校验 | 官方执行适配尚未接通，LATeD官方协议仍需落实；当前内部F1不替代官方成绩 |

作者词图来源：[固定LSTR/ESTR发布](https://huggingface.co/datasets/lumimusta/Low-light_Scene_Text_Dataset/tree/619e3c1d3bb27135da785b77d3c414a946e37000)。MLT独立协议为 `data/annotations/v5/detection/mlt17_train.reference.json`；旧冻结provenance保留。

TotalText新候选及逐项核验位于`data/annotations/v5/candidates/totaltext_20261023/conversion.json`。作者MAT候选全量ZIP校验通过、5个公开示例逐字节匹配，但全量上游身份仍未证实，继续标为`source_candidate`；不据此覆盖冻结测试标注。评测源缓存位于`~/.local/share/seeread-tools/author-evaluation/`，未增加数据集下载。

## 4. 融合顺序：先定位收益，再决定删改

Reading当前路径是“词图增强→TrOCR”；Seeing本身是检测器。**RLLIE的IBL/PRT是为识别设计的图像增强，不是检测头。** 放到Seeing之前属于新的跨任务位置实验；不能预先判定有益或有害。

| 阶段 | 每次只改变的因素 | 判定 |
|---|---|---|
| A：固定接口 | 原图→同一Seeing→裁剪→同一TrOCR | 组合对照起点；固定权重、阈值、裁剪和匹配 |
| B：裁剪后增强 | 完全复用A的检测框；只将crop→固定Self-DACE→同一TrOCR | 比A的暗光spotting F1、GT crop CER，归因识别输入变化 |
| C：检测前增强 | 固定Self-DACE增强整图→同一Seeing；仍从原图裁剪给同一TrOCR | 比A的检测F1和spotting F1，归因检测输入变化 |
| D：删改现有模块 | 先测明确有害因素；候选去denoise或去edge loss，二者不同时改 | 必须同起点/预算验证；先跑完等预算基线，再定位增强的收益或损失来源 |
| E：连接模块 | 已确定较好链路后，只加一个crop/连接修正 | 暗光F1有收益且亮光退化可接受才保留，再考虑下一个模块 |

A/B/C共用相同OCR、检测器和增强器权重。R2训练权重与R4训练权重的整方法比较另报，不能代替位置消融。保留暗/亮spotting F1、检测F1、GT crop raw/lower-alnum CER、失败裁剪数和延迟；token/收敛模块后置，不先堆模块。

固定开发数据已生成：**同100图明暗配对，每支821实例/94ignore**，恢复旧val遗漏6实例；暗图为physical/dark、固定seed1234，原图和暗图与现TT train1152/test300字节SHA交集均0。[协议](../data/annotations/v5/fusion/totaltext_dev_protocol.json)。不使用ESTR或测试集选模块。正式融合验收采用完整目标权重；开发对照允许使用固定中间快照。`tools/v5_eval_spotting.py` 与单份 `configs/v5/fusion_placement_dev.yaml` 已实现；CPU预检通过100对标注和200张图像摘要，当前待填两份固定基线权重。识别器及采样运行异常直接上抛，几何非法裁剪仍纳入CER分母；9项CPU测试通过。

### Connector：实例、裁剪与Read适配（2026-10-23）

单图输出多个实例，每个实例保持自己的`instance_id / poly / score / text / crop_error`。检测依text与center预测的连通性分组，随后重建轮廓；“距离近”不等于同一个词。当前解码器有粘连合并、重建碎片只保留最大轮廓的局限，暂保持See基线，先量化分组问题。CTW行级与TotalText词级的粒度不同，不能直接把整行当作词crop验收；连接实验需明确实例粒度。

本轮只改connector：裁剪累积到rec_chunk立即识别并释放，避免先将整图所有词crop存入显存；跨批次、非法框夹杂、不同区域文字的对应关系已验证。输入Read保持RGB[0,1]及384×384，crop→Self-DACE→TrOCR与现有ReadingPipeline一致；裁剪域与LSTR词图域的差异仍需实际CER验证。

新增可独立消融的`sampling: polygon_boundary`：保留轮廓截面上下边界及顶点折点，再展开给Read。原`local_normal`仍为C0默认，C1配置`configs/v5/fusion_connector_boundary_dev.yaml`仅改变采样；弯曲V形及薄文字带反例表明旧采样可越出轮廓，新方案在这些反例内保持采样点位于轮廓内。两种方案仍要求PCA方向单截面，文字方向/严重折返不靠GT推断。几何候选后续真实OCR结果见下表，直接替换不能改善所有预测轮廓。

开发评估新增逐实例GT覆盖率、预测纯度、IoU及合并/分裂候选，仅用于归因，不回流推理。固定健康step85600旧开发记录的实际统计如下（不是最新权重，也不是正式test）：

| 数据集dev100 | 预测实例 | 合并候选预测 | 分裂或重复候选GT | 无任意单预测覆盖半面积的GT |
|---|---:|---:|---:|---:|
| LATeD | 313 | 4 | 34 | 742/914 |
| CTW | 791 | 4 | 122 | 498/705 |

合并候选指单预测覆盖至少两个GT各50%；分裂/重复候选指至少两个预测各覆盖同GT≥10%、且各预测≥50%面积位于该GT。不是一对一匹配Recall，候选也不等于已确认的语义分词错误。[逐实例诊断](../results/v5/diagnostics/connector_associations_20261023.json)。当前阶段主要仍有大量未覆盖GT，不能把所有损失归咎于裁剪。

TotalText固定100图的727个有效转写GT，原crop可返回723个，失败4个（非法拓扑1、超图3）；失败继续保留CER分母。返回成功不代表边界采样准确。固定同一批723个可裁剪GT，pad=1、64×128坐标采样网格对照如下；漏覆盖面积按采样格union近似测量。

| 几何指标（每词均值） | C0 local_normal | C1 polygon_boundary |
|---|---:|---:|
| 采样点越界比例 | 9.086% | 0% |
| 漏GT覆盖面积 | 2.761% | 0.00991% |
| 漏面积超过5%的词数 | 82 | 0 |

[逐实例几何审计](../results/v5/diagnostics/connector_geometry_20261023.json)记录清单/源码摘要、原图像素0.001容差及失败原因。该几何审计本身没有运行OCR；随后已完成下述独立固定权重OCR对照。连接及采样14项CPU测试通过，两条基线已锁源码摘要仍一致。

下方已完成固定检测、识别权重的真实OCR对照；后续连接因素继续逐项验证，不先改实例分组或基线。

### 固定中间See的实际connector对照

无需等待完整See才做开发诊断。固定健康SynthText **step109600**快照、R4str best step36200、100对明暗dev；同一个检测前向得到的框分别送入两种裁剪。追加条件策略的独立推理也逐图核验了全部检测框、分数、顺序、权重及数据摘要完全一致。未训练模型、未用测试数据。

| 开发数据 | 原local_normal | 直接polygon_boundary | 条件boundary_when_supported |
|---|---:|---:|---:|
| 亮光GT词框raw CER%↓ | 23.3342 | 18.7327 | 18.7327 |
| 暗光GT词框raw CER%↓ | 72.9193 | 70.8826 | 70.8826 |
| 亮光spotting F1%↑ | 14.0411 | 10.1027 | **15.4110** |
| 暗光spotting F1%↑ | 4.6512 | 4.1616 | **5.3856** |

直接边界策略在锯齿/折返的预测轮廓上拒绝更多crop，说明GT几何收益不能直接等同于预测框收益。条件策略仅在`multiple_cross_sections`时使用原local_normal，并记录`crop_strategy/crop_fallback_reason`，其余错误继续报告；没有根据GT、识别结果或置信度选择分支。亮/暗分别恢复95/22个原可识别的预测；仍失败的90/20个多截面和亮光1个非法拓扑继续以空文字参与评估。每组GT词框均727，4个几何失败保留分母。

相对原策略，按原图分组、seed0、2000次配对bootstrap：亮光F1 +1.3699pp，95%区间[+0.3367,+2.5097]pp；暗光F1 +0.7344pp，区间[−0.2632,+1.8380]pp。暗光GT-crop CER −2.0367pp，区间[−3.5154,−0.6406]pp；亮光CER −4.6015pp但区间[−12.7477,+3.4898]pp跨0。候选保留开发验证，暗光F1增益尚需更多证据。

当前中间See暗光检测Recall仅5.5021%、F1 9.7919%，亮光检测F1 33.5616%；三策略检测指标相同，后续仍需完成检测训练。Read在当前合成暗光GT crop域CER约70.88%，也说明LSTR数值接近论文不等于已解决连接后的域适配。

初次实测还发现白区采样产生1.000000119，触发Read输入契约；现仅在connector的有限裁剪输出恢复[0,1]，两条基线源未改。实测重放与17项CPU测试通过。旧失败输出保留。

证据：[原/边界完整记录](../results/v5/fusion/connector_actual_rgb_20261023/run.json)、[条件策略完整记录](../results/v5/fusion/connector_supported_20261023/run.json)、[配对统计与正负实例](../results/v5/fusion/connector_supported_20261023/paired_comparison.json)。均为内部IoU≥0.5、lower-alnum匹配的开发spotting，非官方test。

### 连接因素单独拆分（2026-10-24）

固定相同Syn109600/R4best、100对dev和727有效GT词，完成以下单因素开发诊断。CER包含4个几何失败词；所有检测框、分数、顺序固定。精度修复后严格加载要求另存connector快照；迁移逐项验证全部权重/状态与原快照相同，新无曝光对照的200图框、分数、文字及全部指标与旧pad1.1记录完全一致。[快照迁移与前向等价核验](../results/v5/diagnostics/connector_dsconv_20261024/numeric_migration.json)。

| 条件轮廓裁剪后的路径 | pad | 亮光GT CER%↓ | 暗光GT CER%↓ | 亮光spotting F1%↑ | 暗光spotting F1%↑ |
|---|---:|---:|---:|---:|---:|
| 完整增强 | 1.0 | 18.7327 | 70.8826 | 15.4110 | 5.3856 |
| **完整增强，当前保留候选** | **1.1** | **18.8081** | **68.9464** | **15.5822** | **5.6304** |
| 绕过整个增强器；同一R4识别权重 | 1.0 | 17.9029 | 72.2152 | 15.4110 | 5.3856 |
| p95曝光→完整增强 | 1.1 | 18.9842 | 68.9464 | 15.5822 | 5.6304 |
| 保留提亮、仅移除去噪 | 1.1 | 17.2743 | 70.2037 | 15.2397 | 5.6304 |

pad1.1表示展开高度增加10%，上下各留原高度5%；相对于pad1的暗光CER下降1.9361pp，按原图配对bootstrap95%区间[−3.3370,−0.6236]pp。暗光F1提升0.2448pp，区间[−0.8054,+1.3576]pp跨0，继续作为开发候选，不据此宣称已确立F1提升。[余量对照](../results/v5/fusion/connector_pad110_20261023/run.json) / [配对统计](../results/v5/fusion/connector_pad110_20261023/paired_comparison.json)。

暗光错误审计显示，723个可裁剪GT的原图轮廓内p95亮度中位数0.1105；727词中153次预测为SALE，均错，358个暗光错误词在亮光能识对。依此测试只读取当前crop的固定曝光（亮度p95→0.7、RGB gain限制1–8、进入原增强器前应用），逐实例记录gain；暗光130词改善、130词变差，总CER和F1不变，不纳入保留方案。[错误审计](../results/v5/diagnostics/connector_dark_errors_20261023.json) / [曝光对照及配对统计](../results/v5/fusion/connector_exposure_pair_20261024/p95/paired_comparison.json)。曝光默认关闭，无GT/转写参与推理。

完整增强在当前暗光裁剪域优于整体绕过和去噪移除；保留Self-DACE提亮与去噪，暂不增加增强模块。[同权重整体移除](../results/v5/fusion/connector_identity_20261023/paired_comparison.json) / [去噪移除](../results/v5/fusion/connector_no_denoise_20261024_verified/run.json) / [去噪配对统计](../results/v5/fusion/connector_no_denoise_20261024_verified/paired_comparison.json)。这些是推理路径诊断，不代替独立重训消融。采用`_verified`中实际SelfDACE、denoise_present=false的完整记录。

当前保留候选相对最初local_normal，暗光GT CER由72.9193%降到68.9464%、内部spotting F1由4.6512%到5.6304%；同固定中间See的暗光检测F1只有9.7919%，检测漏检仍是当前spotting上限的重要来源。后续优先用完整See权重复核这一连接方案，再决定是否训练裁剪域适配；官方检测评测与终测继续后置。

当前并行推进connector在固定快照上的裁剪、暗光适配与逐实例调试；官方检测评测接入和终测统一后置。完整目标检测权重上的正式融合验收、重训消融、多种子及真实低光终测尚未完成；RLLIE也未实现。正式结论需要完整基线，开发验证不再把等待See作为前置条件。

### 2026-10-25：完整MLT、目标域迁移与Read裁剪适配

固定R4 best、条件轮廓裁剪/pad1.1、同100对开发图；只更换检测权重，未用test。

| 检测权重 | 亮光检测F1% | 暗光检测F1% | 暗光检测Recall% | 亮光spotting F1% | 暗光spotting F1% |
|---|---:|---:|---:|---:|---:|
| 原SynthText109600 | 33.5616 | 9.7919 | 5.5021 | 15.5822 | 5.6304 |
| 完整MLT100末权重 | 44.3206 | 6.7669 | 3.7139 | 20.7666 | 4.2607 |
| LATeD前41轮开发集best（epoch38） | 9.9562 | 14.8936 | 10.5915 | 2.7352 | 4.0619 |

三组GT-crop CER相同（亮18.8081%、暗68.9464%），因为Read和裁剪图不变。MLT预训练的亮光泛化提高，不能据此声称低光spotting提升；LATeD提高暗光命中却增加误检，检测命中后读对比例27.27%。LATeD暗光309个原始预测中48个裁剪多截面失败、54个有覆盖多个GT的合并候选；这些是诊断计数，不用GT修正实际推理。完整结果：[MLT](../results/v5/fusion/connector_mlt100_20261025/run.json)、[LATeD](../results/v5/fusion/connector_lated41_20261025/run.json)、[实例关联审计](../results/v5/diagnostics/connector_detector_transfer_20261025.json)。不同目标域的检测分数不代替最终完整权重验收。

因此本轮先保留几何策略与增强器，建立独立裁剪域LoRA训练线：从R4 best初始化，固定Self-DACE，仅优化原TrOCR LoRA；训练图使用冻结TotalText train的1152张原图、修正候选转写，生成16934张亮暗训练词图，训练原图与100对dev/test做路径和字节隔离，亮/合成暗1:1。通过实际connector生成384×384裁剪；PNG量化只用于训练/选模，完整spotting复核仍用原图浮点裁剪，并包含几何失败词。训练属于融合适配，不改独立LSTR基线，也不把该候选标注版本称为官方复现。短程已完成1200步；723张有效暗光dev PNG选模CER从71.6919%降至64.6970%（step800最佳）。PNG量化与直接浮点裁剪的输入不同，不将这组数值与先前68.9464%直接相减；已完成同检测框、原图浮点裁剪的明暗spotting复核，见下表。4个无效GT裁剪在完整复核中继续保留CER分母。

| 固定检测器；仅替换Read LoRA | 亮光GT CER%↓ | 暗光GT CER%↓ | 亮光spotting F1%↑ | 暗光spotting F1%↑ |
|---|---:|---:|---:|---:|
| MLT100 + 原R4 | 18.8081 | 68.9464 | 20.7666 | 4.2607 |
| MLT100 + 裁剪适配短程best800 | **14.0558** | **64.8227** | **21.4634** | 4.2607 |
| LATeD epoch38 + 原R4 | 18.8081 | 68.9464 | 2.7352 | 4.0619 |
| LATeD epoch38 + 同适配best800 | **14.0558** | **64.8227** | **3.1729** | **4.4487** |

暗光CER差−4.1237pp，按原图配对bootstrap95%区间[−5.9034,−2.4282]pp；LATeD框下暗光F1差+0.3868pp，区间[−0.3543,+1.1930]pp跨0。识别改善有证据，端到端增益尚待完整检测权重复核。亮光CER差−4.7523pp；固定MLT框的亮光F1差+0.6969pp，区间跨0。逐项核验只有144个LoRA张量变化，增强器与其他全部张量逐位相同，原R4文件不变。[短程权重审计](../results/v5/fusion/read_connector_lora_20261025/short_weight_audit.json)；稳定快照`short_best.pt`；[MLT复核/配对统计](../results/v5/fusion/connector_mlt100_read_lora_short_20261025/paired_comparison.json)、[LATeD复核/配对统计](../results/v5/fusion/connector_lated41_read_lora_short_20261025/paired_comparison.json)。

短程达到预登记扩预算条件：暗光CER改善≥1pp、暗光spotting不降、亮光F1退化≤1pp。长程总预算8468更新（16934词图、batch4、两遍），已从短程接续；用户随后要求优先跑完LATeD，Read暂停在最新健康last，排队到GPU3 CTW后继续，不从头训练。新connector候选仅对原策略仍失败的`multiple_cross_sections`使用凸包边界，正常裁剪不变，并记录`convex_hull_boundary`；CPU连接回归21项通过，实际同权重对照排在Read长程后，在取得增益证据前不替换保留方案。

LATeD本轮故障为geometry head首卷积有限输入→FP16 Inf→BN NaN。相同批BF16/FP32有限；仅geometry head FP32后，真实完整更新loss2.03157、grad norm8.66608、参数全有限，峰值allocated20.96GiB/reserved23.10GiB。[失败批重放](../results/v5/diagnostics/lated_failed3776_20261025.json) / [反向验证](../results/v5/diagnostics/lated_failed3776_backward_green_20261025.json)。为保留运行中的CTW闭包，修复在独立代码副本`results/v5/runtime/lated_geometry_fp32_20261025/`，[补丁](../experiments/autoresearch/v5/lated_geometry_fp32_20261025.patch)包含源码与回归用例；主目录Seeing未变。新LATeD从健康step3690迁移，model/optimizer/scheduler/scaler/RNG/sampler/loader/state八项保持，MLT初始化另存等权重且拓扑一致的副本。恢复已原序重放失败批、到健康step3900，参数全有限且源码吻合；迁移记录分别在两新检测目录的`numeric_migration.json`。补丁待现有固定权重连接对照完成后统一合入，不能把跨源码迁移称作原闭包exact resume。

GPU2加速遵照用户新增授权，原batch10和250轮预算保持：同一健康3900批次，原重算稳定均值0.7742秒/步，保留激活0.6911秒/步，吞吐约+12.0%；峰值allocated20.97→34.58GiB，全部参数有限，首个loss相同。减少重算后的浮点梯度有正常数值差异，不称逐位相同训练轨迹。[重算对照](../results/v5/diagnostics/lated_recompute_benchmark_20261025.json) / [保留激活对照](../results/v5/diagnostics/lated_no_recompute_benchmark_20261025.json)。独立副本`lated_geometry_fast_20261025`同时保留几何头FP32修复；内存上限由.35授权为.80，容器RAM24GiB/CPU8/PID512不变，主目录CTW闭包未改；独立加速副本回归20项及5个子测试通过。从健康3900迁移全部八项状态，最佳epoch38也迁移保留；不使用interrupted.pt续训。[加速补丁](../experiments/autoresearch/v5/lated_no_recompute_20261025.patch)、新目录`numeric_migration.json`与`best_migration.json`记录来源。最初启动被旧.35硬限制挡住，调整授权副本后已实际运行，不把失败尝试记为训练完成。

### 训练轮数是否不足（2026-10-25）

当前容器训练和250轮基线预算保持。epoch必须同时看数据量、batch与更新数：本项目SynthText每轮85875次更新，2轮共171750；LATeD每轮90次更新，即使1000轮也只有90000次更新。不能用其他任务“几千轮”的数字判断这里是否欠训练。

| 训练线 | 当前预算/观察 | 延长判断 |
|---|---|---|
| 独立Read R4/LSTR | 20轮/50740更新；best step36200 dev CER39.7753%，末次50.1642% | 后期已变差；不先盲加轮数，保留原best与独立基线 |
| Read connector裁剪适配 | 16934词图、batch4；健康2400步约0.57轮，dev CER71.6919%→62.6768%；当前目标8468步/2轮 | 更值得检查欠拟合；先跑完2轮及原图连接复核，再考虑5、10轮分段 |
| See CTW | 已完成250轮；epoch87已达54.9801%，最终best55.0292%（epoch217），末轮52.8061% | best增量约0.05pp，有平台迹象；延长须独立对照学习率，不预设千轮有效 |
| See LATeD | 审计时约79轮，后续约87轮，尚未经历epoch100的第一次学习率下降 | 先完成当前250轮，不能现在判定轮数不足 |

本地固定Self-DACE源中，`old_version/stage2/train_2stage.py`第141行确有`num_epochs=1000`默认值，对应独立去噪网络训练；new_version的增强阶段也设1000轮。[固定源脚本](https://github.com/John-Wendell/Self-DACE/blob/ec0c033def6c98a97b99c2938e7c8ffc50ba65fd/codes_SelfDACE/old_version/stage2/train_2stage.py#L141)。当前Read加载公共light/denoise权重后再适配OCR，这个1000轮默认值不能直接用作LoRA/联合微调的轮数要求。

Seeing论文§5.1明确列出SynthText2轮、MLT100轮、batch10与250轮训练，并每100轮将学习率乘0.1：[原文](https://arxiv.org/html/2404.08965v3#S5.SS1)。按本地StepLR从1e-4延到1000轮，学习率会降至1e-14；仅改epochs不构成有效长程试验。Reading正文未找到明确epoch配方，不把本地20轮称为作者指定轮数：[原文](https://arxiv.org/html/2604.23685v1)。

补充实验只看固定开发集，原250轮与Read独立基线封存；追加轮数和调整学习率分开登记。本轮grill-me已确认优先Read裁剪域适配。当前2轮及全部连接对照结束后，保持学习率2e-5、batch4、数据、冻结增强器和原LoRA不变，继续到5轮/21170更新；5轮相对2轮在同完整CTW权重上暗光原图GT CER改善≥1pp、暗光E2E F1不降且亮光F1退化≤1pp时，再到10轮/42340更新。各边界使用硬链接保留best/last和报告；不足门槛记为跳过，不把旧结果冒充10轮训练。See当前250轮计划不变。[曲线审计](../results/v5/diagnostics/training_duration_audit_20261025.json)。

## 5. 运行和证据入口

| 内容 | 位置 |
|---|---|
| 检测旧主链/状态 | [DSConv精度恢复计划](../experiments/autoresearch/v5/queue_20261024_detection_dsconv_fp32.json) / [实时状态](../results/v5/queue_20261024_detection_dsconv_fp32.json)；MLT完成、LATeD3776中断，恢复见新记录 |
| LATeD当前长程 | [授权加速计划](../experiments/autoresearch/v5/queue_20261025_lated_fast_authorized_until250_gpu2.json) / [实时状态](../results/v5/queue_20261025_lated_fast_authorized_until250_gpu2.json)；`LATeD_geometry_fast_20261025`，无短程上限 |
| Read分段延长 | [5→10轮计划](../experiments/autoresearch/v5/queue_20261025_read_5to10_epochs_gpu3.json) / [状态](../results/v5/queue_20261025_read_5to10_epochs_gpu3.json)；等待现有全部对照结束，容器GPU3执行，门槛回归通过 |
| GPU3后续Read/connector | [CTW后续计划](../experiments/autoresearch/v5/queue_20261025_ctw_then_read_connector_gpu3.json) / [状态](../results/v5/queue_20261025_ctw_then_read_connector_gpu3.json)；等待当前CTW完成后继续Read8468与凸包候选对照；随后[完整CTW连接复核](../experiments/autoresearch/v5/queue_20261025_full_ctw_connector_gpu3.json) |
| LATeD数值恢复 | [短程计划](../experiments/autoresearch/v5/queue_20261025_lated_geometry_preflight.json) / [状态](../results/v5/queue_20261025_lated_geometry_preflight.json)；`LATeD_geometry_fp32_20261025`，原失败记录保留 |
| Read裁剪域适配 | [短程计划](../experiments/autoresearch/v5/queue_20261025_connector_read_lora_pilot.json)；输出`results/v5/fusion/read_connector_lora_20261025/` |
| 检测原主链 | `results/v5/detection/SynthText_dsconv_fp32_20261024/` → `MLT17_dsconv_fp32/` → `LATeD_dsconv_fp32/`；日志 `results/v5/queue_logs/20261024_Seeing_dsconv_fp32_full_chain.log` |
| GPU3 CTW并行分支 | [当前计划](../experiments/autoresearch/v5/queue_20261024_connector_no_denoise_verified_then_ctw_gpu3.json) / [当前状态](../results/v5/queue_20261024_connector_no_denoise_verified_then_ctw_gpu3.json)；connector对照已完成、CTW已完成250轮，日志 `results/v5/queue_logs/20261024_CTW_after_no_denoise_verified_gpu3.log`，完成后输出`results/v5/detection/CTW1500_dsconv_fp32/` |
| 识别训练 | [GPU3计划](../experiments/autoresearch/v5/queue_20261022_reading_gpu3.json) / [实时状态](../results/v5/queue_20261022_reading_gpu3.json)；`results/v5/reading/R0str_frozen/、R2str_lora/、R4str_joint/` |
| 识别锁定验收 | [后续计划](../experiments/autoresearch/v5/queue_20261022_reading_acceptance_gpu3.json) / [状态](../results/v5/queue_20261022_reading_acceptance_gpu3.json)；三组锁定官方LSTR测试均完成 |
| 来源/规则 | `results/v5/source_snapshots/`；Self-DACE ec0c033、DSCNet0e35b2b；[执行规则](../experiments/autoresearch/v5/program.md)、[结果索引](../experiments/autoresearch/v5/results.tsv) |

仅使用物理GPU2/3，每卡独立容器、各RAM24GiB/无额外swap、CPU8/PID512/shm4GiB/线程4/默认显存比例.35；本次用户授权GPU2加速副本.80；外部负载出现时只让出自己的任务。队列14项安全测试通过。控制器已脱离SSH会话，断开SSH可继续；服务器/Docker重启后须显式恢复。Reading训练闭包未改，ESTR60未使用。日志/events是实时进度，status.json在调用退出时更新；健康last才是恢复点。历史结果与旧实现结论见03，不在本篇重复展开。

## 6. 到投稿的剩余工作与时间估计

Reading官方test、完整MLT与CTW250均已完成。GPU3继续Read两轮和现有连接复核，随后执行已授权的5轮、条件10轮扩展；GPU2 LATeD还约3.5–4小时。延长阶段沿用原学习率和每200步dev验证，预计5轮增加约40–60分钟，若到10轮再约1–2小时，以实际事件耗时修正。正式评测、多种子和终测另计。

| 剩余步骤 | 交付证据 | 粗估时间 |
|---|---|---|
| 1. 独立基线验收 | Reading已完成锁定test；Seeing继续完整预训练、接通官方LATeD/CTW评测 | 3–7天，取决于检测首次完整结果及官方评测对齐 |
| 2. 固定接口与位置实验 | 同权重A/B/C、暗亮spotting F1、检测指标和GT-crop CER | 1–3天 |
| 3. 删改与一个连接模块 | 定位主要退化来源；同预算消融证明一个模块的增益 | 1–2周，失败候选会延长 |
| 4. 完整验证 | 多随机种子、主要消融、泛化及代价；锁定后做最终测试，真实spotting需完成LATeD转写复核 | 1–2周；标注可并行，取决于人力 |
| 5. 论文与可复现实验包 | 方法、结果表、机制与失败案例、配置/权重/运行说明 | 1–2周，可与步骤4并行 |

在融合产生稳定收益的前提下，完整实验论文初稿按4–6周安排；包含充分真实低光终测和更完整复核时按6–10周安排。这是基于当前工程状态的估计，不把单纯拼接两条基线视为新方法贡献。优先取得可靠的组合F1及一个清楚、可重复的模块收益，再扩大实验矩阵。
