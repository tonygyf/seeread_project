# 04 v5 基线、当前故障与保留实验

## 11月5日：LATeD全部1000张转写草稿准备

已将900训练+100开发图按源GT裁成9115行，并逐项核验ID/多边形及官方1000图并集；500张终测未进入。新审核入口为 [完整1000图](http://100.88.81.10:8765/full/review.html)，旧审核页和用户本地状态保留。物理GPU2的独立PaddleOCR任务批量处理原图/数据包亮版/现有轮廓校正和方向候选，所有置信度及空结果留存，等待人工确认；GPU3 See训练继续。模型、状态和逐行草稿在 `data/derived/v5/lated_str_1000_20261105/paddle_drafts{,_provenance}.json*`。Read分支源代码和完整操作说明见下方独立分支链接；同一浏览器自动迁移旧审核，也可导入旧JSON补缺。


## LATeD STR 独立分支准备（11月4日）

已确定：用户人工审核转写；900训练/100开发/500终测按原图隔离；清晰但倾斜/环绕且OCR错误的样本保留；自动仅提示极端像素候选，不删除；R4固定权重比较原始外接裁剪与connector轮廓校正，随后等训练ID/预算LoRA适配，冻结增强器及非LoRA参数。尚未训练，不占用GPU3。

代码在 `exp/read-scene-str` / `results/v5/worktrees/read_scene_str`，主Read基线未改。现有[20图258行试标](http://100.88.81.10:8765/pilot.html)复用原增强参考和OCR草稿，新增极端候选提示；完整300图3041行包提示260行，pilot提示42行，均等待人工确认。原900/100/500共8201/914/4808实例，目前原始标注无转写，不能直接监督Read或报告真实识别成绩。

具体计划与操作见[分支文档](../results/v5/worktrees/read_scene_str/docs/04_v5_alignment.md)及 `experiments/autoresearch/v5/scene_str_plan.json`（分支内）。新工具记录全部实例、审核状态与裁剪失败，配对训练只取共同训练ID；评分保留所有可确认GT，失败裁剪作空预测，支持按原图配对bootstrap。最终500张不用于筛图规则或模型选择；GT识别收益后另验固定See预测框及整图spotting。此方案是独立STR/裁剪域适配，不能仅凭裁剪+微调宣称新方法。


## 11月4日：保留GPU3续训，停止GPU2从头任务

依用户决策，GPU2在约300步时停止；用户要求的条件是训练超过10万步且配置有明确单因素差异。停止时已用所有权记录验证，仅对本项目GPU2进程发SIGINT，保留日志及最近健康last；没有触碰GPU3或其他用户。GPU3继续原配方。三个阶段除了实验名称、输出目录与各自前驱权重路径外，配置和随机种子一致；两条训练的历史起点不同。

原健康148200断点复制为 `results/v5/detection/SynthText_final_scm_fp32proj_20261102/healthy_step148200.pt`，SHA256与原last相同。`tools/log_health.py`按每20步记录对比断点前后20条日志，梯度范数为裁剪前，批次不同；在续训记录满20条前不判定通过或失败。临时健康监控会在结果齐全后写入 `results/v5/diagnostics/resume_health_148200_20261104.json` 并停止自身。此前双卡及GPU2预计完成时间是历史记录，不再生效。

GPU2仍有后续单因素分叉的价值，但先完成用户要求的健康检查；当前未确认signed_bn或log-height能改进，尚未启动新变体。更换结构时需核对具体参数匹配及优化器状态，`strict=False`不会自动处理同名张量尺寸不匹配，也不能单独保证恢复正确。


## 11月4日18:34：双卡完整实验已启动

用户授权从头训练及最多两张空闲GPU。启动前GPU2/3无计算任务；现各运行一条独立链路，**不是DDP**，两条均保持batch10、640×640、Adam1e-4及原分阶段训练配方。固定代码快照 `results/v5/runtime/seeing_stable_20261104`，后续编辑主目录不影响在跑实验。

| GPU / 用途 | 起点与完整路径 | 初始速度及预计完成（北京时间） |
|---|---|---|
| GPU2，从头主实验 | 公共ImageNet骨干+随机检测头；SynthText858750图×2轮→MLT17训练7200图×100轮→LATeD官方1000图×250轮 | 约2.30秒/步；剩184–225小时，11月12–14日 |
| GPU3，精度迁移对照 | 健康148200步迁移，保留模型/优化器/调度器/scaler/RNG/采样进度；完成余下SynthText及同样MLT/LATeD | 约2.33秒/步；剩86–108小时，11月8–9日；不能标为从头复现 |

主实验总更新268750步，对照起点剩120550步。时间以早期实测速率外推，包含验证与保存余量，尚未实测后续阶段吞吐；[初始ETA](../results/v5/diagnostics/stable_dual_eta_20261104.json)。主实验初始GradScaler发生一次反向梯度拒绝更新，已按现有机制降低scale、恢复BN并重试同批，后续继续；没有本次已知前向NaN中止。

两条链由独立systemd用户服务守护，用户linger已开启，断SSH继续。容器分别限制24GiB主存、8CPU、512PID、4GiB共享内存，无额外swap，进程显存上限80%；只暴露各自GPU。队列每10秒核对GPU占用，出现外来任务只停止自己的训练。每200步保存健康断点；数值故障标记needs_attention后停止，不无限重启。

对照首次预检拒绝了迁移记录与实际CUDA分配器环境不一致，尚未更新参数。补齐 `expandable_segments:True` 后通过严格校验，以r2队列运行；旧失败记录保留。[迁移记录](../results/v5/detection/SynthText_stable_resume_20261104/numeric_migration.json)。

两条链在全部训练结束后自动用固定第250轮last.pt评LATeD500，与旧基线P/R/F1=66.2278/54.4093/59.7397%及论文82.6/57.0/67.1%比较；不根据测试分数选择中间轮次。该500张历史上参与过选择，仍不称独立封存终测。MLT沿用7200训练/1795开发，LATeD1000全部训练、CTW100只作过程诊断，数据划分未临时改变。

SwanLab新增“See | 从头主实验”和“See | 健康断点精度迁移对照”，已启用专用监控定时器，后续MLT/LATeD日志出现后自动接入。队列状态分别位于 `results/v5/queue_logs/stable_fresh_20261104_state.json` 和 `stable_resume_20261104_r2_state.json`；配置为 `experiments/autoresearch/v5/stable_{fresh,resume}_pipeline_20261104.json`。未推送Git。


历史记录截至11月4日双卡启动前：当时所有训练已停止，建议先验证精度修复和 SCM 稳定性；当前状态以上方“双卡完整实验已启动”为准。LATeD 人工转写暂停，未经明确要求不 push。

## 11月4日复核：运行状态、排名与数据隔离

该次复核时未发现11月2日以后的SeeRead训练或新目标集评测，11月4日双卡任务随后启动。Read R2/R4完成50740更新（20轮预算），Connector适配完成42340更新（10轮预算）；旧See完成250轮。新结构两次分别在146366和148356步异常中止，不能将GPU空闲解释为新实验正常完成。

- 当前See已完成候选：官方1000图训练、250轮的高度0.05+OHEM，**last.pt**在LATeD500上F1为59.7397%。相对较早900图训练的best99在同500图上的58.9943%，提高0.7454pp；训练样本数、预算也变化，不能单独归因于结构或某一个模块。
- Read按LSTR测试CER：LoRA 49.4897%略低于Self-DACE+LoRA 49.5191%，相差0.0294pp；词准确率则联合模型40.3157%略高于LoRA的40.2095%。两者没有确定的大幅优劣证据，开发集配对CER差区间跨0。
- 已完成的整链暗化TotalText300图结果为E2E F1 **6.6418%**，使用旧MVP检测器和裁剪适配Read，未换上59.74%的检测候选。原暗图评测因EXIF方向不一致已标为无效，只使用`spotting_mvp_totaltext_test_dark_encoded_20261027`的修正记录；该修正不等于模型训练提升。

本次只读重算57993个当前文件的SHA256：LSTR train/dev/test/ESTR60，LATeD900/100/500及官方1000/500，TotalText1152/100/300，Connector16934/723词图，均未发现跨划分路径或相同文件内容重叠。Connector训练、开发的源图均不进入TotalText测试；Read训练清单摘要仍与训练记录一致。[本次完整检查](/home/gyf/experiment_hub/reports/seeread_audit_20261104.json)。

**泄漏结论的范围：**当前同图、配对图及已登记源图的隔离检查通过；Read原始场景来源不完整，近重复场景及公共预训练语料的交叉未被本检查证明排除。LATeD500曾参与多轮比较，测试选择带来的偏差仍存在，不能恢复为独立终测。后续根据这些分数调整方法时，最终结论需要另一份未参与选择的终测。

SwanLab的8项名称及说明已标明LSTR测试CER、LATeD测试F1、开发集成绩或异常中止，避免把不同阶段的曲线混排；FiftyOne改为保留修正后的暗图记录。此次未启动训练或推送代码。

## 续训依据核对

对照两个实际runtime，已复现的146366/148356失败属于FP16前向溢出，不是已证实的标签或梯度公式错误。旧runtime将原分辨率mask作为 `targets["scm_text"]`，新runtime将同一变量作为 `scm_target` 传入；监督来源相同。数值修复还改变采样后端和浮点计算次序，不能声称训练轨迹逐位相同。健康断点意味着可恢复及已查状态正常，不证明历史每一步没有数值误差，也不证明结构与论文完全一致。

新结构的注意力轴、连接关系和SCM语义监督等是论文未完全公开的重实现假设，独立于这两次FP16故障；现有证据不足以排除它们影响结果。续训对照已越过原148356失败步，但尚未完成SynthText，更无MLT/LATeD最终成绩。两条实验差异包含早期计算精度及训练历史，最终分数差距不能单独归因于某个bug。

## 11月4日数值修复：暂停训练，仅做回归验证

两次已捕获前向故障均来自FP16越界：146366步为末级PM输出投影，148356步为DSConv偏移卷积。降低GradScaler不修复前向溢出。本次统一将PM的普通卷积、双轴投影、Q/K投影、注意力和输出投影，FPN投影/融合、SCM及三个检测头放在FP32；骨干仍可AMP，DSConv保留激活重计算。损失在平方/求和之前转FP32，覆盖语义差值平方及320×320掩码计数的半精度溢出。未新增数值截断、NaN替换或修改损失系数。

主代码 `SeeingDetector(architecture="figure2")` 现已包含之前快照的新结构；默认 `legacy` 保留旧结构。两个真实故障权重均严格加载成功。小尺寸FP32结构对照的8项输出与11月2日快照逐项完全一致；真DSConv和256通道完整模型另由故障重放验证。旧runtime、旧队列和检查点未覆盖，**旧队列不会自动使用本次修复**。

| 回归对象 | 本次结果 |
|---|---|
| 原146366步失败批，batch10、640×640，3次内存Adam更新 | loss 44.4108→42.9099→40.8631；前向/梯度/模型参数/优化器状态有限 |
| 原148356步失败批，同样完整批次、3次内存更新 | loss 40.8792→40.8222→40.8160；同上 |
| PM、SCM、DSConv及标签隔离 | 14项测试通过；新增Q/K和输出投影测试先复现失败后通过 |
| 几何/损失 | 13项通过，包括大幅度语义平方及大尺寸掩码求和 |
| 训练器恢复、AMP拒绝更新和健康断点保护 | 13项通过 |

完整批次峰值allocated58.77GiB、reserved最高63.02GiB，低于本次85,651MiB显卡的80%进程上限。FP32大尺寸展开图另触发cuDNN grid_sample限制，现明确使用PyTorch原生采样，保留同一插值规则。正式训练入口上限与此前授权的80%一致，默认配置仍为35%；pilot短程限制不变。此精度方案不能直接假设能装入论文48GB A6000。

准备了但**未启动** `configs/v5/seeing_figure2_stable.yaml`：与前次SynthText候选逐项核对，训练、数据、解码配方一致，只更改实验标识、显式结构选择和独立输出路径。恢复旧健康断点需显式记录精度/来源迁移；严格resume校验仍会拒绝来源变化及 `interrupted.pt`，未放宽保护。

[论文§3.2–3.4、§5.1](https://arxiv.org/html/2404.08965v3)规定并行Conv/DSC、注意力、重建L1、语义L2和分阶段配方，未规定必须FP16。精度修复不改变这些运算的数学定义。通道注意力轴、双轴融合和额外投影、F1跳接/输出步长解释、语义目标stop-gradient及具体均值归约仍是本地实现假设，不能将“数值有限”写成“作者实现完全复现”。SCM语义项在第二批仍约38.75–38.88；本次未改其优化目标，收敛问题尚未由这些回归测试解决。

证据：[完整批次回归](../results/v5/diagnostics/seeing_precision_replay_20261104.json)、[结构与配置对照](../results/v5/diagnostics/seeing_structure_parity_20261104.json)、[训练器测试](../results/v5/diagnostics/seeing_trainer_regression_20261104.log)。诊断更新只发生在内存，未保存权重、未续训、未推送；测试结束GPU已释放。

## 基线验收

| 模型 / 协议 | 论文 | 本地结果 | 结论 |
|---|---:|---:|---|
| Read R0，LSTR test raw CER | 53.16% | 53.5778% | 差 +0.4178pp，达到3pp容差 |
| Read R2 LoRA，LSTR test raw CER | 52.84% | 49.4897% | 数值更低，差 −3.3503pp |
| Read R4 Self-DACE+LoRA，LSTR test raw CER | 50.02% | 49.5191% | 差 −0.5009pp，达到容差 |
| See，高度0.05+OHEM，LATeD官方500图 P/R/F1 | 82.6 / 57.0 / 67.1% | 66.2278 / 54.4093 / 59.7397% | 差 −16.3722 / −2.5907 / −7.3603pp；仅R在容差内 |
| 旧 MVP 检测器，LATeD内部100开发图 P/R/F1 | 不作论文同协议比较 | 65.4088 / 45.5142 / 53.6774% | 现有整链使用的历史对照 |
| 新结构，SynthText预训练 | 尚无目标集验收 | 148356步非有限loss中止 | 尚未进入本次MLT及LATeD阶段 |

LSTR测试6588词图；训练使用10146/1127的内部train/dev，尚未完整11273 refit。冻结模型和训练best锁定后测试。ESTR60未用于选模。Read RLLIE仍未实现；以上数值接近不表示所有论文组件完成。

旧 See 使用官方1000训练图、250轮。0.05高度系数和OHEM是本地候选选择，不能称为已确认的论文配方。测试采用CTW-VOC规则本地移植，尚未与作者原始LATeD评测逐项核实；500图已按用户授权用于后期轮次比较，不能再称独立封存终测。第250轮报告：[run.json](../results/v5/diagnostics/lated_test500_height_official_train_epoch250_20261027/run.json)。

## GPU0 本次为什么停止

11月2日11:34，新结构在SynthText第148356步报 `Nonfinite detection loss`。输入与标签有限，最早观测到的非有限算子为：

`pm.2.snake_h.operator.offset_conv → gn_offset → tanh → 后续特征与检测头`

| 同一失败批次 | offset Conv输出范围 | 总loss |
|---|---|---:|
| 原FP16 autocast | 最小值−Inf | NaN |
| BF16前向 | 约−66048至8384，有限 | 40.9678 |
| FP32前向 | 约−66026.5至8385.54，有限 | 40.8895 |

FP16有限范围到65504；本次偏移卷积在归一化前越界。此前146366步修的是末级PM的输出投影，本次是**另一个未保护的算子**。未发现显存不足或其他用户任务抢占导致本次中止的证据。

主代码 [dsconv.py](../seeread/v5/dsconv.py) 已将偏移预测及其归一化放到FP32；冻结的旧runtime未改动。原失败批次用内存中的同等修复完成前向、反向和一次Adam更新：loss40.8793、裁剪前梯度范数18.1517，模型参数有限；峰值allocated44.55GiB/reserved48.10GiB。新增CPU FP16回归测试先失败，修复后7项DSConv测试全部通过。

证据：[失败重放](../results/v5/diagnostics/seeing_failed148356_20261102.json)、[单次更新](../results/v5/diagnostics/seeing_failed148356_offset_probe_20261102.json)、[回归结果](../results/v5/diagnostics/offset_regression_green_20261102.log)。没有覆盖原检查点，也没有据此启动长训。

## 是否值得继续

**不建议原样继续新结构长训。** 数值故障可以修复，但该批SCM语义项仍为38.75，占总loss约95%；前面长期曲线中语义项也没有稳定下降。新结构同时改变了FPN步长、SCM接入位置、残差路径和decoder末层，现有证据不能将漂移单独归因于其中某项。

项目仍有可用基础：Read数值接近论文；旧See具有59.74%检测F1；已有整图多实例connector通路。下一步优先固定旧See做connector/Read开发。若继续评估新结构，先从保留的健康断点在独立短程中验证偏移精度修复和SCM幅度/梯度；没有稳定性与开发集收益证据时，不再投入完整三阶段长程。老师后续DarkText源码可作备选，但历史66.74%结果未确认对应具体权重，不能直接当作本地已验收成绩。

## 保留内容与实验入口

| 用途 | 保留位置 |
|---|---|
| 旧 See 正式候选 | `results/v5/detection/LATeD_height005_official_train1000_20261027/{best,last}.pt`；论文对比用last第250轮 |
| 旧整链MVP | `results/v5/detection/LATeD_balanced_ce_20261026/best.pt` |
| 旧预训练/跨域对照 | MLT17末轮、SynthText旧断点、CTW best各一份 |
| Read 独立基线 | `results/v5/reading/{R2str_lora,R4str_joint}/best.pt`；`weights/reading/`保留两组轻量增量参数 |
| Read 裁剪适配 | `results/v5/fusion/read_connector_lora_20261025/best.pt` |
| 新结构故障诊断 | 当前健康last和interrupted、前次故障interrupted与首轮best |
| 更早S4参考 | `results/archive/legacy/s4_rec_v4/best.pt` 一份 |
| 正式识别记录 | `results/v5/final/{R0str_LSTR,R2str_LSTR,R4str_LSTR}/` |

SwanLab [5092](http://100.88.81.10:5092)保留8项主要实验。FiftyOne [5151](http://100.88.81.10:5151)保留所选评测/识别/融合记录和现存代表图，不再导入每轮预测。完整入口与维护见[实验中心说明](/home/gyf/experiment_hub/README.md)。

被清理实验的配置、指标和日志压缩至 `results/archive/experiment_history_20261102.tar.gz`；旧队列/临时配置在 `queue_history_20261102.tar.gz`。删除及保留清单为 `retention_20261102.json`、`queue_retention_20261102.json`。冗余权重和重复渲染已删除，不包含在文本压缩包中。历史长文也已收入压缩包，当前维持四份主文档。

原始数据、人工标注、公共预训练资源和老师仓库未清理。LATeD无人工文字转写，当前不报告真实低光识别/spotting终测；TotalText人工转写及合成暗光可继续用于开发。检测基线达标与真实转写终测仍是后续独立验收步骤。


11月5日完整OCR预填验收：1000图、9115行全部覆盖，共91020候选，约19分钟完成，置信度0至0.999984全部保留。19行最佳草稿为空；65行校正失败仍有8个外接裁剪候选，其余每行10候选。不自动确认任何人工转写。完整页面已包含每行候选，GPU2 OCR任务成功退出并释放，GPU3 See训练继续。[覆盖核对报告](/home/gyf/seeread_project/results/v5/diagnostics/lated_1000_completion_audit_20261105.json)。


## 378条人工审核的首次固定Read诊断（2026-11-05）

原页面仅localStorage，现同源8765服务新增服务器快照同步，浏览器仍保留本地数据；用户刷新后已收到gyf的275 reviewed+103 unreadable及1 pending。271条可用拉丁转写全部来自32张train原图；3条non_latin单列，1条reviewed未选语言留在原备份、暂缓计分。生成的sidecar和本次测试冻结在 `data/annotations/v5/scene_str_reviewed_20261105/`，不覆盖用户原记录。备份保留最近20个内容快照，页面导出仍可用。

固定未做LATeD适配的R4、相同128-token解码上限；本批真值最长24 tokenizer tokens，无标签截断。原始外接裁剪：271行、2890字符、1995编辑、84行全对，CER69.0311%、整行准确率30.9963%。现有轮廓校正：269行成功、2行失败作空预测，仍以271行计分；2441编辑、74行全对，CER84.4637%、准确率27.3063%。CER差B−A=+15.4325pp；按原图bootstrap95%区间约[−2.7322,+40.2581]pp，样本不足以宣称稳定总体退化。未更新权重。报告 `results/v5/reading/scene_str_reviewed_test_20261105/run.json`，配对及长度分层见同目录 `paired_diagnostic.json`；[逐行图片/人工文字/两种预测对照](http://100.88.81.10:8765/full/read_test/comparison.html)。

原始裁剪分层：≤8字符138行CER33.5252%；9–16字符96行CER57.7679%；>16字符37行CER103.7209%，插入错误可令CER超过100%。这只能说明该人工选择子集长文本较难，不能单独归因为裁剪、低光或长度。当前全部来自train分配，但R4尚未用LATeD训练；属于目标域的冻结模型诊断，不是已适配模型训练成绩，更不是dev/test最终结论。

Read定位是一个已裁出的文字实例→一个字符串，可为单词或单行多词；不是整图多区域检测/阅读顺序模块。弯曲、倾斜、环绕在语义上仍可是一行，当前模型和校正器尚未证明稳健。多行段落需额外行分割/排序协议。当前R4父模型为TrOCR-base-STR，LSTR训练10146/dev1127，LSTR test6588，CER49.5191%；ESTR60留作最终真实低光评测。新LATeD分支900/100/500、拉丁文字范围，现仅有train人工子集。父配置max_len64 tokens，测试128，模型输入统一384×384；长度、分辨率和数据域均需控制。[TrOCR官方用途和预训练来源](https://huggingface.co/microsoft/trocr-base-str)。
