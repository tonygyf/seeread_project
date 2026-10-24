# v5 autoresearch 执行规则

数据、论文配方、实验矩阵、结果与待办只维护在 [04 v5 对齐](../../../docs/04_v5_alignment.md)。本文件只保留执行约束。

1. 固定train/dev/test、暗化种子、CER归一化与spotting匹配规则；dev选模，锁定后才做官方test验收；ESTR60最终留用。
2. 新实验独立公共初始化、新输出目录；exact resume须同源码、运行库、数据与配方，保留optimizer/scaler/RNG/sampler。不得迁入旧S3/S4权重。
3. 已授权同时使用物理GPU2和GPU3，每卡独立受限容器、每卡串行一项训练；各24GiB/8CPU/512PID、shm4GiB、无额外swap，显存比例0.35。监控只查询本卡，外部负载时等待或中断经身份核验的自有任务，不碰其他进程与共享服务。
4. 短程1200秒，报告实际更新数、样本和验证耗时。仅完整dev参与keep/discard；方法候选获开发收益后才扩预算。R4b长程门槛为比冻结R1同dev raw CER改善至少1个百分点。
5. 固定论文阶段按预登记目标执行：SynthText2epoch→MLT17 100epoch→目标集250epoch；20分钟数值/资源检查通过后，按已授权的单卡不限时完成，正式配置max_seconds=0。整epoch前无dev，不以train loss下降代替泛化收益。
6. 非有限loss/FP32梯度或连续AMP失败超额即停止；AMP重试不推进sampler/scheduler，成功清零streak；异常快照不覆盖健康last。
7. 保留run/status、逐样本预测、events与checkpoint；结果索引用[results.tsv](results.tsv)。不覆盖已有结果，不修改旧锁摘要，不用官方test挑后续超参或模块。
8. 正式组合验收以两条独立基线为前提；开发阶段可用固定中间快照测connector。连接消融一次只改一个因素，保留暗光收益、亮光退化与归因指标。
9. 论文验收容差为CER/F1相差≤3个百分点，须先匹配数据、实现假设和评测协议；开发集结果不能代替官方测试验收。
10. GPU3新增公开scene-text起点：固定STR revision，R2str与R4str同seed0、batch4、train10146/dev1127。各短程1200秒验证数值/资源后，两支均续至50740总更新（20遍训练数据），按完整dev选best；不因短程CER未改善而取消已登记的等预算基线。方法候选仍遵守第4条开发收益门槛。MLT保持Seeing预训练用途，基线完成后再做同预算有/无MLT消融。

队列计划为本目录`queue_*.json`；实时状态与日志位于`results/v5/`。用本项目`tools/v5_*.py`，不运行upstream GPT trainer或reset/checkout；运行中保持训练源码与冻结协议不变。

11. 2026-10-22检测center head FP16溢出已真实重放，显式从健康step28800迁移为该分支FP32；模型、optimizer/scaler/RNG/sampler等逐项保持。父结果不变；新队列`queue_20261022_detection_fp32.json`完成原2/100/250预算。迁移记录位于新SynthText输出目录，不称父闭包exact resume。
12. 融合先准备固定100图明暗dev与A无增强/B仅crop增强/C仅检测输入增强；三臂共享相同权重，A/B共享检测框。独立检测基线完成后填入固定checkpoint运行，随后只删明显有害因素，再逐个增加有消融收益的连接模块。

13. 2026-10-22 step32962重放确认text head也有FP16卷积溢出，仅增补text head FP32，保留center FP32；几何头/其他AMP不变。最新源989d5900…，从健康32800显式保留状态迁移；该历史检测计划为`queue_20261022_detection_classification_fp32.json`，已在step85672停止。

14. 三组STR独立LSTR test已全部完成并保持锁定；后续模块选择仅看dev。GPU3已完成健康Syn57200快照在CTW/LATeD dev100的阶段P/R/F1，以及固定R4best同权重full/no_denoise/identity推理移除，不作为正式组合或独立重训消融。阶段分数、配对区间和下一步统一见docs04。

15. 2026-10-23 step85672定位PM普通Conv的FP16溢出；三层共用普通Conv/BN分支FP32，其他结构/损失不变。从健康85600显式保留全部训练状态迁移，原坏批重放成功、完成首个epoch；该次主链为`queue_20261023_detection_pm_fp32.json`，继续原2/100/250预算。数值迁移证据在`SynthText_pm_fp32_20261023/numeric_migration.json`；不自动跳过坏批或改写历史锁。
16. TotalText检测ignore遵循作者第6列orientation，GT等价检查允许原样退化轮廓，预测校验保持严格。新train候选单独保存，未覆盖冻结train/test/fusion；MAT全量来源身份未证实前保持source_candidate。官方执行尚未接通时，阶段内部F1不得称官方验收。

17. 用户2026-10-23再次授权双GPU加速。GPU2保持现有SynthText→MLT→LATeD续训，GPU3由`queue_20261023_ctw_branch_gpu3.json`等待共享MLT100完整结束后仅执行CTW250，固定batch10。复用现有stage训练/健康继承检查，共享父目录只读；上游失败时停分支。当前采用目标训练并行，不宣称单链DDP加速。

18. 用户明确要求在Seeing完成前取得connector实测：允许固定健康中间检测checkpoint与固定R4best，仅在冻结100对明暗dev比较local_normal/polygon_boundary，报告GT-crop CER与内部spotting F1。该诊断不替代正式基线验收，不用test选模块。当前GPU3计划`queue_20261023_connector_rgb_then_ctw_gpu3.json`，完整对照结束后自动回到CTW等待；GPU2训练闭包保持。旧对照因RGB采样舍入1+float32 epsilon中断，保留失败记录；裁剪输出恢复[0,1]契约后在同一109600快照重跑。

19. 固定Syn109600/R4best connector完整dev对照已完成。直接boundary在预测多截面轮廓上退化；仅此失败显式采用原local_normal并记录策略/原因的boundary_when_supported，经独立真实推理及200图框/分数/顺序同一性核验，得到亮光F1+1.3699pp、暗光+0.7344pp（暗光区间跨0）。候选不替换正式基线。该次计划queue_20261023_connector_supported_then_ctw_gpu3.json已完成connector并进入CTW等待，后续队列见docs04。

20. 用户2026-10-24要求继续connector，官方检测评测与终测后置。Seeing153502的DSConv卷积FP16溢出已原批复现，使用局部conv/GN FP32及纯算子激活重算，通过原批完整更新后，从健康153400显式迁移八项状态至独立dsconv目录，继续原2/100/250预算。connector109600快照也另存数值迁移副本，保持严格加载检查；新none对照逐图框/分数/文字/指标与旧pad1.1完全一致，再测只由crop像素决定的p95曝光。候选效果、当前队列和证据统一见docs04；不改默认基线或使用test挑选。
