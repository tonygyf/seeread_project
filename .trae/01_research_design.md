# SeeRead：把 *Seeing*（检测）与 *Reading*（识别）合成低光场景文本端到端 Spotting

> 工作名：**SeeRead**（See = 老师的低光检测，Read = Fu et al. 的低光识别）。
> 本文档 = 研究设计书。代码骨架见 `seeread/`，论文骨架见 `paper/`。

---

## 0. 先说三件必须想清楚的事（决定能不能上 A 会）

1. **"把两篇论文拼起来"不是贡献，"为什么必须拼 + 拼出了什么新结论"才是。**
   检测 → 增强 → 识别的级联，错误会累积；两篇论文各自只在自己的子任务上说话（检测 F1 / 识别 CER），没有人回答：
   *在黑暗中，端到端文本 spotting 的错误到底来自哪里、检测与识别如何共享"光照信息"？*
   这就是你的故事主线（见 §2）。
2. **据我两次检索（不穷尽），没有看到公开的"低光场景文本 spotting（检测+识别端到端）"基准与方法**；
   Seeing 是检测、Reading 是识别，Hsu/Lin 是增强+检测。**投稿前必须做系统文献检索**（Google Scholar、arXiv、CVPR/ICCV/ECCV/ACM MM/AAAI 2025–2026），再决定是否写 "first"。
3. **LATeD 目前只有多边形标注，没有文本转写**（Seeing 论文第 4 节只提到 line-level polygon）。
   做 spotting 必须给 13,923 个文本实例补转写 → 这本身就是一个**数据集贡献**（LATeD-Spot），也是最大的工作量与最大的护城河。
   另外 Seeing 说 "code and dataset will be released"——**先向老师确认代码/数据的可用性与署名、使用许可**。

> 关于会议档次：IEEE 模板（IEEEtran）对应的是 IEEE 期刊（TIP / TPAMI / TMM / TCSVT）。CVPR/ICCV/ECCV、ACM MM、AAAI 各有自己的模板（`cvpr.sty` / `acmart` / `aaai`）。
> 按我的印象 CCF 目录里 TIP、TPAMI、IJCV 为 A，TMM 为 B——**以最新 CCF 目录与各会议官网截止日期为准**。
> `paper/main.tex` 的章节是模板无关的，换模板只需改 preamble 和作者块。

---

## 1. 在动手之前：两篇论文里需要你先复现/核对的问题

不是挑老师的错，是"站在别人肩上之前先量一下台阶"。**这些问题审稿人也会发现。**

| # | 来源 | 现象 | 建议 |
|---|------|------|------|
| 1 | Seeing | LATeD 上 "Ours" 在 Table 2 为 P/R/F1 = 82.6/57.0/67.1，Table 4 消融完整模型为 82.8/55.4/66.4；CTW1500 上 Table 3 为 P/R/F1 = 88.7/83.9/86.2，Table 4 完整模型 P 86.2、F1 84.8。 | 以**自己复现的数字**为基线；问老师是哪个 checkpoint / 哪次 run。 |
| 2 | Seeing | 公式 (6) 的 FPS 定义含糊（"离文本中心区域 T 中任意点最远"）。 | 对照官方代码；我实现的是标准 FPS（离已选集合最远）。 |
| 3 | Seeing | DSC 引用写为 CVPR'23；我印象中 Dynamic Snake Convolution 是 ICCV 2023。 | 核对后再引用（`refs.bib` 里按 ICCV 写，请你核实）。 |
| 4 | Reading | 所有方法在 LSTR 上 CER ≈ 50–53%；"冻结" 与 "全量微调" 的 CER 同为 53.16。 | 微调可能没有真正更新/没训够；按其公式 `(k·I)^γ`、γ=5、k≈0.3 估算，满幅信号 ≈ 0.0024，而 AWGN σ 可达 0.02–0.05 → **信号低于噪声底**（我的推算，未验证），说明合成流程可能过于苛刻。→ 我提供了 `physical` 模式（泊松-高斯噪声，光学→传感器→ISP 的顺序）。 |
| 5 | Reading | ESTR 仅 60 张、无训练划分；Table 4 的文字写 "without additional training" 但又写 "jointly fine-tuning SRDACE on ESTR"；RLLIE 描述为 "without backpropagating through TrOCR" 又说端到端联合训练；RLLIE 在 LSTR 上 CER 50.43 劣于 SRDACE 联合训练的 50.02，且没有 IBL/PRT 消融。 | "物理渲染先验"的收益**没有被证明**。SeeRead **不依赖**该叙事（IAE 是曲线增强 + 检测器光照上下文）。 |
| 6 | Reading | TrOCR 的分词器是英文；LATeD 以中英文为主。 | 多语言用 CRNN/SVTR/PARSeq 类识别器；TrOCR 只报英文子集（代码已留接口）。 |

---

## 2. 研究问题、假设、贡献

### 2.1 研究问题（每个都要有"可证伪的预测"）

- **RQ1 错误传播**：在黑暗中，*增强→检测→识别* 级联的失败有多少来自"漏检"，多少来自"检到但读错"？增强是否在把一类错误换成另一类？
  - 预测：LLIE 提高检测召回但识别 CER 不降甚至升（呼应 Reading §4.3 的"亮不等于好认"）。
  - 证伪：若 LLIE 级联在所有光照档位均全面优于端到端，则故事不成立。
- **RQ2 光照上下文**：裁剪后的文本小图丢失了"场景有多黑"的信息；把检测器的**全图光照上下文**喂给文本增强器，是否优于只看裁剪图的增强器？
  - 预测：越暗（extreme 档）增益越大。
- **RQ3 跨光照一致性**：训练期用（暗, 亮）配对约束识别特征（CIC，推理零成本），能否迁移到**没有亮图**的真实暗图？
- **RQ4 "多暗算太暗"**：以光照档位为横轴画 E2E 曲线，是否存在物理下限（信噪比不足、文本不可恢复）？这直接延伸 Reading 的 "How bright is bright enough?"。

### 2.2 贡献（论文里写这四条）

1. **LATeD-Spot + ESTR-v2 基准**：为 LATeD 补文本转写；新采集带"手电/路灯亮帧"配对的真实夜景；固定种子的合成暗化测试集；**光照分档的评测协议**。
2. **SeeRead 方法**：检测（SCM+DSF+TSR，沿用 Seeing）与识别通过 ① 共享光照上下文（IAE）、② 中心线可微矫正（CSR，识别损失回传到高度头）、③ 训练期跨光照一致性（CIC）耦合；推理时 SCM/CIC 不引入开销。
3. **系统性分析**：级联 vs. 端到端、光照档位曲线、错误归因（漏检 / 读错 / 正确）。
4. **复现包**：代码、固定划分、评测脚本、3 随机种子 mean±std。

> 备选方案（如果方法增益不够）：以**基准 + 分析**为主打（NeurIPS Datasets & Benchmarks / ACM MM 数据集方向），方法作为强基线。

---

## 3. 方法设计（对应 `seeread/models/`）

```
            ┌───────────────────────── 训练期专用 ─────────────────────────┐
            │  SCM: L_sr(重建文本位置掩码) + L_ss(语义对齐)                  │
 dark img ─►│ ResNet50 ─► DSF (Conv ∥ SnakeConv → gated attn) ─► heads      │
            │     │                       text / center / height / (cos,sin)│
            │     └─ c5 ─► illum_context z ──────────────┐                  │
            └───────────────────────────────────────────┼──────────────────┘
 TSR(FPS+旋转矩形累加) ─► 中心线点 + 高度 ─► CSR 可微条带矫正 ─► crop(64×256)
                                                         │
                                         IAE(crop; z)  ◄─┘  FiLM 条件化的曲线增强
                                                         │
                                  Recognizer (CRNN-CTC / TrOCR+LoRA) ─► text
 训练期:  CIC:  1 − cos( enc(IAE(dark crop)),  sg[ enc(lit crop) ] )
```

| 模块 | 文件 | 说明 / 与原论文差异 |
|---|---|---|
| DSF | `models/detector.py` | 并行 Conv 与蛇形卷积，门控自注意力。**差异**：`SnakeConvLite`（用 `deform_conv2d` + 累积偏移实现，非官方 DSConv）；自注意力改为**通道注意力**（全图 token 注意力在 1/4 分辨率太贵）；加残差；方向用 (cos,sin)。均需消融或换回官方实现。 |
| SCM | 同上 | 仅训练；`L_sr`=L1(重建掩码, GT 文本掩码)，`L_ss`=L2(主干特征, 辅助语义特征.detach())。detach 方向是我的选择，需消融。 |
| TSR | `models/tsr.py` | 连通域 → FPS 选中心 → 固定宽度旋转矩形累加 → 形态学闭运算 → 轮廓。 |
| IAE | `models/enhancer.py` | AAC 曲线（Reading Eq.7）× 6 次迭代，参数由 FiLM-ResNet 预测，**条件为检测器的光照上下文 z**。`use_ctx=false` 即 RQ2 的对照。 |
| CSR | `models/rectify.py` | 中心线点（不可微）+ 高度（从检测器高度图双线性采样，**可微**）→ 条带矫正，弯曲文本友好。训练时用 `pred_geom_prob` 从 GT 高度逐步过渡到预测高度（课程）。 |
| CIC | `losses.py` | 仅在有亮图配对（合成或真实手电帧）时启用。 |
| Recognizer | `models/recognizer.py` | `CRNNRecognizer`（多语言、无需下载）与 `TrOCRRecognizer`（HF+peft LoRA，英文）。 |

**损失**：`L = λ_det·(L_seg + L_H + L_dir + L_sr + L_ss) + λ_rec·L_rec + λ_enh·L_enh + λ_cic·L_cic`，默认 λ = (1, 1, 0.1, 0.5)，在验证集上扫。

**训练三阶段**（`train.stage`）：`det`（检测器，SynthText+MLT 暗化预训练→LATeD-Spot 微调）→ `rec`（冻结检测器，用 GT 几何训练 IAE+识别器）→ `joint`（全部解冻，几何由 GT 渐进切到预测）。

---

## 4. 数据：训练集 / 验证集 / 测试集（务必先定死再跑实验）

### 4.1 真实数据（核心）
| 名称 | 来源 | 规模 | 标注 | 用途 |
|---|---|---|---|---|
| **LATeD-Spot** | Seeing 的 LATeD + **新增转写** | 1,500 张（官方 1,000/500），13,923 实例 | 多边形 + 转写 + 语种 | train 900 / **val 100（从训练集按场景×光照分层抽，固定种子）** / **test 500（官方，绝不碰）** |
| **ESTR-v2**（新采集） | 夜间街景，建议 300–500 场景 | — | 多边形 + 转写 | **只做测试**（跨域泛化）；其中 ≥200 场景为**三脚架"暗帧 + 手电/补光帧"配对**（参考 Reading 的 Fig.1 手电思路），用于 CIC 与上限分析 |
| ESTR（Reading 原版） | HuggingFace | 60 | 裁剪级 | 与 Reading 对齐的识别测试 |

### 4.2 合成数据（只能辅助，不能当主结论）
| 名称 | 构造 | 用途 |
|---|---|---|
| Dark-SynthText / Dark-MLT17 | **在线**暗化（`SeeReadDataset`，`illum_kind=lit`） | 检测器/识别器预训练，自带亮图配对 → CIC |
| LSTR（Reading） | 11,273 / 6,588 裁剪 | 识别器训练与对齐实验 |
| **Dark-CTW1500 / Dark-Total-Text 测试集** | `tools/make_synth_dark.py`，**固定种子，三档（mild/dark/extreme）** | 受控的光照分档评测、与亮图逐像素配对、"无退化"回归测试 |
| 亮光数据 | CTW1500 / Total-Text / ICDAR2015 | 验证**不退化**（Seeing 已做检测，这里补 E2E） |

> ⚠ 合成测试集必须**离线生成并冻结**；训练用在线随机。真实数据的光照分布用 `tools/illum_stats.py` 统计后，**反过来校准** `PHYS_LEVELS` 与 `BIN_EDGES`，让合成分布贴近真实分布。
> ⚠ 隐私：公开场所图像也要对人脸/车牌做模糊或剔除，数据发布前核对许可。

### 4.3 标注流程（LATeD-Spot 补转写）
预标注（对**增强后**的图跑一个多语种 OCR 得草稿）→ 双人独立校对 → 第三人仲裁 → 报告一致率（字符级 κ/NED）。语种标记 `en/zh/ja/ko/vi/ar`；不可辨认 → `###`（计入 don't-care）。详见 `02_dataset_protocol.md`。

---

## 5. 评测协议与"比拼能力"

**指标**（`metrics/spotting.py`）
- 检测：P / R / F（IoU≥0.5，CTW1500 协议）。
- 识别：在匹配上的实例上算词准确率与 CER（归一化：去标点、不区分大小写）。
- **E2E**：匹配 **且** 文本一致（无词典）；英文子集可另报 Strong/Weak/Generic 词典。
- **光照分档**：实例级亮度（多边形外接框内均值 Y）→ 档位 `extreme/dark/dim/lit`；分档报**召回型**指标（精确率无法分档）。
- **错误归因**：`frac_missed`（漏检）/ `frac_misread`（检到读错）/ 正确 —— 回答 RQ1。
- **效率**：参数量、FPS（同一 GPU、batch=1、含后处理）、显存。
- **统计**：3 个随机种子 mean±std；对 E2E-F 做 bootstrap 95% CI；与最强基线做配对检验。

**对比实验矩阵**（`scripts/run_baselines.sh`）

| 组 | 方法 | 目的 |
|---|---|---|
| A 检测上界 | 用 GT 多边形 + 各识别器 | 隔离识别难度 |
| B 级联 | {无, gamma, CLAHE, SSR, ZeroDCE, SCI, RUAS, SRDACE, GDP} → {Seeing 检测器 / BPN++ / DPText} → {PARSeq / CRNN / TrOCR(LoRA)} | RQ1：增强能不能救 |
| C 端到端 spotter（在暗图上微调） | DeepSolo、TESTR、ESTextSpotter、ABCNet v2（官方实现，需核对可用性） | 与强通用 spotter 比 |
| D Ours | SeeRead | 主结果 |
| E 数据设定 | 仅合成 / 仅真实 / 合成+真实 | sim2real |

**消融**（每项 3 种子）
`±IAE上下文 (use_ctx)`、`±CSR可微高度 (pred_geom_prob)`、`±CIC`、`±SCM`、`±DSF`、`±TSR`、`LoRA/冻结/全微调`、`SCM 的 detach 方向`、`SnakeConvLite vs 官方 DSConv`、训练数据组成、识别器类型。

**必须有的图表**：① E2E-F vs 光照档位曲线（所有方法同图）；② 错误归因堆叠柱；③ 定性对比（含失败案例）；④ 精度-速度散点；⑤ IAE 曲线参数可视化（α/β 图）。

---

## 6. 代码结构

```
seeread_project/
├── configs/            base.yaml + 各消融 yaml（_base_ 继承，--set 覆盖）
├── seeread/
│   ├── config.py
│   ├── losses.py                 检测损失 / SCM / CIC
│   ├── data/   lowlight_synth.py geometry.py datasets.py
│   ├── models/ detector.py tsr.py enhancer.py rectify.py recognizer.py seeread.py
│   ├── metrics/spotting.py       检测/识别/E2E + 光照分档 + 错误归因
│   └── baselines/llie_registry.py
├── tools/  train.py eval.py make_synth_dark.py illum_stats.py align_pairs.py
├── scripts/run_baselines.sh
├── tests/test_smoke.py
├── docs/   01_research_design.md  02_dataset_protocol.md
└── paper/  main.tex  refs.bib   (IEEEtran)
```

状态说明（诚实版，截至本次交付）：
- **已实际运行验证**（沙箱无 GPU/PyTorch，仅验证纯 numpy/cv2/shapely 部分）：低光合成（三档亮度单调递减）、几何与 GT 图生成、TSR 解码与阅读方向排序、评测器（匹配/E2E/分档/错误归因）。
- **仅通过语法编译（`py_compile`），未执行**：全部 PyTorch 代码——检测器、IAE、CSR、识别器、联合训练前向/反向、`train.py`/`eval.py`。`tests/test_smoke.py` 已写好（含"识别损失梯度能到达检测器 ctx 分支""CSR 对高度可微"等断言），**请你第一件事就是在本机跑 `python tests/test_smoke.py`，并修掉可能出现的 shape/设备问题**。
- **未经验证**：`TrOCRRecognizer`（需联网下载权重）；`SnakeConvLite` 与官方 DSConv 的差异；任何实验数字（论文里全是 `XX.X` 占位）。
- **合成暗化需要校准**：`physical` 模式在一张随机测试图上给出平均亮度约 0.11 / 0.06 / 0.04（mild/dark/extreme），其中 extreme 受噪声下限抬升；真实 LATeD 的亮度分布出来后必须重调 `PHYS_LEVELS` 与 `BIN_EDGES`。
- **需要你补**：CTW1500/Total-Text/MLT/SynthText → jsonl 转换器；外部 LLIE 与 spotter 的适配；LATeD-Spot 转写；`eval_oracle.py`（GT 框 + 识别）。

---

## 7. 时间表（约 6 个月，按"先保底"排）

| 周 | 里程碑 | 产出 / 门槛 |
|---|---|---|
| 1–2 | 文献检索 + 向老师确认代码/数据；复现 Seeing 的 LATeD 检测数字 | 复现 F1 与论文差 ≤1 点，否则先查原因 |
| 2–6 | LATeD 转写标注（可与下面并行）；`illum_stats` 校准合成 | LATeD-Spot v1 + 一致率报告 |
| 4–8 | 新采集 ESTR-v2（含配对） | ≥200 配对场景 |
| 6–10 | 级联基线 + 通用 spotter 基线 | 矩阵 A/B/C 全表 |
| 8–14 | SeeRead 三阶段训练 + 消融 | 主表 + 消融表（3 种子） |
| 14–18 | 分析图、失败案例、效率 | 图 1–5 |
| 18–22 | 写作、内部评审（找没参与的人读） | 完整初稿 |
| 22–24 | 补实验、投稿 | 与目标截止日期倒推 |

**里程碑决策点**：第 10 周若 SeeRead 相对"最强级联 + 微调"的 E2E-F 没有 ≥2–3 个点的稳定优势（3 种子），转向"基准 + 分析"论文。

---

## 8. 风险与对策

| 风险 | 对策 |
|---|---|
| 审稿人说"只是工程拼装" | 叙事围绕 RQ1–RQ4 的**结论**，每个模块都有消融证明；强调光照上下文这一"桥" |
| 通用 spotter（DeepSolo 等）在暗图上微调后已很强 | 报告它们的分档曲线；我们的优势应出现在 extreme 档与跨域 ESTR-v2 |
| 合成与真实分布不一致 | 以真实测试集为主结论；合成只作受控分析 |
| 标注工作量 | 预标注 + 双人校对；先做 500 测试图保证基准可用 |
| 老师的代码未释放/不可用 | 用本骨架的 DSF/SCM/TSR 重实现，并在报告中写明与原文差异 |
| 检测 F1 低导致 E2E 被检测拖垮 | 错误归因表本身就是结论；同时给 "GT 框 + 识别" 上界 |
| 作者与署名 | 提前与老师约定：数据集共同署名、代码开源范围 |

---

## 9. 快速开始

```bash
pip install -r requirements.txt
python tests/test_smoke.py                      # CPU 冒烟测试
# 1) 把你的标注转成 jsonl（格式见 02_dataset_protocol.md），再统计光照
python tools/illum_stats.py --ann data/raw/latedspot_train.jsonl --root data --out data/annotations/latedspot_train.jsonl
# 2) 三阶段训练
python tools/train.py --config configs/base.yaml --set train.stage=det  train.out_dir=results/s1_det
python tools/train.py --config configs/base.yaml --set train.stage=rec  train.init_det=results/s1_det/last.pt train.out_dir=results/s2_rec
python tools/train.py --config configs/seeread_joint.yaml --set train.init_det=results/s2_rec/last.pt train.init_rec=results/s2_rec/last.pt
# 3) 评测
python tools/eval.py --config configs/seeread_joint.yaml --ckpt results/seeread_joint/last.pt --ann data/annotations/latedspot_test.jsonl --out results/seeread_joint/test.json
```
