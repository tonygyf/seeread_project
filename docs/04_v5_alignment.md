# 04 LATeD STR 独立分支

本分支只记录Read裁剪适配；See运行记录保留在共享主目录的docs/04_v5_alignment.md，原模型不覆盖。

## 独立 LATeD STR 分支（2026-11-04，代码准备）

分支 `exp/read-scene-str`，主目录和See冻结runtime不动。选定现有R4为父模型，先固定权重比较原始外接裁剪与现有connector轮廓校正，再冻结增强器及非LoRA参数、等预算适配。第一步在新的900训练图+100开发图完整包中复用既有20张/258行pilot；原200+100包保留为审核迁移来源。全体划分900/100/500，新增包由本分支prefill工具的 `--train-images 900`生成，不覆盖旧审核。500张终测暂不导出、不选模型。

与旧connector裁剪域适配的区别：此分支新增真实LATeD人工转写、原始/校正裁剪受控对照、明确识别覆盖率及含裁剪失败的STR分母；尚无新方法收益。LSTR合成暗光与LATeD长行/倾斜/环绕/采样差异需要分层测量，不能预先认定仅因训练图片太清晰。[ASTER](https://ieeexplore.ieee.org/abstract/document/8395027)和[RARE](https://openaccess.thecvf.com/content_cvpr_2016/papers/Shi_Robust_Scene_Text_CVPR_2016_paper.pdf)已有不规则文本校正识别路线，单纯裁剪加微调不作为新颖性主张。

用户审核：OCR识错不删样本。可辨认的困难样本保留；无法确认转写标记unreadable；非拉丁单列。自动工具只标记几乎恒定像素供复核，不自动认定不可读。质量数值记录在原始外接裁剪上，不能用经过增强/放大的锐度给两种方法制造不同分档。训练筛选以后单独消融，dev/test不根据模型是否读对或清晰程度删除已可确认文字。GT轮廓不能唯一确定180°阅读方向，环绕多截面可能导致现有rectifier失败，失败作为结果保留。

执行入口（容器内，从本worktree目录运行；路径均为共享根目录，不另复制权重）：

```bash
# 导入网页导出的人工审核JSON；保留独立sidecar
python tools/data/finalize_lated.py --package /workspace/data/derived/v5/lated_str_1000_20261105 --review /workspace/REVIEW.json --output /workspace/data/annotations/v5/scene_str_review.json
# 未完成pilot全部非ignore状态复核时会阻止导出
python tools/v5_scene_str.py export --package /workspace/data/derived/v5/lated_str_1000_20261105 --sidecar /workspace/data/annotations/v5/scene_str_review.json --pilot --method axis_aligned --output /workspace/data/annotations/v5/scene_str_a
python tools/v5_scene_str.py export --package /workspace/data/derived/v5/lated_str_1000_20261105 --sidecar /workspace/data/annotations/v5/scene_str_review.json --pilot --method boundary_when_supported --output /workspace/data/annotations/v5/scene_str_b
# 两种裁剪的训练ID取共同集，dev仍保留各自全部catalog及失败记录
python tools/v5_scene_str.py pair-train --a /workspace/data/annotations/v5/scene_str_a --b /workspace/data/annotations/v5/scene_str_b --output /workspace/data/annotations/v5/scene_str_paired
# 固定父模型A臂；B臂用对应目录，输出必须独立。max-len先核验，不能截断标签。
CUDA_VISIBLE_DEVICES=0 python tools/v5_adapt_reading.py --mode evaluate --parent /workspace/results/v5/reading/R4str_joint/best.pt --manifests /workspace/data/annotations/v5/scene_str_paired/a --data-root /workspace/data --max-len 128 --output /workspace/results/v5/reading/scene_str_20261104/a_fixed
python tools/v5_scene_str.py score --catalog /workspace/data/annotations/v5/scene_str_paired/a/dev.catalog.jsonl --predictions /workspace/results/v5/reading/scene_str_20261104/a_fixed/dev.records.jsonl --output /workspace/results/v5/reading/scene_str_20261104/a_fixed/all_eligible_score.json
```

真实转写就绪才启动训练：每臂从相同R4独立加载、新优化器，batch4/lr2e-5/seed0/maxlen一致；先20分钟资源检查，再相同5轮，完整dev raw CER改善至少1pp后考虑10轮。训练器内部best仍按成功裁剪dev选择，同一裁剪方法内分母固定；跨裁剪方法必须用 `score` 的全eligible统计。配对比较以原图分组，失败裁剪补空预测、same-ID匹配，不直接用只含成功裁剪的两个CER相减。GT指标达标后，再固定See评预测裁剪和整图spotting。

计划登记：`experiments/autoresearch/v5/scene_str_plan.json`；使用本项目autoresearch固定目标/预算/keep-discard规则，不启动Karpathy的语言模型训练器。此轮仅CPU准备与测试，没有训练、没有生成500张OCR伪真值、没有push。


本次准备验证：900/100/500源图共8201/914/4808实例，原始标注可用转写均为0；精确文件内容跨划分检查通过。现有300图审核包3041行，其中260行近恒定像素候选，20图pilot的258行中42行被提示；所有行仍保留。R4父检查点健康标志及基线源码摘要匹配。CPU测试覆盖人工sidecar→两种真实裁剪导出→同训练ID配对、失败裁剪计分和数据隔离；Read/See训练均未在本分支启动。


## 1000张完整审核包与PaddleOCR（2026-11-05）

用户要求快速为全部1000张生成草稿，沿用数据集GT区域、既有裁剪和审核页。新包 `data/derived/v5/lated_str_1000_20261105` 覆盖900训练图+100开发图、9115行；与官方1000训练图的并集以及所有ID/多边形逐项匹配，500张test未进入。旧300图包及其人工审核保持，旧增强资源以只读用途的链接复用。

使用本地固定PP-OCRv5_server_rec（PaddleOCR3.4.0/Paddle3.2.2，识别模型revision b26c3587fda8da3c8ec0ce357214b4d661ff1558），在独立受限容器的物理GPU2上只做识别。候选来自GT外接裁剪的原始像素、数据集提供的同坐标亮版及复用connector校正亮版；原图/亮版各四个直角方向，校正亮版两个方向。该亮版是转写辅助，不成为Read暗光推理的输入假设。校正不支持的形状保留错误原因，外接裁剪候选仍存在；全部分数和空结果留存，不设置信度阈值、不自动确认语言或文字。

完整入口：[1000图review](http://100.88.81.10:8765/full/review.html)；原[20图pilot](http://100.88.81.10:8765/pilot.html)保持。[新包20图pilot](http://100.88.81.10:8765/full/pilot.html)包含同批试标及新Paddle候选。新包逐ID验证旧任务的原图摘要/多边形/划分后，记录兼容协议；相同浏览器和Tailscale源地址自动补入旧本地审核，不覆盖新审核。可导入已验证旧包JSON补缺，也可按置信度双向排序；候选采用后仍为pending。浏览器测试已验证迁移、全部9115行保留、排序及候选采用，未触碰用户浏览器存储。

机器产物：`paddle_drafts.jsonl`（每行全部候选与原始分数）、`paddle_drafts_provenance.json`（模型与任务摘要、覆盖和进度）、`paddle_human_views.jsonl`（额外人眼亮版）。人工结果仍从页面导出JSON，再用原finalize工具生成独立sidecar；OCR原始输出不是真值。置信度未在LATeD校准，只帮助审核排序。


11月5日完整OCR预填验收：1000图、9115行全部覆盖，共91020候选，约19分钟完成，置信度0至0.999984全部保留。19行最佳草稿为空；65行校正失败仍有8个外接裁剪候选，其余每行10候选。不自动确认任何人工转写。完整页面已包含每行候选，GPU2 OCR任务成功退出并释放，GPU3 See训练继续。[覆盖核对报告](/home/gyf/seeread_project/results/v5/diagnostics/lated_1000_completion_audit_20261105.json)。


## 378条人工审核的首次固定Read诊断（2026-11-05）

原页面仅localStorage，现同源8765服务新增服务器快照同步，浏览器仍保留本地数据；用户刷新后已收到gyf的275 reviewed+103 unreadable及1 pending。271条可用拉丁转写全部来自32张train原图；3条non_latin单列，1条reviewed未选语言留在原备份、暂缓计分。生成的sidecar和本次测试冻结在 `data/annotations/v5/scene_str_reviewed_20261105/`，不覆盖用户原记录。备份保留最近20个内容快照，页面导出仍可用。

固定未做LATeD适配的R4、相同128-token解码上限；本批真值最长24 tokenizer tokens，无标签截断。原始外接裁剪：271行、2890字符、1995编辑、84行全对，CER69.0311%、整行准确率30.9963%。现有轮廓校正：269行成功、2行失败作空预测，仍以271行计分；2441编辑、74行全对，CER84.4637%、准确率27.3063%。CER差B−A=+15.4325pp；按原图bootstrap95%区间约[−2.7322,+40.2581]pp，样本不足以宣称稳定总体退化。未更新权重。报告 `results/v5/reading/scene_str_reviewed_test_20261105/run.json`，配对及长度分层见同目录 `paired_diagnostic.json`；[逐行图片/人工文字/两种预测对照](http://100.88.81.10:8765/full/read_test/comparison.html)。

原始裁剪分层：≤8字符138行CER33.5252%；9–16字符96行CER57.7679%；>16字符37行CER103.7209%，插入错误可令CER超过100%。这只能说明该人工选择子集长文本较难，不能单独归因为裁剪、低光或长度。当前全部来自train分配，但R4尚未用LATeD训练；属于目标域的冻结模型诊断，不是已适配模型训练成绩，更不是dev/test最终结论。

Read定位是一个已裁出的文字实例→一个字符串，可为单词或单行多词；不是整图多区域检测/阅读顺序模块。弯曲、倾斜、环绕在语义上仍可是一行，当前模型和校正器尚未证明稳健。多行段落需额外行分割/排序协议。当前R4父模型为TrOCR-base-STR，LSTR训练10146/dev1127，LSTR test6588，CER49.5191%；ESTR60留作最终真实低光评测。新LATeD分支900/100/500、拉丁文字范围，现仅有train人工子集。父配置max_len64 tokens，测试128，模型输入统一384×384；长度、分辨率和数据域均需控制。[TrOCR官方用途和预训练来源](https://huggingface.co/microsoft/trocr-base-str)。
