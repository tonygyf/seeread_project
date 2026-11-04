# 04 LATeD STR 独立分支

本分支只记录Read裁剪适配；See运行记录保留在共享主目录的docs/04_v5_alignment.md，原模型不覆盖。

## 独立 LATeD STR 分支（2026-11-04，代码准备）

分支 `exp/read-scene-str`，主目录和See冻结runtime不动。选定现有R4为父模型，先固定权重比较原始外接裁剪与现有connector轮廓校正，再冻结增强器及非LoRA参数、等预算适配。第一步复用既有20张/258行pilot，底层包为200训练图+100开发图；全体划分900/100/500，扩展至900训练图由本分支prefill工具的 `--train-images 900`生成新包，不覆盖旧审核。500张终测暂不导出、不选模型。

与旧connector裁剪域适配的区别：此分支新增真实LATeD人工转写、原始/校正裁剪受控对照、明确识别覆盖率及含裁剪失败的STR分母；尚无新方法收益。LSTR合成暗光与LATeD长行/倾斜/环绕/采样差异需要分层测量，不能预先认定仅因训练图片太清晰。[ASTER](https://ieeexplore.ieee.org/abstract/document/8395027)和[RARE](https://openaccess.thecvf.com/content_cvpr_2016/papers/Shi_Robust_Scene_Text_CVPR_2016_paper.pdf)已有不规则文本校正识别路线，单纯裁剪加微调不作为新颖性主张。

用户审核：OCR识错不删样本。可辨认的困难样本保留；无法确认转写标记unreadable；非拉丁单列。自动工具只标记几乎恒定像素供复核，不自动认定不可读。质量数值记录在原始外接裁剪上，不能用经过增强/放大的锐度给两种方法制造不同分档。训练筛选以后单独消融，dev/test不根据模型是否读对或清晰程度删除已可确认文字。GT轮廓不能唯一确定180°阅读方向，环绕多截面可能导致现有rectifier失败，失败作为结果保留。

执行入口（容器内，从本worktree目录运行；路径均为共享根目录，不另复制权重）：

```bash
# 导入网页导出的人工审核JSON；保留独立sidecar
python tools/data/finalize_lated.py --package /workspace/data/derived/v5/lated_mvp_20261026 --review /workspace/REVIEW.json --output /workspace/data/annotations/v5/scene_str_review.json
# 未完成pilot全部非ignore状态复核时会阻止导出
python tools/v5_scene_str.py export --package /workspace/data/derived/v5/lated_mvp_20261026 --sidecar /workspace/data/annotations/v5/scene_str_review.json --pilot --method axis_aligned --output /workspace/data/annotations/v5/scene_str_a
python tools/v5_scene_str.py export --package /workspace/data/derived/v5/lated_mvp_20261026 --sidecar /workspace/data/annotations/v5/scene_str_review.json --pilot --method boundary_when_supported --output /workspace/data/annotations/v5/scene_str_b
# 两种裁剪的训练ID取共同集，dev仍保留各自全部catalog及失败记录
python tools/v5_scene_str.py pair-train --a /workspace/data/annotations/v5/scene_str_a --b /workspace/data/annotations/v5/scene_str_b --output /workspace/data/annotations/v5/scene_str_paired
# 固定父模型A臂；B臂用对应目录，输出必须独立。max-len先核验，不能截断标签。
CUDA_VISIBLE_DEVICES=0 python tools/v5_adapt_reading.py --mode evaluate --parent /workspace/results/v5/reading/R4str_joint/best.pt --manifests /workspace/data/annotations/v5/scene_str_paired/a --data-root /workspace/data --max-len 128 --output /workspace/results/v5/reading/scene_str_20261104/a_fixed
python tools/v5_scene_str.py score --catalog /workspace/data/annotations/v5/scene_str_paired/a/dev.catalog.jsonl --predictions /workspace/results/v5/reading/scene_str_20261104/a_fixed/dev.records.jsonl --output /workspace/results/v5/reading/scene_str_20261104/a_fixed/all_eligible_score.json
```

真实转写就绪才启动训练：每臂从相同R4独立加载、新优化器，batch4/lr2e-5/seed0/maxlen一致；先20分钟资源检查，再相同5轮，完整dev raw CER改善至少1pp后考虑10轮。训练器内部best仍按成功裁剪dev选择，同一裁剪方法内分母固定；跨裁剪方法必须用 `score` 的全eligible统计。配对比较以原图分组，失败裁剪补空预测、same-ID匹配，不直接用只含成功裁剪的两个CER相减。GT指标达标后，再固定See评预测裁剪和整图spotting。

计划登记：`experiments/autoresearch/v5/scene_str_plan.json`；使用本项目autoresearch固定目标/预算/keep-discard规则，不启动Karpathy的语言模型训练器。此轮仅CPU准备与测试，没有训练、没有生成500张OCR伪真值、没有push。


本次准备验证：900/100/500源图共8201/914/4808实例，原始标注可用转写均为0；精确文件内容跨划分检查通过。现有300图审核包3041行，其中260行近恒定像素候选，20图pilot的258行中42行被提示；所有行仍保留。R4父检查点健康标志及基线源码摘要匹配。CPU测试覆盖人工sidecar→两种真实裁剪导出→同训练ID配对、失败裁剪计分和数据隔离；Read/See训练均未在本分支启动。
