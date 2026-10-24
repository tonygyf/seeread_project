# SeeRead：低光场景文本检测与识别

当前主线：**分别复现 Reading 与 Seeing，再验证组合和消融。** 当前状态、论文差距与最新实验统一查看 [04 v5 对齐](docs/04_v5_alignment.md)。

## 文档

| 编号 | 内容 |
|---|---|
| [01 项目设计](docs/01_research_design.md) | 研究顺序、目录结构与开发入口 |
| [02 数据协议](docs/02_dataset_protocol.md) | 数据位置、标注、ignore、划分和评测 |
| [03 历史结论](docs/03_history_and_findings.md) | S1–S4 结果、修复与判断纠偏 |
| [04 v5 对齐](docs/04_v5_alignment.md) | 数据/实现对齐表、实际指标、缺项与运行状态 |

## 开发入口

- `seeread/v5/`、`configs/v5/`、`tools/v5_*.py`：当前独立基线。
- `experiments/autoresearch/v5/`：执行规则、队列计划、锁与结果索引。
- `tools/data/`：通用数据工具；`tools/legacy/`、`configs/legacy/`：历史训练与诊断。
- `tests/v5/`、`tests/legacy/`：当前/历史回归；`tests/test_smoke.py`保留Docker默认入口。
- `data/`：冻结数据；`results/v5/`：当前实验；`results/archive/`：历史结果；`results/logs/`：独立启动日志。
- [weights/reading/](weights/reading/README.md)：选定的R2/R4增量推理权重与加载说明，完整续训检查点保留在服务器。
- [.agents/README.md](.agents/README.md)：项目 skills 来源与调用方式。
- `paper/`：论文草稿，`XX.X` 和 TODO 尚为占位。

## 验证

在现有容器中运行CPU回归，隐藏CUDA以免干扰训练队列：

```bash
docker exec -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH=/workspace -e DSCNET_SOURCE_ROOT=/opt/dscnet -e SELF_DACE_SOURCE_ROOT=/opt/self-dace seeread-v5-run-20261020b python -m pytest -q tests/v5 -k 'not pidfd and not pid_reuse'
python3 -m unittest discover -s tests/v5 -p test_v5_queue.py -k pidfd -k pid_reuse
```

当前训练计划与资源约束见 [04运行入口](docs/04_v5_alignment.md#5-运行与证据入口)。训练期间保持源码、配置和冻结数据不变；新配方另开输出目录。

`pidfd`进程身份测试在宿主机运行；训练分支与模型测试在具备训练依赖的容器内运行。

全量CPU回归可将上面的`tests/v5`替换为`tests`。旧路径查找与本次目录整理见 [01目录导航](docs/01_research_design.md#5-历史路径迁移)。
