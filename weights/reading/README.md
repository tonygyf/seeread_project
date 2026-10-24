# Reading 选定推理权重

只保留已完成基线的增量参数，公共 TrOCR 底座按各 JSON 中的固定配置加载。

| 文件 | 来源 best step | 内容 | 大小 |
|---|---:|---|---:|
| `R2str_lora.pt` | 37400 | TrOCR LoRA | 8.1 MiB |
| `R4str_joint.pt` | 36200 | TrOCR LoRA、Self-DACE 提亮及去噪参数 | 10.8 MiB |

同名 JSON 记录公共初始化配置、完整来源检查点摘要、导出摘要和参数清单。导出时重新构建固定公共模型，加载增量参数后，已逐张量验证识别器与增强器和原 best 检查点完全一致。

在 `Dockerfile.v5` 对应环境中加载；R4 需要配置中的官方 Self-DACE 源码与公共初始化文件：

```python
import json
from pathlib import Path
import torch
from seeread.v5.recognition import build_pipeline

name = "R4str_joint"  # 或 R2str_lora
directory = Path("weights/reading")
info = json.loads((directory / f"{name}.json").read_text())
pipeline = build_pipeline(info["config"])
state = torch.load(directory / f"{name}.pt", map_location="cpu", weights_only=True)
loaded = pipeline.load_state_dict(state, strict=False)
assert not loaded.unexpected_keys
assert not any("lora_" in key or key.startswith("enhancer.") for key in loaded.missing_keys)
pipeline.eval()
# pipeline.predict(crops)：RGB [0,1]，N×3×384×384。
```

这些文件用于推理参数恢复，不包含优化器、采样器或 RNG；续训与现有完整检查点评测仍使用服务器 `results/v5/reading/` 中的完整文件。See 正在训练，其最终选定权重在训练完成后再发布。
