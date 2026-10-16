# data/raw — 原始官方数据集（未转换）

每个数据集独立子目录，统一由 `data/DATASETS.md` 记录来源、状态与大小。
转换器（待补：docs/01 §6 状态说明"需要你补"）将其转为 `data/annotations/*.jsonl` 并按协议平铺到 `data/images`、`data/images_lit`。

| 子目录 | 数据集 | 用途（docs/01 §4） | 获取方式 |
|---|---|---|---|
| LATeD/ | LATeD 1,500 张（暗图+亮图+多边形标注） | 主基准 LATeD-Spot（train 900/val 100/test 500） | 本地复制自 DarkText-private |
| SynthText/ | SynthText 80万合成场景文本 | Dark-SynthText 在线暗化预训练源 | 容器下载 |
| MLT17/ | ICDAR2017 MLT 多语种 | Dark-MLT17 在线暗化预训练源 | 容器下载 |
| CTW1500/ | CTW1500 曲线文本 | Dark-CTW1500 固定种子三档合成暗化测试 + 检测协议 | 容器下载 |
| TotalText/ | Total-Text 任意形状文本 | Dark-Total-Text 三档合成暗化测试 + E2E 不退化验证 | 容器下载 |
| ICDAR2015/ | ICDAR2015 四点标注 | 亮光数据不退化验证 | 容器下载 |
| ESTR/ | ESTR 60 裁剪（Reading 原版） | 与 Reading 对齐的识别测试 | 容器下载（HuggingFace） |
