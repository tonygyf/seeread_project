# data/annotations/ — 协议 JSONL 占位（待生成）

训练/评测直接读取的最终数据形态：一行一图，schema 见 `docs/02_dataset_protocol.md` §1。
`poly` 为 CTW1500 约定（顶边左→右，底边右→左）；`text="###"` 或 `ignore=true` 为 don't-care。

## 约定文件名（与 `configs/base.yaml:3-5` 一一对应）

| 文件 | 内容 | 状态 |
|---|---|---|
| `latedspot_train.jsonl` | LATeD-Spot train 900（官方 1000 去掉 val，场景×光照分层，固定种子） | 待生成 |
| `latedspot_val.jsonl` | LATeD-Spot val 100（从官方 train 抽样；**只用于选超参/早停**） | 待生成 |
| `latedspot_test.jsonl` | LATeD 官方 test 500（**只在定稿时评一次**，见 docs/03 P0-06） | 待生成 |
| `ctw1500_test.jsonl` / `totaltext_test.jsonl` | 亮图测试集，同时是 `make_synth_dark.py` 的合成源 | 待生成 |
| `mlt17_train.jsonl` / `synthtext_train.jsonl` | 预训练（`illum_kind=lit`，在线暗化） | 待生成 |
| `splits.json` | 冻结的划分清单（图像/场景 ID + 哈希，随数据发布） | 待生成 |

## 生成链路

```
data/raw/<name>/ 原始标注 ──▶ 转换器(待补, docs/03 P0-02) ──▶ data/raw/<name>.jsonl
  ──▶ python tools/illum_stats.py --ann data/raw/<name>.jsonl --root data --out data/annotations/<name>.jsonl
      （写入 image 级 meta.illum_y/illum_bin + 实例级 bin，并打印分位数用于校准 BIN_EDGES）
```

## 注意

- `image` / `lit` 字段必须是相对 `data/`（`data.root`）的可解析路径：`seeread/data/datasets.py` 用 `os.path.join(root, image)` 读图。
- 三个 `latedspot_*.jsonl` 与图像未就位前训练无法启动；文档中的 900/100/500 是协议数字，不是已生成数据（docs/03 P0-01）。
- test 划分冻结后不得参与任何模型/阈值选择。
