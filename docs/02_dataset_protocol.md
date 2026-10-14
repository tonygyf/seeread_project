# 数据集协议（LATeD-Spot / ESTR-v2 / Dark-Synth）

## 1. jsonl 格式（每行一张图）
```json
{"image": "images/0001.jpg",
 "lit": "images_lit/0001.jpg",
 "instances": [
   {"poly": [[x,y], ...], "text": "STOP", "lang": "en", "ignore": false, "bin": "dark"}
 ],
 "meta": {"illum_kind": "real|synth|lit", "illum_y": 0.05, "illum_bin": "dark", "scene": "street", "device": "phone-A", "iso": 3200}}
```
- `poly`：**CTW1500 约定**——顶边从左到右，再底边从右到左（偶数个点，≥4）。竖排文本需要额外约定（当前代码按左→右阅读，竖排属已知限制）。
- `illum_kind=lit` 表示**亮光源图，训练时在线暗化**；`real` 为真实暗图；`synth` 为离线固定种子的合成暗图（带 `lit` 配对路径）。
- `text="###"` 或 `ignore=true` 为 don't-care，不计入召回，命中它的预测被吸收。
- `bin` 由 `tools/illum_stats.py` 写入（实例级亮度分档）。

## 2. 光照分档
默认 `extreme<0.04 ≤ dark<0.08 ≤ dim<0.15 ≤ lit`（平均亮度 Y，Rec.709）。**这是占位值**：用真实 LATeD / ESTR-v2 的 Y 分布（`illum_stats.py` 会打印分位数）重新定边界，并把 `PHYS_LEVELS` 调到合成分布与真实分布重合。

## 3. 配对采集规范（ESTR-v2）
1. 三脚架固定，关闭防抖，手动对焦；同一场景拍 ① 环境暗光帧 ② 手电/补光帧（光源不入画面，避免高光死白）。
2. 记录设备、ISO、快门、白平衡；RAW 保留更佳（可做曝光包围）。
3. `tools/align_pairs.py` 做 ECC 对齐检查，偏移 >2 px 的配对剔除或重配准——只标注一次（在亮帧上），多边形复用到暗帧。
4. 覆盖：室内/室外、印刷体/霓虹/反光/手写涂鸦、弯曲文本、多语种；场景去重（同一招牌只保留一个视角）。
5. 隐私：人脸、车牌、门牌模糊；发布前确认许可。

## 4. 转写规范
- 行级转写，保留原始字符，评测时再归一化（去标点、不区分大小写）。
- 无法辨认 → `###`；半可辨认（<50% 字符可读）→ `###`。
- 双人独立 → 不一致项第三人仲裁；报告 NED 一致率与每语种实例数。

## 5. 划分（冻结后写入 `splits.json` 并随数据发布）
| 集合 | 来源 | 张数 | 备注 |
|---|---|---|---|
| train | LATeD 官方 train 去掉 val | 900 | 场景×光照分层 |
| val | LATeD 官方 train 抽样 | 100 | 固定种子；只用于选超参/早停 |
| test | LATeD 官方 test | 500 | 只在定稿时评一次（分档报告） |
| test-xdomain | ESTR-v2 | 300–500 | 跨域，不参与任何训练 |
| test-synth | Dark-CTW1500 / Total-Text × 3 档 | 500×3 / 300×3 | 固定种子、带亮图配对 |
