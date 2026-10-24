# 原始数据

每个数据集保留独立子目录与原始标注。位置和处理入口见 [02 数据协议](../../docs/02_dataset_protocol.md)，当前冻结划分与可选扩展见 [04 v5 对齐](../../docs/04_v5_alignment.md)。图像按manifest路径读取，无需重新平铺或复制。

已完整解压的ZIP在逐文件大小、CRC核对后清理，压缩包SHA256与删除清单保存在[清理记录](../../results/archive/datasets/)。尚未解压的包继续保留；训练使用解压文件和冻结manifest，来源provenance保留历史压缩包摘要。
