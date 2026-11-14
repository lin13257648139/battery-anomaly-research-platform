# 数据审计与任务边界

## GIS

来源：<https://doi.org/10.5281/zenodo.10656500>。本地压缩包已由用户解压到 `GIS/GIS`。

静态检查发现 41 个编号目录，各有 `vin_1.pkl`、`vin_2.pkl`、`vin_3.pkl`，合计 859,865,627 字节。所有文件为 pickle 协议 4，涉及 PyTorch 张量重建。原张量存储指向 CUDA；仅在无网络受限 Docker 容器中进行了 CPU 映射和反序列化。转换后共有 571,685 个行位置，每个目录的三张表宽度均为 7、90、91 列，未见非有限值。

7 列张量均在约 0–1 范围。90 列张量中有 85 列完全相同；91 列张量中有 85 列绝对值低于 1e-10。文件没有可直接读取的字段名、绝对时间戳或故障标签。因此不能把编号目录解释为 41 种故障，也不能把宽张量解释为 90 个独立传感器。它适合后续复现发布方方法的辅助研究，当前不用于主模型评价。

125 个源文件（GIS 123 个、Figshare 两个归档）共 1,152,255,832 字节，SHA-256 与文件大小记录在 `reports/source_hashes.json`。GIS 证据另见 `reports/static_audit.json`、`reports/converted_profile.json` 和 `converted/*/manifest.json`。

## Figshare 充电数据

来源：<https://doi.org/10.6084/m9.figshare.23659323.v1>，发布页许可 CC BY 4.0。已下载第二品牌和第三品牌压缩包。压缩包内路径及链接类型经静态检查，无目录穿越或符号链接。`.pkl` 只在无网络、只读输入、低权限容器内读取，再按车辆写为 Parquet。第二品牌转换为 194,245 个片段、24,863,360 条采样记录，数值均有限。发布方标签清单列出 50 辆车，其中训练集正常车辆 230 没有片段文件；实际有数据的是 49 辆：训练 31 辆、测试 18 辆。测试组含 7 辆异常车。特征行数与片段数一致，所有片段均为 128 个采样点。

第三品牌共有 100 辆车，标签 91 正常、9 异常；29,598 段充电数据。两个品牌的单段样本均含 `volt`、`current`、`soc`、`max_single_volt`、`min_single_volt`、`max_temp`、`min_temp`、`timestamp`。同时保留源文件唯一 `snippet_id`、源元数据的 `charge_segment`、`mileage`、片段标记和车辆标签。`charge_segment` 编号会重复，`snippet_id` 在不同车辆中也可重复，因此按 `(car, snippet_id)` 分组。

`timestamp` 的全局时钟语义尚未核实；不据此宣称提前预警天数。车辆标签是评估真值；片段标记只作元数据，**没有进入模型特征**。第二品牌严格保留发布方固定车辆训练/测试划分；第三品牌以车辆分层五折评估。两品牌没有混合拟合。第二品牌最小测试车辆仅有 5 个片段，已另做最少 20 片段的敏感性分析。

证据：`converted_figshare/brand2/manifest.json`、`converted_figshare/brand3/manifest.json`、`features/brand2_manifest.json`、`features/brand3_manifest.json`、`reports/brand2_comparison.json` 和 `reports/baseline_results.json`。
