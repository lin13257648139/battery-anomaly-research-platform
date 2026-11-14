# 动力电池异常预警研究平台

本项目用公开电动车充电数据，建设可审计的数据转换、车辆级异常检测实验和可运行的本地 API/看板。原始 `.pkl` 是不可信序列化文件，只在受限容器内解析；常规训练与服务只读 Parquet。当前系统是**研究原型**，不用于车辆安全诊断。

## 公开仓库说明

仓库只包含源代码、说明文档和汇总实验报告。原始下载文件、转换后的 Parquet、模型权重、本机运行状态及个人文档不会提交。克隆仓库后需要从下述原始发布页自行获取数据，并运行受限转换与特征脚本，才能启动依赖数据的 API 和前端。仓库中的历史结果是已完成实验的记录，不应视为克隆后自动重现的实时结果。

## 数据和评估

主数据：[Sun 等 Figshare 数据集](https://doi.org/10.6084/m9.figshare.23659323.v1)，CC BY 4.0。第二品牌发布方清单有 50 辆车；实际有片段的 49 辆，其中固定训练组 31 辆、固定测试组 18 辆（7 辆异常）。共 194,245 个充电片段、24,863,360 条采样记录。车辆 230 在训练标签清单中，但没有片段文件。第三品牌另有 100 辆车、29,598 段，用于独立的车辆级五折探索。两品牌不混合训练。

辅助数据：[Cao 等 GIS](https://doi.org/10.5281/zenodo.10656500) 已安全转换。它在本地缺少可确认的列语义、时间戳和故障标签，不进入主评估。详见 [数据审计](docs/DATA_AUDIT.md)。

第二品牌官方测试组的车辆级结果：

| 方法 | AUROC | 平均精度 AP |
|---|---:|---:|
| 最高 5% 片段的单体电压差均值 | 0.805 | 0.848 |
| 正常车训练的 Isolation Forest | 0.630 | 0.484 |
| 正常车训练的时序卷积自编码器 | 0.675 | 0.608 |

每辆车先聚合片段分数，再计算车辆级指标。深度模型用独立正常训练车选择训练轮数，再用全部正常训练车重新拟合；测试车辆仅在模型定稿后读取。18 辆测试车样本很少，分层车辆 Bootstrap 的置信区间很宽；排除仅有 5 段的车辆后，电压差 AUROC 为 0.800。完整结果见 `reports/brand2_comparison.json`。目前以电压差作为解释性主基线，深度模型是可复现对照，不能声称它优于简单基线。

## 本地运行

已生成的 Parquet 和实验报告位于当前目录时：

```powershell
docker compose up -d --build
```

前端控制台：<http://127.0.0.1:18501>；API 文档：<http://127.0.0.1:18080/docs>。前端采用 React + TypeScript + Vite + Recharts，经 Nginx 同源代理访问 FastAPI。包括运行总览、车辆诊断、模型实验、数据管道和系统状态五个工作区；支持搜索、筛选、车辆钻取、信号切换及响应式布局。设计和能力边界见[前端架构说明](docs/FRONTEND_ARCHITECTURE.md)。容器仅绑定本机回环地址。停止：`docker compose down`。

Spark 批处理与历史回放使用官方 `spark:3.5.9-java17-python3` 镜像。先准备特征，再运行 `.\batch\run_spark.ps1` 和 `.\stream\run_replay.ps1`；脚本内固定容器资源限制和网络隔离。回放只模拟消息顺序，不能作为真实车辆采集或提前预警证据。再次运行回放会复用断点和 SQLite 账本，用于检验幂等性。

从下载文件重建核心实验：

```powershell
docker build -f safe_convert/Dockerfile -t gis-safe-convert:1 .
.\safe_convert\run_all.ps1
.\safe_convert\run_figshare.ps1
.\safe_convert\run_brand2.ps1
python model\build_features.py --brand brand2
python model\evaluate_brand2.py
python model\evaluate_temporal_autoencoder.py
python model\compare_brand2.py
python model\operating_point.py
python model\build_features.py --brand brand3
python model\evaluate_baselines.py
python model\bootstrap_ci.py
python tools\hash_sources.py
python tools\record_environment.py
```

上述转换脚本会在禁用网络、限制权限的容器中反序列化原始文件。训练/服务代码不直接读取原始 `.pkl`。模型权重位于 `models/`，原始数据和中间产物不提交到代码仓库。

## 工程状态

AI 与数据工程深化：新增第二品牌特征质量与 SHA-256 审计、固定阈值逐车误报/漏报分析，以及仅用正常训练车拟合的工况条件化残差模型。新模型在原测试组 AUROC 0.727、AP 0.823，仍低于电压差基线 0.805、0.848；这是查看旧测试结果后开展的探索性分析，不作为已验证提升。实施细节与重现命令见 [深化记录](docs/AI_DATA_IMPROVEMENT.md)。

新增按车辆增量特征作业：输入、代码与输出都记录 SHA-256，重复运行会跳过未变化车辆。已用完整 49 车数据验证与原全量特征一致；第二次运行更新 0 辆车。该作业目前生成独立分区，不会直接替换现有训练数据。

`reports/release_manifest.json` 记录 15 项关键代码、数据、模型与报告的内容哈希，可用 `python tools/release_manifest.py --verify` 检查当前工作区是否与该本地快照一致。只读 API 的 `/research/improvements` 可查询本轮质量、误差与新模型报告。

已运行验证：来源审计、受限转换、特征一致性检查、车辆级独立评估与置信区间、Spark 批处理和逐列核对、Spark 历史回放与重启无重复、只读 API、看板和 Docker Compose。本机 API 与看板均返回 HTTP 200，最大测试车辆的 2,513,792 行信号可分批抽样查询。回放的 74,862 个唯一事件全部处理，749 条注入重复没有造成重复告警，重启后事件/告警计数保持 74,862/811。当前告警规则涉及 3/11 辆正常车，误报仍高。

实施中的关卡：告警误报控制、长期运行监控和云端实测。具体架构与验收条件见 [实施方案](docs/PROJECT_PLAN.md)，无云账号时的迁移步骤见 [云端迁移设计](docs/CLOUD_MIGRATION.md)，与赛道要求的对应关系见 [赛道契合度](docs/COMPETITION_ALIGNMENT.md)。答辩 PPTX 与讲稿位于 `deliverables/`。未实际部署的云服务不会作为已完成成果申报。
