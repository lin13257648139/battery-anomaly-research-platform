# 云端迁移设计（尚未部署）

本文件说明将已验证的本地数据与代码迁移到阿里云时的最小路径；**不代表云服务已开通、费用已测量或集群已验证**。

| 本地已验证组件 | 云端候选 | 迁移前的验收 |
|---|---|---|
| 按车 Parquet 与实验报告 | [OSS 对象存储](https://www.alibabacloud.com/help/en/oss/user-guide/what-is-oss/) | 上传后逐文件 SHA-256/对象元数据核对；私有桶与最小权限 |
| Spark 3.5.9 local[4] 特征作业 | [EMR Serverless Spark](https://www.alibabacloud.com/help/en/emr/emr-serverless-spark/product-overview/what-is-emr-serverless-spark) | 相同输入输出键、50 列容差比较；记录资源、耗时、费用 |
| 文件历史回放 | [ApsaraMQ for Kafka](https://www.alibabacloud.com/help/en/apsaramq-for-kafka/) 加 Spark 流作业 | 同一事件标识的重复、乱序、重启测试；仅模拟回放，不宣称真实车辆流 |
| FastAPI + React/Nginx Docker Compose | [ACK Serverless](https://www.alibabacloud.com/help/en/ack/product-overview/product-introduction) 或更简单的容器托管服务 | 私有镜像、限流、健康检查、日志、只读数据、自动停止 |

## 最小落地顺序

1. 确认预算、地域和允许的服务。先设定费用上限与资源生命周期；此时不预留常驻集群。
2. 上传**安全转换后的 Parquet**，原始 pickle 仅作离线归档，不给在线服务读取。核对 `reports/source_hashes.json` 和转换清单。
3. 在云端跑一次 Spark 特征作业，执行与本地 `batch/check_parity.py` 相同的键和列比对；记录耗时、并行度、峰值内存、真实费用。
4. 迁移 API 与看板，健康检查和摘要接口先通过，再做大车辆抽样请求。只授予只读数据权限。
5. 最后接入消息队列历史回放，按 `event_id` 验证幂等、告警持久化、断点续跑。没有真实采集协议与时间语义时，保留“历史模拟回放”标识。

## 计算资源

当前时序自编码器实验可在 CPU 上运行。GPU 可用于下一轮工况条件化或自监督时序模型实验；在新的独立测试集准备好前，不能依据已见测试组反复调参并报告为无偏结果。
