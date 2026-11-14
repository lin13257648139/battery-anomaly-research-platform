# AI 与数据工程深化记录

## 本轮实施

1. `tools/audit_feature_data.py` 对第二品牌片段特征执行数据契约检查：源清单行数、`(car, snippet_id)` 唯一性、每段 128 点、缺失与非有限值、车辆标签和训练/测试分组一致性；同时记录完整 Schema、文件大小与 SHA-256。结果见 `reports/feature_quality.json`。本次 194,245 行、49 车全部通过；`batch/run_spark.ps1` 现在先执行这道质量门槛，失败即停止后续作业。
2. `model/analyze_failures.py` 复用原来从正常训练车确定的探索性终局筛查阈值，不重新调阈值。结果与旧报告的 TP=5、FP=1、FN=2、TN=10 一致。误报车 233；漏报车 239、240。逐车 SOC、电流和温度描述见 `reports/brand2_failure_cases.csv`。这些是事后描述，不构成因果解释。
3. `model/evaluate_conditioned_brand2.py` 只用 22 辆正常训练车训练 ExtraTrees，以 SOC、电流、温度工况预测片段最大单体电压差，再以残差给片段及车辆评分。正常训练车用车辆隔离五折产生残差参照；固定测试组仅用于最终计算指标。结果见 `reports/conditioned_brand2_results.json` 与逐车评分 CSV。
4. `model/build_incremental_features.py` 按车辆从已安全转换的 Parquet 生成特征分区。每个分区记录输入 SHA-256、特征代码 SHA-256、输出 SHA-256 和行数；临时文件完成后原子替换，清单也原子写入。未变化且输出哈希匹配的车辆跳过，输入消失则停止。全量 49 车、194,245 片段与原全量特征逐项一致；第二次运行更新车辆数为 0。产物位于 `features_incremental/brand2/`，不替换当前模型读取的全量特征。
5. `tools/release_manifest.py` 将主数据清单、特征、模型权重、主要报告和相关代码的大小与 SHA-256 固定为 15 项本地快照；`--verify` 可检查后续漂移。它在创建快照前要求质量检查和增量/全量特征一致性通过。结果见 `reports/release_manifest.json`。这是本地内容清单，不是签名或正式模型注册中心。`/research/improvements` 只读接口提供质量、失败分析及新模型报告。

## 同一测试组结果

| 方法 | AUROC | AP |
|---|---:|---:|
| 工况条件化残差 | 0.727 | 0.823 |
| 原电压差物理基线 | 0.805 | 0.848 |

工况模型未超过简单基线。其方案是在已经查看过旧测试结果之后提出，所以即使超过也只能算探索性结果，必须用新的独立车辆验证。当前数据只有车辆级标签，不能推断故障类型或真实提前预警时间。模型输入使用同一完整片段的工况特征，因此属于终局筛查，不是在线早期预警。

## 下一关卡

- AI：对误报与漏报车辆做逐片段可视化和工况匹配；在训练车辆内部确定模型及阈值，再使用未参与选择的新车辆验证。若缺少新标签，不宣称性能提升。
- 数据工程：增量特征与本地发布快照已完成；后续需要把新批次导入、特征、模型与报告串成统一的发布事务。
- 系统：将同一质量门槛接入后续模型训练与 API 产物发布流程；当前已经接入 Spark 批处理入口。

## 重现命令

```powershell
python tools/audit_feature_data.py
python model/analyze_failures.py
python model/evaluate_conditioned_brand2.py
python model/build_incremental_features.py
python tools/release_manifest.py
python tools/release_manifest.py --verify
```
