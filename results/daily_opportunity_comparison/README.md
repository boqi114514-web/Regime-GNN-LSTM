# 股票级日频机会模型：完整账户结果

这是独立日频研究分支，不覆盖原默认流水线。统一区间为 2026-01-01—2026-09-24、178 个交易日、9 个月（9 月未完整），初始现金/新增投入上限 25,000 元、主板 100 股整手、无补款、费用零。

月目标是当月现金账户盈利至少 7,500 元；`budget_return` 分母固定 25,000，`return` 分母为上月末账户权益，两个字段不能混用。

- [重建、实际缺陷及研究结果](../../reports/daily_opportunity_rebuild_2026-10-03.md)
- [全部月份长表](monthly_results.csv)
- [全部月份利润矩阵](monthly_profit_matrix.csv)
- [账户指标](metrics.csv)
- [分支来源及审计状态](accounts.json)
- [汇总指纹](comparison_manifest.json)
- [已完成精确离线重放](../daily_opportunity_offline_replays_20261002/replay_summary.json)
- [月频与日频完整架构](../../reports/monthly_daily_architecture_2026-10-03.md)
- [本轮四组精确离线重放目录](../daily_opportunity_offline_replays_20261003/)
- [未完成路径账户的退市证据](../daily_opportunity_path_band_account_20261003/account/data_blocker_evidence.json)

仅独立账本、输入指纹及预测来源校验通过的完整账户进入表格；尚未完成的分支只记录状态，不补造利润。所有失败保留，不择月拼接。2026 历史已经参与设计，未宣称日频新架构已有三年验证。

截至本轮归档有 14 个完整账户。最新路径收益＋成交额版本盈利 7,497 元、回撤 8.3845%；日频全期最高盈利仍是原图树模型的 14,374.2 元。无成交额门槛路径账户因退市股票生命周期及估值缺口未完成；没有将其缺失行情伪装为连续停牌。

生成：`python scripts/summarize_daily_opportunity.py`。精确重放到新目录：`python scripts/replay_daily_opportunity_accounts.py --out results/daily_opportunity_offline_replays_NEW`。
