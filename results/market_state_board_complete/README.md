# 当前有效研究结果（2026-09-30）

区间 2023-01 至 2026-09-24，45 个月、905 个交易日，9 月不完整。保留候选 `daily_guarded_retry`；固定基线 `state_onset_retry`。这里采用缺板修复后的输入及延迟卖出重试逻辑，不与相邻历史目录混用。

- `metrics.csv`：11 个已完成账户的统一指标，含失败消融。
- `annual_results.csv` / `monthly_results.csv`：各年/各月结果。
- `account_daily_guarded_retry.csv`：保留候选的 45 个月账户。
- `daily_nav_*` / `trades_*` / `corporate_events_*`：日净值、逐笔交易、股份与分红变化。
- `*_protocol.json`：各探索分支规则；有规则文件不表示该分支已完成。
- `verification.json`：独立核账及输入/输出指纹；完整重算需要未上传的本地数据。
- `board_coverage_repairs.csv` / `daily_coverage_repairs.csv`：输入修复审计。

候选期末权益 73,537.51 元，全期增加盈利 14,898 元；四五月仍亏损，月盈利 7,500 元目标仅 3 / 45 个月达到。生产默认入口未切换。详见 [报告](../../reports/market_state_research_2026-09-30.md) 与 [发布/复跑说明](../../docs/research_upgrade_2026-09-30.md)。

研究候选 JSON 中的 `github_pushed` 保留研究完成时状态，不代表本次发布状态。大数据、pickle 和本机特征缓存清单不随仓库发布。
