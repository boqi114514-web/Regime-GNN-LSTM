# 2026 入场与个股目标实验的独立核验

当前为 8 组完整账户、各 9 月/178 交易日，截止 **2026-09-24**。报告见 [完整记录](../../reports/market_state_research_2026-10-02_entries.md)。不替换默认流水线。

- `monthly_results.csv`：全部账户月份，盈利和预算收益率。
- `metrics.csv`：全期账户与四、五月指标。
- `stock_contributions.csv`：逐月逐股盈利，与月账逐项对齐。
- `independent_checks.json`：独立现金/持股/净值、执行报价、整手/主板/限价和投入约束检查及输入 SHA。
- `forecast_ranking_diagnostics.csv`：模型前列组的复权下月标签诊断，**不是账户收益**；9 月没有完整标签。
- `forecast_independent_checks.json`：训练标签截断及保存模型预测重算检查。

## 重建核验

所需本地缓存齐全且八组账户完成后运行；不要与同一账户的重新回测同时运行，否则完成状态及文件哈希会变化。

```powershell
python scripts/audit_dc_theme_accounts.py --accounts results/dc_structure_replay_20261001=dc_structure_2026_retry results/dc_reversal_2026_research=dc_reversal_2026_retry results/dc_weekly_2026_research=dc_weekly_2026_retry results/dc_weekly_reversal_2026_research=dc_weekly_reversal_2026_retry results/dc_forecast_2026_research=dc_forecast_2026_retry results/dc_expected_profit_2026_research=dc_expected_profit_2026_retry results/dc_peer_forecast_2026_research=dc_peer_forecast_2026_retry results/dc_rally_2026_research=dc_rally_2026_retry --output-dir results/dc_entry_comparison
python scripts/audit_dc_forecasts.py --accounts results/dc_forecast_2026_research=dc_forecast_2026_retry results/dc_expected_profit_2026_research=dc_expected_profit_2026_retry results/dc_peer_forecast_2026_research=dc_peer_forecast_2026_retry results/dc_rally_2026_research=dc_rally_2026_retry --output-dir results/dc_entry_comparison
```

分类器排序分数未经校准，不能将兼容列 `forecast_return` 解释为预期收益。各组协议和当时输入固定，全期规则一致，不拼接月份赢家。原始数据及 pickle 缓存仍保留本地，未推送。
