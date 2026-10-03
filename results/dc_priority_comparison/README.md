# 2026 题材内部领涨与配置优先级研究

12 个新增账户 + 上一版结构账户同口径复跑，区间 2026-01-01—2026-09-24，9 个账月、178 个交易日。初始及新增投入上限 25,000 元，不补款、主板 100 股整手、费用零；全部沪深板块参与特征/排序。

保留 `dc_member_relative_leader_2026_retry`：四月 +15,846 元、五月 +8,262 元；全期 +22,182 元、期末 47,182 元、最大日回撤 11.9165%。六月、七月仍亏损。累计盈利最高的另一版本未解决四、五月，不拼接各月最佳版本；历史已参与选型，非未触碰样本验证。

## 证据文件

- [完整报告](../../reports/market_state_research_2026-10-02_priority.md)
- [13 方案指标](metrics.csv) · [全部 117 条月账](monthly_results.csv) · [逐股月利润归因](stock_contributions.csv)
- [月账、日账、报价及资金核验](independent_checks.json) · [13 组逐单元格离线重放](offline_replay_checks.json)
- [保存排序模型重算及标签检查](forecast_independent_checks.json) · [排序诊断](forecast_ranking_diagnostics.csv)
- [基线/中间版/保留版的全部候选资金优先级](capital_priority.csv) · [该诊断的指纹和口径](capital_priority_manifest.json)
- [目录和变体映射](accounts.json)

`capital_priority.csv` 中每股 `budget_max_lots` 是单股按目标预算测算的上限，不是联合组合或剩余现金可同时买入的手数；`diagnostic_month_profit` 仅是已发生持仓归因，不参与选择。未买股票不导出未来收益。最末 9 月日历只要求覆盖实际账户终点。

## 重跑

需本地真实行情、历史主题/成分、已验证公司行动和执行缓存。GitHub 不包含原始行情和 pickle，不能仅下载源码就声称完整离线复现。

```powershell
python src/research_market_states.py run --variant dc_member_relative_leader_2026_retry --output-dir results/dc_member_relative_leader_2026_research --offline
python src/research_market_states.py summary --output-dir results/dc_member_relative_leader_2026_research
python scripts/audit_dc_theme_accounts.py --accounts results/dc_member_relative_leader_2026_research=dc_member_relative_leader_2026_retry --output-dir results/dc_leader_recheck
python scripts/audit_dc_capital_priority.py --accounts results/dc_member_relative_leader_2026_research=dc_member_relative_leader_2026_retry --output-dir results/dc_leader_recheck
python -m unittest discover -s tests -q
```

其他十二组替换路径和变体即可。主题成员与特征 manifests 必须和源指纹一致；源变动后应从已验证历史缓存重新生成证明，不手工修改 SHA。需要补缓存时，API 密钥仅设置在进程环境中，去掉 `--offline`；不得写入源码、结果或报告。

本轮 391 项测试通过；13 组全期离线重放与独立日/月核账通过。默认流水线未切换，本輪未推送。
