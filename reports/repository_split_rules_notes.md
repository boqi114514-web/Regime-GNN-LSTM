# 月频固定规则工程拆分检查

检查日期：2026-10-03。本文只记录源码静态检查；没有修改策略、跑新回测或修改 Git。

## 结论

当前最新纯规则主线是 `dc_member_relative_eligible_hold_20261003`：东财历史题材、月频固定排序与整手入场、每日退出检查、月初仍满足完整入场资格则续持。它不训练收益预测模型。先前的 DC 固定规则版本也应保留，以完整保存研究过程。

`research_dc_themes.py` 中五个 DC 学习实现（forecast、peer_context、peer_forecast、rally_classifier、context_ranker）的引用均处于对应学习变体分支。只运行 DC 固定规则变体可以不复制这些模块。但是不能把整个原项目的前序 `market_state_*` 研究统称为纯固定规则：它含每季度训练的离散跳跃状态模型，部分候选还使用学习型行业预测或股票排序。

## 规则变体范围

下列全部留在固定规则项目：

- `dc_theme_retry`：2025—2026 题材基线；历史数据仍有缺口，应继续标为未完成研究，不能写成已跑通两年。
- `dc_theme_2026_retry`、`dc_liquid_2026_retry`、`dc_affinity_2026_retry`、`dc_structure_2026_retry`。
- `dc_reversal_2026_retry`、`dc_weekly_2026_retry`、`dc_weekly_reversal_2026_retry`：月频主干和周频补充实验。
- `dc_acceleration_2026_retry`、`dc_early_2026_retry`、`dc_member_acceleration_2026_retry`、`dc_member_moderate_2026_retry`。
- `dc_member_relative_2026_retry`、`dc_member_relative_moderate_2026_retry`、`dc_member_leader_2026_retry`、`dc_member_relative_leader_2026_retry`。
- 最新包装器 `dc_member_relative_eligible_hold_20261003`，保留原引擎产物别名 `dc_member_relative_leader_2026_retry`，README 应解释别名以免混读。

下列 DC 变体进入月频机器学习项目：`dc_forecast_2026_retry`、`dc_expected_profit_2026_retry`、`dc_peer_forecast_2026_retry`、`dc_rally_2026_retry`、`dc_context_rank_2026_retry`、`dc_context_member_2026_retry`、`dc_local_rank_2026_retry`、`dc_local_member_2026_retry`。注意 local 也是 LightGBM 排序，不属于规则。

## 启动依赖，不能仅按文件名前缀切分

核心 DC 规则源码：

```text
config.py
research_dc_themes.py
research_dc_specialist.py
research_dc_entry.py
research_dc_entry_policy.py
research_dc_leader_allocation.py
research_dc_eligible_hold.py
research_theme_affinity.py
research_market_states.py
research_leadership.py
research_scoped_actions.py
small_account_backtest.py
s7_budget_portfolio.py
verify_small_account.py
verify_market_states.py
data_pipeline/__init__.py
data_pipeline/execution_data.py
data_pipeline/execution_boundaries.py
data_pipeline/tushare_config.py
data_pipeline/theme_data.py
data_pipeline/public_theme_data.py
```

但原版有两条额外的顶层依赖链：

1. `research_market_states → research_learned_leaders → lightgbm`；前者只是顶层取 `VARIANTS`，实现本来已经在运行对应分支时动态导入。
2. `small_account_backtest / verify_small_account → stock_execution_research → price_risk_pipeline → research_improvements / lightgbm / sklearn.Ridge`。`stock_execution_research` 的 `industry_plans` 只在其 `main()` 中使用，DC 账户只是要 `OUT`。这条链可以通过延迟导入消除，或将共享路径配置从策略模块提取。

另外 `research_market_states` 顶层为注册变体导入 `research_trend_features`、`research_structural_router`、`research_fine_industry`、`research_peer_graph`、`research_daily_reentry`、`research_quality_floor`、`research_liquid_leaders`。若原封不动保留这个混合引擎，必须一起保留这些文件；可把变体注册移到无策略依赖的共享模块，并给固定规则入口加允许列表。直接删除对应模块而保留顶层 import 会令固定规则入口报错。

`sklearn.cluster.KMeans` 在 `research_market_states` 顶层导入，但只供历史状态模型 `fit_jump()` 使用。若不提取混合引擎，requirements 仍须包括 scikit-learn；README 应区分运行环境兼容依赖和实际决策模型。

建议最小诚实方案：DC 规则实现和账户引擎在规则仓库内独立保留；把两条学习型顶层依赖改为局部 import 或共享注册常量；用规则白名单入口明确边界。共享代码可以在两个项目中各存一份并标明来源，不要求跨仓库 import。前序含学习状态/行业预测的实验及成果归月频 ML 历史研究区，规则 README 链接过去。

## 工具脚本及测试

规则脚本：`replay_dc_eligible_hold.py`、`audit_monthly_trail_applicability.py`、`audit_dc_capital_priority.py`、`audit_dc_theme_accounts.py`、`fetch_dc_theme_research.py`、`probe_concept_gateway.py`。采集器依赖最后一个探针模块及 `requests`。`complete_suspension_checks.py` 还依赖日频项目的 `research_daily_graph_data.py`，不能盲目复制脚本而漏其共享数据依赖。

直接规则测试：

```text
test_dc_themes.py
test_dc_specialist.py
test_dc_eligible_hold.py
test_dc_capital_priority.py
test_dc_entry.py
test_dc_entry_policy.py
test_dc_weekly_execution.py
test_dc_theme_collection.py
test_dc_leader_allocation.py
test_monthly_trail_applicability.py
test_theme_affinity.py
test_scoped_actions.py
test_execution_data.py
test_small_account_backtest.py
test_empty_action_verifiers.py
test_verify_market_states.py
test_theme_data.py
test_public_theme_data.py
test_tushare_gateway.py
test_concept_gateway_probe.py
```

其中账户/状态等共享测试可能测试前序非 DC 分支，若收缩共享引擎应按实际保留功能重新选择，而非声明未运行的全套通过。`test_dc_forecast`、`test_dc_peer_forecast_adapter`、`test_dc_peer_context`、`test_dc_rally_classifier`、`test_dc_context_ranker` 归月频 ML。

## 数据、结果、报告

规则结果保留上述固定规则变体对应目录，以及 `dc_member_relative_eligible_hold_replay_20261003`、`dc_structure_replay_20261001`、`dc_structure_replay_20261002`、`dc_theme_comparison`、`dc_entry_inputs`、`dc_leader_capital_review`、`dc_priority_capital_review`。`dc_entry_comparison` / `dc_priority_comparison` 混合规则与学习型对照，可作为完整比较附件保留，但必须注明跨项目比较，不声称全是本仓库策略。

规则首要报告：`monthly_eligible_hold_2026-10-03.md`、`market_state_research_2026-10-01_dc.md`、`market_state_research_2026-10-02_entries.md`、`market_state_research_2026-10-02_priority.md`；后两份含 ML 对照，保留完整背景并补跨仓库链接。`monthly_daily_architecture_2026-10-03.md` 适合总引导页。早期报告可以在规则项目引用，勿整体重新归类为纯规则。

完整本地复跑还需要 `data/raw/execution_v1/`（实际 ROOT 由 execution_data 定义）、`data/raw/dc_theme_research/` 以及各账户目录里的 pickle、scoped_actions、exit_quotes、replacement_limits、dividend_repairs、suspension_events、verified_suspensions；并且汇总核验枚举 `leadership.OUT/dividends`。发布源码与 CSV/JSON 时不可宣称克隆即可完整回测。

## 路径与旧哈希证据

`config.PROJECT_DIR` 目前写死为原绝对路径。必须换为由 `__file__` 推导的新项目根路径，否则所谓独立项目仍在原项目读写。

`verification.json` 的 inputs、`eligible_hold_protocol.json` 的 frozen_source、`theme_feature_manifest.json` 的源文件位置等含原绝对路径。`verify_frozen_inputs()` 不仅校验哈希，还要求新 source 的 `candidates.pkl/plans.pkl/daily.pkl` 绝对路径存在于原 manifest 中。直接复制整个目录后运行会失败（或误读旧项目），而修改源码后原源文件哈希也会不同。历史证据应原样保留，不能把路径迁移后重新生成的校验文件伪称旧证据。

可增加明确的可迁移路径解析层：旧根到新根按相对路径映射，逐个验证数据文件哈希，对迁移代码单独记录原始/当前哈希与迁移原因；或在新目录按当前代码重跑基线形成新的审计，再跑 eligible-hold，并与旧 CSV 做逐单元格对照。README 应分别说明“历史发布证据”“本地原始数据”“迁移后验证”。

## 原入口示例（需在拆分适配后验证）

```powershell
$env:PYTHONPATH = 'src'
python -m research_market_states run --variant dc_member_relative_leader_2026_retry --offline --output-dir results/NEW_BASELINE
python -m research_market_states summary --variant dc_member_relative_leader_2026_retry --output-dir results/NEW_BASELINE
python -m research_dc_eligible_hold --source results/NEW_BASELINE --out results/NEW_ELIGIBLE_HOLD
python scripts/replay_dc_eligible_hold.py --source results/NEW_ELIGIBLE_HOLD --out results/NEW_EXACT_REPLAY
```

基线 `NEW_BASELINE` 事先需准备已验证的题材聚合、清单、daily 及公司行动等输入；上述不是从空目录自动完成数据采集的单一命令。最新 eligible-hold 离线读取冻结候选，不会自动训练或下载。
