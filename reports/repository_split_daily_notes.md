# 日频仓库拆分检查（2026-10-03）

本文件为源码检查笔记，未修改模型源码、数据、原始审计或 Git 状态。

## 范围

日频独立模型的核心入口为 `src/research_daily_opportunity*.py` 和 `src/research_daily_graph_data.py`。保留其所有实验分支，README 主入口建议选 `research_daily_opportunity_path`（路径收益 LightGBM）与 `research_daily_opportunity_path_runner`（成交额/风险过滤与账户）。

`src/research_daily_reentry.py` 和 `tests/test_daily_reentry.py` 属于月频固定规则模型的月内补仓增强：它的 docstring 明确写 retaining monthly strategy，候选池是上一自然月的候选股票，不能按文件名中的 daily 归入新日频模型。

脚本范围为以下六项（`scripts/_probe_daily_token.py` 是临时令牌探测，不建议发布）：

- `scripts/fetch_daily_opportunity_inputs.py`
- `scripts/audit_daily_opportunity_account.py`
- `scripts/audit_daily_opportunity_typed.py`
- `scripts/replay_daily_opportunity_accounts.py`
- `scripts/replay_daily_opportunity_final.py`
- `scripts/summarize_daily_opportunity.py`

直接测试范围为 `tests/test_daily_opportunity*.py`（19 个）和 `tests/test_daily_graph_data.py`（1 个）。共享执行/数据底层对应的测试也可附带，避免声称仅日频 20 个测试文件覆盖整个共享底层。

## 现有本地依赖闭包

按所有函数体内和顶层静态 import 递归检查，核心闭包为 36 个模块（另保留 `data_pipeline/__init__.py`）：

```text
config
data_pipeline.execution_data
data_pipeline.industry_monthly
data_pipeline.market_factors
data_pipeline.theme_data
data_pipeline.tushare_config
price_risk_pipeline
research_daily_graph_data
research_daily_opportunity
research_daily_opportunity_band_filter
research_daily_opportunity_diagnostics
research_daily_opportunity_execution
research_daily_opportunity_extension_filter
research_daily_opportunity_features
research_daily_opportunity_leadership
research_daily_opportunity_liquid_filter
research_daily_opportunity_liquid_runner
research_daily_opportunity_masked_features
research_daily_opportunity_model
research_daily_opportunity_online
research_daily_opportunity_path
research_daily_opportunity_path_runner
research_daily_opportunity_path_targets
research_daily_opportunity_ranked
research_daily_opportunity_runner
research_daily_opportunity_trailing
research_daily_opportunity_trees
research_daily_opportunity_trend_residual
research_improvements
research_leadership
research_scoped_actions
s4_beta_selection
s7_budget_portfolio
small_account_backtest
stock_execution_research
verify_small_account
```

其中较多月频模块是历史耦合，不是日频选股逻辑：

- `research_daily_opportunity_execution` 调用 `small_account_backtest` 的 Ledger、估值、公司行动归一化和未知复权变动检查，另用 `s7_budget_portfolio.is_main_board`。
- `small_account_backtest` 顶层导入 `stock_execution_research.OUT`，后者顶层导入 `price_risk_pipeline.industry_plans`，再导入 `research_improvements`。
- `research_scoped_actions` 导入 `research_leadership` 仅在 `load_scoped_actions` 使用其 `OUT/dividends` 作为缓存回退目录；账务归一化来自 `small_account_backtest`。
- `s7_budget_portfolio` 的 `s4_beta_selection` 属于可选 live 命令；`research_leadership` 的 `verify_small_account` 也是函数内历史路径。

如直接保留兼容闭包，应在 README 标记 shared execution compatibility，不把这些月频入口列作日频模型。若提取共享执行库，需新建结果运行或明确记录代码变更，不能继续宣称旧源码 SHA 与新代码一致。

尤其要修正新仓库副本中的 `config.PROJECT_DIR`：当前是原仓库绝对路径，且导入 config 会创建该路径下若干目录。建议改为根据 `__file__` 解析当前仓库，保留旧仓库供历史只读审计。`data_pipeline.execution_data.ROOT` 来自 config 的数据目录。

Python 依赖包括 numpy、pandas、torch、scipy、lightgbm、scikit-learn、matplotlib、tushare、requests。hmmlearn 不属于此日频闭包；如共享旧模块另保留其历史完整功能，可继续用完整 requirements。

## 最新路径模型的完整流程

命令均从新仓库根目录运行，输出目录需未完成且不能与输入同目录。下例使用新 run 名避免覆盖归档结果。它先构造基本特征，再构造 V3 题材与缺失掩码，直接训练路径收益树，不必先训练旧神经网络。

```powershell
$env:PYTHONPATH='src'
$env:OMP_NUM_THREADS='4'
python -m research_daily_opportunity prepare --inputs data/raw/daily_opportunity_20261002 --out results/daily_base_NEW --allow-partial-graph
python -m research_daily_opportunity_leadership prepare --source results/daily_base_NEW --out results/daily_leadership_NEW
python -m research_daily_opportunity_path --source results/daily_leadership_NEW --out results/daily_path_NEW
python -m research_daily_opportunity_path_runner filter --source results/daily_path_NEW --out results/daily_path_liquid_NEW --mode liquid
python -m research_daily_opportunity_path_runner account --source results/daily_path_liquid_NEW --out results/daily_path_liquid_account_NEW --inputs data/raw/daily_opportunity_20261002
python scripts/replay_daily_opportunity_final.py --source results/daily_path_liquid_account_NEW --out results/daily_replays_NEW
python -m unittest discover -s tests -p 'test_daily_*.py' -q
```

`replay_daily_opportunity_final` 是路径模型正确的离线重放入口：验证原结果指纹，复制该账户自己的公司行动，离线重建账户，使用 typed 独立审计，并对 11 份经济 CSV 规范排序后逐单元格精确对比。`audit_daily_opportunity_typed.py` 当前只有函数，没有 CLI；直接执行这个文件不会做审计。直接使用旧 `audit_daily_opportunity_account.py` 命令可能把路径标签 pickle 当作官方涨跌停数据，路径模型应使用 typed 适配。

filter/account 默认在线补齐官方限制/公司行动，需要本地 `TUSHARE_API_KEY`；完整缓存已经准备好时可分别加 `--offline`。重放脚本强制离线。`--allow-partial-graph` 是显式接受已停止采集但不完整的历史图，缺失情况保存在 manifest；不能写成完整历史图。

## 数据前提及准备

新 clone 不含原始行情、复权因子、历史题材边、特征张量、预测 pickle、检查点和全部交易缓存。prepare 的输入目录需至少包含经过指纹绑定的 `adjustment_manifest.json` 及调整价格 aggregate、`graph_manifest.json` 及相应 aggregate 和 dated snapshot shards。

现有下载器不是从零下载全市场行情的入口；它还需要全市场原始日行情 pickle 和东财历史题材目录 pickle。默认源为 `results/dc_context_inputs_20261002/daily.pkl` 和 `data/raw/dc_theme_research/catalogs.pkl`，拆分 README 必须说明这些前置文件，或提供新仓库内的准备脚本。日行情需要 `date/ts_code/open/high/low/close/volume/amount`；题材目录至少需 `trade_date/ts_code/up_num/down_num` 并保持 dated key 唯一。

```powershell
python scripts/fetch_daily_opportunity_inputs.py adjustments --daily data/raw/source_daily.pkl --output-dir data/raw/daily_opportunity_20261002
python scripts/fetch_daily_opportunity_inputs.py members --catalog data/raw/theme_catalogs.pkl --output-dir data/raw/daily_opportunity_20261002
```

下载器默认行情起点 2024-09-01、终点 2026-09-24；历史图计划范围 2024-12-20 至 2026-08-31。members 遇历史成员缺失会写 incomplete manifest 并报错，支持后续续采；只能显式选择 partial graph，不能抹去失败。

现有输出 manifest 绑定大量绝对路径及 SHA。复制目录后若继续依赖原位置，应明确为历史归档；不应批量改 manifest 路径并仍把旧审计记为新通过。可保留原目录只读副本，给新仓库迁移 manifest 指明源与时间，新的独立运行用新目录产生自身指纹。
