# 月频机器学习拆仓核查（2026-10-03）

本文件只记录拆分归属与 README 建议，未修改模型、原始数据或已冻结实验。

## 推荐主线及命名边界

主线为 `price_risk_pipeline.py` 的 **Ridge + 五种子 LightGBM 行业排名模型**，再接历史行业内的固定选股规则与整手账户。行业分数来自训练，个股层的 50% 跳过近月的十二月动量 + 30% 六月动量 + 20% 六月低波动来自固定规则，README 必须同时交代这两层。

`run_all.py` / `s1_gnn_train.py` / `s2_lstm_b_train.py` 是 GAT + LSTM 的另一条月频机器学习线；原默认入口训练时已停用 HMM。`run_original_architecture.py` 是修复时点后的历史四状态宏观 HMM + GAT + LSTM 复现线。`s0_regime.py` 是三状态成交额 HMM 对照模块。这些均属于机器学习项目，不应误称当前同一模型。

`compare_fixed_weights.py`、`fixed_weight_*` 产物中的 fixed 指 **GNN/LSTM 学习分数的固定集成比例**，不是月频固定规则选股，仍归机器学习项目。`research_improvements.py` 内的 momentum/reversal/low_vol/trend_defensive 是作为机器学习实验对照的规则基准，保留在同一实验目录并明确标注。

## 应复制的主体文件

保留相对路径，避免破坏原导入与归档证据。以下全为当前 Git 已跟踪文件或目录。

- 当前模型与完整资金账户：`src/price_risk_pipeline.py`、`src/research_improvements.py`、`src/verify_improvement_research.py`、`src/stock_execution_research.py`、`src/small_account_backtest.py`、`src/verify_small_account.py`、`src/s7_budget_portfolio.py`。
- 月频神经网络及历史复现：`src/s0_regime.py`、`src/s1_gnn_train.py`、`src/s2_lstm_b_train.py`、`src/s3_ensemble_backtest.py`、`src/s4_beta_selection.py`、`src/s5_etf_backtest.py`、`src/s6_etf_execution_backtest.py`、`src/run_all.py`、`src/run_original_architecture.py`、`src/run_branch.py`、`src/compare_fixed_weights.py`。
- 共用基础设施：`src/config.py`；`src/data_pipeline/__init__.py`、`download.py`、`update.py`、`industry_monthly.py`、`prosperity.py`、`tech_factors.py`、`pattern_factors.py`、`market_factors.py`、`global_indices.py`、`tushare_config.py`、`execution_data.py`、`execution_boundaries.py`、`probe_execution_data.py`、`etf_mapping.py`、`etf_mapping_v2.py`。
- 旧月频应用运行层：`src/live/`、`src/notifier/` 和 Dashboard 可归本项目的历史应用；README 应交代其默认仍是 GAT/LSTM，未切换到新价格模型。是否带打包启动器由主代理统一决定。
- 对应测试：`test_original_architecture.py`、`test_fixed_weight_market_factors.py`、`test_regime_and_signal_timing.py`、`test_industry_monthly_integrity.py`、`test_market_return_alignment.py`、`test_factor_cache_publication.py`、`test_research_improvements.py`、`test_price_risk_pipeline.py`、`test_stock_execution_research.py`、`test_small_account_backtest.py`、`test_budget_portfolio.py`、`test_execution_data.py`、`test_tushare_gateway.py`；共享执行测试可按导入闭包补全。
- 原型辅助 scripts：`_run_3branches_independent.py`、`_run_3branches_mainboard.py`、`_append_mainboard_holdings.py`、`gate1_sw_l2_glasso.py`、`gate3_l2_circ_mv.py`、`gate3_l2_members.py`、`gate3_l2_pe_pb.py`、`gate3_l2_synth_monthly.py`、`probe_l2_two_layer.py`。探测脚本中可能有旧绝对路径，须清点后再提供为可执行入口。

## 后期混合研究的归属

`research_market_states.py` 是一个同时支持规则与学习方案的大型研究调度器，顶层就导入多种策略模块。直接只拷贝名称含 ML 的几个文件会缺依赖。应按依赖闭包保留共享执行支持，或将非本项目方案变为显式可选；不要把顶层依赖存在误解为当前规则策略执行时调用了全部学习器。

真实监督学习模块包括：

- `research_learned_leaders.py`：20 特征的 LightGBM 月频个股排名；滚动 36 个月，至少 12 个成熟标签月，按季度拟合。
- `research_dc_forecast.py`：15 特征 HistGradientBoostingRegressor，预测下月复权收益。
- `research_dc_peer_forecast.py`：上述回归增加 4 个同走势背景，共 19 特征。
- `research_dc_rally_classifier.py`：19 特征 HistGradientBoostingClassifier，预测下月涨幅至少 30% 事件。
- `research_dc_context_ranker.py`：LightGBM LambdaRank 上下文排名。

`research_dc_peer_context.py` 的 KMeans 属无监督统计分组；`research_market_states.py` 的状态聚类也不等同于监督收益预测。`research_dc_specialist.py` 已核对为完全固定的题材/相对加速度排序规则，没有拟合模型，归月频规则；`research_dc_leader_allocation.py` 和 `research_dc_eligible_hold.py` 是规则执行/资金与续持模块。最新月频 `dc_member_relative_eligible_hold_20261003` 的利润不归 ML 主线。

后期 ML 账户结果应完整保留失败记录。2026-01—09-24：收益回归盈利 -7,002.60 元；相同预测改金额目标 -5,091 元；同走势回归 +7,621 元；主升浪分类 -2,586 元，均未超过当阶段规则结构版 +13,282 元。对应 `results/dc_forecast_2026_research/`、`dc_expected_profit_2026_research/`、`dc_peer_forecast_2026_research/`、`dc_rally_2026_research/`，汇总报告 `reports/market_state_research_2026-10-02_entries.md` 包含规则对照，必须明确混合归档。

## 主体报告与可阅读结果

- 报告：`reports/rebuild_2026-09-25.md`、`rebuild_2026-09-26.md`、`original_architecture_2026-09-26.md`、`improvement_research_2026-09-27.md`、`price_risk_pipeline_2026-09-27.md`、`small_account_results_2026-09-28.md`。
- 目录：`results/price_risk_pipeline/`、`results/improvement_research/`、`results/original_architecture/`、`results/stock_execution_research/`。
- 根目录结果：`backtest_*`、`fixed_weight_*`、`predictions_gnn.csv`、`predictions_lstm_b.csv`、`predictions_lstm_b_no_market.csv`、`regime_labels.csv`、`budget_portfolio_*`、`etf_backtest*`、`stock_backtest*`、`bt_*_mainboard.csv`。
- 根目录旧行业回测/图只作为标注清楚的历史资料；不得把旧受数据口径问题影响的年化 24.7% / 31.2% 当作最新可复现成绩。

## README 可用事实

推荐默认介绍价格模型；GAT/LSTM 与 HMM 放在历史对照部分。价格模型输入共 26 项：21 行业价格/风险 + 5 市场背景。

- 行业 21 项：1/2/3/6/9/12 月动量；3/6/12 月跳过最近月动量；3/6/12 月波动；3/6/12 月均线偏离；6/12 月回撤；6 月下行波动；6 月上涨月份比例；12 月 beta；12 月相对市场残差波动。
- 市场 5 项：当月收益、3 月动量、6 月动量、6 月波动、上涨行业比例。
- 目标是下月收益横截面排名减 0.5。Ridge alpha=100；LightGBM 160 树、7 叶、深度 3、学习率 0.03、最小叶样本 80、L2=10，种子 42/142/242/342/442。
- 滚动 72 个信号月，至少 48 个训练月；季度训练；训练标签截止首次预测月之前一月。按月均衡样本权重；Ridge 排名占 50%，五树模型排名均值占 50%。

真实股票账户（归档 CSV、manifest 和报告一致）：2023-01-01—2026-09-24，25,000 元初始及新增投入上限，无追加本金，佣金/滑点/卖出印花税按实验设为零。

| 方案 | 期末权益 | 累计盈利 | 最大月末回撤 |
|---|---:|---:|---:|
| 新模型，不额外降仓 `new_full` | 47,817.46 | 22,817.46 | 26.51% |
| 新模型，波动率控仓 `new_vol` | 43,187.979 | 18,187.979 | 26.57% |
| 修复旧行业模型 + 同一新个股层 `original` | 14,401.8516 | -10,598.1484 | 50.74% |

这是 **月末回撤**，不能与日频/最新月频规则的最大日回撤直接比较。2026 九月不完整；开发过程已查看这些历史，重放不构成新独立样本。行业 Top-5 回测与整手股票账户是不同层级。

## 运行、数据与拆分注意

主线建议流程：

```text
python src/price_risk_pipeline.py
python src/stock_execution_research.py
python src/small_account_backtest.py --offline
python src/verify_small_account.py
```

第二步同时加载 `results/original_architecture/predictions_ensemble.pkl` 以生成旧模型对照；若未复制本地缓存，需先准备历史模型或为新版提供只运行新模型的显式入口。不能声称仅执行四条命令在空数据克隆上就可以复现。

默认 GAT/LSTM 线为 `python src/run_all.py`；原架构复现为 `python src/run_original_architecture.py`。完整消融是 `research_improvements.py --phase develop` 后 `--phase evaluate`，开发协议已存在时不应清空重建。

`src/config.py` 第 24 行当前硬编码旧仓库绝对路径，拆分副本必须改为文件位置推导的新根，且应支持显式数据根配置。`small_account_backtest.py` 共用 `stock_execution_research.OUT`；移植执行器时尤其要防止在另一个仓库覆盖 ML 的历史输出。

原始供应商数据和 `.pkl` 缓存未公开。主线需要 `data/raw/ts_sw_industry_monthly.csv`；完整股票账户还需 `data/raw/execution_v1/`、历史行业成员及研究候选/行业预测等本地缓存。旧神经网络线另需景气/技术/形态因子、股票日线、宏观等文件。不要上传密钥；不要以文件下载可得替代数据来源授权。

修改根路径会改变新代码指纹，既有报告中的冻结指纹应保留为历史事实，拆分后另写来源/迁移清单，不回写旧验证记录冒称重新验证。
