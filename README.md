# Regime-GNN-LSTM 行业轮动模型

## 最新升级（2026-09-30）

本轮发布行情状态识别、行业龙头/大小盘风格、日频止盈止损与补充入场、财报可得性底线、数据覆盖修复及独立核账代码。当前保留的研究候选是 **`daily_guarded_retry`**：保留月度选股主干，仅给日频补充买入加上当时已披露的盈利/净资产底线。默认 `run_all.py` 和 Dashboard 尚未切换。

统一区间为 **2023-01 至 2026-09-24**，初始本金及新增投入上限 25,000 元、不补款、主板 100 股整手，可单股集中；费用按本次实验约定为零。

| 指标 | 固定研究基线 `state_onset_retry` | 保留候选 `daily_guarded_retry` |
|---|---:|---:|
| 期末权益 | 58,639.51 元 | 73,537.51 元 |
| 累计盈利 | 33,639.51 元 | 48,537.51 元 |
| 账户最大日频回撤 | 45.36% | 41.69% |
| 最大峰谷亏损金额 | 16,987.09 元 | 17,303.09 元 |
| 月盈利至少 7,500 元 | 3 / 45 | 3 / 45 |

全期多赚 **14,898 元**，但新增盈利主要来自较早年份，2026 年累计反而少 48 元；**2026 年四、五月仍分别亏损 1,554 元和 2,222 元，问题尚未解决**。百分比回撤下降不等于绝对亏损改善。9 月不完整；这些历史已经参与多轮研究。

- [升级内容、目录导航与复跑条件](docs/research_upgrade_2026-09-30.md)
- [详细报告：年度及全部 45 个月盈利](reports/market_state_research_2026-09-30.md)
- [当前有效结果目录](results/market_state_board_complete/README.md) · [全部已完成实验](results/market_state_board_complete/metrics.csv) · [保留候选账户](results/market_state_board_complete/account_daily_guarded_retry.csv) · [独立核账记录](results/market_state_board_complete/verification.json)

发布前重新运行 **106 项测试全部通过**。原始行情、模型与 pickle 缓存、密钥不上传；GitHub 上的表格可直接阅读，完整离线重放需要本地数据和执行缓存，不能把下载仓库等同于下载完整数据集。

## 前一阶段（2026-09-28）：行业模型与独立账户

最新独立账户实验采用 **Ridge + 五种子 LightGBM 行业模型**、历史时点行业成员与价格动量选股、2.5 万元投入上限及 100 股整手主板配置；优先保留不额外降仓版。完整记录见 [账户实验报告](reports/small_account_results_2026-09-28.md)，逐月账户净值见 [account_new_full.csv](results/stock_execution_research/account_new_full.csv)。该实验尚未替换默认 `run_all.py` 或 Dashboard 流水线，下文 GAT/LSTM 架构描述的是默认路径及历史实验。

代码、测试、研究报告和可阅读回测结果随仓库发布。原始数据、模型及 pickle 缓存、生成的 valuation 大型面板、运行日志和打包启动器不入仓库；离线重放需保留本地数据及缓存。API 密钥仅通过环境变量配置。

下面是原有默认路径及历史实验：GAT + LSTM 的 A 股申万一级行业轮动；HMM 已从训练和集成链路停用。全市场成交额作为可选的 LSTM 量价输入，固定 4:6、5:5、6:4 三组 GNN:LSTM 权重并行评估。不要将下方行业组合收益与本轮整手资金账户收益混用。

## 架构

```
行业景气度 → [GLASSO] → 动态邻接矩阵 → [GAT] ───────────────┐
                                                           ↓ 固定 4:6 / 5:5 / 6:4 对照
行业技术因子 + 全市场成交额/量价 → [LSTM-B] ────────────────┘
                                                              ↓
                                                    Top-K 行业组合
                                                              ↓
                            ┌─────────────┬───────────────────┘
                            ↓             ↓
                  行业内多因子选股   申万行业 → ETF 映射
                  (β + 动量 + quality)  (卡尔曼 β · R² · 规模)
                            ↓             ↓
                       个股组合         ETF 组合
                            ↓             ↓
                        周报 + Dashboard 可视化
```

## 当前实验：无 HMM、固定权重、成交额因子消融

同一套无 HMM GNN 预测，LSTM 分别使用原 12 个行业技术因子、以及另加 5 个全市场成交额/广度因子。2026-09-26 完成行情口径修复与全量重训；2019-01 至 2026-08 连续 92 个信号月，其中 91 个已实现，对应 2019-02 至 2026-08 持有月。Top-5 行业纯多头回测（非 2.5 万元整手回测）：

| LSTM 输入 | GNN:LSTM | 年化 | Sharpe | 最大回撤 | 同期行业等权年化超额 |
|---|---:|---:|---:|---:|---:|
| 原行业因子 | 4:6 | 6.13% | 0.150 | -32.8% | -1.15 个百分点 |
| 原行业因子 | 5:5 | 7.46% | 0.218 | -31.7% | +0.19 个百分点 |
| 原行业因子 | 6:4 | 8.33% | 0.255 | -28.1% | +1.06 个百分点 |
| 加入全市场量价 | 4:6 | 6.22% | 0.149 | -34.9% | -1.06 个百分点 |
| 加入全市场量价 | 5:5 | 6.51% | 0.164 | -37.4% | -0.77 个百分点 |
| 加入全市场量价 | 6:4 | 7.50% | 0.213 | -37.9% | +0.23 个百分点 |

近 44 个持有月（2023-01 至 2026-08），仅原行业因子 6:4 略超同期行业等权；新增全市场量价因子没有稳定改善近期表现。**不因全样本最佳值而自动选定实盘权重**。逐年收益以实际持有月份归年，见 `results/fixed_weight_calendar_returns.csv`；其他指标见 `results/fixed_weight_market_factor_ablation.csv` 与 `results/fixed_weight_market_factor_subperiods.csv`。

### 2026-09-26 数据修复与复跑

原行业月线中的 `pct_chg` 前后混用百分数/小数，且 2025-11 有 29 组行业重复月份。原始文件备份为 `data/raw/ts_sw_industry_monthly.before_repair_20260926.csv`；修复后 5178 → 5149 行，统一以收盘价计算收益，并把 `pct_chg` 规范为百分数。读取、更新和 ETF 行业收益映射共用同一口径。GNN 收益标签从完整行情取信号月的下一月收益；回测对预测缺月、标签错位及重复行业/月直接报错。`results/backtest_monthly_returns.csv` 分别标记信号日和持有月，`results/backtest_manifest.json` 记录本次输入指纹。同一输入重新执行回测，四个主要文本产物的 SHA-256 均相同。详细核验见 `reports/rebuild_2026-09-26.md`。

这次没有解决基本面底表的历史修订值和公告日逐日可得性问题；因此上述仍是研究性行业回测，不能视为已审计的可实盘收益，也不是 2.5 万元整手个股组合回测。

## 上一轮 HMM 实验（仅作对照，不再是当前训练路径）

截至本次数据，个股日线更新至 2026-09-24；完整的行业月度因子与沪深 300 月线更新至 2026-08-31。月末 t 的特征预测 t+1 月收益，因此 2026-08 信号尚无完整的已实现标签。以下回测的最后完整持有月为 2026-08，不是截至 9 月 24 日的日频实盘收益。

HMM 以沪深北全市场日成交额相对于 20/60/252 交易日历史的强弱为主，辅以涨跌家数、沪深 300 趋势/波动、韩国综合指数和滞后可得的美股指数。3 个状态按训练窗口内成交额强弱排列为低成交、常态、高成交/过热风险；高成交不自动等于上涨。集成权重按相同状态下过去已实现的 IC 计算，LSTM 权重下限为 60%。

| 行业策略 | 已实现月数 | 年化收益 | Sharpe | 最大回撤 | 同期行业等权基准年化超额 |
|---|---:|---:|---:|---:|---:|
| 固定 40% GNN / 60% LSTM | 91 | 7.7% | 0.227 | -34.5% | +0.4 个百分点 |
| HMM 条件集成 | 91 | 4.8% | 0.085 | -39.8% | -2.5 个百分点 |

这些是上一轮行业层 Top-5 对照、尚未叠加 2.5 万元整手约束的结果；当前 `results/backtest_summary.csv` 已被无 HMM 实验覆盖，旧数值归档在 `reports/rebuild_2026-09-25.md`。HMM 条件权重未优于简单固定权重，因此已退出当前训练链路。

个股层旧 25 只/月筛选在上一轮 HMM 信号下重跑过，2019-02 至 2026-08 的已完成持有月年化约 8.8%、Sharpe 0.257、最大回撤 -41.6%（原代码成本假设）；**此值不能归属当前无 HMM 实验**。该层财报选用历史最新修订版本、行业归属使用当前映射，尚未完成严格的逐日可得性审计，不能当成可信实盘绩效。2.5 万元整手整数优化目前只产出最新持仓方案，没有对应的历史资金约束回测。

## 旧版回测记录（2019-01 ~ 2026-04，仅作历史对照）

> **旧口径，不可作为实盘依据**：曾存在行业 `pct_chg` 混用百分数与小数、GNN 同月标签问题。24.7% Regime 版本跨 87 个日历月只记录 72 个月，且无法由同一提交保存的预测 CSV 与 Regime 标签重现其净值；下表只保留实验历史，不用于新旧绩效比较。

| 策略 | 年化收益 | Sharpe | 最大回撤 | 超额(vs等权) |
|------|---------|--------|---------|-------------|
| 等权基准 | 2.8% | -0.01 | -33.5% | — |
| GNN | 12.9% | 0.469 | -26.4% | +10.0% |
| LSTM-B | 11.9% | 0.429 | -32.2% | +9.1% |
| 等权集成 | **31.2%** | **1.282** | **-22.1%** | **+28.4%** |
| 自适应集成 | 25.1% | 1.003 | -22.5% | +22.3% |
| Regime集成 | 24.7% | 0.991 | -22.5% | +21.9% |

> 历史最高（早期训练版本）：等权集成 Sharpe 1.38，详见下方分支说明各自的代表数。

## 分支说明

仓库维护 3 个并行实验分支，回应同一个核心问题——**HMM Regime 受宏观数据发布滞后影响**（M1M2/SF 月中才发布，月初到月中 Regime 分类不稳定）。三种解法：

| 分支 | 思路 | Regime 处理 | 集成方式 | Sharpe |
|------|------|------------|---------|--------|
| `main` | 基线方案 | 滚动窗口 HMM + FiLM 调制 | Regime 条件集成（按状态自适应权重） | **1.38** |
| `fix/macro-neutral-fill` | 修缺失/不修结构 | 全量 HMM 替代滚动窗口；宏观缺失用历史均值（非 ffill）做"中性化" | Regime 条件集成 | 1.29 |
| `refactor/equal-weight-ensemble` | 极简化 | **完全移除** HMM / 宏观因子 / FiLM 调制 | 固定 50/50 等权（GNN + LSTM-B） | 1.28 |

**关键 commits（见各分支历史）**

- `fix`：`9e2a7c9` 宏观缺失改用历史均值 → `808f78b` 全量 HMM 替换滚动窗口
- `refactor`：`1c94d80` 移除 Regime 条件集成 → `81304d9` 从 pipeline 完整移除宏观因子/HMM

**旧实验结论（已失效）**：下方分支比较基于修正前的收益率及标签口径，不可据此断言滚动 HMM 优于固定权重。新的同口径回测见上表。

> Dashboard 周报页可通过左侧 **Main / Fix / Refactor** 按钮切换查看三份历史结果对比（需在该分支跑过流水线，跑完会自动归档至 `reports/branches/{branch}.md`）。

## 项目结构

```
src/
├── config.py                       # 配置 + 数据加载
├── s0_regime.py                    # 成交额主导的 HMM 状态识别（3状态滚动训练）
├── s1_gnn_train.py                 # GAT + GLASSO 行业基本面联动
├── s2_lstm_b_train.py              # LSTM-B 技术因子动量预测
├── s3_ensemble_backtest.py         # 自适应/等权/Regime 集成 + 回测
├── s4_beta_selection.py            # 行业内个股选股（β + 动量 + quality 复合）
├── s5_etf_backtest.py              # ETF 实盘可行性验证
├── s6_etf_execution_backtest.py    # ETF 执行回测（含费率/滑点）
├── s7_budget_portfolio.py          # 2.5 万元整手主板组合优化（末端模块）
├── run_all.py                      # 一键回测（s0→s1→s2→s3）
│
├── data_pipeline/                  # 数据自包含
│   ├── download.py                 # 个股日线 + 财务报表下载/迁移
│   ├── update.py                   # 增量拉取（sw行业/csi300/宏观/因子）
│   ├── tech_factors.py             # 价量因子生成（个股→行业中位数）
│   ├── pattern_factors.py          # 走势复刻因子
│   ├── prosperity.py               # 景气度指标（财务三表→TTM→行业聚合）
│   ├── etf_mapping_v2.py           # 申万行业 → ETF 映射（卡尔曼 β + R²）
│   └── tushare_config.py           # Tushare token 配置
│
├── live/                           # 实盘信号系统
│   ├── predict.py                  # 提取最新月份 Top-K 信号
│   ├── monitor.py                  # 对比上期 + 生成周报 markdown
│   ├── infer_incremental.py        # 月度增量推理（无需重训，复用 inference_state）
│   ├── train.py                    # 季末全量重训 / 月末微调
│   ├── scheduler.py                # Python schedule 调度（周日 20:00）
│   ├── trade_cal.py                # 交易日历
│   └── state.py                    # 状态持久化（持仓/模型版本）
│
└── notifier/                       # 告警抽象层（当前: noop + file）

dashboard/                          # 前端 Dashboard（FastAPI + 桌面 webview）
├── main.py                         # 入口：uvicorn + pywebview 桌面窗口
├── api/                            # 路由（report / holdings / backtest / system / runner）
├── utils/data_loader.py            # 周报 .md 解析 + 回测 CSV 加载
└── static/                         # 前端 SPA（HTML + CSS + 原生 JS）

launch_dashboard.exe                # PyInstaller 打包的桌面启动器（不入 git）

data/                               # 数据（未入仓库）
├── raw/                            # tushare 原始数据
├── processed/                      # 景气度 / 技术因子
└── cache/                          # 推理结果缓存
models/                             # 模型快照（current / quarterly / monthly）
reports/                            # 周报（latest.md + 按周/分支归档）
results/                            # 回测结果（csv + png + ckpt）
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 一键回测（s0→s3，约 10 分钟）
cd src
python run_all.py

# 实盘信号（需先跑过一次回测或 live.train）
python -m live.predict
python -m live.monitor        # 生成 reports/latest.md
```

### 2.5 万元组合预览

在项目根目录运行 `python src/s7_budget_portfolio.py`。模块重新生成最新的全市场候选，让创业板、科创板参与上游排序；最终组合只保留沪深主板，按 100 股整数倍、2.5 万元预算、单股最多 30% 和每行业最多一只求整数最优解。结果写入 `results/budget_portfolio_YYYYMM.csv`。`--month YYYY-MM` 可试算已保存的历史候选；旧候选可能已被主板过滤。

默认文件以信号月末收盘价测算，不是委托单。下一交易日可准备 `stock_code,limit_price` 两列 CSV，并运行 `python src/s7_budget_portfolio.py --prices-csv 报价.csv`；优化只使用该文件中有价格的候选，按买入限价核算 2.5 万元硬上限。限价订单可能不成交。当前按使用者要求不计佣金、印花税及滑点；资金约束模块尚未做完整历史执行回测，不据此宣称实盘收益。

## Dashboard

桌面可视化（FastAPI + pywebview，1440×900 窗口，端口 8765）：

```bash
# 直接跑源码（开发用，磁盘改动即时生效）
python dashboard/main.py

# 或双击根目录的 launch_dashboard.exe（PyInstaller 打包，免装 webview）
```

四个页面：**周报**（含分支切换）、**持仓**、**回测**（净值曲线 + 绩效表）、**系统**（数据源时效 + 重训日期）。点 **▶ 运行流水线** 会按选中分支跑完整 pipeline，跑完自动归档至 `reports/branches/{branch}.md` 供后续切换查看。

## 实盘化用法

```bash
# 增量更新行业行情、沪深300及韩美指数（宏观文件仅供旧版兼容）
python -m data_pipeline.update

# 增量更新全部数据（含个股日线/财报/因子，较慢）
python -m data_pipeline.update --processed

# 月度增量推理（无需重训，~30 秒）
python -m live.infer_incremental

# 季末全量重训
python -m live.train --mode quarterly

# 调度器常驻（周日 20:00 自动 update → predict → 周报）
python -m live.scheduler
```

## 原始架构修复对照（2026-09-26）

后续架构/因子验证见 [2026-09-27 改进实验报告](reports/improvement_research_2026-09-27.md)。该实验独立保存在 `results/improvement_research/`，含 36 个初始候选及五随机种子复核；最终比较以 `verified/` 为准，没有自动替换本项目默认模型。

`src/run_original_architecture.py` 单独恢复改动前 `818423c` 的架构：宏观 4 状态 HMM（full 协方差、60 月滚动）→ GLASSO/GAT + 原 12 因子 LSTM → 同状态最近 12 次历史 IC 动态配权。没有固定 4:6 权重，也没有 LSTM 权重下限。

保留月线单位、月份去重、下一月标签和滚动尾窗修复；额外统一宏观利差为 SHIBOR 1 年减 1 月，缺发布日期的宏观发布指标滞后一个月，并让状态独热列避开截面标准化，防止被归零。实际恢复的是提交中的 GAT + 状态拼接实现，不是文档中提到但该实现未使用的 FiLM。

```bash
# 从 HMM 到两个分支重新训练，再与现有六组固定权重结果按相同月份比较
python src/run_original_architecture.py

# 校验源数据/代码/预测指纹后，仅重放集成回测
python src/run_original_architecture.py --backtest-only
```

输出隔离在 `results/original_architecture/`：`annual_comparison.csv`（逐年）、`monthly_returns.csv`（逐月）、`monthly_weights.csv`（动态权重）、`manifest.json`（配置及输入 SHA-256）。不会覆盖当前六组方案的预测。2026 年列是已完成月份累计收益，不外推全年。

## 数据来源

### 独立价量风险模型入口（2026-09-27）

接入状态及待补数据见 [独立入口验证记录](reports/price_risk_pipeline_2026-09-27.md)。

```bash
# 重新训练 Ridge + 五种子 LightGBM，复现行业层逐月收益
python src/price_risk_pipeline.py

# 接受外部提供的当月个股候选，生成金额/整手约束分配计划
# 候选必须含 month、stock_code、ind_code、rank_in_ind；不是账户回测命令
python src/price_risk_pipeline.py --candidates path/to/candidates.csv --month 2026-08 --equity 25000 --capital-cap 25000
```

输出隔离在 `results/price_risk_pipeline/`，未替换默认流水线；这里保存的是行业层结果。候选股生成及完整账户回测已由下方独立账户实验接通，结果另存 `results/stock_execution_research/`，两层收益不可混用。

新 GET 数据接口通过环境变量 `TUSHARE_API_KEY` 启用（密钥不写入源码），默认地址为 `https://tl.kaixin8.top/tushare/pro`，可用 `TUSHARE_GATEWAY_URL` 覆盖；仍支持旧 `TUSHARE_TOKEN`/`TUSHARE_URL` SDK 配置。新入口保持 TLS 校验，返回兼容原调用代码的 DataFrame。设置 `PYTHONPATH=src` 后可运行 `python -m data_pipeline.probe_execution_data` 做小样本连通性检查，结果隔离在 `data/raw/execution_probe/`，不表示历史覆盖完整。

独立账户实验入口（输出在 `results/stock_execution_research/`，不是替换默认流程）：

完整 2023—2026 年账户结果、数据修复证据和核验口径见 [2026-09-28 整手账户实验报告](reports/small_account_results_2026-09-28.md)。

```bash
# 先设置 PYTHONPATH=src 和进程环境中的 TUSHARE_API_KEY
python -m data_pipeline.execution_data --quotes
python -m data_pipeline.execution_boundaries
python src/stock_execution_research.py
python src/small_account_backtest.py
# 所需缓存齐备后，禁用网络重新播放并独立核对账本
python src/small_account_backtest.py --offline
python src/verify_small_account.py
```

账户实验给新旧行业信号配同一个历史成员/复权动量选股器；其中 `original` 指旧行业模型，并非复用有时点问题的旧基本面选股器。完整结果需要三组各 45 个月，并以 `account_manifest.json` 和 `ledger_reconciliation.json` 为完成及核验依据；单独的阶段性 CSV 不代表完整实验已经通过。

- 行业月度行情、沪深300、韩国与美国指数：Tushare（镜像代理）；原始架构对照另使用历史宏观因子
- 行业景气度指标：财务三表 TTM + 券商一致预期
- 技术因子：个股日K线 → 价量因子 + 走势复刻因子 → 行业中位数聚合

> 数据文件未包含在仓库中，需自行通过 Tushare API 获取或从已有项目迁移（`python -m data_pipeline.download --migrate`）。

## 设计决策

- **FiLM 调制**：强制 GAT 每层感知宏观状态，优于简单 concat
- **单层 GAT + 双层 LSTM**：月频 60 样本训练窗口，严格控制参数量
- **Rank IC 损失**：直接优化截面排名相关性
- **滚动 HMM**：避免前瞻偏差，状态通过动量均值排序对齐
- **DropEdge**：图边随机丢弃 10% 防过拟合
- **Year-Month 对齐**：技术因子与行情按月份合并，避免最后交易日 vs 日历月末错位丢数据
- **显式 fwd_ret 目标**：月末因子 → 预测下月收益，消除隐式偏移带来的对齐歧义
- **月度增量推理**：训练时把模型 + scaler 等推理状态打包成 `*_inference_state.pkl`，月度新数据直接 forward pass，无需重训（季末再做一次全量）
- **多因子复合选股（s4）**：行业内按 β（卡尔曼日频估计） + 动量 + quality（ROE/现金流/毛利率）三因子加权打分，叠加持仓惯性 + 换仓缓冲降低换手率
- **基本面时点状态**：旧 s4 个股基本面加载器仍需按公告日及修订版本重建；新的独立账户实验使用历史成员与纯价格特征，不复用该加载器，不把旧路径视为已完成时点修复。
