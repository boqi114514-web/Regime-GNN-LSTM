# Regime-GNN-LSTM 行业轮动模型

基于宏观状态识别（HMM）+ 图注意力网络（GAT）+ LSTM 的 A 股申万一级行业轮动策略，含实盘化信号系统。

## 架构

```
宏观因子 → [HMM 4状态] → Regime Embedding
                             ↓ FiLM 调制
行业景气度 → [GLASSO] → 动态邻接矩阵 → [GAT] → 行业关系嵌入 ──┐
                                                              ↓ 自适应 IC 加权
技术因子（9价量 + 3走势复刻）────────── [LSTM-B] → 动量信号 ──┘
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

## 回测表现（2019-01 ~ 2026-04，main 分支最近一次 s3 输出）

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

**实验结论**：滚动 HMM + Regime 条件集成（main）回测最优。两个简化方向均不如 main——说明状态切换敏感性虽存在，但 Regime 条件带来的收益超过其引入的不稳定性。`refactor` 以损失 ~0.1 Sharpe 换取大幅简化，工程取舍上仍有意义。

> Dashboard 周报页可通过左侧 **Main / Fix / Refactor** 按钮切换查看三份历史结果对比（需在该分支跑过流水线，跑完会自动归档至 `reports/branches/{branch}.md`）。

## 项目结构

```
src/
├── config.py                       # 配置 + 数据加载
├── s0_regime.py                    # HMM 宏观状态识别（4状态滚动训练）
├── s1_gnn_train.py                 # GAT + GLASSO 行业基本面联动
├── s2_lstm_b_train.py              # LSTM-B 技术因子动量预测
├── s3_ensemble_backtest.py         # 自适应/等权/Regime 集成 + 回测
├── s4_beta_selection.py            # 行业内个股选股（β + 动量 + quality 复合）
├── s5_etf_backtest.py              # ETF 实盘可行性验证
├── s6_etf_execution_backtest.py    # ETF 执行回测（含费率/滑点）
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
# 增量更新行情 + 宏观数据
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

## 数据来源

- 行业月度行情、宏观因子、沪深300：Tushare（镜像代理）
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
- **基本面前视卡口**：以 `ann_date` 决定季报可用月份（披露日下月起），避免 Q2/Q4 数据未发布即被使用
