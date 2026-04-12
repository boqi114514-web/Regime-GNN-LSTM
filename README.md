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
                                                    Top-K 行业组合 → 周报
```

## 回测表现（2019-01 ~ 2026-03）

| 策略 | 年化收益 | Sharpe | 最大回撤 | 超额(vs等权) |
|------|---------|--------|---------|-------------|
| 等权基准 | 2.8% | -0.01 | -33.5% | — |
| GNN | 19.5% | 0.77 | -27.2% | +16.7% |
| LSTM-B | 11.0% | 0.39 | -32.2% | +8.2% |
| 等权集成 | **32.7%** | **1.38** | **-17.9%** | **+29.9%** |
| 自适应集成 | 28.9% | 1.14 | -23.7% | +26.1% |
| Regime集成 | 28.1% | 1.10 | -26.1% | +25.3% |

## 项目结构

```
src/
├── config.py                  # 配置 + 数据加载
├── s0_regime.py               # HMM 宏观状态识别（4状态滚动训练）
├── s1_gnn_train.py            # GAT + GLASSO 行业基本面联动
├── s2_lstm_b_train.py         # LSTM-B 技术因子动量预测
├── s3_ensemble_backtest.py    # 自适应集成 + 回测
├── s5_etf_backtest.py         # ETF 实盘可行性验证
├── run_all.py                 # 一键回测（s0→s1→s2→s3）
│
├── data_pipeline/             # 数据自包含
│   ├── download.py            # 个股日线 + 财务报表下载/迁移
│   ├── update.py              # 增量拉取（sw行业/csi300/宏观/因子）
│   ├── tech_factors.py        # 价量因子生成（个股→行业中位数）
│   ├── pattern_factors.py     # 走势复刻因子
│   └── prosperity.py          # 景气度指标（财务三表→TTM→行业聚合）
│
├── live/                      # 实盘信号系统
│   ├── predict.py             # 提取最新月份 Top-K 信号
│   ├── monitor.py             # 对比上期 + 生成周报 markdown
│   ├── train.py               # 季末全量重训 / 月末微调
│   ├── scheduler.py           # Python schedule 调度（周日 20:00）
│   └── state.py               # 状态持久化（持仓/模型版本）
│
└── notifier/                  # 告警抽象层（当前: noop + file）

data/                          # 数据（未入仓库）
├── raw/                       # tushare 原始数据
├── processed/                 # 景气度 / 技术因子
└── cache/                     # 推理结果缓存
models/                        # 模型快照（current / quarterly / monthly）
reports/                       # 周报（latest.md + 归档）
results/                       # 回测结果（csv + png）
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

## 实盘化用法

```bash
# 增量更新行情 + 宏观数据
python -m data_pipeline.update

# 增量更新全部数据（含个股日线/财报/因子，较慢）
python -m data_pipeline.update --processed

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
