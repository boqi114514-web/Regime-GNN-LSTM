# Regime-GNN-LSTM 行业轮动模型

基于宏观状态识别（HMM）+ 图注意力网络（GAT）+ LSTM 的 A 股申万一级行业轮动策略。

## 架构

```
宏观因子 → [HMM] → Regime Embedding
                      ↓ FiLM 调制
行业景气度 → [GLASSO] → 动态邻接矩阵 → [GAT] → 行业关系嵌入
                                                  ↓
技术因子 ──────────────────────────── [LSTM-B] → 动量信号
                                                  ↓
                                         自适应集成 → Top-K 行业组合
```

## 回测表现

| 策略 | 年化收益 | Sharpe | 最大回撤 |
|------|---------|--------|---------|
| 等权基准 | 2.9% | -0.00 | -33.5% |
| GNN | 20.2% | 0.78 | -26.5% |
| LSTM-B | 5.5% | 0.11 | -36.9% |
| 自适应集成 | **24.9%** | **0.97** | -25.4% |

## 项目结构

```
src/
├── config.py                # 配置 + 数据加载
├── s0_regime.py             # HMM 宏观状态识别（滚动训练）
├── s1_gnn_train.py          # GAT + GLASSO 行业基本面联动
├── s2_lstm_b_train.py       # LSTM-B 技术因子动量预测
├── s3_ensemble_backtest.py  # 自适应集成 + 回测
├── s5_etf_backtest.py       # ETF 实盘可行性验证
└── run_all.py               # 一键运行
results/                     # 回测结果（csv + png）
```

## 分支说明

- **main**: 行业轮动（s0-s3 + ETF回测）
- **with-stock-selection**: 在行业轮动基础上增加多因子选股（s4），行业轮动 → 卡尔曼β + 动量 + 质量因子 → 个股组合

## 运行

```bash
cd src
python run_all.py
```

## 依赖

```bash
pip install -r requirements.txt
```

## 数据来源

- 行业月度行情、宏观因子、沪深300：Tushare
- 行业景气度指标：财务三表 + 一致预期
- 技术因子：价量因子 + 走势复刻因子

> 注：数据文件未包含在仓库中，需自行通过 Tushare API 获取。

## 设计决策

- **FiLM 调制**：强制每层感知宏观状态，优于简单 concat
- **单层 GAT + 单层 LSTM**：月频 60 样本训练窗口，严格控制参数量
- **Rank IC 损失**：直接优化截面排名相关性
- **滚动 HMM**：避免前瞻偏差，状态通过动量均值排序对齐
- **DropEdge**：图边随机丢弃 10% 防过拟合
