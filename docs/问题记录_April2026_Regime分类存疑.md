# 问题记录：April 2026 HMM Regime 分类存疑

**记录时间**：2026-05-06  
**状态**：待排查

---

## 现象

`results/regime_labels.pkl` 中 April 2026 被 HMM 分类为 **regime 0（衰退）**，但市场表现与衰退不符。

---

## 矛盾点

| 指标 | 值 | 含义 |
|------|---|------|
| CSI300 April 涨幅 | **+8.03%** | 强势反弹 |
| CSI300 4月收盘 | 4807.31 | **12个月新高**（mkt_pos_12m = 1.0） |
| 3月 PMI | 50.4（ffilled 至 4月） | 制造业扩张区间 |
| pmi_chg | +1.4（ffilled） | PMI 加速改善 |

---

## 根本原因

### 1. April 宏观数据全部 NaN，靠 ffill 撑着

```
date        pmi_mfg  m1_m2_spread  sf_yoy   term_spread
2026-04-30   NaN       NaN          NaN       0.145
```

HMM 实际看到的 April 特征（`s0_regime.py:build_regime_features` ffill 后）：

| 特征 | 值 | 来源 |
|------|---|------|
| pmi_chg | +1.4 | ffill 自 March |
| term_spread | 0.145 | **实际值**（较 March 0.261 明显下降） |
| m1_m2_spread | -3.4 | ffill 自 March（负） |
| sf_yoy | -11.35% | ffill 自 March（负） |
| mkt_mom_1m | +8.03% | 实际 |
| mkt_vol_3m | 6.81% | 实际（3月-5.5%→4月+8%，高波动） |
| mkt_vol_ratio | 1.478 | 实际（短期波动>>长期） |

**负向信号**（m1_m2 < 0、sf_yoy < 0、高波动率、term_spread 低）盖过了**正向信号**（市场动量、PMI 改善）。

### 2. April PMI 实际值未能入库

4月30日国家统计局已发布 April PMI，但 `ts_macro_factors.csv` 中 April PMI = NaN。  
可能原因：
- a. tushare 私有服务器（101.35.233.113:8020）尚未更新 April PMI
- b. `update_macro_factors` 拉取窗口计算有偏差，跳过了 April PMI
- c. tushare `cn_pmi` 接口 4月额度/权限问题

**待验证**：直接调用 `ts.cn_pmi(start_m='202604', end_m='202604')` 确认是否有数据。

### 3. term_spread 下降明显

April term_spread = 0.145（March = 0.261），下降约 44%。这在历史数据中可能与紧缩或衰退阶段相关，是 HMM 分类为衰退的额外推力。

---

## 影响范围

- `results/regime_labels.pkl` April 2026 = 衰退
- GNN 增量推理 April 2026 已用此 regime 标签（但 April 结果只影响**5月**持仓建议）
- 当前周报（2026-05-06）Top-5：房地产、商贸零售、电子、计算机、传媒（衰退状态下的防御型配置）
- 若 April 实为复苏/扩张，持仓推荐可能偏保守

---

## 下一步排查计划

### Step 1：验证 tushare PMI 数据是否可拉
```python
# 在 src/ 目录下
ts_api.cn_pmi(start_m='202604', end_m='202604')
# 同理验证 M1/M2: cn_m(start_m='202604')
# 社融: sf_month(start_m='202604')
```

### Step 2：若数据可拉，手动入库
- 在 `data/raw/ts_macro_factors.csv` 中更新 2026-04-30 行
- 重跑 `s0_regime.py` → 检查 April regime 是否变化

### Step 3：若 regime 改变
- 强制删除 `results/predictions_gnn.pkl` 中 2026-04-30 行
- 重跑增量推理 `s3_infer_incremental.py`
- 重跑 `s4_lgbm_selection.py`（选股层）
- 重新生成周报 `s6_report.py`

### Step 4：若数据拉不到
- 等待 tushare 服务器更新（通常滞后 1-2 周）
- 或手动从 Wind/Choice/国家统计局 录入

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `src/s0_regime.py:build_regime_features` | ffill 逻辑在此 |
| `src/data_pipeline/update.py:update_macro_factors` | 拉取 PMI/M1M2/SF 的逻辑 |
| `data/raw/ts_macro_factors.csv` | 宏观因子原始数据 |
| `results/regime_labels.pkl` | 当前 HMM 输出（April=衰退） |

---

## 背景

本次排查起因：更新系统时 `sw_industry_monthly` 显示"新窗口内全为空"，用户认为影响周报。  
实际查明：SW 数据完整（April 29行），empty 是拉 May 数据（未到月末，正常）。  
真正的问题是宏观数据 April PMI 未入库，导致 regime 判断可能偏保守。
