# -*- coding: utf-8 -*-
"""
Step 2：LSTM-B 分支 —— 技术因子动量预测

沿用 ARIMAX-LSTM 项目的 LSTM-B 方案：
  - 输入：9个价量因子 + 3个走势复刻因子 + 行业 one-hot
  - 所有行业合并训练（共享参数，one-hot 区分行业）
  - Walk-forward 滚动训练

捕捉技术面/动量信号，与 GNN 的基本面分支互补
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import os
import time

from config import *

# ============================================================
#  LSTM-B 模型
# ============================================================

class LSTMBModel(nn.Module):
    """LSTM-B: 技术因子 → 收益率预测"""

    def __init__(self, input_dim, hidden=LSTM_HIDDEN, dropout=LSTM_DROPOUT):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden, hidden // 2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        """x: (batch, lookback, input_dim)"""
        h, _ = self.lstm1(x)
        h = self.drop1(h)
        h, _ = self.lstm2(h)
        out = self.drop2(h[:, -1, :])
        return self.fc(out).squeeze(-1)


# ============================================================
#  数据准备
# ============================================================

def prepare_lstm_b_data(tech_df, mkt, industries, date_range,
                        target_start=None, lookback=LSTM_LOOKBACK,
                        scaler_dict=None):
    """
    准备 LSTM-B 序列数据

    对齐方式（修复数据滞后）：
      - 技术因子和行情按 year-month 合并（避免最后交易日 vs 日历月末差 1 天导致丢行）
      - features[t] = 月 t 的技术因子（月末已知）
      - target = ret[t+1]（下月收益率，即持仓期收益）
      - meta 中记录的 date = 下月（target 月份）的日期
      - 这样在实盘中：月末观察到当月因子 → 预测下月收益 → 月初换仓

    参数：
        tech_df: 技术因子 DataFrame
        mkt: 行业月度行情
        industries: 行业列表
        date_range: 特征数据的日期范围 (start, end)
        target_start: 目标起始日期（仅输出该日期之后的样本）。
                      None 表示从 lookback 之后开始。
        lookback: 时间步长
        scaler_dict: {industry_code: fitted_scaler}，用于预测时复用训练期scaler

    返回：
        X, y, meta, scalers_out
    """
    n_ind = len(industries)
    ind_to_idx = {code: i for i, code in enumerate(industries)}

    factor_cols = [c for c in tech_df.columns if c not in ['ts_code', 'date', 'ym']]
    n_factors = len(factor_cols)

    # 用 year-month 做合并键，解决最后交易日 vs 日历月末的日期错位
    tech_ym = tech_df.copy()
    tech_ym['ym'] = tech_ym['date'].dt.to_period('M')

    ret_ym = mkt[['ts_code', 'date', 'ret']].copy()
    ret_ym['ym'] = ret_ym['date'].dt.to_period('M')

    X_list, y_list, meta = [], [], []
    scalers_out = {}

    for ind_code in industries:
        ind_tech = tech_ym[(tech_ym['ts_code'] == ind_code) &
                           (tech_ym['date'] >= date_range[0]) &
                           (tech_ym['date'] <= date_range[1])].sort_values('date')
        ind_ret = ret_ym[(ret_ym['ts_code'] == ind_code) &
                          (ret_ym['date'] >= date_range[0]) &
                          (ret_ym['date'] <= date_range[1])].sort_values('date')

        if len(ind_tech) < lookback + 1:
            continue

        # 按 year-month 合并，保留行情侧的 date 作为 target_date
        merged = pd.merge(
            ind_tech.drop(columns='date'),
            ind_ret[['ts_code', 'ym', 'date', 'ret']].rename(columns={'date': 'mkt_date'}),
            on=['ts_code', 'ym'], how='inner'
        )
        merged = merged.sort_values('ym').reset_index(drop=True)

        if len(merged) < lookback + 2:  # 需要至少 lookback + 1 个月（+1 给 fwd_ret）
            continue

        # 构造 fwd_ret：下月收益率
        merged['fwd_ret'] = merged['ret'].shift(-1)
        merged['fwd_date'] = merged['mkt_date'].shift(-1)
        # 最后一行没有下月收益，去掉
        merged = merged.dropna(subset=['fwd_ret']).reset_index(drop=True)

        if len(merged) < lookback:
            continue

        # MinMax 归一化
        factor_values = merged[factor_cols].values.astype(float)
        factor_values = np.where(np.isfinite(factor_values), factor_values, 0.0)

        if scaler_dict and ind_code in scaler_dict:
            scaler = scaler_dict[ind_code]
            factor_scaled = scaler.transform(factor_values)
        else:
            scaler = MinMaxScaler()
            factor_scaled = scaler.fit_transform(factor_values)
        scalers_out[ind_code] = scaler

        fwd_ret_values = merged['fwd_ret'].values.astype(float)
        fwd_dates = merged['fwd_date'].values

        # one-hot
        onehot = np.zeros(n_ind, dtype=np.float32)
        onehot[ind_to_idx[ind_code]] = 1.0

        # 序列构建：features[t-lookback+1 : t+1] → predict fwd_ret[t]
        # 即用 t 所在月（含）往前 lookback 个月的因子，预测 t 的下月收益
        for t in range(lookback - 1, len(merged)):
            if target_start is not None and fwd_dates[t] < target_start:
                continue

            seq = np.zeros((lookback, n_factors + n_ind), dtype=np.float32)
            for k in range(lookback):
                seq[k, :n_factors] = factor_scaled[t - lookback + 1 + k]
                seq[k, n_factors:] = onehot

            X_list.append(seq)
            y_list.append(fwd_ret_values[t])
            meta.append((ind_code, fwd_dates[t]))

    if not X_list:
        return None, None, None, scalers_out

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, meta, scalers_out


# ============================================================
#  训练逻辑
# ============================================================

def train_lstm_b_single(X_train, y_train, X_val, y_val, input_dim, seed=42):
    """训练单个 LSTM-B 模型"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cpu')
    model = LSTMBModel(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR, weight_decay=1e-4)

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    n_train = len(X_train)
    batch_size = min(LSTM_BATCH_SIZE, n_train)

    for epoch in range(LSTM_EPOCHS):
        model.train()
        # Shuffle
        perm = torch.randperm(n_train)
        total_loss = 0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]

            xb = X_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)

            pred = model(xb)
            loss = F.mse_loss(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t.to(device))
            val_loss = F.mse_loss(val_pred, y_val_t.to(device)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= LSTM_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss


# ============================================================
#  Walk-Forward 主循环
# ============================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("  LSTM-B 分支：技术因子动量预测")
    print("=" * 60)

    # 加载数据
    print("\n加载数据...")
    mkt = load_industry_monthly()
    tech = load_tech_factors()
    industries = get_industries(mkt)
    n_industries = len(industries)

    factor_cols = [c for c in tech.columns if c not in ['ts_code', 'date']]
    n_factors = len(factor_cols)
    input_dim = n_factors + n_industries
    print(f"  行业数: {n_industries}")
    print(f"  技术因子: {n_factors} → 输入维度: {input_dim}")

    # 可用月份：用 year-month period 取交集（避免日期差 1 天丢月份）
    tech_periods = set(tech['date'].dt.to_period('M'))
    mkt_periods = set(mkt['date'].dt.to_period('M'))
    common_periods = sorted(tech_periods & mkt_periods)

    # 转回 Timestamp（用月末日期），供 walk-forward 索引和 date_range 过滤
    available_months = [p.to_timestamp('M') for p in common_periods]
    print(f"  可用月份: {len(available_months)} ({common_periods[0]} ~ {common_periods[-1]})")

    # Walk-Forward
    total_window = TRAIN_MONTHS + VAL_MONTHS
    all_preds = []
    n_windows = 0

    for start_idx in range(0, len(available_months) - total_window - PREDICT_MONTHS + 1, STEP_MONTHS):
        train_months = available_months[start_idx:start_idx + TRAIN_MONTHS]
        val_months = available_months[start_idx + TRAIN_MONTHS:start_idx + total_window]
        pred_months = available_months[start_idx + total_window:
                                       start_idx + total_window + PREDICT_MONTHS]

        if len(pred_months) == 0:
            continue

        n_windows += 1
        print(f"\n窗口 {n_windows}: 训练 {train_months[0].strftime('%Y-%m')}~"
              f"{train_months[-1].strftime('%Y-%m')}, "
              f"验证 {val_months[0].strftime('%Y-%m')}~{val_months[-1].strftime('%Y-%m')}, "
              f"预测 {pred_months[0].strftime('%Y-%m')}~{pred_months[-1].strftime('%Y-%m')}")

        # 准备训练数据
        train_range = (train_months[0], train_months[-1])
        X_train, y_train, meta_train, train_scalers = prepare_lstm_b_data(
            tech, mkt, industries, train_range)

        if X_train is None or len(X_train) < 50:
            print(f"  跳过（训练数据不足）")
            continue

        # 准备验证数据（用训练期 scaler）
        # 扩展范围以包含 lookback 历史
        val_feat_start = train_months[-LSTM_LOOKBACK] if len(train_months) > LSTM_LOOKBACK else train_months[0]
        val_range = (val_feat_start, val_months[-1])
        X_val, y_val, meta_val, _ = prepare_lstm_b_data(
            tech, mkt, industries, val_range,
            target_start=val_months[0], scaler_dict=train_scalers)

        if X_val is None or len(X_val) < 10:
            print(f"  跳过（验证数据不足）")
            continue

        print(f"  训练样本: {len(X_train)}, 验证样本: {len(X_val)}")

        # 多 seed 训练
        seed_models = []
        for s in range(LSTM_N_SEEDS):
            seed = RANDOM_SEED + s * 100
            model, val_loss = train_lstm_b_single(
                X_train, y_train, X_val, y_val, input_dim, seed=seed)
            print(f"  Seed {s}: Val MSE = {val_loss:.6f}")
            seed_models.append(model)

        # 预测：扩展日期范围以包含 lookback 历史
        pred_feat_start = val_months[-LSTM_LOOKBACK] if len(val_months) > LSTM_LOOKBACK else val_months[0]
        pred_range = (pred_feat_start, pred_months[-1])
        X_pred, y_pred_actual, meta_pred, _ = prepare_lstm_b_data(
            tech, mkt, industries, pred_range,
            target_start=pred_months[0], scaler_dict=train_scalers)

        if X_pred is None or len(X_pred) == 0:
            print("  预测数据不足，跳过")
            continue

        # 多 seed 预测平均
        all_seed_scores = []
        for model in seed_models:
            model.eval()
            with torch.no_grad():
                X_pred_t = torch.tensor(X_pred)
                scores = model(X_pred_t).numpy()
                all_seed_scores.append(scores)

        avg_scores = np.mean(all_seed_scores, axis=0)

        for i, (code, date) in enumerate(meta_pred):
            all_preds.append({
                'ts_code': code,
                'date': date,
                'actual_ret': y_pred_actual[i],
                'pred_lstm_b': avg_scores[i],
            })

    # 保存结果
    pred_df = pd.DataFrame(all_preds)
    pred_df['date'] = pd.to_datetime(pred_df['date'])

    out_path = os.path.join(OUTPUT_DIR, 'predictions_lstm_b.pkl')
    pred_df.to_pickle(out_path)
    pred_df.to_csv(out_path.replace('.pkl', '.csv'), index=False, encoding='utf-8-sig')

    # 评估 IC
    monthly_ics = []
    for dt in pred_df['date'].unique():
        m_data = pred_df[pred_df['date'] == dt]
        if len(m_data) >= 10:
            ic, _ = spearmanr(m_data['pred_lstm_b'], m_data['actual_ret'])
            if np.isfinite(ic):
                monthly_ics.append(ic)

    if monthly_ics:
        mean_ic = np.mean(monthly_ics)
        std_ic = np.std(monthly_ics)
        icir = mean_ic / (std_ic + 1e-8)
        pos_ratio = np.mean([ic > 0 for ic in monthly_ics])
        print(f"\n{'='*60}")
        print(f"  LSTM-B 分支评估")
        print(f"{'='*60}")
        print(f"  Rank IC: {mean_ic:.4f} ± {std_ic:.4f}")
        print(f"  ICIR:    {icir:.4f}")
        print(f"  IC>0:    {pos_ratio:.1%}")
        print(f"  月数:    {len(monthly_ics)}")

    # 保存最后一个窗口的推理状态（供 live/infer_incremental.py 月度增量推理）
    if n_windows > 0:
        import pickle as _pkl
        infer_state = {
            'model_states': [
                {k: v.clone().cpu() for k, v in m.state_dict().items()}
                for m in seed_models
            ],
            'scalers': train_scalers,
            'factor_cols': factor_cols,
            'industries': industries,
            'input_dim': input_dim,
            'train_end_month': pd.Timestamp(train_months[-1]).strftime('%Y-%m-%d'),
            'pred_end_month': pd.Timestamp(pred_months[-1]).strftime('%Y-%m-%d'),
        }
        _state_path = os.path.join(OUTPUT_DIR, 'lstm_b_inference_state.pkl')
        with open(_state_path, 'wb') as _f:
            _pkl.dump(infer_state, _f, protocol=4)
        print(f"  LSTM-B 推理状态已保存 → {_state_path}")

    elapsed = time.time() - t0
    print(f"\nLSTM-B 训练完成，耗时 {elapsed/60:.1f} 分钟")
    print(f"预测保存至: {out_path}")


if __name__ == '__main__':
    main()
