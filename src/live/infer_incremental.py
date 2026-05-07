# -*- coding: utf-8 -*-
"""月度增量推理（Phase 5）

每月末 pv_factors 更新后，无需完整重训，直接用已保存的最后一个窗口权重
对新月份做 forward pass，追加到预测 pkl，更新集成文件。

被 live/monitor.py 在每次生成周报前调用。
"""
import os
import sys
import pickle
import shutil

import numpy as np
import pandas as pd
import torch

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import (
    OUTPUT_DIR, MODELS_CURRENT_DIR, LSTM_LOOKBACK, GLASSO_ROLLING_MONTHS,
    load_industry_monthly, load_prosperity_monthly, load_tech_factors,
    get_industries, zscore_cross_section,
)
from s1_gnn_train import IndustryGAT, build_glasso_graph
from s2_lstm_b_train import LSTMBModel
from s3_ensemble_backtest import simple_rank_ensemble


# ─── 路径 ─────────────────────────────────────────────────────────────────────

_GNN_STATE  = os.path.join(OUTPUT_DIR, 'gnn_inference_state.pkl')
_LSTM_STATE = os.path.join(OUTPUT_DIR, 'lstm_b_inference_state.pkl')
_GNN_PKL    = os.path.join(OUTPUT_DIR, 'predictions_gnn.pkl')
_LSTM_PKL   = os.path.join(OUTPUT_DIR, 'predictions_lstm_b.pkl')
_ENS_PKL    = os.path.join(OUTPUT_DIR, 'predictions_ensemble.pkl')
_EQENS_PKL  = os.path.join(OUTPUT_DIR, 'predictions_ensemble_equal.pkl')

# ─── 工具 ─────────────────────────────────────────────────────────────────────

def _load_state(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def _load_pred_df(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_pickle(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def _find_new_months(gnn_df, tech):
    """返回在 tech 中但尚未在 gnn_df 里有预测的月份列表（月末 Timestamp）。"""
    tech_c = tech.copy()
    tech_c['date'] = pd.to_datetime(tech_c['date'])
    tech_latest = tech_c['date'].max()

    if gnn_df.empty:
        return []
    gnn_latest = gnn_df['date'].max()

    from pandas.tseries.offsets import MonthEnd
    new = []
    m = gnn_latest + MonthEnd(1)
    while m <= tech_latest:
        new.append(m)
        m = m + MonthEnd(1)
    return new


# ─── GNN 推理 ──────────────────────────────────────────────────────────────────

def _infer_gnn(new_months, state, mkt, prosperity):
    """对 new_months 中的每个月运行 GNN forward pass，返回 row 列表。"""
    feature_cols = state['feature_cols']
    in_features  = state['in_features']
    industries   = state['industries']
    n_industries = len(industries)
    ret_pivot_saved = state.get('ret_pivot')

    # 加载模型
    models = []
    for sd in state['model_states']:
        m = IndustryGAT(in_features)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)

    # 构造特征表（含 pe/pb 百分位、regime one-hot）
    mkt_merged = pd.merge(
        mkt[['ts_code', 'date', 'year', 'month', 'ret', 'pe', 'pb']],
        prosperity,
        on=['ts_code', 'year', 'month'],
        how='inner',
    ).sort_values(['date', 'ts_code']).reset_index(drop=True)

    for col in ['pe', 'pb']:
        mkt_merged[f'{col}_pctile'] = mkt_merged.groupby('ts_code')[col].transform(
            lambda x: x.rolling(60, min_periods=12).rank(pct=True)
        )

    mkt_merged = zscore_cross_section(mkt_merged, feature_cols)

    # 收益率 pivot（用于构建 GLASSO 图）
    full_ret_pivot = mkt.pivot_table(
        index='date', columns='ts_code', values='ret', aggfunc='first')
    full_ret_pivot = full_ret_pivot.reindex(columns=industries)
    # 合并已保存的历史数据（防止 mkt CSV 覆盖范围不足）
    if ret_pivot_saved is not None:
        combined_pivot = pd.concat(
            [ret_pivot_saved[~ret_pivot_saved.index.isin(full_ret_pivot.index)],
             full_ret_pivot]
        ).sort_index()
    else:
        combined_pivot = full_ret_pivot

    new_rows = []
    for month in new_months:
        # 特征：优先用当月数据，缺失时前向填充
        month_data = mkt_merged[mkt_merged['date'] == month]
        if month_data.empty:
            last_known = mkt_merged[mkt_merged['date'] < month]
            if last_known.empty:
                print(f"  [GNN增量] {month.strftime('%Y-%m')} 无可用特征，跳过")
                continue
            month_data = last_known[last_known['date'] == last_known['date'].max()].copy()

        month_data = month_data.set_index('ts_code').reindex(industries)
        feat = month_data[feature_cols].values.astype(np.float32)
        feat = np.nan_to_num(feat, nan=0.0)

        # 重新做截面 z-score（前向填充时行业间可能尺度不一）
        for j in range(feat.shape[1]):
            col_vals = feat[:, j]
            mu, sigma = np.nanmean(col_vals), np.nanstd(col_vals)
            feat[:, j] = (col_vals - mu) / (sigma + 1e-8)
            feat[:, j] = np.nan_to_num(feat[:, j], nan=0.0)

        # GLASSO 图
        past_dates = [d for d in combined_pivot.index if d < month]
        if len(past_dates) < 6:
            adj = np.eye(n_industries)
        else:
            window = combined_pivot.loc[past_dates[-GLASSO_ROLLING_MONTHS:]].values
            adj = build_glasso_graph(window)

        feat_t = torch.tensor(feat)
        adj_t  = torch.tensor(adj.astype(np.float32))

        month_ranks = []
        for model in models:
            with torch.no_grad():
                pred = model(feat_t, adj_t).numpy()
            rank = np.argsort(np.argsort(-pred)).astype(float) + 1
            month_ranks.append(rank)
        avg_score = -np.mean(month_ranks, axis=0)

        for i, code in enumerate(industries):
            new_rows.append({
                'ts_code': code,
                'date': month,
                'actual_ret': np.nan,
                'pred_gnn': float(avg_score[i]),
            })
        print(f"  [GNN增量] {month.strftime('%Y-%m')} 推理完成")

    return new_rows


# ─── LSTM-B 推理 ───────────────────────────────────────────────────────────────

def _infer_lstm_b(new_months, state, tech):
    """对 new_months 中的每个月运行 LSTM-B forward pass，返回 row 列表。"""
    factor_cols = state['factor_cols']
    scalers     = state['scalers']
    input_dim   = state['input_dim']
    industries  = state['industries']
    n_ind       = len(industries)
    ind_to_idx  = {code: i for i, code in enumerate(industries)}

    models = []
    for sd in state['model_states']:
        m = LSTMBModel(input_dim)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)

    tech_c = tech.copy()
    tech_c['date'] = pd.to_datetime(tech_c['date'])
    tech_c['ym']   = tech_c['date'].dt.to_period('M')

    new_rows = []
    for month in new_months:
        month_period  = month.to_period('M')
        # 需要 lookback 个月的特征：[month - lookback + 1, ..., month]
        needed = [month_period - (LSTM_LOOKBACK - 1 - k) for k in range(LSTM_LOOKBACK)]

        all_scores = [[] for _ in range(n_ind)]

        for model in models:
            X_list, valid_idx = [], []

            for idx, ind_code in enumerate(industries):
                if ind_code not in scalers:
                    continue
                scaler   = scalers[ind_code]
                ind_tech = tech_c[tech_c['ts_code'] == ind_code].sort_values('ym')
                onehot   = np.zeros(n_ind, dtype=np.float32)
                onehot[ind_to_idx[ind_code]] = 1.0

                seq   = np.zeros((LSTM_LOOKBACK, len(factor_cols) + n_ind), dtype=np.float32)
                valid = True
                for k, period in enumerate(needed):
                    row = ind_tech[ind_tech['ym'] == period]
                    if row.empty:
                        valid = False
                        break
                    factors = row.iloc[0][factor_cols].values.astype(float)
                    factors = np.where(np.isfinite(factors), factors, 0.0)
                    scaled  = scaler.transform(factors.reshape(1, -1))[0].astype(np.float32)
                    seq[k, :len(factor_cols)] = scaled
                    seq[k, len(factor_cols):]  = onehot

                if not valid:
                    continue
                X_list.append(seq)
                valid_idx.append(idx)

            if not X_list:
                continue
            X_t = torch.tensor(np.array(X_list))
            with torch.no_grad():
                scores = model(X_t).numpy()
            for j, idx in enumerate(valid_idx):
                all_scores[idx].append(float(scores[j]))

        for idx, ind_code in enumerate(industries):
            avg = float(np.mean(all_scores[idx])) if all_scores[idx] else 0.0
            new_rows.append({
                'ts_code': ind_code,
                'date': month,
                'actual_ret': np.nan,
                'pred_lstm_b': avg,
            })
        print(f"  [LSTM-B增量] {month.strftime('%Y-%m')} 推理完成")

    return new_rows


# ─── 集成 + 同步 ───────────────────────────────────────────────────────────────

def _rebuild_ensemble(gnn_df, lstm_df):
    """重新生成等权集成 pkl（含新追加行）。_ENS_PKL 与 _EQENS_PKL 均写等权结果。"""
    equal_ens = simple_rank_ensemble(gnn_df, lstm_df)
    if equal_ens is not None:
        equal_ens.to_pickle(_ENS_PKL)
        equal_ens.to_pickle(_EQENS_PKL)
        print(f"  [增量集成] 等权集成更新 → {len(equal_ens)} 行")


def _sync_to_current():
    for fname in ['predictions_gnn.pkl', 'predictions_lstm_b.pkl',
                  'predictions_ensemble.pkl', 'predictions_ensemble_equal.pkl']:
        src = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(MODELS_CURRENT_DIR, fname))
    print("  [增量推理] 已同步到 models/current/")


# ─── 主入口 ────────────────────────────────────────────────────────────────────

def run() -> bool:
    """增量推理主入口。有新月份时返回 True，无则返回 False。"""
    print("  [增量推理] 检查是否有新月份...")

    gnn_state  = _load_state(_GNN_STATE)
    lstm_state = _load_state(_LSTM_STATE)
    if gnn_state is None or lstm_state is None:
        print("  [增量推理] 推理状态文件不存在，跳过（请先运行季度重训以生成状态文件）")
        return False

    gnn_df  = _load_pred_df(_GNN_PKL)
    lstm_df = _load_pred_df(_LSTM_PKL)
    if gnn_df.empty:
        print("  [增量推理] predictions_gnn.pkl 不存在，跳过")
        return False

    tech       = load_tech_factors()
    new_months = _find_new_months(gnn_df, tech)

    if not new_months:
        print(f"  [增量推理] 无新月份（当前最新：{gnn_df['date'].max().strftime('%Y-%m')}）")
        return False

    print(f"  [增量推理] 发现新月份：{[m.strftime('%Y-%m') for m in new_months]}")

    mkt        = load_industry_monthly()
    prosperity = load_prosperity_monthly()

    gnn_rows  = _infer_gnn(new_months, gnn_state, mkt, prosperity)
    lstm_rows = _infer_lstm_b(new_months, lstm_state, tech)

    if not gnn_rows and not lstm_rows:
        print("  [增量推理] 推理结果为空，跳过")
        return False

    if gnn_rows:
        new_gnn = pd.DataFrame(gnn_rows)
        new_gnn['date'] = pd.to_datetime(new_gnn['date'])
        gnn_df = pd.concat([gnn_df, new_gnn], ignore_index=True)
        gnn_df = gnn_df.drop_duplicates(subset=['ts_code', 'date'], keep='last')
        gnn_df.to_pickle(_GNN_PKL)

    if lstm_rows:
        new_lstm = pd.DataFrame(lstm_rows)
        new_lstm['date'] = pd.to_datetime(new_lstm['date'])
        lstm_df = pd.concat([lstm_df, new_lstm], ignore_index=True)
        lstm_df = lstm_df.drop_duplicates(subset=['ts_code', 'date'], keep='last')
        lstm_df.to_pickle(_LSTM_PKL)

    _rebuild_ensemble(gnn_df, lstm_df)
    _sync_to_current()
    return True


def main():
    print('=' * 60)
    print('  live/infer_incremental: 月度增量推理')
    print('=' * 60)
    updated = run()
    print('\n完成' if updated else '\n无更新')


if __name__ == '__main__':
    main()
