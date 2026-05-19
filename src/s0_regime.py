# -*- coding: utf-8 -*-
"""
Step 0：HMM 宏观状态识别

识别4种宏观状态（衰退/复苏/扩张/过热），为 GNN 和集成提供宏观上下文。

输入：宏观因子（PMI、利差、M1-M2、社融）+ 大盘价格信号（动量、波动率）
输出：regime_labels.pkl（每月 regime 标签 + 概率）
"""

import numpy as np
import pandas as pd
import warnings
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import os
import time

warnings.filterwarnings('ignore')

from config import (
    load_macro_factors, load_csi300_monthly,
    HMM_N_STATES, HMM_COVARIANCE, HMM_N_ITER, HMM_TRAIN_WINDOW,
    OUTPUT_DIR,
)


def build_regime_features(macro_df, csi300_df):
    """构建 HMM 观测特征（宏观 + 大盘价格信号）"""
    macro = macro_df.copy().sort_values('date').reset_index(drop=True)
    macro['pmi_chg'] = macro['pmi_mfg'].diff()
    # sf_yoy 原列是"社融月度增量的同比"，季节性噪声极大（全样本标准差 ~1.5），
    # 单月极端值（如 2026-04 的 -0.46）会把 HMM regime 带成虚假衰退。
    # 改用 12 个月滚动社融总量的同比（= 滚动年度社融增速），消除季节性、
    # 口径稳健（标准差降到 ~0.17），与宏观分析常用的社融增速口径一致。
    _sf_roll12 = macro['sf_inc_month'].rolling(12).sum()
    macro['sf_yoy'] = _sf_roll12 / _sf_roll12.shift(12) - 1.0
    macro_cols = ['pmi_chg', 'term_spread', 'm1_m2_spread', 'sf_yoy']

    csi = csi300_df[['date', 'close']].copy().sort_values('date')
    csi['ret'] = csi['close'].pct_change()
    csi['mkt_mom_1m'] = csi['close'].pct_change(1)
    csi['mkt_mom_3m'] = csi['close'].pct_change(3)
    csi['mkt_mom_6m'] = csi['close'].pct_change(6)
    csi['mkt_vol_3m'] = csi['ret'].rolling(3).std()
    csi['mkt_vol_6m'] = csi['ret'].rolling(6).std()
    csi['mkt_vol_ratio'] = csi['mkt_vol_3m'] / (csi['mkt_vol_6m'] + 1e-8)
    ma6 = csi['close'].rolling(6).mean()
    csi['mkt_bias_6m'] = csi['close'] / (ma6 + 1e-8) - 1
    hi_12 = csi['close'].rolling(12).max()
    lo_12 = csi['close'].rolling(12).min()
    csi['mkt_pos_12m'] = (csi['close'] - lo_12) / (hi_12 - lo_12 + 1e-8)

    price_cols = ['mkt_mom_1m', 'mkt_mom_3m', 'mkt_mom_6m',
                  'mkt_vol_3m', 'mkt_vol_6m', 'mkt_vol_ratio',
                  'mkt_bias_6m', 'mkt_pos_12m']

    macro['ym'] = macro['date'].dt.to_period('M')
    csi['ym'] = csi['date'].dt.to_period('M')

    merged = pd.merge(
        macro[['date', 'ym'] + macro_cols],
        csi[['ym'] + price_cols],
        on='ym', how='inner'
    )
    feature_cols = macro_cols + price_cols

    # 前向填充缺失值（宏观指标偶尔缺月，用上期值比直接丢弃好）
    merged = merged.sort_values('date').reset_index(drop=True)
    merged[feature_cols] = merged[feature_cols].ffill()
    # 仅丢弃前向填充后仍全空的行（序列开头无法 ffill 的部分）
    merged = merged.dropna(subset=feature_cols, how='all').reset_index(drop=True)
    return merged, feature_cols


def align_states(hmm_model, X_scaled, n_states):
    """通过动量均值排序保证状态含义一致：0=衰退, 3=过热"""
    labels = hmm_model.predict(X_scaled)
    mom_col_idx = 5  # mkt_mom_3m

    state_means = []
    for s in range(n_states):
        mask = labels == s
        if mask.sum() > 0:
            state_means.append((s, X_scaled[mask, mom_col_idx].mean()))
        else:
            state_means.append((s, 0.0))

    state_means.sort(key=lambda x: x[1])
    return {old: new for new, (old, _) in enumerate(state_means)}


def rolling_hmm(features_df, feature_cols, train_window=HMM_TRAIN_WINDOW,
                n_states=HMM_N_STATES):
    """滚动训练 HMM，逐月产出 regime 标签"""
    n = len(features_df)
    results = []

    for t in range(train_window, n):
        train_slice = features_df.iloc[t - train_window:t]
        X_train = train_slice[feature_cols].values.astype(float)
        X_current = features_df.iloc[t:t + 1][feature_cols].values.astype(float)

        # ffill 后仍可能有残余 NaN（序列开头），用列均值填充
        col_means = np.nanmean(X_train, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        for j in range(X_train.shape[1]):
            mask = np.isnan(X_train[:, j])
            if mask.any():
                X_train[mask, j] = col_means[j]
            if np.isnan(X_current[0, j]):
                X_current[0, j] = col_means[j]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_current_s = scaler.transform(X_current)

        try:
            hmm = GaussianHMM(
                n_components=n_states, covariance_type=HMM_COVARIANCE,
                n_iter=HMM_N_ITER, random_state=42,
            )
            hmm.fit(X_train_s)
            state_map = align_states(hmm, X_train_s, n_states)

            X_full = np.vstack([X_train_s, X_current_s])
            raw_labels = hmm.predict(X_full)
            raw_proba = hmm.predict_proba(X_full)
            aligned_label = state_map[raw_labels[-1]]

            proba = np.zeros(n_states)
            for old, new in state_map.items():
                proba[new] = raw_proba[-1, old]
        except Exception:
            aligned_label = -1
            proba = np.full(n_states, 1.0 / n_states)

        results.append({
            'date': features_df.iloc[t]['date'],
            'regime': aligned_label,
            **{f'regime_prob_{i}': proba[i] for i in range(n_states)},
        })

    return pd.DataFrame(results)


def main():
    t0 = time.time()
    print("=" * 60)
    print("  HMM 宏观状态识别")
    print("=" * 60)

    macro = load_macro_factors()
    csi300 = load_csi300_monthly()

    print("\n构建特征...")
    features, feature_cols = build_regime_features(macro, csi300)
    print(f"  特征数: {len(feature_cols)}, 样本数: {len(features)}")
    print(f"  日期范围: {features['date'].min().strftime('%Y-%m')} ~ "
          f"{features['date'].max().strftime('%Y-%m')}")

    print(f"\n滚动 HMM ({HMM_N_STATES}状态, 窗口={HMM_TRAIN_WINDOW}月)...")
    regime_df = rolling_hmm(features, feature_cols)

    # 统计
    regime_names = {-1: '失败', 0: '衰退', 1: '复苏', 2: '扩张', 3: '过热'}
    print(f"\n  Regime 分布:")
    for r, cnt in regime_df['regime'].value_counts().sort_index().items():
        name = regime_names.get(r, f'状态{r}')
        print(f"    {name}({r}): {cnt} 月 ({cnt/len(regime_df):.0%})")

    # 保存
    out_path = os.path.join(OUTPUT_DIR, 'regime_labels.pkl')
    regime_df.to_pickle(out_path)
    regime_df.to_csv(out_path.replace('.pkl', '.csv'), index=False, encoding='utf-8-sig')

    elapsed = time.time() - t0
    print(f"\n完成，耗时 {elapsed:.0f}秒，保存至 {out_path}")
    return regime_df


if __name__ == '__main__':
    main()
