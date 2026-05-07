# -*- coding: utf-8 -*-
"""
Step 0：HMM 宏观状态识别

识别4种宏观状态（衰退/复苏/扩张/过热），为 GNN 和集成提供宏观上下文。

输入：宏观因子（PMI、利差、M1-M2、社融）+ 大盘价格信号（动量、波动率）
输出：regime_labels.pkl（每月 regime 标签 + 概率）
      hmm_model.pkl（训练好的模型，供月度增量推理复用）

训练策略：全量训练（而非滚动窗口）
  滚动窗口每月重训一次新模型，状态聚类每次都可能重新组合，导致相邻月份
  的 regime 标签含义不一致（同一状态编号在不同月份代表不同的市场环境）。
  全量训练用同一个模型对所有月份预测，状态定义固定，相邻月份完全可比。
  季度重训时重新跑 s0，新数据纳入后标签会整体更新。
"""

import pickle
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
    HMM_N_STATES, HMM_COVARIANCE, HMM_N_ITER,
    OUTPUT_DIR,
)

_MODEL_PATH = os.path.join(OUTPUT_DIR, 'hmm_model.pkl')


def build_regime_features(macro_df, csi300_df):
    """构建 HMM 观测特征（宏观 + 大盘价格信号）"""
    macro = macro_df.copy()
    macro['pmi_chg'] = macro['pmi_mfg'].diff()
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

    merged = merged.sort_values('date').reset_index(drop=True)

    # 宏观列：缺失时用扩展历史均值填充（而非 ffill）。
    # ffill 会将上期特殊值带入本期，与当月市场信号产生虚假矛盾；
    # 历史均值相当于"中性"信号，让市场特征主导 regime 判断。
    for col in macro_cols:
        merged[col] = merged[col].fillna(merged[col].expanding().mean())

    # 价格列：前向填充（市场数据连续，偶尔缺月时适用）
    merged[price_cols] = merged[price_cols].ffill()

    # 丢弃仍全空的行（序列开头无法填充的部分）
    merged = merged.dropna(subset=feature_cols, how='all').reset_index(drop=True)
    return merged, feature_cols


def align_states(hmm_model, X_scaled, n_states):
    """通过动量均值排序保证状态含义一致：0=衰退, 3=过热"""
    labels = hmm_model.predict(X_scaled)
    mom_col_idx = 5  # mkt_mom_3m 在 feature_cols 中的位置

    state_means = []
    for s in range(n_states):
        mask = labels == s
        if mask.sum() > 0:
            state_means.append((s, X_scaled[mask, mom_col_idx].mean()))
        else:
            state_means.append((s, 0.0))

    state_means.sort(key=lambda x: x[1])
    return {old: new for new, (old, _) in enumerate(state_means)}


def _prep_X(features_df, feature_cols):
    """提取特征矩阵，用全局列均值填充残余 NaN（序列开头）"""
    X = features_df[feature_cols].values.astype(float)
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = col_means[j]
    return X


def fit_and_predict(features_df, feature_cols, n_states=HMM_N_STATES):
    """全量训练 HMM 并对所有月份产出 regime 标签。

    同时将训练好的模型（hmm、scaler、state_map、feature_cols）保存到
    hmm_model.pkl，供月度增量推理直接加载复用，无需重训。
    """
    X_raw = _prep_X(features_df, feature_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    hmm = GaussianHMM(
        n_components=n_states, covariance_type=HMM_COVARIANCE,
        n_iter=HMM_N_ITER, random_state=42,
    )
    hmm.fit(X_scaled)
    state_map = align_states(hmm, X_scaled, n_states)

    raw_labels = hmm.predict(X_scaled)
    raw_proba  = hmm.predict_proba(X_scaled)

    results = []
    for i in range(len(features_df)):
        aligned = state_map[raw_labels[i]]
        proba = np.zeros(n_states)
        for old, new in state_map.items():
            proba[new] = raw_proba[i, old]
        results.append({
            'date':   features_df.iloc[i]['date'],
            'regime': aligned,
            **{f'regime_prob_{j}': proba[j] for j in range(n_states)},
        })

    # 保存模型供增量推理复用
    model_bundle = {
        'hmm':          hmm,
        'scaler':       scaler,
        'state_map':    state_map,
        'feature_cols': feature_cols,
    }
    with open(_MODEL_PATH, 'wb') as f:
        pickle.dump(model_bundle, f)

    return pd.DataFrame(results)


def predict_new_month(new_features_row: dict) -> dict:
    """用保存的模型对单个新月份做推理（月度增量，不重训）。

    Args:
        new_features_row: dict，键为 feature_cols 中的列名，值为浮点数。
                          缺失的宏观字段传 None，内部用全历史均值替代。
    Returns:
        {'regime': int, 'regime_prob_0': float, ...}
    """
    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError('hmm_model.pkl 不存在，请先运行 s0_regime.py 完整训练')

    with open(_MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)

    hmm       = bundle['hmm']
    scaler    = bundle['scaler']
    state_map = bundle['state_map']
    fcols     = bundle['feature_cols']

    # 用 scaler 均值替代缺失字段（等价于 z-score 为 0 的中性值）
    x = np.array([new_features_row.get(c, None) for c in fcols], dtype=object)
    scaler_mean = scaler.mean_
    for j, v in enumerate(x):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            x[j] = scaler_mean[j]   # 已在原始空间，transform 后为 0
    x = x.astype(float).reshape(1, -1)
    x_scaled = scaler.transform(x)

    raw_proba = hmm.predict_proba(x_scaled)[0]
    n_states  = hmm.n_components
    proba = np.zeros(n_states)
    for old, new in state_map.items():
        proba[new] = raw_proba[old]
    regime = int(np.argmax(proba))

    return {'regime': regime, **{f'regime_prob_{j}': float(proba[j]) for j in range(n_states)}}


def main():
    t0 = time.time()
    print("=" * 60)
    print("  HMM 宏观状态识别（全量训练）")
    print("=" * 60)

    macro  = load_macro_factors()
    csi300 = load_csi300_monthly()

    print("\n构建特征...")
    features, feature_cols = build_regime_features(macro, csi300)
    print(f"  特征数: {len(feature_cols)}, 样本数: {len(features)}")
    print(f"  日期范围: {features['date'].min().strftime('%Y-%m')} ~ "
          f"{features['date'].max().strftime('%Y-%m')}")

    print(f"\n全量训练 HMM（{HMM_N_STATES} 状态）...")
    regime_df = fit_and_predict(features, feature_cols)

    regime_names = {-1: '失败', 0: '衰退', 1: '复苏', 2: '扩张', 3: '过热'}
    print(f"\n  Regime 分布:")
    for r, cnt in regime_df['regime'].value_counts().sort_index().items():
        name = regime_names.get(r, f'状态{r}')
        print(f"    {name}({r}): {cnt} 月 ({cnt/len(regime_df):.0%})")

    # 保存
    out_path = os.path.join(OUTPUT_DIR, 'regime_labels.pkl')
    regime_df.to_pickle(out_path)
    regime_df.to_csv(out_path.replace('.pkl', '.csv'), index=False, encoding='utf-8-sig')
    print(f"  模型已保存至 {_MODEL_PATH}")

    elapsed = time.time() - t0
    print(f"\n完成，耗时 {elapsed:.0f}秒，标签保存至 {out_path}")
    return regime_df


if __name__ == '__main__':
    main()
