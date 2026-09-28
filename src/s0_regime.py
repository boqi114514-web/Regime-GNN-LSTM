# -*- coding: utf-8 -*-
"""以沪深北全市场成交额为主轴的月末 HMM 状态识别。"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from config import (HMM_COVARIANCE, HMM_N_ITER, HMM_N_STATES,
                    HMM_TRAIN_WINDOW, OUTPUT_DIR, STOCK_DAILY_PATH,
                    LOCAL_DATA_RAW, load_csi300_monthly)

warnings.filterwarnings('ignore')

REGIME_NAMES = {-1: '识别失败', 0: '低成交', 1: '常态', 2: '高成交/过热风险'}
TURNOVER_COLS = ['turnover_rel_20_60', 'turnover_rel_20_252',
                 'turnover_20d_change']
PRICE_COLS = ['breadth_20d', 'csi_mom_3m', 'csi_vol_3m']
GLOBAL_COLS = ['ks11_mom_20d', 'us_mom_20d']


def load_market_daily():
    """amount 原始单位为千元；只需相对量，单位不影响状态。"""
    obj = pd.read_pickle(STOCK_DAILY_PATH)
    stock = obj['df_stock'] if isinstance(obj, dict) else obj
    frame = stock[['date', 'code', 'close', 'amount']].copy()
    frame['date'] = pd.to_datetime(frame['date'])
    frame['amount'] = pd.to_numeric(frame['amount'], errors='coerce')
    frame['close'] = pd.to_numeric(frame['close'], errors='coerce')
    frame = frame.sort_values(['code', 'date'])
    frame['up'] = frame.groupby('code')['close'].pct_change(fill_method=None).gt(0)
    daily = frame.groupby('date').agg(
        turnover=('amount', 'sum'), breadth=('up', 'mean'),
        stocks=('code', 'nunique')).sort_index()
    daily['turnover'] = daily['turnover'].where(daily['turnover'] > 0)
    daily['turnover_rel_20_60'] = np.log(
        daily['turnover'].rolling(20).mean() /
        daily['turnover'].rolling(60).mean())
    daily['turnover_rel_20_252'] = np.log(
        daily['turnover'].rolling(20).mean() /
        daily['turnover'].rolling(252).mean())
    daily['turnover_20d_change'] = np.log(
        daily['turnover'].rolling(20).mean() /
        daily['turnover'].shift(20).rolling(20).mean())
    daily['breadth_20d'] = daily['breadth'].rolling(20).mean()
    return daily.reset_index()


def _global_asof(month_dates, global_daily, code, cutoff_offset):
    """美股收盘晚于中国，当日美股不可用。"""
    g = global_daily[global_daily['ts_code'].eq(code)].sort_values('trade_date').copy()
    g['mom_20d'] = g['close'] / g['close'].shift(20) - 1
    left = pd.DataFrame({'cutoff': month_dates - pd.Timedelta(days=cutoff_offset)})
    aligned = pd.merge_asof(left.sort_values('cutoff'),
                            g[['trade_date', 'mom_20d']],
                            left_on='cutoff', right_on='trade_date',
                            direction='backward')
    return aligned['mom_20d'].to_numpy()


def build_regime_features(market_daily, csi300_df, global_daily):
    """每个月最后一个中国交易日的收盘信息，不含下一交易日数据。"""
    csi = csi300_df[['date', 'close']].copy().sort_values('date')
    csi['date'] = pd.to_datetime(csi['date'])
    csi['csi_mom_3m'] = csi['close'].pct_change(3)
    csi['csi_vol_3m'] = csi['close'].pct_change().rolling(3).std()
    joined = pd.merge_asof(csi, market_daily.sort_values('date'),
                           on='date', direction='backward')
    glob = global_daily.copy()
    glob['trade_date'] = pd.to_datetime(glob['trade_date'].astype(str),
                                        format='%Y%m%d', errors='coerce')
    glob['close'] = pd.to_numeric(glob['close'], errors='coerce')
    glob = glob.dropna(subset=['trade_date', 'close'])
    joined['ks11_mom_20d'] = _global_asof(joined['date'], glob, 'KS11', 0)
    spx = _global_asof(joined['date'], glob, 'SPX', 1)
    ixic = _global_asof(joined['date'], glob, 'IXIC', 1)
    joined['us_mom_20d'] = np.nanmean(np.column_stack([spx, ixic]), axis=1)
    feature_cols = TURNOVER_COLS + PRICE_COLS + GLOBAL_COLS
    joined = joined.dropna(subset=feature_cols).reset_index(drop=True)
    joined['signal_date'] = joined['date']
    joined['date'] = joined['date'].dt.to_period('M').dt.to_timestamp('M')
    return joined, feature_cols


def align_states(hmm_model, x_train, n_states):
    """仅用训练窗内的成交额强弱排列状态，避免看未来。"""
    labels = hmm_model.predict(x_train)
    state_means = []
    for state in range(n_states):
        mask = labels == state
        strength = x_train[mask, :len(TURNOVER_COLS)].mean() if mask.any() else -np.inf
        state_means.append((state, strength))
    state_means.sort(key=lambda item: item[1])
    return {old: new for new, (old, _) in enumerate(state_means)}


def rolling_hmm(features_df, feature_cols, train_window=HMM_TRAIN_WINDOW,
                n_states=HMM_N_STATES):
    results = []
    for t in range(train_window, len(features_df)):
        x_train = features_df.iloc[t-train_window:t][feature_cols].to_numpy(dtype=float)
        x_current = features_df.iloc[t:t+1][feature_cols].to_numpy(dtype=float)
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_current = scaler.transform(x_current)
        try:
            hmm = GaussianHMM(n_components=n_states,
                              covariance_type=HMM_COVARIANCE,
                              n_iter=HMM_N_ITER, random_state=42)
            hmm.fit(x_train)
            state_map = align_states(hmm, x_train, n_states)
            sequence = np.vstack([x_train, x_current])
            label = state_map[hmm.predict(sequence)[-1]]
            raw_prob = hmm.predict_proba(sequence)[-1]
            probabilities = np.zeros(n_states)
            for raw, ordered in state_map.items():
                probabilities[ordered] = raw_prob[raw]
        except (ValueError, FloatingPointError) as exc:
            warnings.warn(f'HMM {features_df.iloc[t]["date"]}: {exc}')
            label = -1
            probabilities = np.full(n_states, 1 / n_states)
        results.append({
            'date': features_df.iloc[t]['date'],
            'signal_date': features_df.iloc[t]['signal_date'],
            'regime': label,
            'regime_name': REGIME_NAMES[label],
            'turnover': features_df.iloc[t]['turnover'],
            **{f'regime_prob_{i}': probabilities[i] for i in range(n_states)},
        })
    return pd.DataFrame(results)


def main():
    started = time.time()
    market = load_market_daily()
    csi = load_csi300_monthly()
    global_path = os.path.join(LOCAL_DATA_RAW, 'ts_global_indices_daily.csv')
    if not os.path.exists(global_path):
        raise FileNotFoundError('缺少韩美指数日线，请先运行 python -m data_pipeline.global_indices')
    global_daily = pd.read_csv(global_path)
    features, feature_cols = build_regime_features(market, csi, global_daily)
    regimes = rolling_hmm(features, feature_cols)
    output = os.path.join(OUTPUT_DIR, 'regime_labels.pkl')
    regimes.to_pickle(output)
    regimes.to_csv(output.replace('.pkl', '.csv'), index=False, encoding='utf-8-sig')
    print(f'特征 {feature_cols}；训练月数 {HMM_TRAIN_WINDOW}；状态数 {HMM_N_STATES}')
    print(regimes.groupby(['regime', 'regime_name']).size().to_string())
    print(f'最新信号: {regimes.iloc[-1]["signal_date"]}，状态 {regimes.iloc[-1]["regime_name"]}')
    print(f'耗时 {time.time() - started:.1f} 秒；保存 {output}')
    return regimes


if __name__ == '__main__':
    main()
