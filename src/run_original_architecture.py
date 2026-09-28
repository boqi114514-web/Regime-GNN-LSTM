"""Rebuild the pre-redesign architecture with corrected data/timing.

Reference: committed 818423c (macro HMM, GAT, 12-factor LSTM, regime IC mix).
No changes to the default fixed-weight pipeline or its saved predictions.
Run: python src/run_original_architecture.py [--backtest-only]
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

import config
from s3_ensemble_backtest import (calc_metrics, run_topk_strategy,
                                  simple_rank_ensemble, validate_backtest_inputs)

OUT = Path(config.OUTPUT_DIR) / 'original_architecture'
MACRO_COLS = ['pmi_chg', 'term_spread', 'm1_m2_spread', 'sf_yoy']
PRICE_COLS = ['mkt_mom_1m', 'mkt_mom_3m', 'mkt_mom_6m', 'mkt_vol_3m',
              'mkt_vol_6m', 'mkt_vol_ratio', 'mkt_bias_6m', 'mkt_pos_12m']


def monthly_unique(frame, end):
    """Canonical month key; rolling windows must not count a month twice."""
    frame = frame.copy().sort_values('date', kind='stable')
    frame['ym'] = pd.to_datetime(frame['date']).dt.to_period('M')
    frame = frame.drop_duplicates('ym', keep='last').set_index('ym')
    frame = frame.loc[frame.index <= pd.Timestamp(end).to_period('M')]
    frame = frame.reindex(pd.period_range(frame.index.min(), frame.index.max(), freq='M'))
    frame['date'] = frame.index.to_timestamp('M')
    return frame


def build_original_features(macro, csi, end):
    macro = monthly_unique(macro, end)
    csi = monthly_unique(csi[['date', 'close']], end)
    # The historical definition is SHIBOR 1y minus 1m, not 1y minus overnight.
    macro['term_spread'] = macro['shibor_1y'] - macro['shibor_1m']
    macro['m1_m2_spread'] = macro['m1_yoy_pct'] - macro['m2_yoy_pct']
    macro['pmi_chg'] = macro['pmi_mfg'].diff()
    sf12 = macro['sf_inc_month'].rolling(12).sum()
    macro['sf_yoy'] = sf12 / sf12.shift(12) - 1
    # No release timestamps in this source: conservatively use previous-month
    # macro releases. Market-observed SHIBOR/CSI remain at the signal month.
    lagged = ['pmi_chg', 'm1_m2_spread', 'sf_yoy']
    macro[lagged] = macro[lagged].shift(1)
    csi['ret'] = csi.close.pct_change(fill_method=None)
    for n in (1, 3, 6):
        csi[f'mkt_mom_{n}m'] = csi.close.pct_change(n, fill_method=None)
    for n in (3, 6):
        csi[f'mkt_vol_{n}m'] = csi.ret.rolling(n).std()
    csi['mkt_vol_ratio'] = csi.mkt_vol_3m / (csi.mkt_vol_6m + 1e-8)
    csi['mkt_bias_6m'] = csi.close / (csi.close.rolling(6).mean() + 1e-8) - 1
    lo, hi = csi.close.rolling(12).min(), csi.close.rolling(12).max()
    csi['mkt_pos_12m'] = (csi.close - lo) / (hi - lo + 1e-8)
    frame = csi[['date'] + PRICE_COLS].join(macro[MACRO_COLS], how='inner')
    cols = MACRO_COLS + PRICE_COLS
    frame[cols] = frame[cols].replace([np.inf, -np.inf], np.nan).ffill()
    return frame.dropna(subset=cols, how='all').reset_index(drop=True), cols


def rolling_original_hmm(features, cols, window=60):
    """Historical full-covariance 4-state HMM; every fit sees only its past."""
    records = []
    logging.getLogger('hmmlearn.base').setLevel(logging.ERROR)
    for t in range(window, len(features)):
        past = features.iloc[t-window:t][cols].to_numpy(float)
        current = features.iloc[t:t+1][cols].to_numpy(float)
        means = pd.DataFrame(past).mean().fillna(0).to_numpy()
        past = np.where(np.isnan(past), means, past)
        current = np.where(np.isnan(current), means, current)
        scaler = StandardScaler().fit(past)
        x, now = scaler.transform(past), scaler.transform(current)
        failure = ''
        converged = False
        try:
            model = GaussianHMM(n_components=4, covariance_type='full',
                                n_iter=200, random_state=42)
            model.fit(x)
            labels = model.predict(x)
            mom_idx = cols.index('mkt_mom_3m')
            order = sorted(range(4), key=lambda s: x[labels == s, mom_idx].mean()
                           if (labels == s).any() else 0.0)
            mapping = {old: new for new, old in enumerate(order)}
            full = np.vstack([x, now])
            state = mapping[model.predict(full)[-1]]
            raw_prob = model.predict_proba(full)[-1]
            prob = np.array([raw_prob[old] for old in order])
            if not np.isfinite(prob).all():
                raise ValueError('nonfinite state probabilities')
            converged = bool(model.monitor_.converged)
        except (ValueError, np.linalg.LinAlgError) as exc:
            state, prob = -1, np.full(4, .25)
            failure = f'{type(exc).__name__}: {exc}'
        records.append({'date': features.iloc[t]['date'], 'regime': state,
                        'converged': converged, 'failure': failure,
                        **{f'regime_prob_{i}': prob[i] for i in range(4)}})
    result = pd.DataFrame(records)
    print(f'HMM: {len(result)} months; failed fits={(result.regime == -1).sum()}', flush=True)
    return result


def original_regime_ensemble(gnn, lstm, regimes, lookback=12):
    """Original nonnegative same-state IC weighting, WITHOUT an LSTM floor."""
    frame = gnn.merge(lstm[['ts_code', 'date', 'pred_lstm_b']],
                      on=['ts_code', 'date'], validate='one_to_one')
    frame = frame.merge(regimes[['date', 'regime']], on='date',
                        how='left', validate='many_to_one')
    if frame.regime.isna().any():
        raise ValueError('Original HMM does not cover every prediction month')
    history = {r: [] for r in range(-1, 4)}
    result = []
    for date, month in frame.sort_values('date').groupby('date', sort=True):
        month = month.copy()
        state = int(month.regime.iloc[0])
        past = history[state][-lookback:]
        weights = np.maximum(0, np.mean(past, axis=0)) if past else np.array([.5, .5])
        weights = weights / weights.sum() if weights.sum() >= 1e-8 else np.array([.5, .5])
        month['rank_gnn'] = month.pred_gnn.rank(ascending=False)
        month['rank_lstm'] = month.pred_lstm_b.rank(ascending=False)
        month['w_gnn'], month['w_lstm'] = weights
        month['rank_ensemble'] = weights[0]*month.rank_gnn + weights[1]*month.rank_lstm
        month['pred_ensemble'] = -month.rank_ensemble
        result.append(month)
        # Update AFTER setting this signal's weights. Its next-month return
        # becomes eligible only for subsequent monthly signals.
        labeled = month.dropna(subset=['actual_ret'])
        if len(labeled) >= 5:
            ics = [spearmanr(labeled[col], labeled.actual_ret).statistic
                   for col in ('pred_gnn', 'pred_lstm_b')]
            history[state].append([v if np.isfinite(v) else 0.0 for v in ics])
    return pd.concat(result, ignore_index=True)


def input_hashes():
    root = Path(config.PROJECT_DIR)
    files = [Path(__file__), root/'src/config.py', root/'src/s1_gnn_train.py',
             root/'src/s2_lstm_b_train.py', root/'src/s3_ensemble_backtest.py',
             root/'src/data_pipeline/industry_monthly.py']
    files += [Path(config.LOCAL_DATA_RAW)/f for f in
              ('ts_macro_factors.csv', 'ts_csi300_monthly.csv', 'ts_sw_industry_monthly.csv')]
    files += [Path(config.LOCAL_DATA_PROCESSED)/f for f in
              ('prosperity_indicators_clean.pkl', 'price_volume_factors.pkl', 'pattern_factors.pkl')]
    files += [Path(config.OUTPUT_DIR)/f for f in
              ('predictions_gnn.pkl', 'predictions_lstm_b.pkl', 'predictions_lstm_b_no_market.pkl')]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def backtest(output):
    mkt = config.load_industry_monthly()
    gnn = pd.read_pickle(output/'predictions_gnn.pkl')
    lstm = pd.read_pickle(output/'predictions_lstm_b.pkl')
    coverage = validate_backtest_inputs(gnn, lstm, mkt)
    regimes = pd.read_pickle(output/'regime_labels.pkl')
    ensemble = original_regime_ensemble(gnn, lstm, regimes)
    ensemble.to_pickle(output/'predictions_ensemble.pkl')
    ensemble.to_csv(output/'predictions_ensemble.csv', index=False, encoding='utf-8-sig')
    weights = ensemble.groupby('date')[['regime', 'w_gnn', 'w_lstm']].first()
    weights.to_csv(output/'monthly_weights.csv', encoding='utf-8-sig')
    original, holdings = run_topk_strategy(ensemble, 'pred_ensemble', inertia=.2)
    holdings.to_csv(output/'holdings.csv', index=False, encoding='utf-8-sig')
    returns = {'original_repaired': original}
    current_gnn = pd.read_pickle(Path(config.OUTPUT_DIR)/'predictions_gnn.pkl')
    for factor_set, name in [('industry12', 'predictions_lstm_b_no_market.pkl'),
                              ('market17', 'predictions_lstm_b.pkl')]:
        other = pd.read_pickle(Path(config.OUTPUT_DIR)/name)
        validate_backtest_inputs(current_gnn, other, mkt)
        for weight, (wg, wl) in config.FIXED_WEIGHT_VARIANTS.items():
            fixed = simple_rank_ensemble(current_gnn, other, wg, wl)
            values, _ = run_topk_strategy(fixed, 'pred_ensemble', inertia=.2)
            returns[f'{factor_set}_{weight}'] = values
    market = mkt.sort_values(['ts_code', 'date']).copy()
    market['fwd_ret'] = market.groupby('ts_code').ret.shift(-1)
    returns['industry_equal_weight'] = market.groupby('date').fwd_ret.mean().reindex(original.index)
    expected = pd.period_range(coverage[0], coverage[-2], freq='M')
    for name, values in returns.items():
        if not values.index.to_period('M').equals(expected) or values.isna().any():
            raise ValueError(f'{name}: incomplete or unequal holding-month coverage')
    monthly = pd.DataFrame(returns).sort_index()
    monthly.index.name = 'signal_date'
    held = monthly.copy()
    held.index += pd.offsets.MonthEnd(1)
    held.index.name = 'holding_month'
    monthly.insert(0, 'holding_month', held.index)
    monthly.to_csv(output/'monthly_returns.csv', encoding='utf-8-sig')
    (1 + held).cumprod().to_csv(output/'nav.csv', encoding='utf-8-sig')
    annual, summaries = [], []
    for strategy in held:
        summaries.append({'strategy': strategy, **calc_metrics(held[strategy])})
        for year, values in held[strategy].groupby(held.index.year):
            annual.append({'strategy': strategy, 'year': year, 'n_months': len(values),
                           'start': values.index.min(), 'end': values.index.max(),
                           'return': (1 + values).prod() - 1})
    annual = pd.DataFrame(annual)
    annual.to_csv(output/'annual_comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(summaries).to_csv(output/'summary.csv', index=False, encoding='utf-8-sig')
    print(annual[annual.year >= 2023].pivot(index='strategy', columns='year', values='return').to_string())
    return {'signal_start': str(coverage[0]), 'signal_end': str(coverage[-1]),
            'holding_start': str(held.index.min().date()), 'holding_end': str(held.index.max().date()),
            'n_realized_months': len(held), 'hmm_failure_months': int((regimes.regime == -1).sum())}


def main(backtest_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    before = input_hashes()
    if backtest_only:
        manifest = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
        if before != manifest['input_sha256']:
            raise ValueError('Inputs changed since training; rerun without --backtest-only')
        for file, digest in manifest['prediction_sha256'].items():
            if hashlib.sha256((OUT/file).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Prediction artifact changed: {file}')
    else:
        print('[1/4] Original macro HMM', flush=True)
        end = config.load_industry_monthly().date.max()
        features, cols = build_original_features(config.load_macro_factors(),
                                                 config.load_csi300_monthly(), end)
        features.to_csv(OUT/'hmm_features.csv', index=False, encoding='utf-8-sig')
        regimes = rolling_original_hmm(features, cols)
        regimes.to_pickle(OUT/'regime_labels.pkl')
        regimes.to_csv(OUT/'regime_labels.csv', index=False, encoding='utf-8-sig')
        print('[2/4] Retrain original GAT with regime one-hot', flush=True)
        from s1_gnn_train import main as train_gnn
        train_gnn(regime_df=regimes, output_dir=str(OUT))
        print('[3/4] Retrain original 12-factor LSTM', flush=True)
        from s2_lstm_b_train import main as train_lstm
        train_lstm(use_market_factors=False, output_dir=str(OUT))
    print('[4/4] Regime IC ensemble and matched annual comparison', flush=True)
    details = backtest(OUT)
    if before != input_hashes():
        raise RuntimeError('Source/data changed during training')
    manifest = {
        'architecture_reference': '818423c',
        'architecture': 'macro HMM(4,full,60) + GLASSO/GAT + industry12 LSTM + regime historical IC',
        'repairs': ['canonical monthly data and close-derived decimal returns',
                    'forward labels with train/validation label cutoffs',
                    'all prediction months including partial final window',
                    'macro monthly deduplication and SHIBOR 1y-minus-1m definition',
                    'one-month lag for macro releases without publication timestamps',
                    'keep regime one-hot out of cross-sectional z-score'],
        'hmm': {'states': 4, 'covariance': 'full', 'window': 60, 'iterations': 200, 'seed': 42},
        'weighting': {'same_state_ic_observations': 12, 'lstm_floor': None},
        'top_k_industries': 5, 'inertia': .2,
        'input_sha256': before,
        'prediction_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [OUT/'predictions_gnn.pkl', OUT/'predictions_lstm_b.pkl',
                                        OUT/'regime_labels.pkl']},
        **details,
    }
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Complete: {OUT}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backtest-only', action='store_true')
    main(parser.parse_args().backtest_only)
