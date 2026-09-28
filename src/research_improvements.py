"""Predeclared, chronological architecture/factor/risk experiments.

First run --phase develop: select using holdings through 2022 ONLY.
Then run --phase evaluate: freeze that decision, reveal 2023-2026 results.
2026 has been discussed before: it is not a pristine untouched holdout.
No changes to default models, raw data or previous experiment artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

import config

OUT = Path(config.OUTPUT_DIR) / 'improvement_research'
START = pd.Timestamp('2019-01-31')
DEV_END = pd.Timestamp('2022-12-31')
LEARNED = ['ridge_price', 'gbdt_price', 'ranker_price', 'gbdt_price_value',
           'ridge_tech12', 'gbdt_tech12']
RULES = ['momentum_6_1', 'reversal_1', 'low_vol_6', 'trend_defensive']
BLENDS = ['blend_ridge_gbdt', 'blend_lstm_gbdt']
BASELINES = ['original_repaired', 'fixed_industry64', 'lstm12_only', 'industry_equal_weight']
RISK_MODES = ['full', 'trend', 'vol_target']
PARAMS = dict(n_estimators=160, learning_rate=.03, num_leaves=7, max_depth=3,
              min_child_samples=80, reg_lambda=10., colsample_bytree=.9,
              random_state=42, n_jobs=1, verbosity=-1, deterministic=True,
              force_col_wise=True)


def input_hashes():
    root = Path(config.PROJECT_DIR)
    files = [Path(__file__), root/'src/config.py',
             root/'src/data_pipeline/industry_monthly.py',
             Path(config.LOCAL_DATA_RAW)/'ts_sw_industry_monthly.csv']
    files += [Path(config.LOCAL_DATA_PROCESSED)/name for name in
              ('price_volume_factors.pkl', 'pattern_factors.pkl')]
    files += [Path(config.OUTPUT_DIR)/name for name in
              ('predictions_gnn.pkl', 'predictions_lstm_b_no_market.pkl',
               'original_architecture/predictions_ensemble.pkl')]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def verify_protocol_inputs(protocol):
    """Permit only the explicitly recorded baseline-measurement correction."""
    current = input_hashes()
    expected = dict(protocol['input_sha256'])
    patch_path = OUT/'measurement_patch.json'
    if current != expected and patch_path.exists():
        patch = json.loads(patch_path.read_text(encoding='utf-8'))
        key = patch['file']
        if expected.get(key) == patch['before_sha256'] and current.get(key) == patch['after_sha256']:
            expected[key] = patch['after_sha256']
    if current != expected:
        raise ValueError('Protocol input hash mismatch')
    return current


def cross_rank(frame, cols):
    ranked = frame.copy()
    for col in cols:
        ranked[col] = frame.groupby('date')[col].rank(pct=True) * 2 - 1
    return ranked


def make_panel(market, tech):
    """Price features use only current/past observations, never full-sample fit."""
    market = market.sort_values(['ts_code', 'date']).copy()
    if market.duplicated(['ts_code', 'date']).any():
        raise ValueError('Duplicate industry-month')
    counts = market.groupby('date').ts_code.nunique()
    benchmark = market.groupby('date').ret.mean().sort_index()
    records = []
    price_cols = []
    for code, part in market.groupby('ts_code', sort=True):
        x = part.copy().set_index('date').sort_index()
        x = x.reindex(pd.date_range(x.index.min(), x.index.max(), freq='ME'))
        x['ts_code'] = code
        close, ret = x.close, x.ret
        feats = {}
        for n in (1, 2, 3, 6, 9, 12):
            feats[f'mom_{n}'] = close.pct_change(n, fill_method=None)
        for n in (3, 6, 12):
            feats[f'mom_{n}_1'] = close.shift(1) / close.shift(n) - 1
            feats[f'vol_{n}'] = ret.rolling(n).std()
            feats[f'bias_{n}'] = close / close.rolling(n).mean() - 1
        for n in (6, 12):
            feats[f'drawdown_{n}'] = close / close.rolling(n).max() - 1
        feats['downside_6'] = ret.clip(upper=0).pow(2).rolling(6).mean().pow(.5)
        feats['positive_6'] = ret.gt(0).astype(float).rolling(6).mean()
        bench = benchmark.reindex(x.index)
        feats['beta_12'] = ret.rolling(12).cov(bench) / bench.rolling(12).var()
        feats['residual_vol_12'] = (ret - bench).rolling(12).std()
        price_cols = list(feats)
        for col, values in feats.items():
            x[col] = values
        x['actual_ret'] = ret.shift(-1)
        x['label_date'] = pd.Series(x.index, index=x.index).shift(-1)
        x['earnings_yield'] = 1 / x.pe.where(x.pe > 0)
        x['book_yield'] = 1 / x.pb.where(x.pb > 0)
        x.index.name = 'date'
        records.append(x.reset_index())
    panel = pd.concat(records, ignore_index=True)
    tech = tech.copy().sort_values('date')
    tech['date'] = pd.to_datetime(tech.date).dt.to_period('M').dt.to_timestamp('M')
    tech = tech.drop_duplicates(['ts_code', 'date'], keep='last')
    tech_cols = [c for c in tech if c not in ('date', 'ts_code')]
    tech = tech.rename(columns={c: f'tech_{c}' for c in tech_cols})
    tech_cols = [f'tech_{c}' for c in tech_cols]
    panel = panel.merge(tech, on=['ts_code', 'date'], how='left', validate='one_to_one')
    # Save raw rule scores before cross-section normalization.
    panel['momentum_6_1'] = panel.mom_6_1
    panel['reversal_1'] = -panel.mom_1
    panel['low_vol_6'] = -panel.vol_6
    panel = cross_rank(panel, price_cols + ['earnings_yield', 'book_yield'] + tech_cols)
    panel['trend_defensive'] = (panel.mom_6_1 + panel.mom_12_1 - panel.vol_6) / 3
    panel['rank_target'] = panel.groupby('date').actual_ret.rank(pct=True) - .5
    # Full contemporaneous universe ranks, not ranks fitted to future data.
    panel['relevance'] = np.floor((panel.rank_target + .5).clip(0, 1) * 4.999)
    common = pd.DataFrame({'date': benchmark.index, 'market_return': benchmark.values})
    nav = (1 + benchmark.fillna(0)).cumprod()
    common['market_mom3'] = nav.pct_change(3).to_numpy()
    common['market_mom6'] = nav.pct_change(6).to_numpy()
    common['market_vol6'] = benchmark.rolling(6).std().to_numpy()
    common['market_breadth'] = market.assign(up=market.ret.gt(0)).groupby('date').up.mean().reindex(benchmark.index).to_numpy()
    market_cols = ['market_return', 'market_mom3', 'market_mom6', 'market_vol6', 'market_breadth']
    panel = panel.merge(common, on='date', how='left', validate='many_to_one')
    feature_sets = {'price': price_cols + market_cols,
                    'price_value': price_cols + ['earnings_yield', 'book_yield'] + market_cols,
                    'tech12': tech_cols}
    for cols in feature_sets.values():
        panel[cols] = panel[cols].replace([np.inf, -np.inf], np.nan).fillna(0.)
    panel = panel.sort_values(['date', 'ts_code']).reset_index(drop=True)
    return panel, feature_sets, benchmark, counts


def rolling_predictions(panel, feature_sets, signal_end):
    dates = sorted(panel.loc[panel.date.between(START, signal_end), 'date'].unique())
    output, audit, importance = [], [], []
    for start in range(0, len(dates), 3):
        pred_dates = dates[start:start+3]
        first = pd.Timestamp(pred_dates[0])
        label_cutoff = first - pd.offsets.MonthEnd(1)
        lower = first - pd.DateOffset(months=72)
        train = panel[(panel.date >= lower) & (panel.label_date <= label_cutoff)
                      & panel.actual_ret.notna()].copy()
        test = panel[panel.date.isin(pred_dates)].copy()
        if train.date.nunique() < 48 or test.empty:
            raise ValueError(f'Insufficient training window at {first}')
        # Equal weight per training month, even when early industry coverage differs.
        weights = 1. / train.groupby('date').ts_code.transform('count')
        weights = weights / weights.mean()
        audit.append({'first_signal': first, 'last_signal': pred_dates[-1],
                      'train_first_signal': train.date.min(), 'train_last_signal': train.date.max(),
                      'latest_training_label': train.label_date.max(),
                      'label_cutoff': label_cutoff, 'n_months': train.date.nunique(),
                      'n_rows': len(train)})
        for name in LEARNED:
            family, feature_set = name.split('_', 1)
            cols = feature_sets[feature_set]
            X = train[cols].to_numpy(float)
            target = train.rank_target.to_numpy()
            if family == 'ridge':
                model = Ridge(alpha=100.)
                model.fit(X, target, sample_weight=weights)
            elif family == 'gbdt':
                model = lgb.LGBMRegressor(objective='regression', **PARAMS)
                model.fit(X, target, sample_weight=weights)
            else:
                model = lgb.LGBMRanker(objective='lambdarank', label_gain=[0, 1, 2, 3, 4],
                                      lambdarank_truncation_level=5, **PARAMS)
                model.fit(X, train.relevance.astype(int), group=train.groupby('date', sort=True).size().to_numpy(),
                          sample_weight=weights)
            test[name] = model.predict(test[cols].to_numpy(float))
            if hasattr(model, 'feature_importances_'):
                importance.extend({'first_signal': first, 'model': name, 'feature': col, 'splits': float(value)}
                                  for col, value in zip(cols, model.feature_importances_))
        output.append(test[['ts_code', 'date', 'actual_ret'] + RULES + LEARNED])
        print(f'fit {start//3+1}: {first:%Y-%m} -> {pd.Timestamp(pred_dates[-1]):%Y-%m}; '
              f'labels <= {train.label_date.max():%Y-%m}', flush=True)
    return pd.concat(output, ignore_index=True), pd.DataFrame(audit), pd.DataFrame(importance)


def add_baselines(pred):
    root = Path(config.OUTPUT_DIR)
    lstm = pd.read_pickle(root/'predictions_lstm_b_no_market.pkl')
    gnn = pd.read_pickle(root/'predictions_gnn.pkl')
    original = pd.read_pickle(root/'original_architecture/predictions_ensemble.pkl')
    for other, col, name in [(lstm, 'pred_lstm_b', 'lstm12_only'),
                              (gnn, 'pred_gnn', 'gnn_current'),
                              (original, 'pred_ensemble', 'original_repaired')]:
        pred = pred.merge(other[['date', 'ts_code', col]].rename(columns={col: name}),
                          on=['date', 'ts_code'], how='left', validate='one_to_one')
    if pred[['lstm12_only', 'gnn_current', 'original_repaired']].isna().any().any():
        raise ValueError('Baseline prediction missing')
    ranks = pred.groupby('date')[LEARNED + ['lstm12_only', 'gnn_current']].rank(pct=True)
    descending = pred.groupby('date')[['gnn_current', 'lstm12_only']].rank(ascending=False)
    pred['fixed_industry64'] = -(.6 * descending.gnn_current + .4 * descending.lstm12_only)
    pred['blend_ridge_gbdt'] = .5 * ranks.ridge_price + .5 * ranks.gbdt_price
    pred['blend_lstm_gbdt'] = .5 * ranks.lstm12_only + .5 * ranks.gbdt_price
    return pred


def exposure_series(benchmark, mode):
    if mode == 'full':
        return pd.Series(1., index=benchmark.index)
    if mode == 'trend':
        # A predeclared 10-month moving-average filter, 50%/100%, no leverage.
        nav = (1 + benchmark.fillna(0)).cumprod()
        return pd.Series(np.where(nav >= nav.rolling(10).mean(), 1., .5), index=nav.index)
    if mode == 'vol_target':
        # Target 15% annualized volatility, estimated from six past/current months.
        # Risk estimate describes the benchmark, not a claim of exact portfolio vol.
        return (.15 / (benchmark.rolling(6).std() * np.sqrt(12))).clip(0, 1).fillna(0.)
    raise ValueError(mode)


def portfolio_returns(pred, score, exposure, k=5, inertia=.2):
    records, holdings = [], []
    previous = set()
    for date, month in pred.groupby('date', sort=True):
        if month.actual_ret.isna().all():
            continue
        if month.actual_ret.isna().any() or month[score].isna().any():
            raise ValueError(f'Partial label/score month {date}')
        month = month.copy()
        bonus = inertia * month[score].std()
        month.loc[month.ts_code.isin(previous), score] += bonus
        # Keep the existing engine's sorting/tie convention for exact baselines.
        chosen = month.sort_values(score, ascending=False).head(k)
        stocks = chosen.ts_code.tolist()
        w = float(exposure.loc[date])
        turnover = 1. - len(previous & set(stocks)) / k if previous else 1.
        records.append({'date': date, 'ret': chosen.actual_ret.mean() * w,
                        'exposure': w, 'turnover': turnover})
        holdings.extend({'date': date, 'ts_code': row.ts_code, 'weight': w/k,
                          'actual_ret': row.actual_ret} for row in chosen.itertuples())
        previous = set(stocks)
    return pd.DataFrame(records).set_index('date'), pd.DataFrame(holdings)


def metrics(returns):
    returns = returns.dropna()
    nav = (1 + returns).cumprod()
    peak = nav.cummax().clip(lower=1.)
    ann = nav.iloc[-1] ** (12 / len(returns)) - 1
    vol = returns.std() * np.sqrt(12)
    return {'n_months': len(returns), 'cumulative_return': nav.iloc[-1]-1,
            'annual_return': ann, 'sharpe': (ann-.03)/vol if vol else 0.,
            'max_drawdown': (nav/peak-1).min(), 'win_rate': returns.gt(0).mean()}


def evaluate(pred, benchmark, output, top_k=5, save_holdings=True):
    returns, details, holdings_all = {}, [], []
    for name in RULES + LEARNED + BLENDS + BASELINES[:-1]:
        score_input = pred
        if name == 'original_repaired':
            # Legacy rank scores tie often. Preserve the archived prediction row
            # order instead of accidentally changing tie-breaking via a merge.
            source = pd.read_pickle(Path(config.OUTPUT_DIR)/'original_architecture/predictions_ensemble.pkl')
            score_input = source[source.date.isin(pred.date.unique())].rename(columns={'pred_ensemble': name})
        modes = RISK_MODES if name not in BASELINES else ['full']
        for mode in modes:
            label = name if mode == 'full' else f'{name}__{mode}'
            data, holdings = portfolio_returns(score_input, name, exposure_series(benchmark, mode), k=top_k)
            returns[label] = data.ret
            details.append({'strategy': label, 'avg_exposure': data.exposure.mean(),
                            'avg_turnover': data.turnover.mean()})
            if save_holdings:
                holdings['strategy'] = label
                holdings_all.append(holdings)
    dates = next(iter(returns.values())).index
    returns['industry_equal_weight'] = pred.groupby('date').actual_ret.mean().reindex(dates)
    monthly = pd.DataFrame(returns).sort_index()
    if monthly.isna().any().any():
        raise ValueError('Unequal realized-month coverage')
    periods = monthly.index.to_period('M')
    if not periods.equals(pd.period_range(periods.min(), periods.max(), freq='M')):
        raise ValueError('Skipped realized months')
    monthly.index += pd.offsets.MonthEnd(1)
    monthly.index.name = 'holding_month'
    annual, summary = [], []
    for name in monthly:
        for period, lo, hi in [('development', '2019-01-01', '2022-12-31'),
                               ('evaluation_2023_2025', '2023-01-01', '2025-12-31'),
                               ('check_2026', '2026-01-01', '2026-12-31'),
                               ('all', '2019-01-01', '2026-12-31')]:
            values = monthly[name].loc[lo:hi]
            if len(values):
                summary.append({'strategy': name, 'period': period, **metrics(values)})
        for year, values in monthly[name].groupby(monthly.index.year):
            annual.append({'strategy': name, 'year': year, 'n_months': len(values),
                           'return': (1 + values).prod()-1})
    monthly.to_csv(output/'monthly_returns.csv', encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(output/'summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(annual).to_csv(output/'annual.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(details).to_csv(output/'portfolio_diagnostics.csv', index=False, encoding='utf-8-sig')
    if holdings_all:
        pd.concat(holdings_all).to_csv(output/'holdings.csv', index=False, encoding='utf-8-sig')
    return monthly, pd.DataFrame(summary), pd.DataFrame(annual)


def main(phase):
    OUT.mkdir(parents=True, exist_ok=True)
    current_hashes = input_hashes()
    protocol_path = OUT/'protocol.json'
    if phase == 'develop':
        if protocol_path.exists():
            raise ValueError('Protocol already exists; do not silently restart a selection experiment')
        protocol = {'selection_period': 'holding 2019-02 through 2022-12',
                    'evaluation_period': '2023-2025', 'later_check': '2026-01 through 2026-08',
                    'later_check_already_seen_in_prior_work': True,
                    'features': 'price21 plus market5; optional PE/PB yields; cached industry12 ablation',
                    'models': LEARNED, 'rules': RULES, 'blends': BLENDS,
                    'risk_modes': RISK_MODES, 'tree_parameters': PARAMS, 'ridge_alpha': 100.,
                    'fit': 'past 72 signal months, quarterly; labels end before first prediction month',
                    'selection': 'highest development Sharpe among new candidates; no post-2022 tuning',
                    'top_k': 5, 'inertia': .2, 'cash_return': 0., 'leverage': False,
                    'input_sha256': current_hashes}
        protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    else:
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        verify_protocol_inputs(protocol)
        if 'selected_strategy' not in protocol:
            raise ValueError('Complete development selection before evaluation')
    market = config.load_industry_monthly()
    panel, features, benchmark, counts = make_panel(market, config.load_tech_factors(False))
    signal_end = DEV_END - pd.offsets.MonthEnd(1) if phase == 'develop' else market.date.max()
    output = OUT/phase
    output.mkdir(exist_ok=True)
    pred, audit, importance = rolling_predictions(panel, features, signal_end)
    pred = add_baselines(pred)
    pred.to_pickle(output/'predictions.pkl')
    audit.to_csv(output/'training_audit.csv', index=False)
    importance.to_csv(output/'feature_importance.csv', index=False)
    monthly, summary, annual = evaluate(pred, benchmark, output)
    if phase == 'develop':
        eligible = summary[(summary.period == 'development') & ~summary.strategy.isin(BASELINES)]
        winner = eligible.sort_values(['sharpe', 'strategy'], ascending=[False, True]).iloc[0]
        protocol['selected_strategy'] = winner.strategy
        protocol['development_sharpe'] = winner.sharpe
        protocol['feature_columns'] = features
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
        print('LOCKED SELECTION:', winner.to_dict(), flush=True)
        print(eligible.sort_values('sharpe', ascending=False).head(10).to_string(index=False))
    else:
        print('Previously locked selection:', protocol['selected_strategy'])
        print(annual[annual.year >= 2023].pivot(index='strategy', columns='year', values='return').to_string())
        print(summary[summary.strategy.eq(protocol['selected_strategy'])].to_string(index=False))
    if input_hashes() != current_hashes:
        raise ValueError('Inputs changed during experiment')
    print(f'COMPLETE {phase}: {output}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['develop', 'evaluate'], required=True)
    main(p.parse_args().phase)
