"""Standalone price-risk model and explicit small-account allocation adapter.

No dependence on old GNN/LSTM predictions or cached stock technical factors.
The default command retrains and reproduces the verified industry experiment.
--candidates accepts a dated stock-selection table; it is NOT an account backtest.
"""
import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import config
from research_improvements import make_panel, exposure_series, portfolio_returns, metrics
from s7_budget_portfolio import attach_asof_prices, optimize_portfolio, apply_limit_prices

OUT = Path(config.OUTPUT_DIR)/'price_risk_pipeline'
SEEDS = [42, 142, 242, 342, 442]
TREE_PARAMS = dict(n_estimators=160, learning_rate=.03, num_leaves=7, max_depth=3,
                   min_child_samples=80, reg_lambda=10., colsample_bytree=.9,
                   n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True)


def source_hashes():
    root = Path(config.PROJECT_DIR)
    paths = [Path(__file__), root/'src/research_improvements.py', root/'src/config.py',
             root/'src/s7_budget_portfolio.py', root/'src/data_pipeline/industry_monthly.py',
             Path(config.LOCAL_DATA_RAW)/'ts_sw_industry_monthly.csv']
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def price_panel(market):
    # Empty optional table: the new model does not load the old 12 factors.
    empty_tech = pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]'),
                               'ts_code': pd.Series(dtype=str)})
    panel, sets, benchmark, _ = make_panel(market, empty_tech)
    return panel, sets['price'], benchmark


def train_predictions(market, start='2019-01-31'):
    panel, cols, benchmark = price_panel(market)
    dates = sorted(panel.loc[panel.date >= pd.Timestamp(start), 'date'].unique())
    records, audits = [], []
    state = None
    for i in range(0, len(dates), 3):
        window = dates[i:i+3]
        first = pd.Timestamp(window[0])
        label_cutoff = first - pd.offsets.MonthEnd(1)
        train = panel[(panel.date >= first-pd.DateOffset(months=72)) &
                      (panel.label_date <= label_cutoff) & panel.actual_ret.notna()]
        test = panel[panel.date.isin(window)].copy()
        if train.date.nunique() < 48:
            raise ValueError(f'Insufficient history at {first}')
        sample_weight = 1/train.groupby('date').ts_code.transform('count')
        sample_weight /= sample_weight.mean()
        ridge = Ridge(alpha=100.).fit(train[cols].to_numpy(), train.rank_target,
                                     sample_weight=sample_weight)
        test['ridge_score'] = ridge.predict(test[cols].to_numpy())
        trees = []
        for seed in SEEDS:
            tree = lgb.LGBMRegressor(objective='regression', random_state=seed, **TREE_PARAMS)
            tree.fit(train[cols].to_numpy(), train.rank_target, sample_weight=sample_weight)
            test[f'tree_{seed}'] = tree.predict(test[cols].to_numpy())
            trees.append(tree)
        tree_cols = [f'tree_{s}' for s in SEEDS]
        ranks = test.groupby('date')[['ridge_score']+tree_cols].rank(pct=True)
        test['pred_ensemble'] = .5*ranks.ridge_score + .5*ranks[tree_cols].mean(axis=1)
        records.append(test[['date', 'ts_code', 'actual_ret', 'pred_ensemble']])
        audits.append({'first_prediction': first, 'last_prediction': window[-1],
                       'latest_training_label': train.label_date.max(), 'training_rows': len(train)})
        state = {'ridge': ridge, 'trees': trees, 'feature_columns': cols, 'seeds': SEEDS,
                 'fit_first_prediction': first, 'latest_training_label': train.label_date.max(),
                 'last_prediction': window[-1], 'tree_params': TREE_PARAMS,
                 'ridge_alpha': 100., 'recipe': 'half ridge rank, half mean tree ranks'}
        print(f'window {i//3+1}: {first:%Y-%m} -> {pd.Timestamp(window[-1]):%Y-%m}', flush=True)
    predictions = pd.concat(records, ignore_index=True)
    exposure = exposure_series(benchmark, 'vol_target')
    predictions['risk_exposure'] = predictions.date.map(exposure)
    return predictions, pd.DataFrame(audits), state, benchmark


def industry_plans(predictions):
    """Include the latest unlabelled signal without using its future return."""
    plans, previous = [], set()
    for date, frame in predictions.groupby('date', sort=True):
        frame = frame.copy()
        if len(frame) < 5 or frame.pred_ensemble.isna().any():
            raise ValueError(f'Invalid prediction month {date}')
        frame['selection_score'] = frame.pred_ensemble
        frame.loc[frame.ts_code.isin(previous), 'selection_score'] += .2*frame.pred_ensemble.std()
        top = frame.sort_values('selection_score', ascending=False).head(5).copy()
        top['industry_rank'] = np.arange(1, 6)
        top['target_equity_weight'] = top.risk_exposure/5
        plans.append(top)
        previous = set(top.ts_code)
    return pd.concat(plans, ignore_index=True)


def make_budget_plan(candidates, plans, daily, signal_month, equity=25000.,
                     capital_cap=25000., limit_quotes=None):
    """Target cap is exposure * min(account equity, fixed 25k capital cap)."""
    if not np.isfinite(equity) or equity <= 0 or not np.isfinite(capital_cap) or capital_cap <= 0:
        raise ValueError('Equity/capital cap must be finite and positive')
    month = pd.Period(signal_month, freq='M')
    signal_date = month.to_timestamp('M')
    plan = plans[pd.to_datetime(plans.date).dt.to_period('M').eq(month)].copy()
    if len(plan) != 5 or plan.ts_code.duplicated().any():
        raise ValueError('A complete five-industry signal is required')
    if plan.risk_exposure.nunique() != 1:
        raise ValueError('Inconsistent risk exposure within month')
    exposure = float(plan.risk_exposure.iloc[0])
    if not 0 < exposure <= 1:
        raise ValueError('No positive valid risky allocation; retain cash')
    if 'month' not in candidates or 'rank_in_ind' not in candidates:
        raise ValueError('Stock candidates require dated month and rank_in_ind')
    candidates = candidates.copy()
    candidates['month'] = pd.to_datetime(candidates.month)
    candidates['rank_in_ind'] = pd.to_numeric(candidates.rank_in_ind, errors='coerce')
    if not candidates.month.dt.to_period('M').eq(month).all():
        raise ValueError('Candidate month differs from selected signal month')
    if not np.isfinite(pd.to_numeric(candidates.rank_in_ind, errors='coerce')).all() or (candidates.rank_in_ind < 1).any():
        raise ValueError('Invalid candidate ranks')
    scores = plan.set_index('ts_code').selection_score
    candidates = candidates[candidates.ind_code.isin(scores.index)].copy()
    candidates['ind_score'] = candidates.ind_code.map(scores)
    # Do not accept stale prices already attached by a previous workflow.
    candidates = candidates.drop(columns=['reference_price', 'price_date'], errors='ignore')
    candidates = attach_asof_prices(candidates, daily, signal_date)
    all_dates = pd.to_datetime(daily.date)
    last_day = all_dates[all_dates <= signal_date].max()
    candidates = candidates[candidates.price_date.eq(last_day)].copy()
    if limit_quotes is not None:
        candidates = apply_limit_prices(candidates, limit_quotes)
    if candidates.empty:
        raise ValueError('No eligible candidates with fresh prices')
    target = min(equity, capital_cap)*exposure
    selected, summary = optimize_portfolio(candidates, budget=target, max_names=5,
                                           min_names=4, max_stock_weight=.30)
    selected['account_weight'] = selected.planned_amount/equity
    summary.update({'signal_month': str(month), 'account_equity': equity,
                    'capital_cap': capital_cap, 'risk_exposure': exposure,
                    'risky_budget': target, 'account_cash_after_plan': equity-summary['planned_amount'],
                    'account_utilization': summary['planned_amount']/equity,
                    'status': 'allocation_plan_not_filled_orders',
                    'candidate_scoring': 'externally supplied dated within-industry ranks',
                    'price_basis': 'provided limit quotes' if limit_quotes is not None else 'signal-month closing quotes'})
    if summary['planned_amount'] > target+.01 or summary['account_cash_after_plan'] < -.01:
        raise AssertionError('Allocation exceeded account or risk budget')
    return selected, summary


def require_execution_data(paths):
    """Fail closed; never silently report raw-price account returns as repaired."""
    required = ('corporate_actions', 'adjustment_factors', 'daily_price_limits')
    missing = [key for key in required if key not in paths or not Path(paths[key]).is_file()]
    if missing:
        raise ValueError('Account backtest is unavailable until execution datasets are provided: '+', '.join(missing))


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    before = source_hashes()
    if args.candidates:
        path = Path(args.candidates)
        cand = pd.read_pickle(path) if path.suffix == '.pkl' else pd.read_csv(path, dtype={'stock_code': str})
        plans = pd.read_pickle(OUT/'industry_plans.pkl')
        month = args.month or str(pd.to_datetime(plans.date).max().to_period('M'))
        daily = pd.read_pickle(config.STOCK_DAILY_PATH)['df_stock']
        quotes = pd.read_csv(args.limit_quotes, dtype={'stock_code': str}) if args.limit_quotes else None
        selected, summary = make_budget_plan(cand, plans, daily, month, args.equity, args.capital_cap, quotes)
        selected.to_csv(OUT/f'budget_plan_{month}.csv', index=False, encoding='utf-8-sig')
        (OUT/f'budget_plan_{month}.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    market = config.load_industry_monthly()
    predictions, audit, state, benchmark = train_predictions(market)
    plans = industry_plans(predictions)
    predictions.to_pickle(OUT/'predictions_ensemble.pkl')
    predictions.to_csv(OUT/'predictions_ensemble.csv', index=False, encoding='utf-8-sig')
    plans.to_pickle(OUT/'industry_plans.pkl')
    plans.to_csv(OUT/'industry_plans.csv', index=False, encoding='utf-8-sig')
    pd.to_pickle(state, OUT/'model_state.pkl')
    audit.to_csv(OUT/'training_audit.csv', index=False)
    returns = {}
    for risk in ('full', 'vol_target'):
        data, _ = portfolio_returns(predictions, 'pred_ensemble', exposure_series(benchmark, risk))
        returns[risk] = data.ret
    monthly = pd.DataFrame(returns)
    monthly.index += pd.offsets.MonthEnd(1)
    monthly.index.name = 'holding_month'
    monthly.to_csv(OUT/'industry_monthly_returns.csv', encoding='utf-8-sig')
    annual = [{'risk': name, 'year': year, 'n_months': len(x), 'return': (1+x).prod()-1}
              for name in monthly for year, x in monthly[name].groupby(monthly.index.year)]
    pd.DataFrame(annual).to_csv(OUT/'industry_annual_returns.csv', index=False)
    # Compare to previous research, but do not require it as a training input.
    reference = Path(config.OUTPUT_DIR)/'improvement_research/verified/monthly_returns.csv'
    checked = False
    if reference.exists():
        saved = pd.read_csv(reference, index_col=0, parse_dates=True)
        for mode in returns:
            np.testing.assert_allclose(monthly[mode], saved[f'blend_five_seed__k5__{mode}'].reindex(monthly.index), atol=1e-12)
        checked = True
    if source_hashes() != before:
        raise ValueError('Training inputs changed')
    manifest = {'recipe': 'price26_ridge100_gbdt7leaf_five_seed_half_rank', 'seeds': SEEDS,
                'feature_columns': state['feature_columns'], 'source_sha256': before,
                'verified_research_return_match': checked, 'n_realized_months': len(monthly),
                'last_signal_month': str(predictions.date.max().to_period('M')),
                'last_holding_month': str(monthly.index.max().to_period('M')),
                'account_backtest_status': 'not_run_missing_corporate_actions_adjustments_limits',
                'original_default_pipeline_changed': False}
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(pd.DataFrame(annual).query('year >= 2023').to_string(index=False))
    print(f'Industry reproduction verified={checked}; output={OUT}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', help='Dated stock candidate CSV/PKL with month,stock_code,ind_code,rank_in_ind')
    parser.add_argument('--month')
    parser.add_argument('--equity', type=float, default=25000.)
    parser.add_argument('--capital-cap', type=float, default=25000.)
    parser.add_argument('--limit-quotes', help='Optional stock_code,limit_price CSV')
    main(parser.parse_args())
