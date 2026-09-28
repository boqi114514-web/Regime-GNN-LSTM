"""Second-stage robustness checks; explicitly exploratory after batch-1 review.

Never replace the failed pre-2023 selection with a retroactively selected winner.
All five seeds, all 3/5/8 holding counts, and all predefined risk modes are saved.
Baseline replay preserves source row order, important for legacy score ties.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
import research_improvements as research
from s3_ensemble_backtest import run_topk_strategy

OUT = research.OUT/'verified'
SEEDS = [42, 142, 242, 342, 442]
TOP_K = [3, 5, 8]


def year_records(values, **metadata):
    return [{**metadata, 'year': year, 'n_months': len(part),
             'return': (1+part).prod()-1}
            for year, part in values.groupby(values.index.year)]


def held(values):
    result = values.copy()
    result.index = pd.DatetimeIndex(result.index) + pd.offsets.MonthEnd(1)
    result.index.name = 'holding_month'
    return result


def baseline_replay(index):
    root = Path(config.OUTPUT_DIR)
    original = pd.read_pickle(root/'original_architecture/predictions_ensemble.pkl')
    original_ret, _ = run_topk_strategy(original, 'pred_ensemble', inertia=.2)
    original_ret = held(original_ret).reindex(index)
    saved = pd.read_csv(root/'original_architecture/monthly_returns.csv', parse_dates=['holding_month'])
    saved = saved.set_index('holding_month')
    np.testing.assert_allclose(original_ret, saved.original_repaired.reindex(index), atol=1e-14)
    fixed = saved.industry12_64.reindex(index)
    benchmark = saved.industry_equal_weight.reindex(index)
    return pd.DataFrame({'original_repaired': original_ret, 'fixed_industry64': fixed,
                         'industry_equal_weight': benchmark}, index=index)


def bootstrap_log_difference(candidate, baseline, trials=10000, block=3):
    """Conditional descriptive CI, not a multiple-testing-adjusted significance test."""
    data = np.log1p(candidate.to_numpy()) - np.log1p(baseline.to_numpy())
    n = len(data)
    rng = np.random.default_rng(42)
    starts = rng.integers(0, n, (trials, int(np.ceil(n/block))))
    idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(trials, -1)[:, :n]
    differences = data[idx].mean(axis=1) * 12
    return {'annual_log_relative_mean': data.mean()*12,
            'ci95_low': float(np.quantile(differences, .025)),
            'ci95_high': float(np.quantile(differences, .975)),
            'bootstrap_positive_fraction': float((differences > 0).mean()),
            'n_months': n, 'block_months': block, 'bootstrap_draws': trials}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((research.OUT/'protocol.json').read_text(encoding='utf-8'))
    accepted_inputs = research.verify_protocol_inputs(protocol)
    verification_protocol = {
        'status': 'exploratory robustness, chosen after inspecting batch-1 evaluation',
        'first_stage_locked_choice': protocol['selected_strategy'],
        'seeds': SEEDS, 'top_k_sensitivity': TOP_K, 'risk_sensitivity': research.RISK_MODES,
        'families': ['equal_rank_ridge_gbdt_price', 'lambdarank_price'],
        'fixed_ensemble': '50% Ridge rank + 50% mean LightGBM rank across five seeds',
        'training': 'same 72-month rolling window and pre-signal label cutoff as batch 1',
        'baseline_correction': 'Replay original source row order; its tied scores were reordered '
                               'by the batch-1 feature merge. New candidates and locked choice unchanged.',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'batch1_input_sha256': protocol['input_sha256'],
        'measurement_corrected_input_sha256': accepted_inputs,
    }
    (OUT/'protocol.json').write_text(json.dumps(verification_protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    market = config.load_industry_monthly()
    panel, sets, benchmark, _ = research.make_panel(market, config.load_tech_factors(False))
    original_pred = pd.read_pickle(research.OUT/'evaluate/predictions.pkl')
    base = original_pred[['date', 'ts_code', 'actual_ret', 'ridge_price']].copy()
    research.LEARNED = ['gbdt_price', 'ranker_price']
    seed_predictions, seed_records, seed_summaries = [], [], []
    for seed in SEEDS:
        research.PARAMS['random_state'] = seed
        pred, audit, _ = research.rolling_predictions(panel, sets, market.date.max())
        pred = pred[['date', 'ts_code', 'gbdt_price', 'ranker_price']]
        audit.to_csv(OUT/f'training_audit_seed{seed}.csv', index=False)
        pred = base.merge(pred, on=['date', 'ts_code'], validate='one_to_one')
        ranks = pred.groupby('date')[['ridge_price', 'gbdt_price', 'ranker_price']].rank(pct=True)
        pred['blend'] = .5*ranks.ridge_price + .5*ranks.gbdt_price
        pred['gbdt_rank'] = ranks.gbdt_price
        pred['ranker_rank'] = ranks.ranker_price
        pred['seed'] = seed
        seed_predictions.append(pred)
        for family in ('blend', 'ranker_price'):
            for k in TOP_K:
                for mode in research.RISK_MODES:
                    data, _ = research.portfolio_returns(pred, family, research.exposure_series(benchmark, mode), k=k)
                    returns = held(data.ret)
                    seed_records.extend(year_records(returns, seed=seed, family=family, top_k=k, risk=mode))
                    for period, lo, hi in [('all', '2019-01-01', '2026-12-31'),
                                           ('recent', '2023-01-01', '2026-12-31')]:
                        seed_summaries.append({'seed': seed, 'family': family, 'top_k': k, 'risk': mode,
                                               'period': period, **research.metrics(returns.loc[lo:hi])})
        print(f'Completed robustness seed {seed}', flush=True)
    seeds = pd.concat(seed_predictions, ignore_index=True)
    seeds.to_pickle(OUT/'seed_predictions.pkl')
    pd.DataFrame(seed_records).to_csv(OUT/'seed_annual_returns.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(seed_summaries).to_csv(OUT/'seed_summary.csv', index=False, encoding='utf-8-sig')
    avg = seeds.groupby(['date', 'ts_code'])[['gbdt_rank', 'ranker_rank']].mean().reset_index()
    pred = base.merge(avg, on=['date', 'ts_code'], validate='one_to_one')
    pred['blend_five_seed'] = .5*pred.groupby('date').ridge_price.rank(pct=True) + .5*pred.gbdt_rank
    pred['ranker_five_seed'] = pred.ranker_rank
    pred.to_pickle(OUT/'ensemble_predictions.pkl')
    returns_dict, sensitivity, holdings_rows = {}, [], []
    for family in ('blend_five_seed', 'ranker_five_seed'):
        for k in TOP_K:
            for mode in research.RISK_MODES:
                data, holdings = research.portfolio_returns(pred, family, research.exposure_series(benchmark, mode), k=k)
                returns = held(data.ret)
                name = f'{family}__k{k}__{mode}'
                returns_dict[name] = returns
                sensitivity.extend(year_records(returns, strategy=name))
                holdings['strategy'] = name
                holdings_rows.append(holdings)
    monthly = pd.DataFrame(returns_dict)
    bases = baseline_replay(monthly.index)
    monthly = monthly.join(bases)
    first_stage = pd.read_csv(research.OUT/'evaluate/monthly_returns.csv', index_col=0, parse_dates=True)
    # Keep all 36 initial alternatives, not just those that looked good later.
    for name in first_stage:
        if name not in monthly:
            monthly[name] = first_stage[name]
    if monthly.isna().any().any():
        raise ValueError('Missing comparison returns')
    annual, summary, bootstrap = [], [], []
    for name in monthly:
        annual.extend(year_records(monthly[name], strategy=name))
        for period, lo, hi in [('all', '2019-01-01', '2026-12-31'),
                               ('development', '2019-01-01', '2022-12-31'),
                               ('recent', '2023-01-01', '2026-12-31')]:
            summary.append({'strategy': name, 'period': period, **research.metrics(monthly[name].loc[lo:hi])})
    candidate = 'blend_five_seed__k5__full'
    recent = monthly.loc['2023-01-01':]
    for baseline in bases:
        for block in (3, 6):
            bootstrap.append({'candidate': candidate, 'baseline': baseline,
                              **bootstrap_log_difference(recent[candidate], recent[baseline], block=block)})
    ics = []
    for date, month in pred.groupby('date'):
        if month.actual_ret.notna().all():
            ics.append({'date': date, 'ic': spearmanr(month.blend_five_seed, month.actual_ret).statistic})
    pd.DataFrame(ics).to_csv(OUT/'monthly_rank_ic.csv', index=False)
    pd.DataFrame(bootstrap).to_csv(OUT/'conditional_bootstrap.csv', index=False)
    pd.DataFrame(annual).to_csv(OUT/'annual_comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(OUT/'summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(sensitivity).to_csv(OUT/'ensemble_sensitivity.csv', index=False, encoding='utf-8-sig')
    pd.concat(holdings_rows).to_csv(OUT/'ensemble_holdings.csv', index=False, encoding='utf-8-sig')
    monthly.to_csv(OUT/'monthly_returns.csv', encoding='utf-8-sig')
    # Replay artifacts: counts, dates and source checksums are explicit.
    verification_protocol.update({'n_realized_months': len(monthly), 'holding_start': str(monthly.index.min()),
                                  'holding_end': str(monthly.index.max()),
                                  'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                    for p in OUT.iterdir() if p.suffix in ('.csv', '.pkl')}})
    (OUT/'protocol.json').write_text(json.dumps(verification_protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    display = ['blend_five_seed__k5__full', 'blend_five_seed__k5__vol_target',
               'ranker_five_seed__k5__full', 'original_repaired', 'fixed_industry64',
               'industry_equal_weight', protocol['selected_strategy']]
    annual = pd.DataFrame(annual)
    print(annual[annual.strategy.isin(display) & annual.year.ge(2023)]
          .pivot(index='strategy', columns='year', values='return').to_string())
    print(pd.DataFrame(summary).query('strategy in @display').to_string(index=False))
    print(pd.DataFrame(bootstrap).to_string(index=False))
    print('VERIFIED COMPLETE', flush=True)


if __name__ == '__main__':
    main()
