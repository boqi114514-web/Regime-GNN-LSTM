"""Past-only stock trend features; isolated monthly-selection ablations.

These are monthly endpoint analogues, NOT a claim to implement Alpha158 or
the original daily 52-week-high strategy. No return-based parameter search.
"""
import json

import numpy as np
import pandas as pd


VARIANTS = ('trend_features_retry', 'trend_open_retry')
PROTOCOL = dict(
    stage='Exploration on already examined 2023-2026 history; not untouched validation',
    variants=list(VARIANTS),
    hypothesis='Separate persistent individual leadership from sector and size hard gates',
    features='Adjusted monthly-close proximity to 12-month endpoint high; signed 6-month log-price path efficiency; L2-relative 3-month momentum; mean daily amount / preceding 3-month mean',
    score='30% mom3 rank + 15% mom6 rank + 15% high proximity rank + 15% path efficiency rank + 15% L2-relative rank + 10% amount expansion rank',
    rescore='trend_features_retry preserves existing size and phase eligibility; replaces score only',
    open='trend_open_retry also admits stocks with mom1>0, mom3>0, mom3 percentile>=85%, high proximity>=.9 and efficiency>0; admitted stocks use advance exits',
    missing='A missing feature receives neutral .5 rank; missing high/efficiency cannot unlock the gate; no backward fill',
    unchanged='Monthly entry, state_onset exposure, corrected deferred sells, existing whole-lot optimizer, 25000 initial/cap, zero costs, no topups, mainboard purchases only',
    sources=['https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py',
             'https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2004.00695.x'])


def monthly_features(monthly, daily):
    m = monthly.copy()
    m['month'] = m.date.dt.to_period('M').dt.to_timestamp('M')
    if m.duplicated(['month', 'ts_code']).any():
        raise ValueError('Duplicate monthly price keys')
    p = m.pivot(index='month', columns='ts_code', values='close')
    a = m.pivot(index='month', columns='ts_code', values='adj_factor')
    p = (p*a).reindex(pd.date_range(p.index.min(), p.index.max(), freq='ME'))
    if (p <= 0).any().any():
        raise ValueError('Nonpositive adjusted prices')
    logp = np.log(p)
    path = logp.diff().abs().rolling(6, min_periods=6).sum()
    efficiency = (logp-logp.shift(6))/path.replace(0, np.nan)
    high = p/p.rolling(12, min_periods=12).max()
    d = daily[['date', 'ts_code', 'amount']].copy()
    if d.duplicated(['date', 'ts_code']).any():
        raise ValueError('Duplicate daily amount keys')
    d['month'] = d.date.dt.to_period('M').dt.to_timestamp('M')
    amt = d.groupby(['month', 'ts_code']).amount.mean().unstack()
    amt = amt.reindex(pd.date_range(amt.index.min(), amt.index.max(), freq='ME'))
    expansion = amt/amt.shift(1).rolling(3, min_periods=3).mean().replace(0, np.nan)
    blocks = []
    for name, frame in [('high_proximity12', high), ('efficiency6', efficiency),
                        ('amount_expansion', expansion)]:
        frame.index.name = 'month'; frame.columns.name = 'ts_code'
        blocks.append(frame.stack().rename(name))
    return pd.concat(blocks, axis=1).reset_index()


def score_candidates(candidates, features, name):
    if name not in VARIANTS:
        raise ValueError(name)
    c = candidates.merge(features, on=['month', 'ts_code'], how='left', validate='one_to_one')
    c['original_leadership_score'] = c.leadership_score
    c['original_phase'] = c.phase
    peer = c.groupby(['month', 'l2_code']).mom3.transform('median')
    count = c.groupby(['month', 'l2_code']).mom3.transform('count')
    c['relative_mom3'] = (c.mom3-peer).where(count >= 5)
    weights = dict(mom3=.30, mom6=.15, high_proximity12=.15,
                   efficiency6=.15, relative_mom3=.15, amount_expansion=.10)
    c['leadership_score'] = 0.
    for col, weight in weights.items():
        rank = c.groupby('month')[col].rank(pct=True).fillna(.5)
        c[f'{col}_feature_rank'] = rank
        c['leadership_score'] += weight*rank
    c['eligible'] = c.size_bucket.eq(c.chosen_style) & ~c.phase.isin(['retreat', 'overheat'])
    c['feature_override'] = False
    if name == 'trend_open_retry':
        strong = (c.mom1.gt(0) & c.mom3.gt(0) & c.mom3_feature_rank.ge(.85)
                  & c.high_proximity12.ge(.9) & c.efficiency6.gt(0))
        c.loc[strong, 'feature_override'] = True
        c.loc[strong, 'eligible'] = True
        c.loc[strong, 'phase'] = 'advance'
    return c


def prepare_candidates(out, root, name):
    path = out/'trend_feature_protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != PROTOCOL:
        raise ValueError('Do not change an existing feature experiment')
    path.write_text(json.dumps(PROTOCOL, indent=2), encoding='utf-8')
    f = monthly_features(pd.read_pickle(root/'stock_month_end_verified.pkl'),
                         pd.read_pickle(out/'daily.pkl'))
    f.to_pickle(out/'trend_features.pkl')
    c = score_candidates(pd.read_pickle(out/'candidates.pkl'), f, name)
    c.to_pickle(out/f'candidate_audit_{name}.pkl')
    return c[c.eligible].copy()
