"""Fixed, causal stock-selection recipe for the isolated account experiment.

No parameter search: 50% 12-minus-1-month momentum, 30% 6-month momentum,
20% low 6-month volatility, ranked within historical industries. All SH/SZ
boards participate in ranking; main-board restriction applies at allocation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data_pipeline.execution_data import ROOT
from price_risk_pipeline import industry_plans

OUT = Path(config.OUTPUT_DIR)/'stock_execution_research'


def historical_members(members, date):
    m = members.copy()
    for column in ('in_date', 'out_date'):
        m[column] = pd.to_datetime(m[column].astype('string').str.replace(r'\.0$', '', regex=True), format='%Y%m%d', errors='coerce')
    m = m[m.in_date.notna() & (m.in_date <= date) & (m.out_date.isna() | (m.out_date > date))]
    m = m[['ts_code', 'l1_code']].drop_duplicates()
    if m.ts_code.duplicated().any():
        raise ValueError(f'Conflicting historical membership at {date}')
    return m.rename(columns={'l1_code': 'ind_code'})


def stock_features(monthly):
    m = monthly.copy()
    m['month'] = m.date.dt.to_period('M')
    if m.duplicated(['month', 'ts_code']).any():
        raise ValueError('Duplicate monthly quotes')
    adjusted = m.pivot(index='month', columns='ts_code', values='close') * m.pivot(index='month', columns='ts_code', values='adj_factor')
    adjusted = adjusted.reindex(pd.period_range(adjusted.index.min(), adjusted.index.max(), freq='M'))
    returns = adjusted.pct_change(fill_method=None)
    features = {'mom12skip1': adjusted.shift(1)/adjusted.shift(12)-1,
                'mom6': adjusted/adjusted.shift(6)-1,
                'lowvol6': -returns.rolling(6, min_periods=6).std()}
    for name, values in features.items():
        s = values.stack().rename(name)
        s.index.names = ['month', 'ts_code']
        m = m.merge(s.reset_index(), on=['month', 'ts_code'], how='left', validate='one_to_one')
    return m.dropna(subset=list(features))


def candidates_by_month(monthly, members):
    f = stock_features(monthly)
    result, audit = [], []
    for month, frame in f.groupby('month', sort=True):
        if month < pd.Period('2022-12'):
            continue
        date = frame.date.max()
        mapping = historical_members(members, date)
        merged = frame.merge(mapping, on='ts_code', how='inner', validate='one_to_one')
        merged = merged[~merged.ind_code.isin(config.SW_EXCLUDE)].copy()
        # Only contemporaneous liquidity, not current names or future status.
        merged = merged[pd.to_numeric(merged.amount, errors='coerce') >= 20000].copy()
        ranks = merged.groupby('ind_code')[['mom12skip1', 'mom6', 'lowvol6']].rank(pct=True)
        merged['score'] = .5*ranks.mom12skip1 + .3*ranks.mom6 + .2*ranks.lowvol6
        merged = merged.sort_values(['ind_code', 'score', 'ts_code'], ascending=[True, False, True])
        merged['rank_in_ind'] = merged.groupby('ind_code').cumcount()+1
        merged['stock_code'] = merged.ts_code.str[:6]
        merged['month'] = month.to_timestamp('M')
        result.append(merged)
        audit.append(dict(month=str(month), feature_stocks=len(frame), ranked_stocks=len(merged),
                          missing_membership=int((~frame.ts_code.isin(mapping.ts_code)).sum())))
    return pd.concat(result, ignore_index=True), pd.DataFrame(audit)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(stock_score={'mom12skip1': .5, 'mom6': .3, 'lowvol6': .2},
                    ranking='all SH/SZ boards within historical industry',
                    minimum_signal_day_amount_thousand=20000, starting_cash=25000,
                    capital_cap=25000, fees=0, max_names=5, min_names=4,
                    max_stock_weight=.30, training_or_tuning='none at stock selection layer',
                    comparison='original sector model versus new full and vol-target; same stock layer',
                    status='candidate_generation_only_not_account_returns')
    p = OUT/'protocol.json'
    if p.exists() and json.loads(p.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Existing protocol differs')
    p.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    monthly = pd.read_pickle(ROOT/'stock_month_end_verified.pkl')
    members = pd.read_csv(Path(config.LOCAL_DATA_RAW)/'ts_sw_members.csv', dtype=str)
    candidates, audit = candidates_by_month(monthly, members)
    candidates.to_pickle(OUT/'candidates.pkl')
    audit.to_csv(OUT/'membership_audit.csv', index=False)
    new = pd.read_pickle(Path(config.OUTPUT_DIR)/'price_risk_pipeline/predictions_ensemble.pkl')
    original = pd.read_pickle(Path(config.OUTPUT_DIR)/'original_architecture/predictions_ensemble.pkl')
    original['risk_exposure'] = 1.
    plans = []
    for name, pred in [('new_vol', new), ('new_full', new.assign(risk_exposure=1.)), ('original', original)]:
        plan = industry_plans(pred)
        plan['strategy'] = name
        plans.append(plan)
    pd.concat(plans, ignore_index=True).to_pickle(OUT/'industry_plans.pkl')
    print(audit.to_string(index=False))
    print(f'Candidates: {len(candidates)}; results: {OUT}')


if __name__ == '__main__':
    main()
