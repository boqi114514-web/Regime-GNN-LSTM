"""Walk-forward stock rank learning, using only matured historical labels."""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from research_trend_features import monthly_features, score_candidates

VARIANTS = ('learned_leader_retry',)
PARAMS = dict(n_estimators=120, learning_rate=.03, num_leaves=7, max_depth=3,
              min_child_samples=1000, reg_lambda=50., n_jobs=2, verbosity=-1,
              random_state=42, deterministic=True, force_col_wise=True)
COLS = ['mom1', 'mom3', 'mom6', 'mom12skip1', 'lowvol6', 'score', 'sector_rank',
        'peer_rank', 'size_rank', 'high_proximity12', 'efficiency6',
        'amount_expansion', 'relative_mom3', 'l2_breadth', 'l2_median',
        'market_mom1', 'market_mom3', 'market_mom6', 'market_breadth', 'market_amount_ratio']
PROTOCOL = dict(
    stage='Exploration on previously examined history; no claim of untouched test',
    variants=list(VARIANTS), model=PARAMS, features=COLS,
    target='Next calendar-month adjusted close-to-close cross-sectional return rank minus .5; not a fill-price return',
    training='Rolling 36 signal months, at least 12 matured signal months, quarterly refit, label month <= month before first prediction; no random time split',
    weighting='Each training month has equal total sample weight',
    fallback='Use unchanged state_onset selection until 12 historical label months are available',
    selection='All boards participate in training and prediction; positive predicted relative-return scores are eligible, divided by same-month maximum for MILP numerical scale; no industry-phase or size-bucket gate',
    exits='Own mom1>0 and mom3>0 uses advance rules; other selected stocks use range rules',
    unchanged='Monthly entries, state_onset exposure, corrected deferred liquidation, 25000 cap, no topups, existing mainboard/whole-lot execution and exit thresholds',
    missing_labels='Missing future prices are not filled; retain coverage audit; no label from current/future prediction window')


def label_panel(candidates, monthly):
    m=monthly.copy()
    m['month']=m.date.dt.to_period('M').dt.to_timestamp('M')
    m['adjusted']=m.close*m.adj_factor
    price=m.pivot(index='month',columns='ts_code',values='adjusted').sort_index()
    price=price.reindex(pd.date_range(price.index.min(),price.index.max(),freq='ME'))
    ret=price.shift(-1)/price-1
    ret.index.name='month'; ret.columns.name='ts_code'
    labels=ret.stack().rename('future_return').reset_index()
    p=candidates.merge(labels,on=['month','ts_code'],how='left',validate='one_to_one')
    p['label_date']=p.month+pd.offsets.MonthEnd(1)
    p['target']=p.groupby('month').future_return.rank(pct=True)-.5
    return p


def walkforward(panel, columns=COLS, params=PARAMS, min_months=12):
    records=[]; audits=[]; models={}; model=None; next_fit=None
    for day,frame in panel.groupby('month',sort=True):
        cutoff=day-pd.offsets.MonthEnd(1)
        if model is None or day>=next_fit:
            train=panel[panel.month.ge(day-pd.DateOffset(months=36)) & panel.label_date.le(cutoff)
                        & panel.target.notna()].copy()
            if train.month.nunique()>=min_months:
                weight=1/train.groupby('month').ts_code.transform('size')
                weight/=weight.mean()
                model=lgb.LGBMRegressor(objective='regression',**params)
                model.fit(train[columns],train.target,sample_weight=weight)
                next_fit=day+pd.offsets.MonthEnd(3)
                models[str(day.date())]=model
                audits.append(dict(first_prediction=day,latest_training_label=train.label_date.max(),
                                   training_months=train.month.nunique(),training_rows=len(train),
                                   refit_after=next_fit))
        block=frame[['month','ts_code']].copy()
        block['prediction']=model.predict(frame[columns]) if model is not None else np.nan
        block['fallback']=model is None
        records.append(block)
    return pd.concat(records,ignore_index=True),pd.DataFrame(audits),models


def prepare_candidates(out,root):
    path=out/'learned_leader_protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=PROTOCOL:
        raise ValueError('Do not change an existing learned-leader protocol')
    path.write_text(json.dumps(PROTOCOL,indent=2),encoding='utf-8')
    monthly=pd.read_pickle(root/'stock_month_end_verified.pkl')
    features=monthly_features(monthly,pd.read_pickle(out/'daily.pkl'))
    c=score_candidates(pd.read_pickle(out/'candidates.pkl'),features,'trend_features_retry')
    c['l2_breadth']=c.groupby(['month','l2_code']).mom3.transform(lambda s:s.gt(0).mean())
    c['l2_median']=c.groupby(['month','l2_code']).mom3.transform('median')
    market=pd.read_csv(out/'market_features.csv',parse_dates=['month']).set_index('month')
    for col in ('mom1','mom3','mom6','breadth','amount_ratio'):
        c['market_'+col]=c.month.map(market[col])
    c[COLS]=c[COLS].replace([np.inf,-np.inf],np.nan)
    panel=label_panel(c,monthly)
    panel.groupby('month').future_return.agg(['count','size']).to_csv(out/'learned_label_coverage.csv')
    pred,audit,models=walkforward(panel)
    audit.to_csv(out/'learned_training_audit.csv',index=False)
    pd.to_pickle(models,out/'learned_leader_models.pkl')
    c=c.merge(pred,on=['month','ts_code'],validate='one_to_one')
    use=~c.fallback
    peak=c.groupby('month').prediction.transform('max')
    c['leadership_score']=c.original_leadership_score
    c.loc[use,'leadership_score']=(c.prediction/peak.where(peak.gt(0))).clip(0,1)
    c.loc[use,'eligible']=c.loc[use,'prediction'].gt(0)
    c.loc[use,'phase']=np.where(c.loc[use,'mom1'].gt(0)&c.loc[use,'mom3'].gt(0),'advance','range')
    c.to_pickle(out/'candidate_audit_learned_leader_retry.pkl')
    print('learned leader fits',len(audit),'fallback months',c.loc[c.fallback,'month'].nunique(),flush=True)
    return c[c.eligible].copy()
