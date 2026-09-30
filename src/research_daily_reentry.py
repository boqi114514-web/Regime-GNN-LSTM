"""Daily close signals for idle cash after an exit, retaining monthly strategy.

Returns deliberately measure open-to-close buying pressure, not multi-day raw
close returns. All same-day OHLC ratios are invariant to price rescaling, so
splits/cash-dividend overnight gaps cannot create artificial trend signals.
"""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

VARIANTS=('daily_pressure_retry',)
PROTOCOL=dict(
    variants=list(VARIANTS),
    stage='Exploration on already viewed 2023-2026 history; no untouched holdout claim',
    objective='Supplement monthly selection when at least half the risk budget is available as idle cash, fewer than 5 holdings and no pending sells; retain residual holdings, no midmonth liquidation',
    timing='Signal after close; next trading session open; wait at least 5 sessions after last sale; no same-day use of closing features',
    universe='Previous calendar-month candidate universe and historical L2; all boards in features, mainboard only for buys',
    features='5/20-session products of close/open minus 1 (intraday pressure, NOT total returns), mean close location in daily range over 5 sessions, mean 5-day amount / preceding 20-day mean amount, L2 median 20-day pressure',
    eligibility='All 25 sessions present; pressure5>5%, pressure20>10%, mean5 close-location>=.65, amount expansion>=1.2; L2 at least 5 stocks with median pressure20>0 and positive-pressure20 breadth>=.6',
    score='25% pressure5 percentile +25% pressure20 percentile +30% L2 pressure percentile +20% amount expansion percentile',
    unchanged='Original state_onset monthly entries/exposure, original daily hard/trailing exits, corrected deferred sales, 25000 initial/new-investment cap, no topups, mainboard whole lots, zero costs',
    replacement_phase='advance; no extra leverage or cancellation of pending exits',
    provenance='Hypothesis inspired by price/volume feature families, not a published-performance replication',
    source='https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py')


def pressure_features(daily):
    if daily.duplicated(['date','ts_code']).any(): raise ValueError('Duplicate daily keys')
    dates=pd.DatetimeIndex(sorted(daily.date.unique()))
    def wide(col): return daily.pivot(index='date',columns='ts_code',values=col).reindex(dates)
    opening,close=wide('open'),wide('close')
    log_pressure=np.log(close/opening)
    p5=np.expm1(log_pressure.rolling(5,min_periods=5).sum())
    p20=np.expm1(log_pressure.rolling(20,min_periods=20).sum())
    high,low=wide('high'),wide('low')
    location=((close-low)/(high-low).replace(0,np.nan)).fillna(.5).where(close.notna())
    loc5=location.rolling(5,min_periods=5).mean()
    amount=wide('amount')
    expansion=amount.rolling(5,min_periods=5).mean()/amount.shift(5).rolling(20,min_periods=20).mean().replace(0,np.nan)
    valid=opening.rolling(25,min_periods=25).count().ge(25)&wide('volume').rolling(25,min_periods=25).min().gt(0)
    blocks=[]
    for name,frame in [('pressure5',p5),('pressure20',p20),('location5',loc5),('pressure_amount',expansion)]:
        frame=frame.where(valid); frame.index.name='signal_day'; frame.columns.name='ts_code'
        blocks.append(frame.stack().rename(name))
    return pd.concat(blocks,axis=1).dropna().reset_index()


def candidates_by_signal(candidates, features):
    f=features.copy()
    # Month-start cash reentry can only use the same lagged universe as its
    # monthly strategy. Each signal maps to its NEXT execution session outside.
    lookup={m.to_period('M'):g for m,g in candidates.groupby('month')}
    blocks=[]
    for day,g in f.groupby('signal_day',sort=True):
        period=day.to_period('M')-1
        if period not in lookup: continue
        c=lookup[period].merge(g,on='ts_code',how='inner',validate='one_to_one')
        grouped=c.groupby('l2_code').pressure20
        c['pressure_peer_count']=grouped.transform('count')
        c['pressure_peer']=grouped.transform('median')
        c['pressure_peer_breadth']=grouped.transform(lambda s:s.gt(0).mean())
        ranks=c[['pressure5','pressure20','pressure_peer','pressure_amount']].rank(pct=True)
        c['leadership_score']=.25*ranks.pressure5+.25*ranks.pressure20+.3*ranks.pressure_peer+.2*ranks.pressure_amount
        eligible=(c.pressure5.gt(.05)&c.pressure20.gt(.1)&c.location5.ge(.65)&c.pressure_amount.ge(1.2)
            &c.pressure_peer_count.ge(5)&c.pressure_peer.gt(0)&c.pressure_peer_breadth.ge(.6))
        c=c[eligible].copy(); c['phase']='advance'
        blocks.append(c)
    return pd.concat(blocks,ignore_index=True) if blocks else pd.DataFrame()


def can_reenter(day, previous_day, session_index, last_sale_index, capacity_available, firsts):
    return (day not in firsts and previous_day is not None and capacity_available and last_sale_index is not None
            and session_index-last_sale_index>=5 and previous_day.to_period('M')==day.to_period('M'))


def prepare_candidates(out):
    path=out/'daily_pressure_protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=PROTOCOL:
        raise ValueError('Daily pressure protocol changed')
    path.write_text(json.dumps(PROTOCOL,indent=2),encoding='utf-8')
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024),b''): h.update(block)
        return h.hexdigest()
    inputs={str(p):digest(p) for p in [out/'daily.pkl',out/'candidates.pkl',Path(__file__)]}
    cache=out/'candidate_audit_daily_pressure_retry.pkl'
    manifest=out/'daily_pressure_feature_manifest.json'
    if cache.exists() and manifest.exists():
        previous=json.loads(manifest.read_text(encoding='utf-8'))
        if previous.get('inputs')==inputs and previous.get('output')==digest(cache):
            c=pd.read_pickle(cache)
            return {day:g for day,g in c.groupby('signal_day')}
    f=pressure_features(pd.read_pickle(out/'daily.pkl'))
    c=candidates_by_signal(pd.read_pickle(out/'candidates.pkl'),f)
    c.to_pickle(cache)
    manifest.write_text(json.dumps(dict(inputs=inputs,output=digest(cache)),indent=2),encoding='utf-8')
    print('daily pressure eligible rows',len(c),'signal days',c.signal_day.nunique(),flush=True)
    return {day:g for day,g in c.groupby('signal_day')}
