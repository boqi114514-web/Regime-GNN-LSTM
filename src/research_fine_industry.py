"""Dated fine-industry leadership research, isolated from production selection.

Historical interval records are reconstructed today, not a timestamped archive
of releases. Current-only membership must never substitute for missing history.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

VARIANTS = ('fine_leader_retry', 'fine_router_retry')
PROTOCOL = dict(
    variants=list(VARIANTS),
    stage='Exploration on previously inspected history; not a new holdout or a parameter sweep',
    membership='Current and exited L3 intervals, in_date <= signal trading date < out_date; require agreement with dated L2; ambiguous/missing membership stays unavailable, never current-label backfill',
    universe='All existing SH/SZ candidate boards form signals; mainboard only at execution',
    groups='At least 5 candidate stocks; equal-weight L3 median adjusted mom1/mom3/mom6 and positive mom3 breadth',
    trend='L3 median mom1>0, mom3>0, mom6>0 and breadth>=.6',
    group_score='20% rank(median mom1) + 50% rank(median mom3) + 30% rank(median mom6), across eligible-size L3 groups',
    stock_score='50% group_score + 50% existing stock fast_score; replaces score, no ticker/theme/size whitelist',
    stock_gate='Positive stock mom1/mom3/mom6, positive L3 trend, group_score in top 20% of eligible-size L3 groups; advance exits',
    routing='fine_leader uses fine selection every month; fine_router uses it only when aggregate L1 breadth<50% AND at least one fine group passes; otherwise unchanged state_onset',
    unchanged='State_onset exposure, monthly rebalancing, daily next-open original stops, deferred sell retries, 25000 initial/new-investment cap, no topups, zero costs, 100-share buys',
    sources=['https://tushare.pro/document/2?doc_id=335',
             'https://arxiv.org/abs/2110.13716'],
    note='Concept-shared-information motivation only; this is not an implementation or replication of HIST')


def validate_members(frame, l1=None):
    required = ['l1_code','l2_code','l3_code','ts_code','in_date','out_date','is_new']
    if frame.empty or not set(required).issubset(frame):
        raise ValueError('Empty or incomplete membership response')
    f = frame.copy()
    if l1 is not None and (len(f)>=2000 or not f.l1_code.eq(l1).all()):
        raise ValueError('Wrong L1 filter or possibly truncated response')
    for col in ('in_date','out_date'):
        s = f[col].astype('string').str.replace(r'\.0$', '', regex=True)
        f[col] = pd.to_datetime(s, format='%Y%m%d', errors='coerce')
        if (s.notna() & f[col].isna()).any():
            raise ValueError('Invalid membership date')
    if f.in_date.isna().any() or (f.out_date.notna() & f.out_date.le(f.in_date)).any():
        raise ValueError('Invalid membership interval')
    if not f.is_new.isin(['Y','N']).all() or (f.is_new.eq('N') & f.out_date.isna()).any():
        raise ValueError('Exited members lack exit dates')
    return f


def collect_global_current(out, raw_dir):
    """Gateway sometimes ignores L1 filters; only accept disjoint complete pages."""
    from data_pipeline.tushare_config import get_pro
    folder=out/'fine_membership_history'
    pages=out/'fine_membership_pages'
    pages.mkdir(parents=True,exist_ok=True)
    frames=[]; seen=set()
    for offset in range(0,18000,3000):
        path=pages/f'current_{offset}.pkl'
        if path.exists(): x=pd.read_pickle(path)
        else:
            errors=[]
            for limit in (3000,6000,5000):
                try:
                    x=get_pro().query('index_member_all',is_new='Y',offset=offset,limit=limit)
                    if not x.empty: validate_members(x)
                    if len(x)>3000 or (not x.empty and not x.is_new.eq('Y').all()):
                        raise ValueError('Unexpected current page size or flags')
                    break
                except Exception as exc: errors.append(str(exc))
            else: raise RuntimeError(f'Current page {offset}: {errors}')
        if x.empty:
            if not frames: raise ValueError('Empty current membership universe')
            break
        keys=set(zip(x.ts_code,x.l3_code,x.in_date))
        if len(keys)!=len(x) or keys&seen: raise ValueError('Repeated page keys; offset ignored')
        seen|=keys; frames.append(x); x.to_pickle(path)
        print('current global membership page',offset,len(x),flush=True)
        if len(x)<3000: break
    else: raise ValueError('Current pages never terminated')
    current=pd.concat(frames,ignore_index=True)
    old=pd.concat([pd.read_pickle(p) for p in (raw_dir/'_l2_members_cache').glob('*.pkl')])
    if current.ts_code.nunique()<.95*old.ts_code.nunique(): raise ValueError('Truncated global current universe')
    for l1 in pd.read_pickle(out/'candidates.pkl').ind_code.unique():
        x=current[current.l1_code.eq(l1)].copy(); validate_members(x,l1)
        if x.ts_code.nunique()<.9*old[old.l1_code.eq(l1)].ts_code.nunique():
            raise ValueError('Incomplete current industry '+l1)
    for l1 in pd.read_pickle(out/'candidates.pkl').ind_code.unique():
        x=current[current.l1_code.eq(l1)].copy()
        path=folder/f'{l1}_Y.pkl'; x.to_pickle(path)
        path.with_suffix('.json').write_text(json.dumps(dict(api='index_member_all',
            method='validated disjoint global pages',offsets=list(range(0,len(frames)*3000,3000)),
            rows=len(x)),indent=2),encoding='utf-8')


def collect_history(out, raw_dir, offline=False):
    from data_pipeline.tushare_config import GatewayClient, get_pro
    cached = sorted((raw_dir/'_l2_members_cache').glob('*.pkl'))
    current = pd.concat([pd.read_pickle(p) for p in cached], ignore_index=True).drop_duplicates()
    validate_members(current)
    folder = out/'fine_membership_history'
    folder.mkdir(parents=True, exist_ok=True)
    required_l1=set(pd.read_pickle(out/'candidates.pkl').ind_code.unique())
    l1s = sorted(set(current.l1_code)&required_l1)
    if set(l1s)!=required_l1: raise ValueError('Candidate industry missing from hierarchy inventory')
    def fetch(l1, flag):
        path = folder/(f'{l1}.pkl' if flag=='N' else f'{l1}_Y.pkl')
        if path.exists():
            x = pd.read_pickle(path); validate_members(x,l1); return x
        if offline: raise ValueError(f'Missing L3 membership: {l1} {flag}')
        shared = get_pro()
        # Each worker owns its requests.Session.
        pro = GatewayClient(shared._key, shared._url) if isinstance(shared,GatewayClient) else shared
        errors = []
        for extra in (dict(offset=0,limit=2000),dict(limit=2000),dict(offset=0),{},
                      dict(limit=5000),dict(offset=0,limit=5000),dict(limit=10000)):
            try:
                x = pro.query('index_member_all',l1_code=l1,is_new=flag,**extra)
                validate_members(x,l1)
                # A current-only response cannot certify historical coverage.
                if not x.is_new.eq(flag).any(): raise ValueError(f'No {flag} records in response')
                if flag=='Y':
                    expected=current[current.l1_code.eq(l1)].ts_code.nunique()
                    if x.loc[x.is_new.eq('Y'),'ts_code'].nunique()<.9*expected:
                        raise ValueError('Current response omits too many previously known members')
                x.to_pickle(path)
                path.with_suffix('.json').write_text(json.dumps(dict(api='index_member_all',
                    parameters=dict(l1_code=l1,is_new=flag,**extra),rows=len(x),
                    exited=int(x.is_new.eq('N').sum())),indent=2),encoding='utf-8')
                print('fine membership',l1,flag,len(x),flush=True)
                return x
            except Exception as exc: errors.append(str(exc))
        raise RuntimeError(f'{l1} {flag}: {errors}')
    records=[]; failures=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(fetch,l1,flag):(l1,flag) for l1 in l1s for flag in ('N','Y')}
        for job in as_completed(jobs):
            try: records.append(job.result())
            except Exception as exc: failures.append(dict(l1_code=jobs[job][0],is_new=jobs[job][1],error=str(exc)))
    (out/'fine_membership_pending.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
    if failures: raise RuntimeError(f'{len(failures)} historical industry queries unresolved')
    f = validate_members(pd.concat(records,ignore_index=True).drop_duplicates())
    f = f.drop_duplicates(['ts_code','l1_code','l2_code','l3_code','in_date','out_date'])
    f.to_pickle(out/'fine_memberships.pkl')
    return f


def attach_membership(candidates, members):
    blocks=[]; audits=[]
    for month, block in candidates.groupby('month',sort=True):
        date=block.date.max()
        m=members[members.in_date.le(date)&(members.out_date.isna()|members.out_date.gt(date))]
        m=m[['ts_code','l2_code','l3_code']].drop_duplicates()
        merged=block.merge(m,on=['ts_code','l2_code'],how='left')
        conflicts=merged.groupby('ts_code').l3_code.nunique().gt(1)
        bad=set(conflicts[conflicts].index)
        merged.loc[merged.ts_code.isin(bad),'l3_code']=pd.NA
        merged=merged.drop_duplicates(['month','ts_code'])
        if len(merged)!=len(block): raise AssertionError('Membership changed stock universe')
        audits.append(dict(month=month,candidates=len(block),classified=int(merged.l3_code.notna().sum()),
                           ambiguous=len(bad),signal_date=date))
        blocks.append(merged)
    return pd.concat(blocks,ignore_index=True),pd.DataFrame(audits)


def fine_scores(candidates, members, market, name):
    if name not in VARIANTS: raise ValueError(name)
    c, coverage=attach_membership(candidates,members)
    groups=c.groupby(['month','l3_code']).agg(
        fine_count=('mom3','count'),fine_mom1=('mom1','median'),fine_mom3=('mom3','median'),
        fine_mom6=('mom6','median'),fine_breadth=('mom3',lambda s:s.gt(0).mean()))
    groups=groups[groups.fine_count.ge(5)].copy()
    rank=groups.groupby('month')[['fine_mom1','fine_mom3','fine_mom6']].rank(pct=True)
    groups['fine_score']=.2*rank.fine_mom1+.5*rank.fine_mom3+.3*rank.fine_mom6
    groups['fine_score_rank']=groups.groupby('month').fine_score.rank(pct=True)
    groups['fine_trend']=(groups.fine_mom1.gt(0)&groups.fine_mom3.gt(0)&groups.fine_mom6.gt(0)
                          &groups.fine_breadth.ge(.6)&groups.fine_score_rank.ge(.8))
    c=c.merge(groups.reset_index(),on=['month','l3_code'],how='left',validate='many_to_one')
    c['original_leadership_score']=c.leadership_score
    c['original_phase']=c.phase
    qualifies=c.fine_trend.eq(True)&c.mom1.gt(0)&c.mom3.gt(0)&c.mom6.gt(0)
    if name=='fine_router_retry':
        available=qualifies.groupby(c.month).any()
        active=c.month.map(market.breadth).lt(.5)&c.month.map(available).fillna(False)
    else: active=pd.Series(True,index=c.index)
    c['fine_route']=active
    c['eligible']=c.size_bucket.eq(c.chosen_style)&~c.phase.isin(['retreat','overheat'])
    c.loc[active,'eligible']=qualifies[active]
    c.loc[active,'leadership_score']=.5*c.loc[active,'fine_score']+.5*c.loc[active,'fast_score']
    c.loc[active&qualifies,'phase']='advance'
    return c,coverage,groups.reset_index()


def prepare_candidates(out, name):
    if json.loads((out/'fine_membership_pending.json').read_text(encoding='utf-8')):
        raise ValueError('Historical L3 membership download is incomplete')
    # A prior current-only inventory is dated May 2026; require refreshed current
    # records as well as exited history before running through September.
    for l1 in pd.read_pickle(out/'candidates.pkl').ind_code.unique():
        if not (out/'fine_membership_history'/f'{l1}_Y.pkl').exists():
            raise ValueError('Current L3 membership not refreshed: '+l1)
    path=out/'fine_industry_protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=PROTOCOL:
        raise ValueError('Fine-industry protocol changed')
    path.write_text(json.dumps(PROTOCOL,indent=2),encoding='utf-8')
    c,coverage,groups=fine_scores(pd.read_pickle(out/'candidates.pkl'),
        pd.read_pickle(out/'fine_memberships.pkl'),
        pd.read_csv(out/'market_features.csv',parse_dates=['month']).set_index('month'),name)
    coverage.to_csv(out/'fine_membership_coverage.csv',index=False)
    groups.to_csv(out/'fine_group_signals.csv',index=False)
    c.to_pickle(out/f'candidate_audit_{name}.pkl')
    return c[c.eligible].copy()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--global-current',action='store_true')
    args=parser.parse_args()
    import config
    if args.global_current:
        collect_global_current(Path(config.OUTPUT_DIR)/'market_state_board_complete',Path(config.LOCAL_DATA_RAW))
    collect_history(Path(config.OUTPUT_DIR)/'market_state_board_complete',Path(config.LOCAL_DATA_RAW),args.offline)
