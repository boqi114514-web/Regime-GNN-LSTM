"""Causal market-state / size-style / next-open exit research, isolated outputs.

Discrete jump-model objective is independently implemented from the published
method: within-state squared error + a penalty for transitions. Online inference
uses prefix end-state costs, never full-test-sequence smoothing.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import config
import research_leadership as leadership
import small_account_backtest as engine
from data_pipeline.execution_data import ROOT, fetch_variants, validate_rows
from data_pipeline.tushare_config import get_pro
from s7_budget_portfolio import optimize_portfolio as original_optimizer
from research_trend_features import VARIANTS as FEATURE_VARIANTS
from research_learned_leaders import VARIANTS as LEARNED_VARIANTS
from research_structural_router import VARIANTS as ROUTER_VARIANTS
from research_fine_industry import VARIANTS as FINE_VARIANTS
from research_peer_graph import VARIANTS as GRAPH_VARIANTS
from research_daily_reentry import VARIANTS as DAILY_VARIANTS
from research_quality_floor import VARIANTS as QUALITY_VARIANTS

# Revised-input research has its own default. Historical result snapshots stay put.
OUT = Path(config.OUTPUT_DIR)/'market_state_board_complete'
VARIANTS = ('baseline_stops','state_only','state_stops','core_stops','small_stops')
SOFT_VARIANTS = ('soft_size','leader_override','soft_leader')
NARROW_VARIANTS = ('narrow_leader','narrow_heat')
RETRY_VARIANTS = ('state_onset_retry','narrow_heat_retry')
HOLD_VARIANTS = ('state_hold_retry','narrow_hold_retry')
ADVANCE_HOLD_VARIANTS = ('narrow_advance_hold_retry',)
EXTENSIONS = ('state_onset',)+SOFT_VARIANTS+NARROW_VARIANTS+RETRY_VARIANTS+FEATURE_VARIANTS+LEARNED_VARIANTS+HOLD_VARIANTS+ADVANCE_HOLD_VARIANTS+ROUTER_VARIANTS+FINE_VARIANTS+GRAPH_VARIANTS+DAILY_VARIANTS+QUALITY_VARIANTS
NARROW_PROTOCOL = dict(
    stage='Fourth-stage exploration after all three soft-selection accounts; not an unseen test',
    variants=list(NARROW_VARIANTS),fixed_before_new_account_results=True,
    hypothesis='Unconditional fast ranking hurts broad/ordinary markets; local leadership must bypass both sector and size penalties only in narrow markets',
    narrow='Previous-month fraction of positive-three-month L1 industries <50%',
    strong='mom1>0, mom3>15%, mom6>0, all-board mom3 percentile>=85%, L2 count>=5; either L2 median>0 and breadth>55%, or stock mom3 exceeds L2 median by >=20pp',
    score='For promoted stocks only: .60 fast_score + .40 within-month rank(stock mom3 - L2 median mom3); no L1 penalty',
    eligibility='Preserve state_onset size/phase gate for other stocks; promoted stocks may bypass it and use advance exits',
    heat_ablation='narrow_heat excludes stocks with mom3>150% AND mom1>30% only in narrow markets; narrow_leader has no new heat exclusion',
    unchanged='state_onset exposure, 25000 initial and investment cap, no topup, all original exit thresholds, whole-lot mainboard execution',
    evaluation='All 45 months, every year and daily drawdown, not just 2026 April-May')
SOFT_PROTOCOL = dict(
    stage='Third-stage exploration after state_onset results; all history previously examined',
    variants=list(SOFT_VARIANTS), fixed_before_new_account_results=True,
    unchanged='state_onset exposure, whole-lot allocation, 25000 cap, monthly entry, next-open stops, all 45 months',
    soft_size='Remove size-bucket eligibility gate; .95 existing leadership score + .05 preferred-style indicator',
    leader_override='Keep size gate; allow strong stocks through retreat and promote range/retreat to advance',
    strong='mom1>0, mom3>15%, mom6>0, within-month all-board mom3 percentile>=85%; L2 >=5 stocks, median mom3>0 and positive-mom3 breadth>55%',
    strong_score='Same pre-existing advance score: .75*(.55 fast+.25 peer+.20 original)+.25 sector_rank',
    soft_leader='Combine soft_size and leader_override; no stock/theme/month whitelist',
    overheat='Keep existing overheat exclusion; no observation so far',
    evaluation='All years, all months, daily drawdown and 7500 profit target; publish failures too')
PROTOCOL = dict(
    stage='Exploration on previously examined history; not a new untouched holdout',
    period='2023-01-01 through 2026-09-24', initial_cash=25000, capital_cap=25000,
    variants=list(VARIANTS), fees=0, external_topups=False,
    market_model='2-state discrete jump model, monthly past 72 observations, penalty 5, quarterly refit',
    features=['industry median mom1','industry median mom3','industry median mom6',
              'industry 6-month volatility median','fraction industries positive mom3','SH/SZ amount 1/6-month ratio'],
    inference='Standardization and centroids fit before prediction; online prefix costs; no future path smoothing',
    sector_phase='advance: mom3>15%, mom1>0, stock breadth>55%, amount ratio>1.05; retreat: mom1<-8% and mom3<0; otherwise range',
    overheat='mom1>30% and stock breadth<50%; keep out of new concentrated entries',
    exposure='market favorable 100%, unfavorable 35%; no leverage',
    stock_score='advance .55 fast+.25 peer+.20 original; range .70 original+.30 lowvol; .75 stock+.25 sector hybrid rank',
    size='historical L2 circ_mv rank: core top30%, small bottom30%, middle remainder; no current market cap backfill',
    adaptive='small bucket only when its median mom1 exceeds core by >3pp and its median mom3 is positive; otherwise core',
    allocation='1-5 names, no stock or sector capital cap, same whole-lot optimizer, top120 feasible scores',
    hard_stop=.08, trailing_activation=.20, advance_trail=.15, range_trail=.10,
    range_take_profit=.30, advance_take_profit=None,
    exit_execution='Close signal, subsequent executable opening quote; gap loss retained, lower-limit exits deferred; no intraday high/low ordering assumptions',
    reentry='No new entries after a stop until next monthly rebalance',
    evaluation='All months, calendar-year profits, daily drawdown, monthly profit/25000 and 7500 target hits; report all completed variants')


def dp_path(x, centers, penalty, initial=None):
    loss=((x[:,None,:]-centers[None,:,:])**2).sum(axis=2)/2
    k=len(centers)
    transition=penalty*(1-np.eye(k))
    values=np.empty_like(loss)
    parent=np.zeros(loss.shape,dtype=int)
    values[0]=loss[0] if initial is None else loss[0]+(initial[:,None]+transition).min(axis=0)
    for i in range(1,len(x)):
        costs=values[i-1,:,None]+transition
        parent[i]=costs.argmin(axis=0)
        values[i]=loss[i]+costs.min(axis=0)
        values[i]-=values[i].min()
    path=np.zeros(len(x),dtype=int); path[-1]=values[-1].argmin()
    for i in range(len(x)-2,-1,-1): path[i]=parent[i+1,path[i+1]]
    return path,values


def fit_jump(x, penalty=5.):
    best=None
    for seed in (42,142,242):
        centers=KMeans(n_clusters=2,n_init=1,random_state=seed).fit(x).cluster_centers_
        previous=None
        for _ in range(30):
            path,cost=dp_path(x,centers,penalty)
            if previous is not None and np.array_equal(path,previous): break
            previous=path.copy()
            centers=np.array([x[path==k].mean(axis=0) if (path==k).any() else centers[k] for k in range(2)])
        objective=.5*np.sum((x-centers[path])**2)+penalty*np.sum(path[1:]!=path[:-1])
        if best is None or objective<best[0]: best=(objective,centers.copy(),path.copy())
    return best[1],best[2]


def online_states(features):
    records=[]; centers=mean=std=cost=None; good=None; cutoff=None
    for date,row in features[features.index>=pd.Timestamp('2022-12-31')].iterrows():
        if centers is None or date.month in (3,6,9,12):
            train=features[features.index<date].tail(72)
            if len(train)<48: raise ValueError('Insufficient pre-signal regime training history')
            mean=train.mean().to_numpy(); std=train.std().clip(lower=1e-6).to_numpy()
            x=(train.to_numpy()-mean)/std
            centers,path=fit_jump(x)
            # Semantic state assignment from observed training momentum only.
            good=int(np.argmax([train.iloc[np.flatnonzero(path==k)].mom3.mean() if (path==k).any() else -np.inf for k in range(2)]))
            _,values=dp_path(x,centers,5.)
            cost=values[-1]; cutoff=train.index.max()
        _,values=dp_path(((row.to_numpy()-mean)/std)[None,:],centers,5.,cost)
        cost=values[-1]; cost=cost-cost.min()
        state=int(cost.argmin())
        records.append(dict(month=date,favorable=state==good,exposure=1. if state==good else .35,
                            fit_cutoff=cutoff,training_end_before_signal=cutoff<date))
    return pd.DataFrame(records)


def size_snapshot(candidates,pro):
    source=pd.read_csv(Path(config.LOCAL_DATA_RAW)/'stock_circ_mv_monthly.csv')
    source['date']=pd.to_datetime(source.trade_date.astype(str))
    directory=OUT/'size_snapshots'; directory.mkdir(parents=True,exist_ok=True)
    blocks=[]
    for month,g in candidates.groupby('month',sort=True):
        day=g.date.max(); prior=source.loc[source.date<=day,'date'].max()
        if pd.isna(prior) or (day-prior).days>45:
            path=directory/f'{day:%Y%m%d}.pkl'
            raw=pd.read_pickle(path) if path.exists() else fetch_variants(pro,'daily_basic',[
                dict(trade_date=f'{day:%Y%m%d}',offset=0,limit=6000),
                dict(trade_date=f'{day:%Y%m%d}',fields='ts_code,trade_date,circ_mv')],
                ['ts_code','trade_date','circ_mv'],f'{day:%Y%m%d}')
            raw=validate_rows(raw,['ts_code','trade_date'],['circ_mv'],f'{day:%Y%m%d}')
            raw.to_pickle(path); snap=raw[['ts_code','circ_mv']].copy(); prior=day
        else:
            snap=source[source.date.eq(prior)][['ts_code','circ_mv']].drop_duplicates()
        if snap.ts_code.duplicated().any(): raise ValueError('Conflicting market-cap snapshot')
        frame=g[['ts_code','month']].merge(snap,on='ts_code',how='left',validate='one_to_one')
        coverage=frame.circ_mv.notna().mean()
        if coverage<.95: raise ValueError(f'Insufficient size coverage {month}: {coverage}')
        frame['size_date']=prior
        blocks.append(frame)
    return pd.concat(blocks,ignore_index=True)


def repair_daily_coverage(raw,pro):
    counts=raw.groupby('date').code.nunique().sort_index()
    expected=counts.rolling(20,min_periods=10).median().shift(1)
    # This targets severe truncation, not ordinary participation/suspension changes.
    suspect=counts[counts<.60*expected]
    folder=OUT/'bulk_daily_repairs'; folder.mkdir(parents=True,exist_ok=True)
    audit=[]; replacements=[]
    for day,n in suspect.items():
        print('repair truncated market day',str(day.date()),int(n),flush=True)
        path=folder/f'{day:%Y%m%d}.pkl'
        frame=pd.read_pickle(path) if path.exists() else fetch_variants(pro,'daily',[
            dict(trade_date=f'{day:%Y%m%d}',offset=0,limit=6000),
            dict(trade_date=f'{day:%Y%m%d}')],['ts_code','trade_date','open','close','vol','amount'],f'{day:%Y%m%d}')
        frame=validate_rows(frame,['ts_code','trade_date'],['open','high','low','close'],f'{day:%Y%m%d}')
        frame=frame[frame.ts_code.str.startswith(('0','3','6'))]
        if len(frame)<.95*expected.loc[day]: raise ValueError('Remote bulk day also incomplete')
        frame.to_pickle(path)
        replacement=frame.rename(columns={'vol':'volume'}).assign(date=day,code=lambda x:x.ts_code.str[:6])
        replacements.append(replacement[raw.columns])
        audit.append(dict(date=day,local_count=int(n),expected_count=float(expected.loc[day]),remote_count=len(frame)))
    pd.DataFrame(audit).to_csv(OUT/'daily_coverage_repairs.csv',index=False)
    if replacements: raw=pd.concat([raw[~raw.date.isin(suspect.index)]]+replacements,ignore_index=True)
    return raw


def board_counts(raw):
    code=raw.code.astype(str).str.zfill(6)
    board=np.select([code.str.startswith(('688','689')),code.str.startswith('3'),
                     code.str.startswith('6'),code.str.startswith('0')],
                    ['star','growth','sh_main','sz_main'],default='other')
    return raw.assign(board=board).groupby(['date','board']).code.nunique().unstack(fill_value=0).sort_index()


def board_coverage_gaps(raw):
    counts=board_counts(raw)
    # A long outage must not become the new "normal" after a short rolling median.
    # Detect loss of an entire board even if total market coverage still exceeds 85%.
    expected=counts.rolling(252,min_periods=1).max().shift(1)
    gaps=(counts<.90*expected)&expected.ge(50)
    return counts,expected,gaps


def repair_board_coverage(raw,pro):
    from concurrent.futures import ThreadPoolExecutor
    counts,expected,gaps=board_coverage_gaps(raw)
    days=gaps.index[gaps.any(axis=1)]
    folder=OUT/'board_daily_repairs'; folder.mkdir(parents=True,exist_ok=True)
    def repair(day):
        path=folder/f'{day:%Y%m%d}.pkl'
        def validate(frame):
            frame=validate_rows(frame,['ts_code','trade_date'],['open','high','low','close'],f'{day:%Y%m%d}')
            frame=frame[frame.ts_code.str.startswith(('0','3','6'))].copy()
            replacement=frame.rename(columns={'vol':'volume'}).assign(date=day,code=lambda x:x.ts_code.str[:6])
            after=board_counts(replacement).reindex(columns=counts.columns,fill_value=0).iloc[0]
            if ((after<.95*expected.loc[day])&expected.loc[day].ge(50)).any():
                raise ValueError(f'Remote board coverage insufficient: {day}')
            if not np.isfinite(pd.to_numeric(replacement.amount,errors='coerce')).all() or replacement.amount.lt(0).any():
                raise ValueError('Invalid repaired amount')
            return frame,replacement
        if path.exists():
            frame,replacement=validate(pd.read_pickle(path))
        elif (ROOT/'boundaries'/f'daily_{day:%Y%m%d}.pkl').exists():
            frame,replacement=validate(pd.read_pickle(ROOT/'boundaries'/f'daily_{day:%Y%m%d}.pkl'))
        else:
            # Each concurrent HTTP worker owns a separate requests session.
            client=copy.copy(pro)
            if hasattr(client,'_session'):
                import requests
                client._session=requests.Session()
            try:
                errors=[]
                variants=[dict(trade_date=f'{day:%Y%m%d}',offset=0,limit=n) for n in (6000,8000,5000,7000,10000)]
                variants += [dict(trade_date=f'{day:%Y%m%d}'),dict(trade_date=f'{day:%Y%m%d}',offset=0),
                             dict(start_date=f'{day:%Y%m%d}',end_date=f'{day:%Y%m%d}',offset=0,limit=6000)]
                for params in variants:
                    try:
                        frame,replacement=validate(client.query('daily',**params))
                        break
                    except Exception as exc:
                        errors.append(str(exc))
                else:
                    raise RuntimeError(f'No complete board quote {day}: {errors}')
            finally:
                if hasattr(client,'_session'): client._session.close()
        frame.to_pickle(path)
        print('repair board coverage',str(day.date()),int(counts.loc[day].sum()),'->',len(frame),flush=True)
        audit=dict(date=day,missing_boards=','.join(gaps.columns[gaps.loc[day]]),
                   local_count=int(counts.loc[day].sum()),remote_count=len(frame),
                   added_amount=float(replacement.amount.sum()-raw.loc[raw.date.eq(day),'amount'].sum()))
        return replacement[raw.columns],audit
    def attempt(day):
        try: return repair(day)
        except Exception as exc:
            print('board repair pending',str(day.date()),str(exc),flush=True)
            return None,dict(date=day,error=str(exc))
    with ThreadPoolExecutor(max_workers=3) as pool:
        attempts=list(pool.map(attempt,days))
    failures=[a for r,a in attempts if r is None]
    pd.DataFrame(failures).to_csv(OUT/'board_coverage_pending.csv',index=False)
    repaired=[(r,a) for r,a in attempts if r is not None]
    pd.DataFrame([a for _,a in repaired]).to_csv(OUT/'board_coverage_repairs.csv',index=False)
    if failures: raise ValueError(f'Board coverage repair incomplete: {len(failures)} days; see board_coverage_pending.csv')
    if repaired:
        raw=pd.concat([raw[~raw.date.isin(days)]]+[r for r,_ in repaired],ignore_index=True)
    _,_,remaining=board_coverage_gaps(raw)
    if remaining.any().any(): raise ValueError('Unresolved board coverage gaps')
    return raw


def prepare(offline=False):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=PROTOCOL:
        raise ValueError('Do not silently change an existing experiment protocol')
    path.write_text(json.dumps(PROTOCOL,ensure_ascii=False,indent=2),encoding='utf-8')
    industry=pd.read_csv(Path(config.LOCAL_DATA_RAW)/'ts_sw_industry_monthly.csv',parse_dates=['date'])
    industry=industry[~industry.ts_code.isin(config.SW_EXCLUDE)]
    price=industry.pivot(index='date',columns='ts_code',values='close').sort_index()
    price=price.reindex(pd.date_range(price.index.min(),price.index.max(),freq='ME'))
    mom={i:price/price.shift(i)-1 for i in (1,3,6)}
    vol=mom[1].rolling(6,min_periods=6).std()
    raw=pd.read_pickle(config.STOCK_DAILY_PATH)['df_stock']
    # 2016 supplies the 6-month turnover warmup plus all 72 pre-2023 fit months.
    raw=raw[raw.code.astype(str).str.startswith(('0','3','6')) & raw.date.ge('2016-01-01')].copy()
    raw=repair_daily_coverage(raw,engine.OfflineClient() if offline else get_pro())
    raw=repair_board_coverage(raw,engine.OfflineClient() if offline else get_pro())
    totals=raw.groupby('date').amount.sum()
    amount=totals.resample('ME').mean()
    features=pd.DataFrame({f'mom{i}':v.median(axis=1) for i,v in mom.items()})
    features['vol']=vol.median(axis=1)
    features['breadth']=mom[3].gt(0).sum(axis=1)/mom[3].notna().sum(axis=1)
    features['amount_ratio']=amount/amount.rolling(6,min_periods=6).mean()
    features=features.replace([np.inf,-np.inf],np.nan).dropna()
    states=online_states(features)
    states.to_csv(OUT/'market_states.csv',index=False)
    features.to_csv(OUT/'market_features.csv',index_label='month')
    # Raw daily prices only for close-trigger / next-open execution, never momentum across ex dates.
    raw=raw[raw.date.between('2022-12-01','2026-09-24')].copy()
    raw['ts_code']=raw.code.astype(str).str.zfill(6)+np.where(raw.code.astype(str).str.startswith('6'),'.SH','.SZ')
    if raw.duplicated(['date','ts_code']).any(): raise ValueError('Duplicate daily price keys')
    invalid=(raw[['open','high','low','close']]<=0).any(axis=1)
    if (invalid & raw.volume.gt(0)).any(): raise ValueError('Invalid local traded OHLC')
    raw[invalid].to_csv(OUT/'zero_volume_placeholder_audit.csv',index=False)
    # Not executable observations; held missing quotes still require suspension evidence.
    raw=raw[~invalid].copy()
    raw.to_pickle(OUT/'daily.pkl')
    c=pd.read_pickle(leadership.OUT/'candidates.pkl')
    cap=size_snapshot(c,engine.OfflineClient() if offline else get_pro())
    c=c.merge(cap,on=['month','ts_code'],validate='one_to_one')
    c['size_rank']=c.groupby(['month','l2_code']).circ_mv.rank(pct=True)
    c['size_bucket']=np.select([c.size_rank.ge(.7),c.size_rank.le(.3)],['core','small'],default='middle')
    c.loc[c.circ_mv.isna() | c.l2_code.isna(),'size_bucket']='unknown'
    pred=pd.read_pickle(Path(config.OUTPUT_DIR)/'price_risk_pipeline/predictions_ensemble.pkl')
    sector=[]
    for i,p in price.iterrows():
        g=c[c.month.eq(i)]
        if g.empty: continue
        one=pd.DataFrame(dict(ind_code=price.columns,month=i,mom1=mom[1].loc[i],mom3=mom[3].loc[i],mom6=mom[6].loc[i])).reset_index(drop=True)
        one['breadth']=one.ind_code.map(g.groupby('ind_code').mom3.apply(lambda x:(x>0).mean()))
        one['amount_ratio']=float(features.loc[i,'amount_ratio'])
        one['phase']='range'
        one.loc[(one.mom3>.15)&(one.mom1>0)&(one.breadth>.55)&(one.amount_ratio>1.05),'phase']='advance'
        one.loc[(one.mom1<-.08)&(one.mom3<0),'phase']='retreat'
        one.loc[(one.mom1>.30)&(one.breadth<.5),'phase']='overheat'
        rank=one[['mom1','mom3','mom6']].rank(pct=True)
        one['trend_score']=.2*rank.mom1+.5*rank.mom3+.3*rank.mom6
        sector.append(one)
    sector=pd.concat(sector,ignore_index=True).merge(pred[['date','ts_code','pred_ensemble']].rename(columns={'date':'month','ts_code':'ind_code'}),on=['month','ind_code'],validate='one_to_one')
    sector['selection_score']=.5*sector.trend_score+.5*sector.pred_ensemble
    sector['sector_rank']=sector.groupby('month').selection_score.rank(pct=True)
    c=c.merge(sector[['month','ind_code','phase','sector_rank','selection_score']],on=['month','ind_code'],validate='many_to_one')
    lowvol=c.groupby('month').lowvol6.rank(pct=True)
    score=np.where(c.phase.eq('advance'),.55*c.fast_score+.25*c.peer_score+.20*c.score,.70*c.score+.30*lowvol)
    c['leadership_score']=.75*score+.25*c.sector_rank
    styles=c[c.size_bucket.isin(['core','small'])].groupby(['month','size_bucket'])[['mom1','mom3']].median().unstack()
    choice=np.where((styles[('mom1','small')]-styles[('mom1','core')]>.03)&styles[('mom3','small')].gt(0),'small','core')
    styles['chosen']=choice
    styles.to_csv(OUT/'style_signals.csv')
    c['chosen_style']=c.month.map(pd.Series(choice,index=styles.index))
    c.to_pickle(OUT/'candidates.pkl')
    sector=sector.merge(states[['month','exposure']],on='month',validate='many_to_one')
    sector.rename(columns={'month':'date','ind_code':'ts_code','exposure':'risk_exposure'}).to_pickle(OUT/'plans.pkl')
    sector.to_csv(OUT/'sector_states.csv',index=False)
    print('prepared',len(c),'candidates; market favorable counts',states.favorable.value_counts().to_dict(),flush=True)


def exit_reason(close,entry,peak,phase):
    if min(close,entry,peak)<=0: raise ValueError('Exit prices must be positive')
    if close<=.92*entry+1e-8: return 'hard_stop'
    if peak>=1.20*entry-1e-8:
        trail=.15 if phase=='advance' else .10
        if close<=(1-trail)*peak+1e-8: return 'trailing_profit'
    if phase!='advance' and close>=1.30*entry-1e-8: return 'range_take_profit'
    return None


def soft_candidates(candidates, name):
    """Cross-sectional rules only; do not mutate cached inputs or peek ahead."""
    if name not in SOFT_VARIANTS:
        raise ValueError('Unknown soft-selection variant')
    c=candidates.copy()
    c['original_phase']=c.phase
    peers=c.groupby(['month','l2_code']).mom3
    c['l2_mom3']=peers.transform('median')
    c['l2_count']=peers.transform('count')
    c['l2_breadth']=peers.transform(lambda x:x.gt(0).mean())
    c['mom3_percentile']=c.groupby('month').mom3.rank(pct=True)
    strong=(c.mom1.gt(0)&c.mom3.gt(.15)&c.mom6.gt(0)&c.mom3_percentile.ge(.85)
            &c.l2_count.ge(5)&c.l2_mom3.gt(0)&c.l2_breadth.gt(.55))
    c['strong_override']=strong & c.phase.ne('overheat') if name!='soft_size' else False
    c.loc[c.strong_override,'phase']='advance'
    c.loc[c.strong_override,'leadership_score']=(.75*(.55*c.fast_score+.25*c.peer_score+.20*c.score)+.25*c.sector_rank)
    size_match=c.size_bucket.eq(c.chosen_style)
    c['eligible']=~c.phase.isin(['retreat','overheat'])
    if name=='leader_override':
        c['eligible'] &= size_match
    else:
        c['leadership_score']=.95*c.leadership_score+.05*size_match.astype(float)
    return c


def save_soft_protocol():
    path=OUT/'soft_selection_protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=SOFT_PROTOCOL:
        raise ValueError('Do not silently change the soft-selection experiment')
    path.write_text(json.dumps(SOFT_PROTOCOL,indent=2),encoding='utf-8')


def narrow_candidates(candidates,market,name):
    if name not in NARROW_VARIANTS: raise ValueError('Unknown narrow-market variant')
    c=soft_candidates(candidates,'soft_leader')
    narrow=c.month.map(market.breadth).lt(.5)
    if c.month.map(market.breadth).isna().any(): raise ValueError('Missing market breadth')
    excess=c.mom3-c.l2_mom3
    excess_rank=excess.groupby(c.month).rank(pct=True)
    local=(c.l2_mom3.gt(0)&c.l2_breadth.gt(.55))|excess.ge(.20)
    strong=(narrow&c.mom1.gt(0)&c.mom3.gt(.15)&c.mom6.gt(0)&c.mom3_percentile.ge(.85)&c.l2_count.ge(5)&local)
    hot=narrow&c.mom3.gt(1.5)&c.mom1.gt(.3) if name=='narrow_heat' else pd.Series(False,index=c.index)
    strong &= ~hot & c.original_phase.ne('overheat')
    c['phase']=c.original_phase
    c['leadership_score']=candidates.leadership_score
    c['eligible']=c.size_bucket.eq(c.chosen_style)&~c.phase.isin(['retreat','overheat'])
    c['narrow_market']=narrow
    c['strong_override']=strong
    c['heat_excluded']=hot
    c['relative_mom3_rank']=excess_rank
    c.loc[strong,'phase']='advance'
    c.loc[strong,'leadership_score']=.6*c.fast_score+.4*excess_rank
    c.loc[strong,'eligible']=True
    c.loc[hot,'eligible']=False
    return c


def save_narrow_protocol():
    path=OUT/'narrow_selection_protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=NARROW_PROTOCOL:
        raise ValueError('Do not silently change the narrow-market experiment')
    path.write_text(json.dumps(NARROW_PROTOCOL,indent=2),encoding='utf-8')


def queue_deferred_rebalance(pending,code,day,remaining):
    """Do not cancel an already-triggered risk exit or forget locked residue."""
    if remaining>0:
        pending.setdefault(code,('rebalance_deferred',day))


def normal_limit_evidence(day,code,quote,history,recent_quotes,boundary_day,boundary_quote,boundary_limit):
    """Rule-derived ordinary-mainboard band, only with corroborated continuity.

    Separate from vendor limit observations. Refuse ST, IPO/relisting/resumption,
    missing evidence, changed status, or inconsistent price/reference data.
    """
    if not pd.Timestamp('2023-04-10')<=day<pd.Timestamp('2026-07-06'):
        raise ValueError('Outside verified rule version')
    if not leadership.is_main_board(code[:6]): raise ValueError('Not ordinary mainboard')
    if not {'ts_code','name','start_date','end_date'}<=set(history) or not history.ts_code.eq(code).all():
        raise ValueError('Missing historical name evidence')
    start=pd.to_datetime(history.start_date,errors='coerce'); end=pd.to_datetime(history.end_date,errors='coerce')
    active=history[start.le(boundary_day)&(end.isna()|end.ge(day))].drop_duplicates()
    if len(active)!=1 or active.name.astype(str).str.contains('ST|退|^N|^C',case=False,regex=True).any():
        raise ValueError('Not a single unchanged ordinary status interval')
    if (day-pd.to_datetime(active.start_date.iloc[0])).days<180:
        raise ValueError('Recent listing/status change needs exact limit data')
    if not boundary_day<day or (day-boundary_day).days>31:
        raise ValueError('No recent independently verified limit anchor')
    if len(recent_quotes)!=5 or any(not d<day or volume<=0 or close<=0 for d,close,volume in recent_quotes):
        raise ValueError('Missing five immediately preceding traded sessions')
    if len({d for d,_,_ in recent_quotes})!=5: raise ValueError('Duplicate continuity dates')
    def rounded(price,multiplier):
        return float((Decimal(str(price))*Decimal(multiplier)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
    ref=float(quote.pre_close); old_ref=float(boundary_quote.pre_close)
    if min(ref,old_ref)<=0: raise ValueError('Invalid reference price')
    if abs(float(boundary_limit.down_limit)-rounded(old_ref,'.90'))>.005 or abs(float(boundary_limit.up_limit)-rounded(old_ref,'1.10'))>.005:
        raise ValueError('Verified anchor is not an ordinary 10% band')
    down,up=rounded(ref,'.90'),rounded(ref,'1.10')
    if float(quote.open)<down-.005 or float(quote.open)>up+.005:
        raise ValueError('Observed opening incompatible with derived ordinary band')
    return dict(method='rule_derived_10pct_band_with_historical_name_continuity_and_verified_anchor',
        not_a_vendor_limit_quote=True,down_limit=down,up_limit=up,pre_close=ref,
        name=str(active.name.iloc[0]),anchor_date=str(boundary_day.date()),
        preceding_sessions=[str(d.date()) for d,_,_ in recent_quotes],
        rule_source=('https://docs.static.szse.cn/www/lawrules/index/rule/W020230217564423808793.pdf'
                     if code.endswith('.SZ') else 'https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/c_20250612_10824490.shtml'))


def execution_quote(pro,day,code,local_open,reference=None,continuity=None):
    """Validate an additional (non-month-boundary) exit against remote daily/limits."""
    folder=OUT/'exit_quotes'; folder.mkdir(parents=True,exist_ok=True)
    if reference is not None and reference>0:
        bound=float((Decimal(str(reference))*Decimal('.95')).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
        if local_open>bound+.005:
            (folder/f'check_{code}_{day:%Y%m%d}.json').write_text(json.dumps(dict(
                method='local_open_strictly_above_conservative_5pct_lower_bound',open=local_open,
                reference_close=reference,reference_adjusted_for_corporate_actions=True,lower_bound=bound,
                not_an_official_limit_quote=True,price_source='daily.pkl; no intraday high/low used for fill',
                note='Uses a sufficient sellability condition, not a missing-limit imputation'),indent=2),encoding='utf-8')
            return local_open,bound
    frames={}
    for api in ('daily','stk_limit'):
        if api=='stk_limit' and 'pre_close' in frames['daily']:
            # Ordinary main-board names have a 10%-or-wider band. Establish
            # historical status, never infer it from today's stock name.
            history_path=folder/f'namechange_{code}.pkl'
            try:
                history=pd.read_pickle(history_path) if history_path.exists() else fetch_variants(pro,'namechange',[
                    dict(ts_code=code),dict(ts_code=code,offset=0,limit=5000),
                    dict(ts_code=code,offset=0),dict(ts_code=code,offset=0,limit=10000)],['ts_code','name','start_date','end_date'])
                if not {'ts_code','name','start_date','end_date'}<=set(history): raise ValueError('Missing historical name schema')
                if not history.ts_code.eq(code).all(): raise ValueError('Wrong name-history stock')
                start=pd.to_datetime(history.start_date,errors='coerce'); end=pd.to_datetime(history.end_date,errors='coerce')
                active=history[start.le(day)&(end.isna()|end.ge(day))].drop_duplicates()
                if len(active)!=1: raise ValueError('Ambiguous historical stock status')
                history.to_pickle(history_path)
                if not active.name.astype(str).str.contains('ST|退',case=False,regex=True).any():
                    reference=float(frames['daily'].pre_close)
                    bound=float((Decimal(str(reference))*Decimal('.90')).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
                    opening=float(frames['daily'].open)
                    if reference>0 and opening>bound+.005:
                        if abs(opening-local_open)>.005: raise ValueError('Local/remote exit open mismatch')
                        (folder/f'check_{code}_{day:%Y%m%d}.json').write_text(json.dumps(dict(
                            method='historical_non_ST_name_and_10pct_sufficient_bound',open=opening,pre_close=reference,
                            lower_bound=bound,name=str(active.name.iloc[0]),not_an_official_limit_quote=True),ensure_ascii=False,indent=2),encoding='utf-8')
                        return opening,bound
            except (RuntimeError,ValueError):
                pass  # Fall back to an exact official limit; never assume missing history means normal.
        path=folder/f'{api}_{code}_{day:%Y%m%d}.pkl'
        derived_path=folder/f'derived_limit_{code}_{day:%Y%m%d}.json'
        if api=='stk_limit' and not path.exists() and derived_path.exists() and continuity is not None:
            history=pd.read_pickle(folder/f'namechange_{code}.pkl')
            bday,bquote,blimit,recent=continuity
            proof=normal_limit_evidence(day,code,frames['daily'],history,recent,bday,bquote,blimit)
            if abs(float(frames['daily'].open)-local_open)>.005: raise ValueError('Local/remote exit open mismatch')
            return local_open,proof['down_limit']
        if path.exists(): frame=pd.read_pickle(path)
        else:
            print('validate exit',api,code,f'{day:%Y%m%d}',flush=True)
            errors=[]; frame=None
            variants=[dict(trade_date=f'{day:%Y%m%d}',offset=0,limit=5000),
                dict(trade_date=f'{day:%Y%m%d}',offset=0,limit=6000),
                dict(trade_date=f'{day:%Y%m%d}',offset=0,limit=8000),
                dict(trade_date=f'{day:%Y%m%d}',offset=0,limit=7000),
                dict(trade_date=f'{day:%Y%m%d}',offset=0,limit=10000),
                dict(trade_date=f'{day:%Y%m%d}'),
                dict(ts_code=code,trade_date=f'{day:%Y%m%d}'),
                dict(ts_code=code,start_date=f'{day:%Y%m%d}',end_date=f'{day:%Y%m%d}',offset=0),
                dict(ts_code=code,start_date=f'{day:%Y}0101',end_date=f'{day:%Y}1231',offset=0,limit=6000)]
            for params in variants:
                try:
                    response=pro.query(api,**params)
                    if not {'ts_code','trade_date'}<=set(response): raise ValueError('Missing exit schema')
                    selected=response[response.ts_code.eq(code)&response.trade_date.astype(str).eq(f'{day:%Y%m%d}')]
                    if len(selected)!=1: raise ValueError('Requested exit observation absent/conflicting')
                    frame=selected.copy(); break
                except Exception as exc: errors.append(str(exc))
            if frame is None:
                if api=='stk_limit' and continuity is not None and (folder/f'namechange_{code}.pkl').exists():
                    history=pd.read_pickle(folder/f'namechange_{code}.pkl')
                    bday,bquote,blimit,recent=continuity
                    proof=normal_limit_evidence(day,code,frames['daily'],history,recent,bday,bquote,blimit)
                    opening=float(frames['daily'].open)
                    if abs(opening-local_open)>.005: raise ValueError('Local/remote exit open mismatch')
                    (folder/f'derived_limit_{code}_{day:%Y%m%d}.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
                    return opening,proof['down_limit']
                raise RuntimeError(f'{api} {code} {day.date()}: {errors}')
        required=['open','close'] if api=='daily' else ['up_limit','down_limit']
        frame=frame[frame.ts_code.eq(code)].copy()
        frame=validate_rows(frame,['ts_code','trade_date'],required,f'{day:%Y%m%d}')
        if len(frame)!=1 or frame.ts_code.iloc[0]!=code: raise ValueError('Wrong stock in exit quote')
        frame.to_pickle(path); frames[api]=frame.iloc[0]
        if api=='daily' and 'pre_close' in frame and float(frame.pre_close.iloc[0])>0:
            reference=float(frame.pre_close.iloc[0])
            bound=float((Decimal(str(reference))*Decimal('.95')).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
            opening=float(frame.open.iloc[0])
            # A strict sufficient condition, NOT an invented official lower limit:
            # main-board 5%-or-wider bands imply actual lower limit <= this bound.
            if opening>bound+.005:
                if abs(opening-local_open)>.005: raise ValueError('Local/remote exit open mismatch')
                (folder/f'check_{code}_{day:%Y%m%d}.json').write_text(json.dumps(dict(
                    method='strictly_above_conservative_5pct_lower_bound',open=opening,pre_close=reference,
                    lower_bound=bound,not_an_official_limit_quote=True,
                    rule_source='https://docs.static.szse.cn/www/lawrules/index/rule/W020230217564423808793.pdf'),indent=2),encoding='utf-8')
                return opening,bound
    if abs(frames['daily'].open-local_open)>.005: raise ValueError('Local/remote exit open mismatch')
    return float(frames['daily'].open),float(frames['stk_limit'].down_limit)


def checked_marks(ledger,quotes,day,field,actions,pro,audit):
    """Repair a missing local traded quote from the validated remote response."""
    manual_marks={}
    interval_path=OUT/'verified_suspensions.json'
    intervals=json.loads(interval_path.read_text(encoding='utf-8')) if interval_path.exists() else []
    for interval in intervals:
        code=interval['code']
        if ledger.shares.get(code,0) and code not in quotes.index and pd.Timestamp(interval['start'])<=day<=pd.Timestamp(interval['end']):
            # Last quoted close is an observed price, not a hand-entered fill.
            last=engine.boundary_frame(pd.Timestamp(interval['last_quote']))
            price=float(last.loc[code,'close'])
            if not actions.empty:
                relevant=actions[actions.ts_code.eq(code)&actions.ex_date.gt(pd.Timestamp(interval['last_quote']))&actions.ex_date.le(day)]
                for _,a in relevant.groupby('ex_date'): price=(price-a.cash.sum())/(1+a.stock.sum())
            manual_marks[code]=price
            audit.append(dict(date=day,code=code,field=field,mark=price,reason='company_announcement_confirmed_suspension',source=interval['source']))
    if manual_marks:
        ledger=copy.copy(ledger)
        ledger.shares={code:n for code,n in ledger.shares.items() if code not in manual_marks}
    suspension_root=OUT/'suspension_events'; suspension_root.mkdir(parents=True,exist_ok=True)
    for code,n in ledger.shares.items():
        cached=suspension_root/f'{code}.pkl'
        target=ROOT/'suspension_checks'/f'{code}_{day:%Y%m%d}_suspend.pkl'
        if n and code not in quotes.index and cached.exists() and not target.exists():
            history=pd.read_pickle(cached)
            history[pd.to_datetime(history.trade_date)<=day].to_pickle(target)
    for _ in range(6):
        try:
            marks=engine.held_marks(ledger,quotes,day,field,actions,pro,audit)
            marks.update(manual_marks)
            return marks
        except RuntimeError as exc:
            if 'suspend_d' not in str(exc): raise
            repaired=False
            for code,n in ledger.shares.items():
                if not n or code in quotes.index: continue
                print('validate suspension history',code,str(day.date()),flush=True)
                history=fetch_variants(pro,'suspend_d',[
                    dict(ts_code=code,offset=0,limit=6000),dict(ts_code=code),
                    dict(ts_code=code,start_date='20230101',end_date='20260924',offset=0)],
                    ['ts_code','trade_date','suspend_type'])
                if not history.ts_code.eq(code).all(): raise ValueError('Wrong suspension stock')
                history=history.drop_duplicates()
                history.to_pickle(suspension_root/f'{code}.pkl')
                history[pd.to_datetime(history.trade_date)<=day].to_pickle(ROOT/'suspension_checks'/f'{code}_{day:%Y%m%d}_suspend.pkl')
                repaired=True
            if not repaired: raise
        except ValueError as exc:
            prefix='Bulk quote missing a trading stock; repair data first: '
            if not str(exc).startswith(prefix): raise
            code=str(exc).removeprefix(prefix)
            path=ROOT/'suspension_checks'/f'{code}_{day:%Y%m%d}_daily.pkl'
            raw=pd.read_pickle(path)
            row=raw[(raw.ts_code==code)&pd.to_datetime(raw.trade_date).eq(day)]
            row=validate_rows(row,['ts_code','trade_date'],['open','high','low','close'])
            if len(row)!=1: raise ValueError('Cannot uniquely repair missing daily quote')
            for column in ('open','high','low','close'):
                quotes.loc[code,column]=float(row[column].iloc[0])
            audit.append(dict(date=day,code=code,field=field,reason='local_quote_gap_repaired_from_remote',source=str(path)))
    raise ValueError('Exceeded held quote repair bound')


def continue_trend(row):
    """Month-end evidence only; a pending exit always takes priority at caller."""
    return row is not None and all(pd.notna(row.get(k)) and row[k]>0 for k in ('mom1','mom3','mom6'))


def may_continue_position(row,entry_phase,has_pending_exit,advance_only=False):
    return not has_pending_exit and (not advance_only or entry_phase=='advance') and continue_trend(row)


def replacement_limits(pro,day,candidates,quotes):
    folder=OUT/'replacement_limits'; folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{day:%Y%m%d}.pkl'
    needed=set(candidates.loc[candidates.stock_code.map(leadership.is_main_board),'ts_code'])&set(quotes.index)
    def validate(frame):
        frame=validate_rows(frame,['ts_code','trade_date'],[],f'{day:%Y%m%d}')
        if not needed.issubset(set(frame.ts_code)): raise ValueError('Replacement limits omit candidate stocks')
        selected=frame[frame.ts_code.isin(needed)]
        validate_rows(selected,['ts_code','trade_date'],['up_limit','down_limit'],f'{day:%Y%m%d}')
        return frame.set_index('ts_code')
    if path.exists(): return validate(pd.read_pickle(path))
    errors=[]
    variants=[dict(trade_date=f'{day:%Y%m%d}',exchange=exchange,limit=n) for n in (6000,5000) for exchange in ('SSE','SZSE')]
    variants += [dict(trade_date=f'{day:%Y%m%d}',offset=0,limit=n) for n in (6000,8000,7000,10000,5000)]
    variants += [dict(trade_date=f'{day:%Y%m%d}'),dict(trade_date=f'{day:%Y%m%d}',limit=6000),
                 dict(trade_date=f'{day:%Y%m%d}',fields='ts_code,trade_date,up_limit,down_limit'),
                 dict(start_date=f'{day:%Y%m%d}',end_date=f'{day:%Y%m%d}',offset=0,limit=6000)]
    for params in variants:
        try:
            frame=pro.query('stk_limit',**params)
            result=validate(frame); frame.to_pickle(path)
            return result
        except Exception as exc: errors.append(str(exc))
    raise RuntimeError(f'No valid deferred replacement limits {day}: {errors}')


def run(name,offline=False):
    selection_name=name.removesuffix('_retry') if name in RETRY_VARIANTS else name
    hold_winners=name in HOLD_VARIANTS+ADVANCE_HOLD_VARIANTS
    retry_rebalance=name in RETRY_VARIANTS+FEATURE_VARIANTS+LEARNED_VARIANTS+HOLD_VARIANTS+ADVANCE_HOLD_VARIANTS+ROUTER_VARIANTS+FINE_VARIANTS+GRAPH_VARIANTS+DAILY_VARIANTS+QUALITY_VARIANTS
    daily_enabled=name in DAILY_VARIANTS or name in ('daily_quality_retry','daily_report_quality_retry','daily_guarded_retry')
    replace_deferred=name=='structural_reentry_retry'
    if name in ROUTER_VARIANTS+FINE_VARIANTS+GRAPH_VARIANTS+DAILY_VARIANTS+QUALITY_VARIANTS: selection_name='state_onset'
    if hold_winners:
        selection_name='state_onset' if name=='state_hold_retry' else 'narrow_heat'
        protocol=dict(variants=list(HOLD_VARIANTS),
            stage='Exploration on already examined history; same rule for every month, no theme/stock whitelist',
            hypothesis='Monthly forced turnover can cut off the medium-term trend captured by stock selection',
            rule='At rebalance retain a holding only if its last completed signal month has positive adjusted mom1, mom3 and mom6; a pre-existing pending exit is never cancelled',
            exits='Existing close-trigger hard/trailing/take-profit rules remain unchanged; no new signal may cancel them',
            budget='Appreciated retained positions need not be cut to 25000; new buys use cash and nonnegative residual of the existing risk budget',
            baseline='state_hold uses state_onset entries; narrow_hold uses narrow_heat entries; only month-start liquidation differs')
        p=OUT/'trend_holding_protocol.json'
        if name in ADVANCE_HOLD_VARIANTS:
            protocol['variants']=list(ADVANCE_HOLD_VARIANTS)
            protocol['additional_gate']='Only holdings already classified advance at entry may continue; range holdings still rebalance monthly'
            protocol['stage']='Follow-up after broad positive-trend continuation trials began; prevents weak range holdings from bypassing monthly reselection; not untouched validation'
            p=OUT/'advance_holding_protocol.json'
        if p.exists() and json.loads(p.read_text(encoding='utf-8'))!=protocol: raise ValueError('Holding protocol changed')
        p.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    if name in RETRY_VARIANTS:
        protocol=dict(variants=list(RETRY_VARIANTS),
            purpose='Correct deferred monthly liquidation; isolate execution from selection changes',
            rule='If planned month-start sale cannot execute, keep a sell order for subsequent executable openings; retain full gap/limit risk; no mid-month replacement buys',
            unchanged='All selection signals, budget, exit thresholds and current source prices',
            baseline='state_onset and narrow_heat retain historical once-monthly blocked-sale convention for attribution')
        p=OUT/'retry_rebalance_protocol.json'
        if p.exists() and json.loads(p.read_text(encoding='utf-8'))!=protocol:
            raise ValueError('Execution protocol changed')
        p.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    pro=engine.OfflineClient() if offline else get_pro()
    raw=pd.read_pickle(OUT/'daily.pkl')
    dayframes={pd.Timestamp(d):g.set_index('ts_code') for d,g in raw.groupby('date')}
    sessions=json.loads((ROOT/'boundaries/sessions.json').read_text(encoding='utf-8'))
    firsts={pd.Timestamp(s['first']) for s in sessions}; lasts={pd.Timestamp(s['last']) for s in sessions}
    session_dates=sorted(dayframes)
    session_positions={d:i for i,d in enumerate(session_dates)}
    anchor_quotes={}; anchor_limits={}
    if name.startswith('baseline_'):
        c=pd.read_pickle(leadership.BASE/'candidates.pkl').assign(phase='range')
        plans=pd.read_pickle(leadership.BASE/'industry_plans.pkl').query("strategy=='new_full'")
        optimizer=original_optimizer
    else:
        c=pd.read_pickle(OUT/'candidates.pkl')
        if name in QUALITY_VARIANTS:
            from research_quality_floor import prepare_candidates
            c=prepare_candidates(OUT,Path(config.LOCAL_DATA_RAW),name)
        elif name in GRAPH_VARIANTS:
            from research_peer_graph import prepare_candidates
            c=prepare_candidates(OUT,ROOT)
        elif name in FINE_VARIANTS:
            from research_fine_industry import prepare_candidates
            c=prepare_candidates(OUT,name)
        elif name in ROUTER_VARIANTS:
            from research_structural_router import prepare_candidates
            c=prepare_candidates(OUT)
        elif name in LEARNED_VARIANTS:
            from research_learned_leaders import prepare_candidates
            c=prepare_candidates(OUT,ROOT)
        elif name in FEATURE_VARIANTS:
            from research_trend_features import prepare_candidates
            c=prepare_candidates(OUT,ROOT,name)
        elif selection_name in SOFT_VARIANTS+NARROW_VARIANTS:
            if selection_name in SOFT_VARIANTS:
                save_soft_protocol()
                c=soft_candidates(c,selection_name)
            else:
                save_narrow_protocol()
                market=pd.read_csv(OUT/'market_features.csv',parse_dates=['month']).set_index('month')
                c=narrow_candidates(c,market,selection_name)
            c.to_pickle(OUT/f'candidate_audit_{name}.pkl')
            c=c[c.eligible]
        else:
            bucket=name.split('_')[0] if name.startswith(('core_','small_')) else None
            c=c[c.size_bucket.eq(bucket) if bucket else c.size_bucket.eq(c.chosen_style)]
            c=c[~c.phase.isin(['retreat','overheat'])]
        plans=pd.read_pickle(OUT/'plans.pkl')
        if selection_name=='state_onset' or selection_name in SOFT_VARIANTS+NARROW_VARIANTS+FEATURE_VARIANTS+LEARNED_VARIANTS:
            market=pd.read_csv(OUT/'market_features.csv',parse_dates=['month']).set_index('month')
            onset=(market.mom1>0)&(market.breadth>.55)&(market.amount_ratio>1.05)
            plans=plans.copy()
            plans.loc[plans.date.map(onset).fillna(False),'risk_exposure']=1.
        optimizer=lambda candidates,budget,**kw: leadership.leader_optimizer(candidates,budget,
            stock_cap=1.,sector_cap=1.,minimum_names=1,sector_name_limit=None,**kw)
    candidate_lookup={d.to_period('M'):g for d,g in c.groupby('month')}
    plan_lookup={d.to_period('M'):g for d,g in plans.groupby('date')}
    continuation_lookup={}
    daily_candidate_lookup={}; last_sale_index=None
    if daily_enabled:
        from research_daily_reentry import prepare_candidates, can_reenter
        daily_candidate_lookup=prepare_candidates(OUT)
        if name in ('daily_quality_retry','daily_report_quality_retry','daily_guarded_retry'):
            from research_quality_floor import filter_daily
            daily_candidate_lookup=filter_daily(OUT,daily_candidate_lookup,name)
    if hold_winners:
        full=pd.read_pickle(OUT/'candidates.pkl')
        continuation_lookup={d.to_period('M'):g.set_index('ts_code') for d,g in full.groupby('month')}
    ledger=engine.Ledger(); actions=pd.DataFrame(); known=set(); anchors={}; pending={}
    trades=[]; allocations=[]; holdings=[]; navs=[]; daily_nav=[]; signals=[]; stale=[]; continuations=[]
    previous_mark=pd.Timestamp('2022-12-30'); previous_factors=engine.factor_snapshot(pro,previous_mark); month_codes=set()
    previous_closes={}
    for day in pd.date_range('2023-01-01','2026-09-24'):
        ledger.morning(day,actions)
        for code,a in anchors.items():
            if not actions.empty:
                changes=actions[actions.ts_code.eq(code)&actions.ex_date.eq(day)]
                if len(changes):
                    a['entry']=(a['entry']-changes.cash.sum())/(1+changes.stock.sum())
                    a['peak']=(a['peak']-changes.cash.sum())/(1+changes.stock.sum())
                    if code in previous_closes:
                        previous_closes[code]=(previous_closes[code]-changes.cash.sum())/(1+changes.stock.sum())
        if day not in dayframes:
            ledger.close_record(day,actions); continue
        q=dayframes[day]; q=q[q.volume>0].copy()
        if day in firsts or day in lasts:
            verified=engine.boundary_frame(day)
            overlap=q.index.intersection(verified.index)
            held=[x for x,n in ledger.shares.items() if n and x in overlap]
            if held and not np.allclose(q.loc[held,'close'],verified.loc[held,'close'],atol=.005,rtol=0):
                raise ValueError('Daily/monthly held quote mismatch')
            q=verified
        prices=checked_marks(ledger,q,day,'open',actions,pro,stale)
        limits=engine.boundary_frame(day,'stk_limit') if day in firsts else None
        deferred_sale=False
        # Pending exits are never filled at yesterday's trigger price.
        for code,(reason,signal_day) in list(pending.items()):
            qty=ledger.shares.get(code,0)-ledger.blocked_shares(code)
            if qty<=0 or code not in q.index: continue
            if limits is not None and code in limits.index:
                price,down=float(q.loc[code,'open']),float(limits.loc[code,'down_limit'])
            else:
                continuity=None
                if float(q.loc[code,'open'])<=previous_closes.get(code,0)*.95+.01:
                    index=session_positions[day]
                    prior=session_dates[max(0,index-5):index]
                    recent=[(d,float(dayframes[d].loc[code,'close']),float(dayframes[d].loc[code,'volume']))
                            for d in prior if code in dayframes[d].index]
                    anchors_before=[d for d in firsts if d<day]
                    if anchors_before:
                        bday=max(anchors_before)
                        if bday not in anchor_quotes:
                            anchor_quotes[bday]=engine.boundary_frame(bday)
                            anchor_limits[bday]=engine.boundary_frame(bday,'stk_limit')
                        if code in anchor_quotes[bday].index and code in anchor_limits[bday].index:
                            continuity=(bday,anchor_quotes[bday].loc[code],anchor_limits[bday].loc[code],recent)
                price,down=execution_quote(pro,day,code,float(q.loc[code,'open']),previous_closes.get(code),continuity)
            if price<=down+.005: continue
            ledger.cash+=qty*price; ledger.shares[code]-=qty
            last_sale_index=session_positions[day]
            deferred_sale=deferred_sale or reason=='rebalance_deferred'
            trades.append(dict(date=day,code=code,side='sell',shares=qty,price=price,reason=reason,signal_date=signal_day))
            if not ledger.shares[code]: anchors.pop(code,None); pending.pop(code,None)
        previous_day=session_dates[session_positions[day]-1] if session_positions[day]>0 else None
        daily_capacity=False
        if daily_enabled and not pending:
            exposure=float(plan_lookup[day.to_period('M')-1].risk_exposure.iloc[0])
            risk_budget=min(ledger.nav(prices),25000.)*exposure
            held_value=sum(qty*prices[code] for code,qty in ledger.shares.items() if qty)
            daily_capacity=(risk_budget>0 and min(ledger.cash,risk_budget-held_value)>=.5*risk_budget
                            and sum(qty>0 for qty in ledger.shares.values())<5)
        daily_buy=(daily_enabled and can_reenter(day,previous_day,session_positions[day],
            last_sale_index,daily_capacity,firsts)
            and previous_day in daily_candidate_lookup)
        if day in firsts or (replace_deferred and deferred_sale) or daily_buy:
            signal=day.to_period('M')-1
            plan=plan_lookup[signal]; cand=candidate_lookup.get(signal,c.iloc[:0])
            if daily_buy: cand=daily_candidate_lookup[previous_day]
            if limits is None: limits=replacement_limits(pro,day,cand,q)
            equity=ledger.nav(prices); target=min(equity,25000.)*float(plan.risk_exposure.iloc[0])
            if day in firsts: month_codes={code for code,n in ledger.shares.items() if n}
            for code,qty in (list(ledger.shares.items()) if day in firsts else []):
                held_qty=qty
                signal_rows=continuation_lookup.get(signal)
                row=signal_rows.loc[code] if signal_rows is not None and code in signal_rows.index else None
                if qty>0 and hold_winners and may_continue_position(row,anchors.get(code,{}).get('phase'),
                        code in pending,name in ADVANCE_HOLD_VARIANTS):
                    continuations.append(dict(date=day,code=code,shares=qty,signal_date=row.date,
                                              mom1=row.mom1,mom3=row.mom3,mom6=row.mom6))
                    continue
                qty-=ledger.blocked_shares(code)
                if qty<=0 or code not in q.index or code not in limits.index:
                    if retry_rebalance: queue_deferred_rebalance(pending,code,day,held_qty)
                    continue
                price=float(q.loc[code,'open'])
                if price<=limits.loc[code,'down_limit']+.005:
                    if retry_rebalance: queue_deferred_rebalance(pending,code,day,held_qty)
                    continue
                ledger.cash+=qty*price; ledger.shares[code]-=qty
                last_sale_index=session_positions[day]
                trades.append(dict(date=day,code=code,side='sell',shares=qty,price=price,reason='rebalance',signal_date=pd.NaT))
                if not ledger.shares[code]: anchors.pop(code,None); pending.pop(code,None)
                elif retry_rebalance: queue_deferred_rebalance(pending,code,day,ledger.shares[code])
            retained={code for code,n in ledger.shares.items() if n}
            retained_value=sum(ledger.shares[code]*prices[code] for code in retained)
            available=min(ledger.cash,max(0.,target-retained_value))
            buy=engine.buy_candidates(cand,plan,q,limits)
            buy=buy[~buy.ts_code.isin(retained)]
            if name.startswith('baseline_'):
                current_industry=cand.set_index('ts_code').ind_code.to_dict()
                buy=buy[~buy.ind_code.isin({current_industry.get(code,ledger.industry[code]) for code in retained})]
            selected=pd.DataFrame()
            if available>=100 and len(retained)<5 and len(buy):
                try: selected,_=optimizer(buy,available,max_names=5-len(retained),min_names=max(1,4-len(retained)),max_stock_weight=.30)
                except ValueError as exc:
                    if not any(x in str(exc) for x in ('所有候选的一手价格','找不到可买组合')): raise
            for row in selected.itertuples():
                code=row.ts_code
                if code not in known:
                    actions=pd.concat([actions,leadership.research_actions(pro,code)],ignore_index=True); known.add(code)
                ledger.cash-=row.shares*row.reference_price; ledger.shares[code]=ledger.shares.get(code,0)+row.shares
                ledger.industry[code]=row.ind_code; month_codes.add(code)
                anchors[code]=dict(entry=row.reference_price,peak=row.reference_price,phase=row.phase)
                trades.append(dict(date=day,code=code,side='buy',shares=row.shares,price=row.reference_price,
                                   reason='daily_reentry' if daily_buy else ('rebalance' if day in firsts else 'deferred_replacement'),
                                   signal_date=previous_day if daily_buy else (day.to_period('M')-1).to_timestamp('M')))
            invested=sum(ledger.shares[code]*prices[code] for code,n in ledger.shares.items() if n)
            if ledger.cash<-.01 or (retained_value<=target and invested>target+.01): raise AssertionError('Budget exceeded')
            allocations.append(dict(date=day,equity=equity,target=target,invested=invested,cash=ledger.cash,
                                    holdings=sum(n>0 for n in ledger.shares.values()),retained=len(retained)))
        ledger.close_record(day,actions)
        close=checked_marks(ledger,q,day,'close',actions,pro,stale)
        nav=ledger.nav(close)
        previous_closes={code:close[code] for code,n in ledger.shares.items() if n}
        daily_nav.append(dict(date=day,equity=nav,cash=ledger.cash))
        if name not in ('state_only','baseline_replay'):
            for code,a in anchors.items():
                if not ledger.shares.get(code,0): continue
                a['peak']=max(a['peak'],close[code])
                reason=exit_reason(close[code],a['entry'],a['peak'],a['phase'])
                if reason and code not in pending:
                    pending[code]=(reason,day)
                    signals.append(dict(date=day,code=code,reason=reason,close=close[code],entry=a['entry'],peak=a['peak'],phase=a['phase']))
        if day in lasts:
            factors=engine.factor_snapshot(pro,day)
            engine.check_unexplained_actions(month_codes,previous_factors,factors,actions,previous_mark,day)
            previous_mark,previous_factors=day,factors
            navs.append(dict(date=day,equity=nav,cash=ledger.cash,dividend_receivable=sum(x[0] for x in ledger.receivable.values()),holdings=sum(n>0 for n in ledger.shares.values())))
            holdings.extend(dict(date=day,code=code,shares=n,close=close[code],value=n*close[code]) for code,n in ledger.shares.items() if n)
            print(name,str(day.date()),f'equity={nav:.2f}',flush=True)
    account=pd.DataFrame(navs).set_index('date')
    account['return']=account.equity/account.equity.shift(1,fill_value=25000)-1
    account['profit']=account.equity-account.equity.shift(1,fill_value=25000)
    account['budget_return_pct']=account.profit/25000*100
    if len(account)!=45 or account.index.max()!=pd.Timestamp('2026-09-24'):
        raise AssertionError('Incomplete monthly account output')
    if name=='baseline_replay':
        original=pd.read_csv(leadership.BASE/'account_new_full.csv',index_col='date',parse_dates=True)
        pd.testing.assert_index_equal(account.index,original.index)
        np.testing.assert_allclose(account.equity,original.equity,atol=1e-6,rtol=0)
        print('DAILY ENGINE REPRODUCES ALL 45 RELEASED MONTH-END EQUITIES',flush=True)
    account.to_csv(OUT/f'account_{name}.csv')
    for label,rows in [('trades',trades),('allocations',allocations),('holdings',holdings),('daily_nav',daily_nav),('exit_signals',signals),('suspension_marks',stale),('continuations',continuations)]:
        pd.DataFrame(rows).to_csv(OUT/f'{label}_{name}.csv',index=False)
    pd.DataFrame(ledger.events,columns=['date','event','code','payment_date','cash_entitlement','bonus_shares','fractional_bonus_discarded']).to_csv(OUT/f'corporate_events_{name}.csv',index=False)
    return account


def summarize():
    import verify_small_account as verifier
    from verify_market_states import verify_daily
    verifier.OUT=OUT
    annual=[]; metrics=[]; monthly=[]; checks=[]
    for name in ('baseline_replay',)+VARIANTS+EXTENSIONS:
        path=OUT/f'account_{name}.csv'
        if not path.exists(): continue
        run_status=OUT/f'run_status_{name}.json'
        if run_status.exists() and not json.loads(run_status.read_text(encoding='utf-8')).get('account_complete'):
            raise ValueError('An incomplete rerun must not silently reuse an older account: '+name)
        check=verifier.verify(name); checks.append(check)
        a=pd.read_csv(path,parse_dates=['date'])
        daily=pd.read_csv(OUT/f'daily_nav_{name}.csv',parse_dates=['date'])
        nav=np.r_[25000.,daily.equity.to_numpy()]
        dd=float(np.min(nav/np.maximum.accumulate(nav)-1))
        a['strategy']=name; a['target_hit']=a.profit>=7500.-1e-7; monthly.append(a)
        trades=pd.read_csv(OUT/f'trades_{name}.csv',parse_dates=['date','signal_date'])
        exits=trades[trades.side.eq('sell') & trades.reason.ne('rebalance')]
        if not exits.date.gt(exits.signal_date).all(): raise AssertionError('Same-day stop fill')
        metrics.append(dict(strategy=name,ending_equity=a.equity.iloc[-1],total_profit=a.profit.sum(),
            total_return_pct=100*(a.equity.iloc[-1]/25000-1),daily_drawdown_pct=100*dd,
            target_hit_months=int(a.target_hit.sum()),losing_months=int(a.profit.lt(0).sum()),
            stop_exits=int(exits.reason.ne('rebalance_deferred').sum()),
            deferred_rebalance_exits=int(exits.reason.eq('rebalance_deferred').sum())))
        for year,g in a.groupby(a.date.dt.year):
            annual.append(dict(strategy=name,year=year,months=len(g),profit=g.profit.sum(),
                account_return_pct=100*((1+g['return']).prod()-1),ending_equity=g.equity.iloc[-1],
                target_hit_months=int(g.target_hit.sum())))
    if not monthly: raise ValueError('No completed variants')
    pd.concat(monthly).to_csv(OUT/'monthly_results.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'annual_results.csv',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    daily_checks=verify_daily(OUT,ROOT,[item['strategy'] for item in checks])
    files=[Path(__file__),Path(engine.__file__),Path(leadership.__file__),
        Path(verifier.__file__),Path(__file__).with_name('verify_market_states.py'),
        Path(config.__file__),Path(__file__).with_name('s7_budget_portfolio.py'),
        OUT/'protocol.json',OUT/'candidates.pkl',OUT/'plans.pkl',OUT/'daily.pkl',Path(config.STOCK_DAILY_PATH),
        Path(config.LOCAL_DATA_RAW)/'stock_circ_mv_monthly.csv',
        Path(config.LOCAL_DATA_RAW)/'ts_sw_industry_monthly.csv']
    files+=list((OUT/'exit_quotes').glob('*.pkl'))+list((OUT/'exit_quotes').glob('*.json'))+list((OUT/'size_snapshots').glob('*.pkl'))
    files+=[OUT/'verified_suspensions.json']+list((ROOT/'dividends').glob('*.pkl'))+list((leadership.OUT/'dividends').glob('*.pkl'))
    files+=list((ROOT/'suspension_checks').glob('*.pkl'))+list((ROOT/'adj_month_end').glob('*.pkl'))
    files+=list(OUT.glob('*protocol.json'))+list((OUT/'dividend_repairs').glob('*.pkl'))+list((OUT/'dividend_repairs').glob('*.json'))
    files+=list((OUT/'suspension_events').glob('*.pkl'))+list((OUT/'bulk_daily_repairs').glob('*.pkl'))
    files+=list((OUT/'board_daily_repairs').glob('*.pkl'))
    files+=list((OUT/'replacement_limits').glob('*.pkl'))
    files+=[OUT/'market_features.csv',OUT/'market_states.csv',OUT/'style_signals.csv',
            OUT/'sector_states.csv',leadership.BASE/'candidates.pkl',leadership.BASE/'industry_plans.pkl',
            ROOT/'boundaries/sessions.json',Path(__file__).parent/'data_pipeline/execution_data.py']
    files+=list((ROOT/'boundaries').glob('*.pkl'))
    if (OUT/'research_reference.json').exists(): files.append(OUT/'research_reference.json')
    if (OUT/'trend_features.pkl').exists():
        files += [OUT/'trend_features.pkl', ROOT/'stock_month_end_verified.pkl',
                  Path(__file__).with_name('research_trend_features.py')]
    if (OUT/'learned_leader_models.pkl').exists():
        files += [OUT/'learned_leader_models.pkl', ROOT/'stock_month_end_verified.pkl',
                  Path(__file__).with_name('research_learned_leaders.py')]
    if (OUT/'structural_router_protocol.json').exists():
        files += [Path(__file__).with_name('research_structural_router.py'),OUT/'candidate_audit_structural_router.pkl']
    if (OUT/'fine_memberships.pkl').exists():
        files += [Path(__file__).with_name('research_fine_industry.py'),OUT/'fine_memberships.pkl']
        files += list((OUT/'fine_membership_history').glob('*'))
        files += list(OUT.glob('candidate_audit_fine_*.pkl'))
    if (OUT/'peer_graph_features.pkl').exists():
        files += [Path(__file__).with_name('research_peer_graph.py'),OUT/'peer_graph_features.pkl',
                  OUT/'candidate_audit_peer_graph_retry.pkl',ROOT/'stock_month_end_verified.pkl']
    if (OUT/'candidate_audit_daily_pressure_retry.pkl').exists():
        files += [Path(__file__).with_name('research_daily_reentry.py'),OUT/'candidate_audit_daily_pressure_retry.pkl']
    if (OUT/'quality_floor_features.pkl').exists():
        files += [Path(__file__).with_name('research_quality_floor.py'),OUT/'quality_floor_features.pkl',
                  Path(config.LOCAL_DATA_RAW)/'raw_income.pkl',Path(config.LOCAL_DATA_RAW)/'raw_balancesheet.pkl']
        files += list(OUT.glob('candidate_audit_*quality_retry.pkl'))
        files += list(OUT.glob('candidate_audit_*quality_retry_monthly.pkl'))
        files += list(OUT.glob('candidate_audit_daily_guarded_retry*.pkl'))
    files += list(OUT.glob('run_status_*.json'))
    if (OUT/'daily_pressure_feature_manifest.json').exists(): files.append(OUT/'daily_pressure_feature_manifest.json')
    (OUT/'verification.json').write_text(json.dumps(dict(reconciliations=checks,daily_reconciliations=daily_checks,
        inputs={str(p.relative_to(config.PROJECT_DIR)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.csv')}),indent=2),encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False))
    print(pd.DataFrame(annual).to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','summary'])
    parser.add_argument('--variant',choices=VARIANTS+EXTENSIONS+('baseline_replay',))
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--output-dir',type=Path,help='Isolate a revised-input experiment from prior outputs')
    args=parser.parse_args()
    if args.output_dir is not None:
        OUT=args.output_dir.resolve()
        OUT.mkdir(parents=True,exist_ok=True)
    if args.action=='prepare': prepare(args.offline)
    elif args.action=='summary': summarize()
    else:
        if not args.variant: parser.error('--variant required')
        status_path=OUT/f'run_status_{args.variant}.json'
        status=dict(variant=args.variant,status='running',started_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                    account_complete=False)
        status_path.write_text(json.dumps(status,indent=2),encoding='utf-8')
        try:
            result=run(args.variant,args.offline)
            status.update(status='completed',account_complete=True,months=len(result),
                          ending_equity=float(result.equity.iloc[-1]))
        except BaseException as exc:
            status.update(status='incomplete',error_type=type(exc).__name__,error=str(exc))
            raise
        finally:
            status['finished_utc']=pd.Timestamp.now(tz='UTC').isoformat()
            status_path.write_text(json.dumps(status,indent=2),encoding='utf-8')
