"""Past-only price-neighbor signal, independent of current L3 classifications."""
import json

import numpy as np
import pandas as pd

VARIANTS=('peer_graph_retry',)
PROTOCOL=dict(
    variants=list(VARIANTS),
    stage='Exploratory follow-up, all 45 months already examined; no unseen-test claim',
    hypothesis='Dated L2 industries can contain several unrelated subtrends; shared price behavior identifies a narrower contemporary peer group',
    graph='Within each signal-date historical L2, correlate the last 12 complete calendar-month adjusted returns; no future rows; exclude self; strongest up to 10 peers with Pearson correlation>=.5; at least 4 required',
    features='Equal-weight peer median adjusted mom1/mom3/mom6, positive-mom3 fraction; all candidate SH/SZ boards participate',
    score='Peer composite = .2 rank(peer mom1)+.5 rank(peer mom3)+.3 rank(peer mom6); final score=.5 peer composite+.5 existing fast stock score',
    gate='Own and peer mom1/mom3/mom6 positive, peer breadth>=.6, peer composite in highest 20%; classify advance',
    routing='Replace selection only when L1 positive 3-month breadth<50% and a qualifying peer-group candidate exists; unchanged state_onset otherwise',
    unchanged='Original state_onset exposure, monthly entry, original daily next-open exits, deferred sells, 25000 initial/new-investment cap, whole lots, no topups, mainboard purchases only, zero fees',
    source='https://arxiv.org/abs/2110.13716',
    note='Shared-information motivation only, not a HIST implementation; graph is linear correlation, not a trained GNN')


def peer_features(candidates, monthly):
    m=monthly.copy()
    m['month']=m.date.dt.to_period('M').dt.to_timestamp('M')
    if m.duplicated(['month','ts_code']).any(): raise ValueError('Duplicate monthly prices')
    prices=m.pivot(index='month',columns='ts_code',values='close')*m.pivot(index='month',columns='ts_code',values='adj_factor')
    prices=prices.reindex(pd.date_range(prices.index.min(),prices.index.max(),freq='ME'))
    returns=prices/prices.shift(1)-1
    records=[]
    for month, pool in candidates.groupby('month',sort=True):
        history=returns.loc[:month].tail(12)
        if len(history)<12: continue
        for l2, g in pool.dropna(subset=['l2_code']).groupby('l2_code',sort=True):
            g=g.sort_values('ts_code').set_index('ts_code')
            h=history.reindex(columns=g.index).dropna(axis=1)
            h=h.loc[:,h.std().gt(1e-12)]
            if h.shape[1]<5: continue
            codes=h.columns.to_numpy()
            corr=np.corrcoef(h.to_numpy().T)
            np.fill_diagonal(corr,-np.inf)
            for i,code in enumerate(codes):
                # Stable ties retain ascending stock code order, never a future outcome.
                positions=np.argsort(-corr[i],kind='stable')[:10]
                positions=positions[corr[i,positions]>=.5]
                if len(positions)<4: continue
                peers=g.loc[codes[positions]]
                records.append(dict(month=month,ts_code=code,peer_count=len(peers),
                    graph_mom1=float(peers.mom1.median()),graph_mom3=float(peers.mom3.median()),
                    graph_mom6=float(peers.mom6.median()),graph_breadth=float(peers.mom3.gt(0).mean()),
                    mean_correlation=float(corr[i,positions].mean()),neighbors=','.join(codes[positions]),
                    history_start=history.index.min(),history_end=history.index.max()))
    columns=['month','ts_code','peer_count','graph_mom1','graph_mom3','graph_mom6','graph_breadth',
             'mean_correlation','neighbors','history_start','history_end']
    return pd.DataFrame(records,columns=columns)


def score_candidates(candidates, features, market):
    c=candidates.merge(features,on=['month','ts_code'],how='left',validate='one_to_one')
    ranks=c.groupby('month')[['graph_mom1','graph_mom3','graph_mom6']].rank(pct=True)
    c['graph_score']=.2*ranks.graph_mom1+.5*ranks.graph_mom3+.3*ranks.graph_mom6
    rank=c.groupby('month').graph_score.rank(pct=True)
    qualifies=(c.peer_count.ge(4)&c.graph_mom1.gt(0)&c.graph_mom3.gt(0)&c.graph_mom6.gt(0)
        &c.graph_breadth.ge(.6)&rank.ge(.8)&c.mom1.gt(0)&c.mom3.gt(0)&c.mom6.gt(0))
    if c.month.map(market.breadth).isna().any(): raise ValueError('Missing aggregate breadth')
    available=qualifies.groupby(c.month).any()
    c['graph_route']=c.month.map(market.breadth).lt(.5)&c.month.map(available).fillna(False)
    c['original_leadership_score']=c.leadership_score
    c['original_phase']=c.phase
    c['eligible']=c.size_bucket.eq(c.chosen_style)&~c.phase.isin(['retreat','overheat'])
    active=c.graph_route
    c.loc[active,'eligible']=qualifies[active]
    c.loc[active,'leadership_score']=.5*c.loc[active,'graph_score']+.5*c.loc[active,'fast_score']
    c.loc[active&qualifies,'phase']='advance'
    return c


def prepare_candidates(out, root):
    path=out/'peer_graph_protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=PROTOCOL:
        raise ValueError('Graph protocol changed')
    path.write_text(json.dumps(PROTOCOL,indent=2),encoding='utf-8')
    c=pd.read_pickle(out/'candidates.pkl')
    features=peer_features(c,pd.read_pickle(root/'stock_month_end_verified.pkl'))
    features.to_pickle(out/'peer_graph_features.pkl')
    market=pd.read_csv(out/'market_features.csv',parse_dates=['month']).set_index('month')
    c=score_candidates(c,features,market)
    c.to_pickle(out/'candidate_audit_peer_graph_retry.pkl')
    return c[c.eligible].copy()
