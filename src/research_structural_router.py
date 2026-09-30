"""Sparse-leadership state routing and delayed-rebalance replacement research."""
import json
import pandas as pd

VARIANTS=('structural_router_retry','structural_reentry_retry')
PROTOCOL=dict(
    variants=list(VARIANTS),
    stage='Exploratory follow-up on already examined history, not untouched validation; no stock/theme/calendar whitelist',
    state='Candidate-pool 3-month momentum median<0, 99th percentile>50%, positive L1-industry breadth<50%, SH/SZ amount ratio>=1',
    meaning='A small leading tail is strong while the typical stock/industry is weak and aggregate turnover is not contracting',
    selection='Use existing narrow_heat stock selection only in this state; otherwise use unchanged state_onset selection',
    exposure='Keep state_onset exposure unchanged; no extra capital or leverage',
    holding='Monthly rebalancing and original daily stop rules; no continuation override',
    delayed_replacement='structural_reentry_retry may allocate available cash at the same executable opening that finally clears a deferred monthly sell; use the same previous-month signal, actual opening and verified daily price limits',
    no_reentry='No same-day repurchase after a close-triggered stop; ordinary stop sales do not trigger replacement',
    capital='25000 initial and new-investment cap, no topups, mainboard 100-share buys, zero costs')


def structural_states(candidates,market):
    f=candidates.groupby('month').mom3.agg(stock_median='median',leader_q99=lambda s:s.quantile(.99))
    f=f.join(market[['breadth','amount_ratio']])
    f['structural']=(f.stock_median.lt(0)&f.leader_q99.gt(.5)&f.breadth.lt(.5)&f.amount_ratio.ge(1.))
    return f


def prepare_candidates(out):
    from research_market_states import narrow_candidates
    p=out/'structural_router_protocol.json'
    if p.exists() and json.loads(p.read_text(encoding='utf-8'))!=PROTOCOL:
        raise ValueError('Structural router protocol changed')
    p.write_text(json.dumps(PROTOCOL,indent=2),encoding='utf-8')
    c=pd.read_pickle(out/'candidates.pkl')
    market=pd.read_csv(out/'market_features.csv',parse_dates=['month']).set_index('month')
    states=structural_states(c,market)
    states.to_csv(out/'structural_router_states.csv')
    c['eligible']=c.size_bucket.eq(c.chosen_style)&~c.phase.isin(['retreat','overheat'])
    specialist=narrow_candidates(c,market,'narrow_heat')
    active=c.month.map(states.structural)
    for col in ('eligible','phase','leadership_score'):
        c.loc[active,col]=specialist.loc[active,col]
    c['structural_route']=active
    c.to_pickle(out/'candidate_audit_structural_router.pkl')
    return c[c.eligible].copy()
