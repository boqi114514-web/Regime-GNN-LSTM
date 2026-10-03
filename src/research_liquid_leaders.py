"""Liquid trend specialists: bypass coarse industry gates without ticker lists."""
import json

import pandas as pd

BASE_VARIANTS = ('liquid_leader_retry', 'liquid_leader_hold_retry')
VARIANTS = BASE_VARIANTS + ('liquid_phase_router_retry',)
PROTOCOL = dict(
    variants=list(BASE_VARIANTS),
    stage='Exploration after user-supplied PCB examples and previously examined history; no untouched-holdout claim',
    hypothesis='A weak broad industry must not veto liquid individual leaders; avoid a global size bucket and defensive low-volatility penalty in narrow markets',
    routing='Previous completed signal month L1 positive-mom3 breadth<50%, and at least one qualified candidate; otherwise unchanged state_onset monthly policy',
    liquidity='Mean positive traded daily amount in the signal calendar month, at or before each stock actual signal date; at least 10 observations; all SH/SZ candidate boards rank together',
    qualification='Mean amount in top 10% of the all-board monthly universe; own adjusted mom1,mom3,mom6>0; retain pre-existing exclusion mom3>150% AND mom1>30%',
    ranking='Within liquid top-decile candidates: .45 amount percentile + .35 mom1 percentile + .20 mom3 percentile; no original low-volatility or L1 score',
    phase='Qualified specialist entries are advance; no calendar, ticker, PCB or optical-module whitelist',
    continuation='Only hold variant: retain a position entered as a liquid specialist while the new monthly specialist rank stays <=10; preserve original entry and peak, never cancel pending exits',
    unchanged='daily_guarded financial floor and daily supplementary entries; state_onset exposure; original next-open stop rules; deferred sells; 25000 initial and new-investment cap; no topups; mainboard whole lots; zero costs',
    selection='Run both full 45-month accounts, report every completed variant and cash drawdown, no best-month splicing',
    sources=['https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00146',
             'https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1104491'],
    note='Literature motivates examining industry versus individual price trends; these precise rules are new experiments, not a reproduction of published profits')

PHASE_PROTOCOL = dict(
    variant='liquid_phase_router_retry',
    stage='Follow-up after both initial liquid variants: April/May improved but June and full-period performance deteriorated; exploratory feedback, not an unseen test',
    diagnosis='Wholesale narrow-market replacement also discards leaders already correctly identified by the original advance policy',
    rule='Keep original monthly selection if its highest-score eligible mainboard stock, affordable by signal close within 25000, is advance with positive mom1/mom3. Otherwise apply the original liquid specialist routing',
    no_lookahead='Priority is determined from the prior signal month only; never use month-ahead performance or named stocks/months',
    unchanged='Same original daily_guarded add-on, exposure, next-open execution, original exits, whole lots and fixed capital cap; no continuation override')


def liquidity_features(candidates, daily):
    if candidates.duplicated(['month', 'ts_code']).any():
        raise ValueError('Duplicate candidate keys')
    if daily.duplicated(['date', 'ts_code']).any():
        raise ValueError('Duplicate daily keys')
    c = candidates[['month', 'ts_code', 'date']].copy()
    if not c.date.dt.to_period('M').eq(c.month.dt.to_period('M')).all():
        raise ValueError('Signal date outside signal month')
    d = daily.loc[daily.volume.gt(0) & daily.amount.gt(0),
                  ['date', 'ts_code', 'amount']].copy()
    d['month'] = d.date.dt.to_period('M').dt.to_timestamp('M')
    blocks = []
    for month, pool in c.groupby('month', sort=True):
        one = d[d.month.eq(month)].merge(
            pool.rename(columns={'date': 'signal_date'}),
            on=['month', 'ts_code'], how='inner', validate='many_to_one')
        one = one[one.date.le(one.signal_date)]
        f = one.groupby(['month', 'ts_code']).agg(
            signal_amount=('amount', 'mean'), liquidity_days=('amount', 'size'),
            liquidity_last_day=('date', 'max')).reset_index()
        blocks.append(f)
    return pd.concat(blocks, ignore_index=True)


def score_candidates(candidates, features, market):
    if candidates.duplicated(['month', 'ts_code']).any():
        raise ValueError('Duplicate candidate keys')
    c = candidates.merge(features, on=['month', 'ts_code'], how='left', validate='one_to_one')
    breadth = c.month.map(market.breadth)
    if breadth.isna().any():
        raise ValueError('Missing market breadth')
    if c.liquidity_last_day.gt(c.date).any():
        raise ValueError('Liquidity contains future observations')
    c['original_leadership_score'] = c.leadership_score
    c['original_phase'] = c.phase
    amount = c.signal_amount.where(c.liquidity_days.ge(10))
    c['liquidity_percentile'] = amount.groupby(c.month).rank(pct=True)
    liquid = c.liquidity_percentile.ge(.9)
    ranks = c.loc[liquid].groupby('month')[['signal_amount', 'mom1', 'mom3']].rank(pct=True)
    c['liquid_score'] = .45*ranks.signal_amount + .35*ranks.mom1 + .20*ranks.mom3
    c['liquid_qualifies'] = (liquid & c.mom1.gt(0) & c.mom3.gt(0) & c.mom6.gt(0)
                             & ~(c.mom3.gt(1.5) & c.mom1.gt(.3)))
    c['liquid_route'] = breadth.lt(.5) & c.month.map(c.groupby('month').liquid_qualifies.any())
    c['liquid_rank'] = float('nan')
    ranked = c[c.liquid_qualifies].sort_values(['month', 'liquid_score', 'ts_code'],
                                              ascending=[True, False, True])
    c.loc[ranked.index, 'liquid_rank'] = ranked.groupby('month').cumcount()+1
    c['eligible'] = c.size_bucket.eq(c.chosen_style) & ~c.phase.isin(['retreat', 'overheat'])
    active = c.liquid_route
    c.loc[active, 'eligible'] = c.loc[active, 'liquid_qualifies']
    c.loc[active, 'leadership_score'] = c.loc[active, 'liquid_score']
    c.loc[active & c.liquid_qualifies, 'phase'] = 'advance'
    return c


def continue_liquid_position(row, pending):
    return (row is not None and not pending and bool(row.liquid_route)
            and bool(row.liquid_qualifies) and 1 <= row.liquid_rank <= 10)


def phase_priority_candidates(scored):
    """Repair a wrong phase switch, not every month with low aggregate breadth."""
    from s7_budget_portfolio import is_main_board
    c = scored.copy()
    original_eligible = c.size_bucket.eq(c.chosen_style) & ~c.original_phase.isin(['retreat', 'overheat'])
    pool = c[original_eligible & c.stock_code.map(is_main_board) & c.close.gt(0) & (100*c.close).le(25000)]
    top = pool.sort_values(['month', 'original_leadership_score', 'ts_code'],
                           ascending=[True, False, True]).drop_duplicates('month').set_index('month')
    priority = top.original_phase.eq('advance') & top.mom1.gt(0) & top.mom3.gt(0)
    c['core_advance_priority'] = c.month.map(priority).eq(True)
    restore = c.liquid_route & c.core_advance_priority
    c.loc[restore, 'eligible'] = original_eligible[restore]
    c.loc[restore, 'phase'] = c.loc[restore, 'original_phase']
    c.loc[restore, 'leadership_score'] = c.loc[restore, 'original_leadership_score']
    c.loc[restore, 'liquid_route'] = False
    return c


def prepare_candidates(out, name):
    if name not in VARIANTS:
        raise ValueError(name)
    path = out/'liquid_leader_protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != PROTOCOL:
        raise ValueError('Liquid specialist protocol changed')
    path.write_text(json.dumps(PROTOCOL, indent=2), encoding='utf-8')
    c = pd.read_pickle(out/'candidates.pkl')
    f = liquidity_features(c, pd.read_pickle(out/'daily.pkl'))
    f.to_pickle(out/'liquid_features.pkl')
    market = pd.read_csv(out/'market_features.csv', parse_dates=['month']).set_index('month')
    c = score_candidates(c, f, market)
    if name == 'liquid_phase_router_retry':
        phase_path = out/'liquid_phase_router_protocol.json'
        if phase_path.exists() and json.loads(phase_path.read_text(encoding='utf-8')) != PHASE_PROTOCOL:
            raise ValueError('Phase priority protocol changed')
        phase_path.write_text(json.dumps(PHASE_PROTOCOL, indent=2), encoding='utf-8')
        c = phase_priority_candidates(c)
    c.to_pickle(out/f'candidate_audit_{name}.pkl')
    c.groupby('month').agg(candidates=('ts_code', 'size'),
                          liquid_qualifies=('liquid_qualifies', 'sum'),
                          routed=('liquid_route', 'first')).to_csv(out/f'liquid_route_audit_{name}.csv')
    return c[c.eligible].copy()
