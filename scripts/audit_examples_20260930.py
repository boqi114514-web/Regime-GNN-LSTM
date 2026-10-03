"""Read-only source audit of named examples; never a tradable stock whitelist.

All writes are isolated under results/_example_audit_20260930.
"""
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'src'))
import small_account_backtest as engine
from data_pipeline.execution_data import ROOT
from research_daily_reentry import pressure_features
from s7_budget_portfolio import is_main_board

SOURCE = BASE / 'results/market_state_board_complete'
OUT = BASE / 'results/_example_audit_20260930'
EXAMPLES = {'600183.SH': '生益科技', '002384.SZ': '东山精密'}


def action_history(code):
    choices = [ROOT / 'dividends' / f'{code}.canonical.pkl',
               ROOT / 'dividends' / f'{code}.pkl',
               BASE / 'results/leadership_research/dividends' / f'{code}.pkl']
    path = next(p for p in choices if p.exists())
    return engine.normalize_actions(pd.read_pickle(path), code), path


def buy_hold(code, start, end, label):
    actions, path = action_history(code)
    opening = engine.boundary_frame(start).loc[code]
    limits = engine.boundary_frame(start, 'stk_limit').loc[code]
    assert limits.down_limit + .005 < opening.open < limits.up_limit - .005
    assert (limits.up_limit - limits.down_limit) / ((limits.up_limit + limits.down_limit) / 2) > .15
    shares = math.floor(25000 / (100 * opening.open)) * 100
    assert shares > 0
    ledger = engine.Ledger()
    ledger.cash -= shares * opening.open
    ledger.shares[code] = shares
    month_ends = {pd.Timestamp(s['last']) for s in json.loads((ROOT / 'boundaries/sessions.json').read_text())}
    prior_nav = 25000.
    rows = []
    for day in pd.date_range(start, end):
        ledger.morning(day, actions)
        ledger.close_record(day, actions)
        if day in month_ends:
            price = float(engine.boundary_frame(day).loc[code, 'close'])
            nav = ledger.nav({code: price})
            rows.append(dict(code=code, name=EXAMPLES[code], scenario=label,
                start=start, mark_date=day, opening_price=opening.open, mark_close=price,
                initial_shares=shares, current_shares=ledger.shares[code], cash=ledger.cash,
                monthly_profit=nav-prior_nav, monthly_profit_over_25000=(nav-prior_nav)/25000,
                cumulative_profit=nav-25000, cumulative_return=(nav-25000)/25000,
                equity=nav, dividend_receivable=sum(a[0] for a in ledger.receivable.values()),
                action_source=str(path.relative_to(BASE))))
            prior_nav = nav
    first_factor = pd.read_pickle(ROOT / 'adj_month_end/20260331.pkl').set_index('ts_code').adj_factor[code]
    last_factor = pd.read_pickle(ROOT / 'adj_month_end' / f'{end:%Y%m%d}.pkl').set_index('ts_code').adj_factor[code]
    relevant = actions[actions.ex_date.between(pd.Timestamp('2026-04-01'), end)]
    if abs(first_factor-last_factor) > .001001:
        assert len(relevant), 'Unexplained adjustment-factor change'
    return rows, relevant


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sessions = json.loads((ROOT / 'boundaries/sessions.json').read_text())
    q2 = [s for s in sessions if s['first'].startswith(('2026-04', '2026-05'))]
    returns, actions = [], []
    for code in EXAMPLES:
        for s in q2:
            r, a = buy_hold(code, pd.Timestamp(s['first']), pd.Timestamp(s['last']), 'independent_month_25000')
            returns.extend(r)
            actions.append(a)
        r, a = buy_hold(code, pd.Timestamp(q2[0]['first']), pd.Timestamp(q2[-1]['last']), 'april_buy_hold_through_may')
        returns.extend(r)
        actions.append(a)
    returns = pd.DataFrame(returns)
    returns.to_csv(OUT / 'buy_hold.csv', index=False)
    pd.concat(actions).drop_duplicates().to_csv(OUT / 'corporate_actions.csv', index=False)
    c = pd.read_pickle(SOURCE / 'candidates.pkl')
    c['monthly_eligible'] = c.size_bucket.eq(c.chosen_style) & ~c.phase.isin(['retreat', 'overheat'])
    c['leadership_all_rank'] = c.groupby('month').leadership_score.rank(ascending=False, method='min')
    lowvol_rank = c.groupby('month').lowvol6.rank(pct=True)
    c['lowvol_percentile'] = lowvol_rank
    c['hypothetical_advance_score'] = .75 * (.55*c.fast_score+.25*c.peer_score+.20*c.score)+.25*c.sector_rank
    c['hypothetical_advance_rank'] = c.groupby('month').hypothetical_advance_score.rank(ascending=False, method='min')
    c['mainboard'] = c.stock_code.map(is_main_board)
    eligible = c[c.monthly_eligible & c.mainboard].copy()
    eligible['eligible_mainboard_rank'] = eligible.groupby('month').leadership_score.rank(ascending=False, method='min')
    c = c.merge(eligible[['month','ts_code','eligible_mainboard_rank']], on=['month','ts_code'], how='left', validate='one_to_one')
    selected = c[c.ts_code.isin(EXAMPLES) & c.month.between('2026-02-01', '2026-05-31')].copy()
    selected.to_csv(OUT / 'monthly_candidates.csv', index=False)
    sector = pd.read_csv(SOURCE / 'sector_states.csv', parse_dates=['month'])
    sector[sector.ind_code.eq('801080.SI') & sector.month.between('2026-02-01','2026-05-31')].to_csv(OUT / 'electronics_phase.csv', index=False)
    for variant in ['state_onset_retry', 'daily_guarded_retry']:
        t = pd.read_csv(SOURCE / f'trades_{variant}.csv', parse_dates=['date'])
        t[t.date.between('2026-04-01','2026-05-31')].to_csv(OUT / f'trades_{variant}.csv', index=False)
        t[t.code.isin(EXAMPLES)].to_csv(OUT / f'example_trades_{variant}.csv', index=False)
    # Include every contemporaneous peer in each relevant L2; no winner-only peer ranks.
    q2_candidates = c[c.month.between('2026-03-01','2026-04-30')].copy()
    peer_codes = set(q2_candidates[q2_candidates.l2_code.isin(selected.l2_code)].ts_code)
    all_raw = pd.read_pickle(SOURCE / 'daily.pkl')
    raw = all_raw[all_raw.ts_code.isin(peer_codes) & all_raw.date.between('2026-02-01','2026-05-31')]
    feature = pressure_features(raw)
    daily_rows = []
    for signal_day, g in feature[feature.signal_day.between('2026-04-01','2026-05-31')].groupby('signal_day'):
        universe = q2_candidates[q2_candidates.month.dt.to_period('M').eq(signal_day.to_period('M')-1)]
        cg = universe.merge(g, on='ts_code', validate='one_to_one')
        peers = cg.groupby('l2_code').pressure20
        cg['peer_count'] = peers.transform('count')
        cg['peer_median'] = peers.transform('median')
        cg['peer_breadth'] = peers.transform(lambda x: x.gt(0).mean())
        cg = cg[cg.ts_code.isin(EXAMPLES)].copy()
        checks = dict(pressure5_ok=cg.pressure5.gt(.05), pressure20_ok=cg.pressure20.gt(.10),
            location_ok=cg.location5.ge(.65), amount_ok=cg.pressure_amount.ge(1.2),
            peer_count_ok=cg.peer_count.ge(5), peer_median_ok=cg.peer_median.gt(0), breadth_ok=cg.peer_breadth.ge(.6))
        for key, value in checks.items():
            cg[key] = value
        cg['all_price_volume_pass'] = cg[list(checks)].all(axis=1)
        daily_rows.append(cg[['ts_code','signal_day','pressure5','pressure20','location5','pressure_amount',
            'peer_count','peer_median','peer_breadth',*checks,'all_price_volume_pass']])
    daily = pd.concat(daily_rows, ignore_index=True)
    guarded = pd.read_pickle(SOURCE / 'candidate_audit_daily_guarded_retry.pkl')
    guarded_keys = set(zip(guarded.ts_code, guarded.signal_day))
    daily['guarded_candidate'] = [key in guarded_keys for key in zip(daily.ts_code,daily.signal_day)]
    daily.to_csv(OUT / 'daily_signal_gates.csv', index=False)
    guarded = guarded[guarded.stock_code.map(is_main_board)].copy()
    guarded['candidate_rank_mainboard'] = guarded.groupby('signal_day').leadership_score.rank(ascending=False, method='min')
    ranked_examples = guarded[guarded.ts_code.isin(EXAMPLES) & guarded.signal_day.between('2026-04-01','2026-05-31')]
    ranked_examples[['signal_day','ts_code','leadership_score','candidate_rank_mainboard']].to_csv(OUT/'daily_example_ranks.csv', index=False)
    t = pd.read_csv(SOURCE / 'trades_daily_guarded_retry.csv', parse_dates=['date'])
    events = pd.read_csv(SOURCE / 'corporate_events_daily_guarded_retry.csv', parse_dates=['date','payment_date'])
    # No April-May action payments: source ledger cash can be reconciled exactly.
    assert not events.payment_date.between('2026-04-01','2026-05-31').any()
    navs = pd.read_csv(SOURCE / 'daily_nav_daily_guarded_retry.csv', parse_dates=['date']).set_index('date')
    dates = pd.DatetimeIndex(sorted(all_raw.date.unique()))
    rows = []
    for signal_day in sorted(daily.loc[daily.guarded_candidate,'signal_day'].unique()):
        signal_day = pd.Timestamp(signal_day)
        execution_day = dates[dates.get_loc(signal_day)+1]
        if execution_day.month not in (4,5):
            continue
        history = t[(t.date<execution_day) | (t.date.eq(execution_day)&t.side.eq('sell'))]
        holdings = history.assign(signed_shares=np.where(history.side.eq('buy'),history.shares,-history.shares)).groupby('code').signed_shares.sum()
        bonuses = events[events.date.le(execution_day)].groupby('code').bonus_shares.sum()
        holdings = holdings.add(bonuses,fill_value=0)
        holdings = holdings[holdings.gt(0)]
        quotes = all_raw[all_raw.date.eq(execution_day)].set_index('ts_code')
        invested = sum(qty*quotes.loc[code,'open'] for code,qty in holdings.items())
        same_day_sells = history[history.date.eq(execution_day)]
        cash = navs.loc[signal_day,'cash'] + (same_day_sells.shares*same_day_sells.price).sum()
        previous_sale = history.loc[history.side.eq('sell'),'date'].max()
        cooldown = dates.get_loc(execution_day)-dates.get_loc(previous_sale)
        available = min(cash, 25000.-invested)
        rows.append(dict(signal_day=signal_day,execution_day=execution_day,
            examples=','.join(daily.loc[daily.signal_day.eq(signal_day)&daily.guarded_candidate,'ts_code']),
            held_open_value=invested,cash=cash,residual_25000_budget=available,
            enough_half_budget=available>=12500,last_sale=previous_sale,sessions_since_sale=cooldown,
            cooldown_ok=cooldown>=5))
    pd.DataFrame(rows).to_csv(OUT / 'daily_capacity_blocks.csv', index=False)
    print('BUY AND HOLD (independent months and uninterrupted April-May)')
    print(returns[['code','scenario','start','mark_date','opening_price','mark_close','initial_shares','monthly_profit','monthly_profit_over_25000','cumulative_profit']].to_string(index=False))
    print('MONTHLY CANDIDATES')
    print(selected[['ts_code','month','mom1','mom3','mom6','phase','size_bucket','chosen_style','score','lowvol_percentile','leadership_score','monthly_eligible','eligible_mainboard_rank','hypothetical_advance_score','hypothetical_advance_rank']].to_string(index=False))
    print('DAILY PASS COUNTS')
    print(daily.groupby('ts_code')[[name for name in daily if name.endswith('_ok')]+['all_price_volume_pass','guarded_candidate']].sum().to_string())
    print('DAILY ELIGIBLE DATES')
    print(daily[daily.all_price_volume_pass | daily.guarded_candidate].to_string(index=False))
    print('DAILY CAPACITY BLOCKS')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
