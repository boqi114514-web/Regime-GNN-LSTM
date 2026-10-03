"""Event ledger with monthly execution/valuation; zero costs as requested.

This first account comparison uses a common fixed stock rule, not the legacy
fundamental stock selector. NAV is sampled at month ends (not daily drawdown).
"""
import json
import uuid
import math
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.execution_data import ROOT, fetch, fetch_pages, fetch_variants, validate_rows
from data_pipeline.tushare_config import get_pro
from s7_budget_portfolio import optimize_portfolio, is_main_board
from stock_execution_research import OUT


def normalize_actions(frame, code, start='2023-01-01', end='2026-09-24'):
    columns = ['ts_code', 'record_date', 'ex_date', 'pay_date', 'div_listdate', 'cash', 'stock', 'event']
    if frame.empty:
        return pd.DataFrame(columns=columns)
    needed = {'ts_code', 'end_date', 'div_proc', 'record_date', 'ex_date', 'pay_date', 'div_listdate', 'cash_div_tax', 'stk_div'}
    if not needed <= set(frame):
        raise ValueError('Incomplete corporate-action schema')
    if not frame.ts_code.eq(code).all():
        raise ValueError('Corporate action response contains other stocks')
    d = frame[frame.div_proc.eq('实施')].copy()
    for col in ('record_date', 'ex_date', 'pay_date', 'div_listdate'):
        d[col] = pd.to_datetime(d[col], errors='coerce')
    d = d[d.ex_date.between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    if not d.end_date.astype(str).str.fullmatch(r'\d{8}').all():
        raise ValueError('Noncanonical dividend response: missing report period')
    if {'stk_bo_rate', 'stk_co_rate'} <= set(d):
        parts = pd.to_numeric(d.stk_bo_rate, errors='coerce').fillna(0)+pd.to_numeric(d.stk_co_rate, errors='coerce').fillna(0)
        total = pd.to_numeric(d.stk_div, errors='coerce').fillna(0)
        if ((parts > 0) & ((parts-total).abs() > 1e-6)).any():
            raise ValueError('Noncanonical dividend response: inconsistent stock distribution units')
    d['cash'] = pd.to_numeric(d.cash_div_tax, errors='coerce').fillna(0.)
    d['stock'] = pd.to_numeric(d.stk_div, errors='coerce').fillna(0.)
    if not np.isfinite(d[['cash', 'stock']].to_numpy()).all() or (d[['cash', 'stock']] < 0).any().any():
        raise ValueError('Invalid corporate action amounts')
    d = d[(d.cash > 0) | (d.stock > 0)]
    if d.record_date.isna().any() or (d.record_date >= d.ex_date).any():
        raise ValueError('Invalid dividend record/ex dates')
    if ((d.cash > 0) & (d.pay_date.isna() | (d.pay_date < d.ex_date))).any():
        raise ValueError('Missing/invalid dividend payment date')
    if ((d.stock > 0) & (d.div_listdate.isna() | (d.div_listdate < d.ex_date))).any():
        raise ValueError('Missing/invalid bonus-share listing date')
    d.loc[d.stock.eq(0), 'div_listdate'] = pd.NaT
    d.loc[d.cash.eq(0), 'pay_date'] = pd.NaT
    d = d[['ts_code', 'end_date', 'record_date', 'ex_date', 'pay_date', 'div_listdate', 'cash', 'stock']].drop_duplicates()
    # Annual and quarterly dividends can share one ex-date; they are separate
    # entitlements, whereas repeated announcements of one period are not.
    if d.duplicated(['ts_code', 'end_date', 'ex_date']).any():
        raise ValueError('Conflicting implemented corporate actions')
    d['event'] = d.ts_code + ':' + d.ex_date.dt.strftime('%Y%m%d') + ':' + d.end_date.astype(str)
    return d[columns]


class Ledger:
    def __init__(self):
        self.cash = 25000.
        self.shares = {}
        self.industry = {}
        self.recorded = {}
        self.receivable = {}
        self.locked = {}
        self.events = []

    def morning(self, date, actions):
        for a in actions.itertuples():
            if a.ex_date == date:
                entitled = self.recorded.get(a.event, 0)
                cash = entitled*a.cash
                bonus = math.floor(entitled*a.stock+1e-8)
                if cash:
                    self.receivable[a.event] = (cash, a.pay_date)
                if bonus:
                    self.shares[a.ts_code] = self.shares.get(a.ts_code, 0)+bonus
                    self.locked[a.event] = (a.ts_code, bonus, a.div_listdate)
                if cash or bonus:
                    self.events.append(dict(date=str(date.date()), event=a.event, code=a.ts_code,
                                            payment_date=str(a.pay_date.date()) if pd.notna(a.pay_date) else None,
                                            cash_entitlement=cash, bonus_shares=bonus,
                                            fractional_bonus_discarded=entitled*a.stock-bonus))
        for event, (cash, pay_date) in list(self.receivable.items()):
            if pay_date <= date:
                self.cash += cash
                del self.receivable[event]
        for event, (_, _, listing) in list(self.locked.items()):
            if listing <= date:
                del self.locked[event]

    def close_record(self, date, actions):
        for a in actions.itertuples():
            if a.record_date == date:
                self.recorded[a.event] = self.shares.get(a.ts_code, 0)

    def blocked_shares(self, code):
        return sum(n for c, n, _ in self.locked.values() if c == code)

    def nav(self, prices):
        missing = [c for c, n in self.shares.items() if n and c not in prices]
        if missing:
            raise ValueError('Missing held-stock valuation: '+','.join(missing))
        return self.cash + sum(self.shares[c]*prices[c] for c in self.shares if self.shares[c]) + sum(x[0] for x in self.receivable.values())


def load_actions(pro, code):
    root = ROOT/'dividends'
    root.mkdir(parents=True, exist_ok=True)
    path = root/f'{code}.pkl'
    canonical = root/f'{code}.canonical.pkl'
    if canonical.exists():
        return normalize_actions(pd.read_pickle(canonical), code)
    if not path.exists():
        print('fetch corporate actions', code, flush=True)
    raw = pd.read_pickle(path) if path.exists() else fetch_variants(pro, 'dividend', [
        dict(ts_code=code),
        dict(ts_code=code, div_proc='实施'),
        dict(ts_code=code, fields='ts_code,end_date,ann_date,div_proc,stk_div,cash_div_tax,record_date,ex_date,pay_date,div_listdate,imp_ann_date'),
        dict(ts_code=code, limit=6000, offset=0)],
        ['ts_code', 'end_date', 'div_proc', 'ex_date', 'cash_div_tax', 'stk_div'])
    raw.to_pickle(path)
    try:
        actions = normalize_actions(raw, code)
    except ValueError:
        # Preserve the noncanonical raw response as evidence. Query the same
        # events with explicit report periods; never guess a unit conversion.
        expected = pd.to_datetime(raw.loc[raw.div_proc.eq('实施'), 'ex_date'], errors='coerce')
        expected = set(expected[expected.between('2023-01-01', '2026-09-24')])
        if not expected:
            raise
        directory = ROOT/'dividend_periods'
        print('repair dividend report-period schema', code, flush=True)
        directory.mkdir(parents=True, exist_ok=True)
        annual = sorted({f'{date.year-1}1231' for date in expected})
        quarters = [d.strftime('%Y%m%d') for d in pd.date_range('2022-03-31', '2026-06-30', freq='QE')
                    if d.strftime('%Y%m%d') not in annual]
        frames, covered, requested = [], set(), []
        for period in annual+quarters:
            p = directory/f'{code}_{period}.pkl'
            frame = pd.read_pickle(p) if p.exists() else fetch(pro, 'dividend', ts_code=code, end_date=period)
            frame.to_pickle(p)
            requested.append(period)
            if frame.empty:
                continue
            normalized = normalize_actions(frame, code)
            frames.append(frame)
            covered.update(normalized.ex_date)
            if expected <= covered:
                break
        if not expected <= covered:
            raise ValueError('Report-period dividend repair lacks original ex-dates: '+code)
        repaired = pd.concat(frames, ignore_index=True).drop_duplicates()
        actions = normalize_actions(repaired, code)
        repaired.to_pickle(canonical)
        (root/f'{code}.repair.json').write_text(json.dumps(dict(
            code=code, method='explicit report-period standard responses, no manual scaling',
            periods_requested=requested, ex_dates_covered=sorted(str(x.date()) for x in covered)), indent=2), encoding='utf-8')
    return actions


def boundary_frame(day, api='daily'):
    frame = pd.read_pickle(ROOT/'boundaries'/f'{api}_{day:%Y%m%d}.pkl')
    return frame.set_index('ts_code', verify_integrity=True)


def factor_snapshot(pro, day):
    path = ROOT/'adj_month_end'/f'{day:%Y%m%d}.pkl'
    cached = path.exists()
    d = pd.read_pickle(path) if cached else fetch_pages(pro, 'adj_factor', trade_date=day.strftime('%Y%m%d'))
    d = validate_rows(d, ['ts_code', 'trade_date'], ['adj_factor'], day.strftime('%Y%m%d'))
    # Reading a validated shared cache must not truncate it during another
    # account's replay. New snapshots are published atomically as well.
    if not cached:
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_name(path.name+'.'+uuid.uuid4().hex+'.staging')
        try:
            d.to_pickle(staged)
            staged.replace(path)
        finally:
            if staged.exists():
                staged.unlink()
    return d.set_index('ts_code').adj_factor.to_dict()


def check_unexplained_actions(codes, before, after, actions, begin, end):
    for code in codes:
        if code not in before or code not in after:
            raise ValueError('Missing corporate-action cross-check factor: '+code)
        # Gateway mixes 3- and 4-decimal factors (e.g. 5.361 / 5.3613).
        # Only changes exceeding the sum of two 3-decimal rounding half-units
        # trigger the missing-event guard; cash/share events are always booked.
        if abs(after[code]-before[code]) > .001001:
            relevant = actions[(actions.ts_code == code) & (actions.ex_date > begin) & (actions.ex_date <= end)]
            if relevant.empty:
                raise ValueError(f'Unexplained factor change (possible rights/missing dividend): {code} {end}')


def held_marks(ledger, quotes, day, field, actions, pro, audit):
    prices = quotes[field].to_dict()
    for code, qty in ledger.shares.items():
        if not qty or code in prices:
            continue
        root = ROOT/'suspension_checks'
        root.mkdir(parents=True, exist_ok=True)
        p = root/f'{code}_{day:%Y%m%d}_daily.pkl'
        raw = pd.read_pickle(p) if p.exists() else fetch_pages(pro, 'daily', ts_code=code,
            start_date=(day-pd.Timedelta(days=90)).strftime('%Y%m%d'), end_date=day.strftime('%Y%m%d'))
        if not {'ts_code', 'trade_date', 'close'} <= set(raw) or raw.empty or not raw.ts_code.eq(code).all():
            raise ValueError('No recent valid quote for missing holding '+code)
        raw = raw.assign(date=pd.to_datetime(raw.trade_date)).sort_values('date')
        if (raw.date > day).any():
            raise ValueError('Future quote in suspension check')
        raw.to_pickle(p)
        last = raw.iloc[-1]
        if last.date == day:
            raise ValueError('Bulk quote missing a trading stock; repair data first: '+code)
        sp = root/f'{code}_{day:%Y%m%d}_suspend.pkl'
        sus = pd.read_pickle(sp) if sp.exists() else fetch_variants(pro, 'suspend_d', [
            dict(ts_code=code, trade_date=day.strftime('%Y%m%d'), fields='ts_code,trade_date,suspend_timing,suspend_type'),
            dict(ts_code=code, start_date=(last.date+pd.Timedelta(days=1)).strftime('%Y%m%d'), end_date=day.strftime('%Y%m%d'), offset=0),
            dict(ts_code=code, start_date=(last.date+pd.Timedelta(days=1)).strftime('%Y%m%d'), end_date=day.strftime('%Y%m%d'))],
            ['ts_code', 'trade_date', 'suspend_type'])
        if not {'ts_code', 'trade_date', 'suspend_type'} <= set(sus) or not sus.ts_code.eq(code).all():
            raise ValueError('Missing suspension evidence '+code)
        sus = sus.assign(date=pd.to_datetime(sus.trade_date))
        sus = sus[sus.date.between(last.date+pd.Timedelta(days=1), day)]
        s = sus.loc[sus.suspend_type.eq('S'), 'date'].max()
        r = sus.loc[sus.suspend_type.eq('R'), 'date'].max()
        if pd.isna(s) or (pd.notna(r) and r >= s):
            raise ValueError('Cannot confirm continuing suspension '+code)
        sus.to_pickle(sp)
        price = float(last.close)
        relevant = actions[(actions.ts_code == code) & (actions.ex_date > last.date) & (actions.ex_date <= day)]
        for _, a in relevant.groupby('ex_date', sort=True):
            price = (price-a.cash.sum())/(1+a.stock.sum())
        if price <= 0:
            raise ValueError('Invalid suspended-stock mark')
        prices[code] = price
        audit.append(dict(date=day, code=code, quote_date=last.date, mark=price, field=field,
                          reason='confirmed_suspension_last_close_adjusted_for_corporate_actions'))
    return prices


def buy_candidates(candidates, plan, quotes, limits):
    c = candidates[candidates.ind_code.isin(plan.ts_code)].copy()
    c = c[c.stock_code.map(is_main_board)]
    c['ind_score'] = c.ind_code.map(plan.set_index('ts_code').selection_score)
    c = c.drop(columns=['open', 'up_limit', 'down_limit'], errors='ignore')
    c = c.merge(quotes[['open']], left_on='ts_code', right_index=True)
    c = c.merge(limits[['up_limit', 'down_limit']], left_on='ts_code', right_index=True)
    # Ignore intraday outcomes: execution eligibility uses opening quote/limits,
    # not that day's high, low, close or realized volume.
    c = c[(c.open < c.up_limit-.005) & (c.open > c.down_limit+.005)]
    # Exclude contemporaneous 5%-limit risk-warning stocks, without using names today.
    c = c[(c.up_limit-c.down_limit)/((c.up_limit+c.down_limit)/2) > .15]
    c['reference_price'] = c.open
    # Preserve the all-board within-industry ranks; candidate truncation is not needed.
    return c


def run(strategy, candidates, plans, sessions, pro):
    ledger, actions, known = Ledger(), pd.DataFrame(), set()
    navs, trades, allocations, holdings, stale_marks = [], [], [], [], []
    plan_lookup = {d.to_period('M'): g for d, g in plans[plans.strategy.eq(strategy)].groupby('date')}
    candidate_lookup = {d.to_period('M'): g for d, g in candidates.groupby('month')}
    firsts = {pd.Timestamp(s['first']): s for s in sessions}
    lasts = {pd.Timestamp(s['last']): s for s in sessions}
    previous_mark = pd.Timestamp('2022-12-30')
    previous_factors = factor_snapshot(pro, previous_mark)
    month_codes = set()
    for day in pd.date_range('2023-01-01', sessions[-1]['last']):
        ledger.morning(day, actions)
        if day in firsts:
            month_codes = {c for c, n in ledger.shares.items() if n}
            signal = day.to_period('M')-1
            plan, cand = plan_lookup[signal], candidate_lookup[signal]
            quotes, limits = boundary_frame(day), boundary_frame(day, 'stk_limit')
            prices = held_marks(ledger, quotes, day, 'open', actions, pro, stale_marks)
            equity = ledger.nav(prices)
            target = min(equity, 25000.)*float(plan.risk_exposure.iloc[0])
            # Full monthly reset among liquid positions; costs intentionally zero.
            for code, qty in list(ledger.shares.items()):
                sellable = qty-ledger.blocked_shares(code)
                if sellable <= 0 or code not in quotes.index or code not in limits.index:
                    continue
                price = prices[code]
                if price <= limits.loc[code, 'down_limit']+.005:
                    continue
                ledger.cash += sellable*price
                ledger.shares[code] -= sellable
                trades.append(dict(date=day, code=code, side='sell', shares=sellable, price=price))
            retained = {c for c, n in ledger.shares.items() if n}
            retained_value = sum(ledger.shares[c]*prices[c] for c in retained)
            available = min(ledger.cash, max(0., target-retained_value))
            c = buy_candidates(cand, plan, quotes, limits)
            current_industry = cand.set_index('ts_code').ind_code.to_dict()
            retained_inds = {current_industry.get(code, ledger.industry[code]) for code in retained}
            c = c[~c.ind_code.isin(retained_inds) & ~c.ts_code.isin(retained)]
            selected = pd.DataFrame()
            if available >= 100 and len(retained) < 5 and len(c):
                try:
                    selected, _ = optimize_portfolio(c, available, max_names=5-len(retained),
                        min_names=max(1, 4-len(retained)), max_stock_weight=.30)
                except ValueError as exc:
                    if not any(x in str(exc) for x in ['所有候选的一手价格', '找不到可买组合']):
                        raise
            for row in selected.itertuples():
                code = row.stock_code + ('.SH' if row.stock_code.startswith('6') else '.SZ')
                if code not in known:
                    new_actions = load_actions(pro, code)
                    actions = pd.concat([actions, new_actions], ignore_index=True)
                    known.add(code)
                ledger.cash -= row.shares*row.reference_price
                ledger.shares[code] = ledger.shares.get(code, 0)+row.shares
                ledger.industry[code] = row.ind_code
                month_codes.add(code)
                trades.append(dict(date=day, code=code, side='buy', shares=row.shares, price=row.reference_price))
            if ledger.cash < -.01:
                raise AssertionError('Negative cash')
            spent = sum(ledger.shares[c]*prices[c] for c, n in ledger.shares.items() if n)
            if spent > target+.01 and retained_value <= target:
                raise AssertionError('Risk cap exceeded by purchases')
            allocations.append(dict(date=day, equity=equity, target=target, invested=spent, cash=ledger.cash,
                                    holdings=sum(n > 0 for n in ledger.shares.values()), retained=len(retained)))
        ledger.close_record(day, actions)
        if day in lasts:
            factors = factor_snapshot(pro, day)
            check_unexplained_actions(month_codes, previous_factors, factors, actions, previous_mark, day)
            previous_mark, previous_factors = day, factors
            quotes = boundary_frame(day)
            prices = held_marks(ledger, quotes, day, 'close', actions, pro, stale_marks)
            nav = ledger.nav(prices)
            navs.append(dict(date=day, equity=nav, cash=ledger.cash,
                             dividend_receivable=sum(x[0] for x in ledger.receivable.values()),
                             holdings=sum(n > 0 for n in ledger.shares.values())))
            holdings.extend(dict(date=day, code=code, shares=qty, close=prices[code],
                                 value=qty*prices[code])
                            for code, qty in ledger.shares.items() if qty)
            print(strategy, str(day.date()), f'equity={nav:.2f}', flush=True)
    result = pd.DataFrame(navs).set_index('date')
    result['return'] = result.equity.div(result.equity.shift(1).fillna(25000.))-1
    result.to_csv(OUT/f'account_{strategy}.csv')
    pd.DataFrame(trades).to_csv(OUT/f'trades_{strategy}.csv', index=False)
    pd.DataFrame(allocations).to_csv(OUT/f'allocations_{strategy}.csv', index=False)
    pd.DataFrame(ledger.events).to_csv(OUT/f'corporate_events_{strategy}.csv', index=False)
    pd.DataFrame(holdings).to_csv(OUT/f'holdings_{strategy}.csv', index=False)
    pd.DataFrame(stale_marks).to_csv(OUT/f'suspension_marks_{strategy}.csv', index=False)
    return result


class OfflineClient:
    def query(self, api, **kwargs):
        raise RuntimeError('Offline replay missing cache: '+api)


def input_hashes():
    import config
    files = list(ROOT.rglob('*.pkl')) + [ROOT/'boundaries/sessions.json', OUT/'protocol.json',
        OUT/'candidates.pkl', OUT/'industry_plans.pkl', Path(__file__),
        Path(config.PROJECT_DIR)/'src/stock_execution_research.py',
        Path(config.PROJECT_DIR)/'src/s7_budget_portfolio.py']
    return {str(p.relative_to(config.PROJECT_DIR)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def main(offline=False):
    sessions = json.loads((ROOT/'boundaries/sessions.json').read_text(encoding='utf-8'))
    if len(sessions) != 45 or sessions[-1]['last'] != '2026-09-24':
        raise ValueError('Boundary dataset incomplete')
    candidates = pd.read_pickle(OUT/'candidates.pkl')
    plans = pd.read_pickle(OUT/'industry_plans.pkl')
    results = {}
    for strategy in ('original', 'new_full', 'new_vol'):
        results[strategy] = run(strategy, candidates, plans, sessions, OfflineClient() if offline else get_pro())
    annual = []
    for name, frame in results.items():
        for year, block in frame.groupby(frame.index.year):
            annual.append(dict(strategy=name, year=year, months=len(block), return_=(1+block['return']).prod()-1,
                               ending_equity=block.equity.iloc[-1]))
    pd.DataFrame(annual).to_csv(OUT/'account_annual_comparison.csv', index=False)
    manifest = dict(start='2023-01-01', end='2026-09-24', months=45, initial_cash=25000,
                    capital_cap=25000, original_means='original industry signal with shared new stock rule',
                    fees=0, cash_dividends='gross before withholding',
                    fractional_bonus='rounded down per event', valuation='month-end raw close',
                    execution='first trading-day opening quote; opening limit restrictions',
                    drawdown_frequency='monthly, not daily', offline_replay=offline,
                    input_sha256=input_hashes())
    (OUT/'account_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(pd.DataFrame(annual).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    main(parser.parse_args().offline)
