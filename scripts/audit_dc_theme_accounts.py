"""Reconcile isolated DC accounts and attribute monthly profit from exported evidence.

No selection policy or trading ledger is called. Prices are the recorded source
quotes, not an independent vendor feed. Membership truth is not certified here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from data_pipeline.execution_data import ROOT
from s7_budget_portfolio import is_main_board
from verify_market_states import verify_daily
import verify_small_account as monthly_verifier


def buy_limit_evidence(out, root, day, code, reason='rebalance'):
    """Resolve and validate dated buy-limit evidence, including weekly caches.

    Prefer the execution route's own file. A present but malformed preferred
    file fails closed rather than being hidden by a secondary source.
    """
    day = pd.Timestamp(day)
    boundary = root/'boundaries'/f'stk_limit_{day:%Y%m%d}.pkl'
    replacement = out/'replacement_limits'/f'{day:%Y%m%d}.pkl'
    supplementary = reason in ('weekly_reentry', 'daily_reentry', 'deferred_replacement')
    paths = (replacement, boundary) if supplementary else (boundary, replacement)
    path = next((p for p in paths if p.exists()), None)
    if path is None:
        raise AssertionError('Missing buy limit evidence file')
    limits = pd.read_pickle(path)
    required = {'ts_code', 'trade_date', 'up_limit', 'down_limit'}
    if not required.issubset(limits.columns):
        raise AssertionError('Missing buy limit evidence columns')
    dates = pd.to_datetime(limits.trade_date.astype(str), format='%Y%m%d', errors='raise')
    if not dates.eq(day).all() or limits.duplicated(['ts_code', 'trade_date']).any():
        raise AssertionError('Wrong-date or duplicate buy limit evidence')
    selected = limits[limits.ts_code.eq(code)]
    if len(selected) != 1:
        raise AssertionError('Missing or ambiguous buy limit evidence row')
    lower, upper = float(selected.down_limit.iloc[0]), float(selected.up_limit.iloc[0])
    if not np.isfinite([lower, upper]).all() or not 0 < lower < upper:
        raise AssertionError('Invalid buy limit evidence prices')
    return path, lower, upper


def audit(out, name):
    status = json.loads((out/f'run_status_{name}.json').read_text(encoding='utf-8'))
    if not status.get('account_complete') or status.get('months') != 9:
        raise ValueError('Incomplete nine-month account')
    old = monthly_verifier.OUT
    monthly_verifier.OUT = out
    try:
        monthly = monthly_verifier.verify(name, expected_months=9, end_date='2026-09-24')
    finally:
        monthly_verifier.OUT = old
    daily = verify_daily(out, ROOT, [name])[0]
    t = pd.read_csv(out/f'trades_{name}.csv', parse_dates=['date', 'signal_date'])
    a = pd.read_csv(out/f'account_{name}.csv', parse_dates=['date'])
    h = pd.read_csv(out/f'holdings_{name}.csv', parse_dates=['date'])
    e = pd.read_csv(out/f'corporate_events_{name}.csv', parse_dates=['date'])
    e['cash_entitlement'] = pd.to_numeric(e.cash_entitlement, errors='raise').astype(float)
    allocations = pd.read_csv(out/f'allocations_{name}.csv')
    if (allocations.target.gt(np.minimum(allocations.equity,25000.)+.01).any()
            or allocations.invested.gt(allocations.target+.01).any()):
        raise AssertionError('This experiment has allocation over its capped budget; appreciated-retention cases need separate opening-mark reconciliation')
    b = t[t.side.eq('buy')]
    if (not b.code.str[:6].map(is_main_board).all()
            or b.shares.le(0).any() or b.shares.mod(100).ne(0).any()):
        raise AssertionError('Non-mainboard or non-whole-lot purchase')
    if t.signal_date.dropna().ge(t.loc[t.signal_date.notna(), 'date']).any():
        raise AssertionError('Signal does not predate execution')
    raw = pd.read_pickle(out/'daily.pkl')
    raw = raw[raw.ts_code.isin(t.code.unique())]
    quote_lookup = raw.set_index(['date', 'ts_code'])
    sources = set()
    max_open_error = 0.
    for row in t.itertuples():
        boundary = ROOT/'boundaries'/f'daily_{row.date:%Y%m%d}.pkl'
        repair = out/'exit_quotes'/f'daily_{row.code}_{row.date:%Y%m%d}.pkl'
        if boundary.exists() or repair.exists():
            path = boundary if boundary.exists() else repair
            q = pd.read_pickle(path)
            q = q[q.ts_code.eq(row.code)]
            if 'trade_date' in q:
                dates = pd.to_datetime(q.trade_date.astype(str), format='%Y%m%d', errors='raise')
                q = q[dates.eq(row.date)]
            if len(q) != 1:
                raise AssertionError('Missing or ambiguous source execution quote')
            opening = float(q.open.iloc[0])
            sources.add(path)
        else:
            q = quote_lookup.loc[(row.date, row.code)]
            if isinstance(q, pd.DataFrame):
                raise AssertionError('Duplicate local execution quote')
            opening = float(q.open)
            sources.add(out/'daily.pkl')
        max_open_error = max(max_open_error, abs(opening-row.price))
        np.testing.assert_allclose(opening, row.price, atol=.005, rtol=0)
        if row.side == 'buy':
            path, lower, upper = buy_limit_evidence(out, ROOT, row.date, row.code, row.reason)
            if not lower+.005 < opening < upper-.005:
                raise AssertionError('Purchase at price limit')
            sources.add(path)
    t['flow'] = t.shares*t.price*np.where(t.side.eq('sell'), 1., -1.)
    rows = []
    previous_date = pd.Timestamp('2025-12-31')
    for month in a.itertuples():
        current = h[h.date.eq(month.date)].groupby('code').value.sum()
        previous = h[h.date.eq(previous_date)].groupby('code').value.sum()
        flow = t[t.date.gt(previous_date)&t.date.le(month.date)].groupby('code').flow.sum()
        entitlement = e[e.date.gt(previous_date)&e.date.le(month.date)].groupby('code').cash_entitlement.sum()
        contribution = current.subtract(previous, fill_value=0).add(flow, fill_value=0).add(entitlement, fill_value=0)
        np.testing.assert_allclose(contribution.sum(), month.profit, atol=1e-6, rtol=0)
        rows.extend(dict(strategy=name, date=month.date, code=code, profit=float(value))
                    for code, value in contribution.items())
        previous_date = month.date
    nav = pd.read_csv(out/f'daily_nav_{name}.csv').equity.to_numpy()
    nav = np.r_[25000., nav]
    metrics = dict(strategy=name, ending_equity=float(a.equity.iloc[-1]),
        profit=float(a.profit.sum()), return_pct=100*(a.equity.iloc[-1]/25000.-1),
        daily_drawdown_pct=100*(1-nav/np.maximum.accumulate(nav)).max(),
        peak_loss_cny=float((np.maximum.accumulate(nav)-nav).max()),
        target_hit_months=int(a.profit.ge(7500.-1e-7).sum()),
        april_profit=float(a[a.date.dt.month.eq(4)].profit.iloc[0]),
        may_profit=float(a[a.date.dt.month.eq(5)].profit.iloc[0]))
    inputs = [out/f'{label}_{name}.csv' for label in
        ('trades', 'account', 'holdings', 'corporate_events', 'daily_nav', 'allocations')]
    inputs += list(sources)+[Path(__file__).resolve()]
    proof = dict(monthly=monthly, daily=daily, trades_checked=len(t),
        max_execution_open_error=max_open_error, whole_lot_mainboard_buys=True,
        allocation_cap_checked=True, attribution_reconciled=True, metrics=metrics,
        inputs={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    return a.assign(strategy=name), pd.DataFrame(rows), proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accounts', nargs='+', required=True,
                        help='directory=variant pairs')
    parser.add_argument('--output-dir', type=Path, default=Path('results/dc_theme_comparison'))
    args = parser.parse_args()
    accounts, contributions, proofs = [], [], []
    for spec in args.accounts:
        directory, name = spec.rsplit('=', 1)
        a, c, p = audit(Path(directory), name)
        accounts.append(a); contributions.append(c); proofs.append(p)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(accounts, ignore_index=True).to_csv(args.output_dir/'monthly_results.csv', index=False)
    pd.concat(contributions, ignore_index=True).to_csv(args.output_dir/'stock_contributions.csv', index=False)
    metrics = pd.DataFrame([p['metrics'] for p in proofs])
    metrics.to_csv(args.output_dir/'metrics.csv', index=False)
    (args.output_dir/'independent_checks.json').write_text(json.dumps(proofs, indent=2), encoding='utf-8')
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()
