"""Independently reconcile exported trades/actions against month-end ledgers."""
import json
import ast
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import config
from stock_execution_research import OUT
from s7_budget_portfolio import is_main_board


def verify(name, expected_months=45, end_date='2026-09-24'):
    nav = pd.read_csv(OUT/f'account_{name}.csv', parse_dates=['date'])
    trades = pd.read_csv(OUT/f'trades_{name}.csv', parse_dates=['date'])
    events = pd.read_csv(OUT/f'corporate_events_{name}.csv', parse_dates=['date', 'payment_date'])
    holdings = pd.read_csv(OUT/f'holdings_{name}.csv', parse_dates=['date'])
    # Header-only action CSVs otherwise have object dtype; pandas' empty
    # groupby then propagates it into numeric inventory reconciliation.
    for col in ('cash_entitlement', 'bonus_shares'):
        events[col] = pd.to_numeric(events[col], errors='raise').astype(float)
        if not np.isfinite(events[col]).all():
            raise ValueError('Invalid corporate event '+col)
    for col in ('shares', 'close', 'value'):
        holdings[col] = pd.to_numeric(holdings[col], errors='raise').astype(float)
    allocations = pd.read_csv(OUT/f'allocations_{name}.csv', parse_dates=['date'])
    if len(nav) != expected_months or nav.date.max() != pd.Timestamp(end_date):
        raise AssertionError('Incomplete performance period')
    buys = trades[trades.side.eq('buy')]
    assert (buys.shares > 0).all() and (buys.shares % 100 == 0).all()
    assert buys.code.str[:6].map(is_main_board).all()
    assert (trades.price > 0).all()
    assert (allocations.cash >= -.01).all() and (allocations.holdings <= 5).all()
    trades['cashflow'] = trades.shares*trades.price*np.where(trades.side.eq('sell'), 1, -1)
    trades['shareflow'] = trades.shares*np.where(trades.side.eq('buy'), 1, -1)
    for row in nav.itertuples():
        t = trades[trades.date <= row.date]
        e = events[events.date <= row.date]
        paid = e.loc[e.payment_date <= row.date, 'cash_entitlement'].sum()
        pending = e.loc[e.payment_date > row.date, 'cash_entitlement'].sum()
        cash = 25000.+t.cashflow.sum()+paid
        np.testing.assert_allclose(cash, row.cash, atol=1e-6, rtol=0)
        np.testing.assert_allclose(pending, row.dividend_receivable, atol=1e-6, rtol=0)
        shares = t.groupby('code').shareflow.sum().add(e.groupby('code').bonus_shares.sum(), fill_value=0)
        h = holdings[holdings.date.eq(row.date)].set_index('code')
        expected = shares[shares.ne(0)].sort_index()
        np.testing.assert_allclose(expected.to_numpy(dtype=float),
                                   h.shares.reindex(expected.index).to_numpy(dtype=float), atol=1e-8, rtol=0)
        assert set(expected.index) == set(h.index)
        np.testing.assert_allclose(h.value, h.shares*h.close, atol=1e-6, rtol=0)
        np.testing.assert_allclose(cash+pending+h.value.sum(), row.equity, atol=1e-6, rtol=0)
    return dict(strategy=name, months=len(nav), trades=len(trades), corporate_events=len(events),
                ending_equity=float(nav.equity.iloc[-1]), status='cash_shares_nav_reconciled')


if __name__ == '__main__':
    manifest = json.loads((OUT/'account_manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['input_sha256'].items():
        actual = hashlib.sha256((Path(config.PROJECT_DIR)/relative).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError('Input/source fingerprint changed: '+relative)
    source = pd.read_csv(Path(config.OUTPUT_DIR)/'original_architecture/holdings.csv', parse_dates=['date'])
    plans = pd.read_pickle(OUT/'industry_plans.pkl')
    original = plans[plans.strategy.eq('original')]
    for row in source.itertuples():
        assert set(original.loc[original.date.eq(row.date), 'ts_code']) == set(ast.literal_eval(row.holdings))
    print(f'Input/source hashes verified: {len(manifest["input_sha256"])}; original industry sets: {len(source)}')
    results = [verify(name) for name in ('original', 'new_full', 'new_vol')]
    (OUT/'ledger_reconciliation.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
