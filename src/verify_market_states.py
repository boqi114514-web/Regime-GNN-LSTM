"""Independent daily cash/share replay of exported market-state experiments.

Does not call the trading ledger or rerun the selection/exit policy. Valuation
uses the same source quotes; confirmed suspension marks remain shared evidence.
"""
import json

import numpy as np
import pandas as pd


def verify_reentry_timing(trades, dates):
    """Independently check exported reentry timestamps, not the signal function."""
    if 'reason' not in trades: return 0
    entries=trades[trades.side.eq('buy') & trades.reason.eq('daily_reentry')].copy()
    if entries.empty: return 0
    entries['signal_date']=pd.to_datetime(entries.signal_date)
    positions={day:i for i,day in enumerate(dates)}
    firsts=set(pd.Series(dates,index=dates).groupby(dates.to_period('M')).min())
    for row in entries.itertuples():
        idx=positions[row.date]
        if idx==0 or row.signal_date!=dates[idx-1] or row.date in firsts:
            raise AssertionError('Reentry not based on previous session close')
        sales=trades[trades.side.eq('sell') & trades.date.le(row.date)]
        if sales.empty or idx-positions[sales.date.max()]<5:
            raise AssertionError('Reentry violates five-session cooldown')
    return len(entries)


def verify_weekly_reentry_timing(trades, dates):
    """Check weekly idle-cash entries independently from the entry scheduler.

    A completed W-SUN week is observed at its last actual session's close;
    execution is allowed only at the first actual session of the next week.
    Month-opening trades retain the separate monthly rebalance convention.
    """
    if 'reason' not in trades:
        return 0
    entries = trades[trades.side.eq('buy') & trades.reason.eq('weekly_reentry')].copy()
    if entries.empty:
        return 0
    dates = pd.DatetimeIndex(dates)
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise AssertionError('Invalid weekly verification session calendar')
    positions = {day: i for i, day in enumerate(dates)}
    firsts = set(pd.Series(dates, index=dates).groupby(dates.to_period('M')).min())
    entries['date'] = pd.to_datetime(entries.date, errors='raise')
    entries['signal_date'] = pd.to_datetime(entries.signal_date, errors='raise')
    sales = pd.to_datetime(trades.loc[trades.side.eq('sell'), 'date'], errors='raise')
    for row in entries.itertuples():
        if row.date not in positions:
            raise AssertionError('Weekly entry not on a verified trading session')
        idx = positions[row.date]
        if idx == 0 or row.signal_date != dates[idx-1] or row.date in firsts:
            raise AssertionError('Weekly entry not based on previous session close')
        if row.date.to_period('W-SUN') == row.signal_date.to_period('W-SUN'):
            raise AssertionError('Weekly entry not at first session after completed week')
        previous_sales = sales[sales.le(row.date)]
        if len(previous_sales):
            most_recent = previous_sales.max()
            if most_recent not in positions or idx-positions[most_recent] < 2:
                raise AssertionError('Weekly entry violates two-session sale cooldown')
    return len(entries)


def verify_daily(out, root, names):
    trades = {n: pd.read_csv(out/f'trades_{n}.csv', parse_dates=['date']) for n in names}
    codes = set().union(*(set(t.code) for t in trades.values()))
    raw = pd.read_pickle(out/'daily.pkl')
    raw = raw[raw.ts_code.isin(codes) & raw.volume.gt(0)]
    prices = raw.pivot(index='date', columns='ts_code', values='close')
    del raw
    sessions = json.loads((root/'boundaries/sessions.json').read_text(encoding='utf-8'))
    for date in sorted({s[k] for s in sessions for k in ('first', 'last')}):
        day = pd.Timestamp(date)
        q = pd.read_pickle(root/'boundaries'/f'daily_{day:%Y%m%d}.pkl').set_index('ts_code')
        available = q.index.intersection(prices.columns)
        prices.loc[day, available] = q.loc[available, 'close']
    checks = []
    for name, t in trades.items():
        daily = pd.read_csv(out/f'daily_nav_{name}.csv', parse_dates=['date']).set_index('date')
        reentries=verify_reentry_timing(t,daily.index)
        weekly_reentries=verify_weekly_reentry_timing(t,daily.index)
        events = pd.read_csv(out/f'corporate_events_{name}.csv', parse_dates=['date', 'payment_date'])
        for col in ('cash_entitlement', 'bonus_shares'):
            events[col] = pd.to_numeric(events[col], errors='raise').astype(float)
            if not np.isfinite(events[col]).all():
                raise ValueError('Invalid corporate event '+col)
        p = prices.copy()
        marks_path = out/f'suspension_marks_{name}.csv'
        if marks_path.stat().st_size > 5:
            marks = pd.read_csv(marks_path, parse_dates=['date'])
            for row in marks[marks.field.eq('close')].itertuples():
                if row.reason == 'local_quote_gap_repaired_from_remote':
                    q = pd.read_pickle(row.source)
                    selected = q[q.ts_code.eq(row.code) & pd.to_datetime(q.trade_date).eq(row.date)]
                    if len(selected) != 1:
                        raise AssertionError('Ambiguous repaired valuation quote')
                    price = float(selected.close.iloc[0])
                else:
                    if 'suspension' not in row.reason:
                        raise AssertionError('Unrecognized valuation evidence')
                    price = float(row.mark)
                p.loc[row.date, row.code] = price
        t = t.copy()
        t['cashflow'] = t.shares*t.price*np.where(t.side.eq('sell'), 1., -1.)
        t['shareflow'] = t.shares*np.where(t.side.eq('buy'), 1., -1.)
        max_cash_error = max_nav_error = 0.
        for date, row in daily.iterrows():
            history = t[t.date.le(date)]
            e = events[events.date.le(date)]
            paid = e.loc[e.payment_date.le(date), 'cash_entitlement'].sum()
            receivable = e.loc[e.payment_date.gt(date), 'cash_entitlement'].sum()
            cash = 25000.+history.cashflow.sum()+paid
            if cash<-.01: raise AssertionError('Negative cash / borrowing')
            shares = history.groupby('code').shareflow.sum().add(e.groupby('code').bonus_shares.sum(), fill_value=0)
            if shares.lt(0).any():
                raise AssertionError('Negative inventory')
            shares = shares[shares.gt(0)]
            close = p.loc[date].reindex(shares.index)
            if close.isna().any() or close.le(0).any():
                raise AssertionError(f'Missing daily valuation: {name} {date}')
            nav = cash+receivable+(shares*close).sum()
            max_cash_error = max(max_cash_error, abs(cash-row.cash))
            max_nav_error = max(max_nav_error, abs(nav-row.equity))
            np.testing.assert_allclose([cash, nav], [row.cash, row.equity], atol=1e-6, rtol=0)
            # Sale inventory must predate today's purchases (A-share T+1).
            sells = t[t.date.eq(date) & t.side.eq('sell')].groupby('code').shares.sum()
            start = t[t.date.lt(date)].groupby('code').shareflow.sum().add(e.groupby('code').bonus_shares.sum(), fill_value=0)
            if (sells > start.reindex(sells.index).fillna(0)+1e-8).any():
                raise AssertionError('Same-day purchased shares sold')
        checks.append(dict(strategy=name, days=len(daily), max_cash_error=max_cash_error,
                           max_nav_error=max_nav_error, status='independent_daily_cash_shares_nav_reconciled',
                           verified_daily_reentry_trades=reentries,
                           verified_weekly_reentry_trades=weekly_reentries,
                           valuation='source quotes plus documented suspension marks, no invented fills'))
    return checks
