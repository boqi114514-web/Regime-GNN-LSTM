"""Read-only capital-priority diagnostics for completed nine-month DC accounts.

This does not invoke a selector, optimizer, ledger, or remote data provider.
Execution filters use dated opening prices only. Month-end contribution is a
separate, explicitly retrospective diagnostic and never changes the ranking.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_pipeline.execution_data import ROOT


MAINBOARD = re.compile(r'^(?:600|601|603|605|000|001|002|003)\d{3}\.(?:SH|SZ)$')
BUDGET_NOTE = ('target_max_lots is a per-stock upper bound from the recorded '
               'allocation target, not free cash, residual buying capacity, '
               'or a jointly feasible portfolio; retained/blocked positions '
               'and other allocations can reduce it')


def _required(frame, columns, name):
    if not set(columns).issubset(frame.columns):
        raise ValueError(f'{name}: missing required columns')


def _account_months(account):
    _required(account, ['date', 'profit'], 'account')
    dates = pd.to_datetime(account.date, errors='raise')
    periods = pd.PeriodIndex(dates, freq='M')
    if dates.isna().any() or dates.duplicated().any() or periods.duplicated().any():
        raise ValueError('Account has missing or duplicate month-end dates')
    if not dates.is_monotonic_increasing:
        raise ValueError('Account month ends are not increasing')
    if len(periods) == 0 or not periods.equals(pd.period_range(periods[0], periods[-1], freq='M')):
        raise ValueError('Account months are not consecutive')
    return dates, periods


def _dated_frame(frame, day, columns, name):
    _required(frame, ['ts_code', 'trade_date'] + columns, name)
    result = frame[['ts_code', 'trade_date'] + columns].copy()
    dates = pd.to_datetime(result.trade_date.astype(str), format='%Y%m%d', errors='raise')
    if dates.isna().any() or not dates.eq(day).all() or result.ts_code.isna().any():
        raise ValueError(f'{name}: wrong-date or missing evidence key')
    if result.ts_code.duplicated().any():
        raise ValueError(f'{name}: duplicate code evidence')
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors='raise')
        if not np.isfinite(result[column]).all() or result[column].le(0).any():
            raise ValueError(f'{name}: invalid price evidence')
    if 'up_limit' in columns and result.down_limit.ge(result.up_limit).any():
        raise ValueError(f'{name}: invalid price-limit ordering')
    return result.drop(columns='trade_date')


def monthly_contributions(account, trades, holdings, events):
    """Independently attribute and reconcile exported monthly account profit."""
    dates, _ = _account_months(account)
    _required(trades, ['date', 'code', 'side', 'shares', 'price'], 'trades')
    _required(holdings, ['date', 'code', 'value'], 'holdings')
    _required(events, ['date', 'code', 'cash_entitlement'], 'corporate events')
    t, h, e = trades.copy(), holdings.copy(), events.copy()
    for frame in (t, h, e):
        frame['date'] = pd.to_datetime(frame.date, errors='raise')
    if not t.side.isin(['buy', 'sell']).all():
        raise ValueError('Unknown trade side')
    for frame, columns in ((t, ['shares', 'price']), (h, ['value']), (e, ['cash_entitlement'])):
        for column in columns:
            frame[column] = pd.to_numeric(frame[column], errors='raise').astype(float)
            if not np.isfinite(frame[column]).all():
                raise ValueError('Nonfinite attribution evidence')
    t['flow'] = t.shares * t.price * np.where(t.side.eq('sell'), 1., -1.)
    previous = (dates.iloc[0].to_period('M') - 1).to_timestamp('M')
    if t.date.le(previous).any() or t.date.gt(dates.iloc[-1]).any():
        raise ValueError('Trade falls outside the initially empty account window')
    if h.date.lt(dates.iloc[0]).any():
        raise ValueError('Initial holdings require a separate opening-position attribution')
    result = []
    for month in account.itertuples(index=False):
        end = pd.Timestamp(month.date)
        current = h[h.date.eq(end)].groupby('code').value.sum()
        earlier = h[h.date.eq(previous)].groupby('code').value.sum()
        flow = t[t.date.gt(previous) & t.date.le(end)].groupby('code').flow.sum()
        cash = e[e.date.gt(previous) & e.date.le(end)].groupby('code').cash_entitlement.sum()
        profit = current.subtract(earlier, fill_value=0).add(flow, fill_value=0).add(cash, fill_value=0)
        np.testing.assert_allclose(profit.sum(), float(month.profit), atol=1e-6, rtol=0)
        result.extend(dict(date=end, code=code, profit=float(value)) for code, value in profit.items())
        previous = end
    return pd.DataFrame(result, columns=['date', 'code', 'profit'])


def capital_priority(candidates, allocations, trades, monthly_account,
                     quote_frames, limit_frames, contributions=None):
    """Audit monthly candidates without modifying selection or using future returns.

    quote_frames/limit_frames map actual month-opening session timestamps to
    dated source frames. Actual monthly purchases and month-opening purchases
    are reported separately, so later weekly/replacement buys are not relabeled.
    """
    dates, months = _account_months(monthly_account)
    _required(candidates, ['month', 'signal_date', 'ts_code', 'eligible', 'leadership_score'], 'candidates')
    _required(allocations, ['date', 'target', 'cash', 'retained'], 'allocations')
    _required(trades, ['date', 'code', 'side', 'shares', 'price'], 'trades')
    c, a, t = candidates.copy(), allocations.copy(), trades.copy()
    c['signal_date'] = pd.to_datetime(c.signal_date, errors='raise')
    c['signal_month'] = pd.to_datetime(c.month, errors='raise').dt.to_period('M')
    a['date'] = pd.to_datetime(a.date, errors='raise')
    t['date'] = pd.to_datetime(t.date, errors='raise')
    if c[['ts_code', 'signal_date', 'signal_month']].isna().any().any():
        raise ValueError('Missing candidate key')
    if c.duplicated(['signal_month', 'ts_code']).any():
        raise ValueError('Duplicate monthly candidate')
    if not c.signal_date.dt.to_period('M').eq(c.signal_month).all():
        raise ValueError('Candidate signal date is outside its signal month')
    if not c.eligible.isin([True, False]).all():
        raise ValueError('Candidate eligibility must be complete boolean evidence')
    if not t.side.isin(['buy', 'sell']).all() or a.date.isna().any() or a.date.duplicated().any():
        raise ValueError('Invalid trade sides or duplicate allocation dates')
    for frame, column in ((a, 'target'), (a, 'cash'), (a, 'retained'), (t, 'shares'), (t, 'price')):
        frame[column] = pd.to_numeric(frame[column], errors='raise')
        if not np.isfinite(frame[column]).all() or frame[column].lt(0).any():
            raise ValueError('Invalid allocation/trade number')
    if t.shares.le(0).any() or t.price.le(0).any() or t.shares.mod(1).ne(0).any():
        raise ValueError('Invalid trade shares/prices')
    buys = t[t.side.eq('buy')].copy()
    if buys.shares.mod(100).ne(0).any() or not buys.code.astype(str).map(lambda value: bool(MAINBOARD.fullmatch(value))).all():
        raise ValueError('Non-mainboard or non-whole-lot buy')
    qmap = {pd.Timestamp(key): value for key, value in quote_frames.items()}
    lmap = {pd.Timestamp(key): value for key, value in limit_frames.items()}
    diagnostic = None
    if contributions is not None:
        _required(contributions, ['date', 'code', 'profit'], 'contributions')
        diagnostic = contributions.copy()
        diagnostic['date'] = pd.to_datetime(diagnostic.date, errors='raise')
        diagnostic['profit'] = pd.to_numeric(diagnostic.profit, errors='raise')
        if diagnostic.duplicated(['date', 'code']).any() or not np.isfinite(diagnostic.profit).all():
            raise ValueError('Invalid contribution diagnostic')
    results = []
    for end, month in zip(dates, months):
        allocations_in_month = a[a.date.dt.to_period('M').eq(month)].sort_values('date')
        if allocations_in_month.empty:
            raise ValueError('Missing month-opening allocation')
        allocation = allocations_in_month.iloc[0]
        day = pd.Timestamp(allocation.date)
        quote_days = sorted(key for key in qmap if key.to_period('M') == month)
        if not quote_days or quote_days[0] != day or day not in lmap:
            raise ValueError('Allocation does not match supplied actual month-opening evidence')
        block = c[c.signal_month.eq(month - 1)].copy()
        if block.empty or block.signal_date.nunique() != 1 or not block.signal_date.lt(day).all():
            raise ValueError('Missing/ambiguous completed signal month before execution')
        # Only as-of candidate fields are exported here; future label columns
        # are deliberately not copied from the input candidate frame.
        fields = ['signal_date', 'ts_code', 'eligible', 'leadership_score']
        fields += [column for column in block.columns if column.startswith(('affinity_', 'member_'))]
        fields += [column for column in ('theme_name', 'theme_score', 'theme_phase', 'ind_code',
                  'classification_basis', 'ranking_basis', 'forecast_return', 'forecast_percentile')
                  if column in block.columns]
        block = block[list(dict.fromkeys(fields))].rename(columns={'eligible': 'original_eligible'})
        block['leadership_score'] = pd.to_numeric(block.leadership_score, errors='coerce')
        quotes = _dated_frame(qmap[day], day, ['open'], 'opening quotes')
        limits = _dated_frame(lmap[day], day, ['up_limit', 'down_limit'], 'opening limits')
        block = block.merge(quotes, on='ts_code', how='left', validate='one_to_one')
        block = block.merge(limits, on='ts_code', how='left', validate='one_to_one')
        block['entry_date'], block['attribution_month_end'] = day, end
        block['mainboard'] = block.ts_code.astype(str).map(lambda value: bool(MAINBOARD.fullmatch(value)))
        block['opening_quote_available'] = block.open.notna()
        block['price_limits_available'] = block[['up_limit', 'down_limit']].notna().all(axis=1)
        block['opening_lot_cost'] = (100 * block.open).round(2)
        block['allocation_target'] = float(allocation.target)
        block['allocation_cash_after_buys'] = float(allocation.cash)
        block['allocation_retained_names'] = int(allocation.retained)
        block['target_max_lots'] = np.floor((float(allocation.target) + 1e-7) / block.opening_lot_cost).astype('Int64')
        block['limit_open_executable'] = (block.open.lt(block.up_limit - .005)
                                        & block.open.gt(block.down_limit + .005))
        block['risk_warning_band_excluded'] = ((block.up_limit - block.down_limit)
                                              / ((block.up_limit + block.down_limit) / 2)).le(.15)
        block['finite_score'] = np.isfinite(block.leadership_score)
        block['audited_executable_candidate'] = (block.original_eligible & block.mainboard
            & block.opening_quote_available & block.price_limits_available
            & block.limit_open_executable & ~block.risk_warning_band_excluded
            & block.target_max_lots.ge(1).fillna(False) & block.finite_score)
        block['mainboard_executable_score_rank'] = pd.Series(pd.NA, index=block.index, dtype='Int64')
        ranked = block[block.audited_executable_candidate].sort_values(
            ['leadership_score', 'ts_code'], ascending=[False, True], kind='stable')
        block.loc[ranked.index, 'mainboard_executable_score_rank'] = np.arange(1, len(ranked) + 1)
        month_buys = buys[buys.date.dt.to_period('M').eq(month)]
        entry_buys = month_buys[month_buys.date.eq(day)]
        block['actual_entry_bought_shares'] = block.ts_code.map(entry_buys.groupby('code').shares.sum()).fillna(0).astype(int)
        block['actual_month_bought_shares'] = block.ts_code.map(month_buys.groupby('code').shares.sum()).fillna(0).astype(int)
        block['actual_entry_bought'] = block.actual_entry_bought_shares.gt(0)
        block['actual_month_bought'] = block.actual_month_bought_shares.gt(0)
        if not entry_buys.code.isin(block.ts_code).all():
            raise ValueError('Month-opening purchase absent from candidate audit')
        if (block.actual_entry_bought_shares.gt(block.target_max_lots.fillna(0) * 100)).any():
            raise ValueError('Aggregated purchases exceed per-stock recorded target upper bound')
        for trade in entry_buys.itertuples(index=False):
            evidence = block[block.ts_code.eq(trade.code)].iloc[0]
            if not evidence.audited_executable_candidate:
                raise ValueError('Month-opening purchase fails audited candidate/execution filters')
            np.testing.assert_allclose(trade.price, evidence.open, atol=.005, rtol=0)
            if trade.shares > int(evidence.target_max_lots) * 100:
                raise ValueError('Purchase exceeds per-stock recorded target upper bound')
        block['entry_allocation_explanation'] = np.select([
            block.actual_entry_bought, ~block.original_eligible, ~block.mainboard,
            ~block.opening_quote_available, ~block.price_limits_available,
            ~block.limit_open_executable, block.risk_warning_band_excluded,
            ~block.target_max_lots.ge(1).fillna(False), ~block.finite_score], [
            'actually_bought_at_month_open', 'not_originally_eligible', 'account_board_permission',
            'missing_opening_quote', 'missing_dated_limits', 'opening_at_or_beyond_limit',
            'risk_warning_price_band', 'one_lot_exceeds_recorded_target', 'nonfinite_score'],
            default='passed_audited_filters_not_allocated_optimizer_not_replayed')
        block['diagnostic_month_profit'] = np.nan
        if diagnostic is not None:
            profits = diagnostic[diagnostic.date.eq(end)].set_index('code').profit
            block['diagnostic_month_profit'] = block.ts_code.map(profits)
        results.append(block)
    return pd.concat(results, ignore_index=True).sort_values(['entry_date', 'ts_code']).reset_index(drop=True)


def audit_account(out: Path, variant: str, root: Path = ROOT):
    """Read one completed account, returning candidate diagnostics and input proof."""
    out, root = Path(out), Path(root)
    status_path = out / f'run_status_{variant}.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    if (status.get('status') != 'completed' or status.get('variant') != variant
            or status.get('account_complete') is not True or status.get('months') != 9):
        raise ValueError('Only status-completed nine-month accounts may be audited')
    files = {label: out / f'{label}_{variant}.csv' for label in
             ('allocations', 'trades', 'account', 'holdings', 'corporate_events')}
    candidate_path = out / f'candidate_audit_{variant}.pkl'
    inputs = [status_path, candidate_path] + list(files.values()) + [Path(__file__).resolve()]
    frames = {label: pd.read_csv(path) for label, path in files.items()}
    account_dates, months = _account_months(frames['account'])
    if len(months) != 9:
        raise ValueError('Exported account does not contain nine months')
    qframes, lframes = {}, {}
    allocations = frames['allocations'].copy()
    allocations['date'] = pd.to_datetime(allocations.date, errors='raise')
    for account_end, month in zip(account_dates, months):
        calendar_path = root / 'boundaries' / f'calendar_{month}.pkl'
        calendar = pd.read_pickle(calendar_path)
        _required(calendar, ['cal_date', 'is_open'], 'calendar')
        calendar_dates = pd.to_datetime(calendar.cal_date.astype(str), format='%Y%m%d', errors='raise')
        open_values = pd.to_numeric(calendar.is_open, errors='raise')
        if calendar_dates.isna().any() or not calendar_dates.dt.to_period('M').eq(month).all() or not open_values.isin([0, 1]).all():
            raise ValueError('Calendar is not valid dated monthly evidence')
        # The final account month may end at the last available data session.
        # Require a complete calendar prefix through its recorded endpoint;
        # future sessions are not required or used to determine execution day.
        expected_dates = pd.date_range(month.start_time.normalize(), account_end.normalize(), freq='D')
        if not set(expected_dates).issubset(set(calendar_dates)):
            raise ValueError('Calendar does not cover every day through the account endpoint')
        flags_by_date = pd.DataFrame({'date': calendar_dates, 'open': open_values}).groupby('date').open.nunique()
        if flags_by_date.gt(1).any():
            raise ValueError('Calendar has conflicting open flags')
        sessions = calendar_dates[open_values.eq(1) & calendar_dates.le(account_end)]
        if sessions.empty:
            raise ValueError('Monthly calendar has no open session')
        day = pd.Timestamp(sessions.min())
        month_allocations = allocations[allocations.date.dt.to_period('M').eq(month)]
        if month_allocations.empty or month_allocations.date.min() != day:
            raise ValueError('Allocation does not occur on actual first trading session')
        quote_path = root / 'boundaries' / f'daily_{day:%Y%m%d}.pkl'
        limit_path = root / 'boundaries' / f'stk_limit_{day:%Y%m%d}.pkl'
        qframes[day], lframes[day] = pd.read_pickle(quote_path), pd.read_pickle(limit_path)
        inputs.extend([calendar_path, quote_path, limit_path])
    contributions = monthly_contributions(frames['account'], frames['trades'], frames['holdings'], frames['corporate_events'])
    result = capital_priority(pd.read_pickle(candidate_path), frames['allocations'], frames['trades'],
                              frames['account'], qframes, lframes, contributions)
    result.insert(0, 'strategy', variant)
    proof = dict(strategy=variant, account_directory=str(out.resolve()), status='completed_account_priority_audited',
        months=9, start=str(result.entry_date.min().date()), end=str(result.attribution_month_end.max().date()),
        candidate_rows=len(result), month_open_bought_rows=int(result.actual_entry_bought.sum()),
        passed_filters_unallocated_rows=int((result.audited_executable_candidate & ~result.actual_entry_bought).sum()),
        missing_quote_rows=int((~result.opening_quote_available).sum()), missing_limit_rows=int((~result.price_limits_available).sum()),
        attribution_reconciled=True, rank_order='leadership_score descending, ts_code ascending; original eligible mainboard executable affordable candidates only',
        budget_note=BUDGET_NOTE, future_diagnostic='diagnostic_month_profit is exported account attribution only; no unheld-stock future return or optimizer replay',
        inputs={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs})
    return result, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accounts', nargs='+', required=True, help='directory=variant pairs')
    parser.add_argument('--output-dir', type=Path, default=Path('results/dc_priority_comparison'))
    args = parser.parse_args()
    accounts, proofs = [], []
    for spec in args.accounts:
        directory, variant = spec.rsplit('=', 1)
        result, proof = audit_account(Path(directory), variant)
        accounts.append(result)
        proofs.append(proof)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / 'capital_priority.csv'
    pd.concat(accounts, ignore_index=True).to_csv(output, index=False)
    manifest = dict(accounts=proofs, budget_note=BUDGET_NOTE,
                    output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    (args.output_dir / 'capital_priority_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(pd.DataFrame([{key: value for key, value in proof.items() if key in
        ('strategy', 'candidate_rows', 'month_open_bought_rows', 'passed_filters_unallocated_rows')} for proof in proofs]).to_string(index=False))


if __name__ == '__main__':
    main()
