"""Read-only applicability check: 8% trailing distance on an existing account.

Preserve the existing 20% activation and hard stop. Reconstruct adjusted
entry/peak anchors from raw daily closes and actual corporate actions. A zero
signal difference proves this single stop change leaves the original path
unchanged; if any signal differs, this script does not claim a counterfactual
account return and a full account replay is required.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from research_scoped_actions import validate_history
from data_pipeline.execution_data import ROOT


NAME = 'dc_member_relative_leader_2026_retry'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stop_reason(close, entry, peak, phase, advance_trail=.15):
    if not np.isfinite([close, entry, peak]).all() or min(close, entry, peak) <= 0:
        raise ValueError('Stop anchors and closes must be finite positive prices')
    if close <= .92*entry+1e-8:
        return 'hard_stop'
    if peak >= 1.20*entry-1e-8:
        distance = advance_trail if phase == 'advance' else .10
        if close <= (1-distance)*peak+1e-8:
            return 'trailing_profit'
    if phase != 'advance' and close >= 1.30*entry-1e-8:
        return 'range_take_profit'
    return None


def scan_holding(quotes, actions, entry_price, phase):
    quotes = quotes.sort_values('date').copy()
    if quotes.empty or quotes.date.duplicated().any():
        raise ValueError('Unique actual held-session closes required')
    entry = peak = float(entry_price)
    first_activation = None
    lowest_active_drawdown = 0.
    maximum_gain = 0.
    baseline = alternative = None
    observations = []
    quote_map = quotes.set_index('date').close
    actions = actions[actions.ex_date.between(quotes.date.min(), quotes.date.max())] if len(actions) else actions
    dates = sorted(set(quotes.date) | set(actions.ex_date if len(actions) else []))
    for day in dates:
        changes = actions[actions.ex_date.eq(day)] if len(actions) else actions
        if len(changes):
            entry = (entry-float(changes.cash.sum()))/(1+float(changes.stock.sum()))
            peak = (peak-float(changes.cash.sum()))/(1+float(changes.stock.sum()))
        if day not in quote_map.index:
            continue
        close = float(quote_map.loc[day])
        peak = max(peak, close)
        gain = peak/entry-1
        maximum_gain = max(maximum_gain, gain)
        active = peak >= 1.20*entry-1e-8
        drawdown = close/peak-1
        if active:
            first_activation = first_activation or day
            lowest_active_drawdown = min(lowest_active_drawdown, drawdown)
        old = stop_reason(close, entry, peak, phase, .15)
        new = stop_reason(close, entry, peak, phase, .08)
        if old is not None and baseline is None:
            baseline = dict(date=str(day.date()), reason=old)
        if new is not None and alternative is None:
            alternative = dict(date=str(day.date()), reason=new)
        observations.append(dict(date=str(day.date()), close=close, adjusted_entry=entry,
                                 adjusted_peak=peak, trailing_active=bool(active), drawdown_from_peak=drawdown))
    return dict(held_sessions=len(quotes), entry_price=float(entry_price), phase=phase,
                first_held_close_date=str(quotes.date.min().date()), last_held_close_date=str(quotes.date.max().date()),
                actual_maximum_raw_close=float(quotes.close.max()), maximum_adjusted_peak_gain=maximum_gain,
                ending_adjusted_entry=entry, ending_adjusted_peak=peak,
                trailing_activation_date=None if first_activation is None else str(first_activation.date()),
                minimum_drawdown_after_activation=lowest_active_drawdown,
                original_first_risk_signal=baseline, proposed_first_risk_signal=alternative,
                signal_changed=baseline != alternative, observations=observations)


def audit(source, output):
    source, output = Path(source), Path(output)
    if source.resolve() == output.parent.resolve():
        raise ValueError('Write applicability evidence outside the original account')
    verification_path = source/'verification.json'
    verified = json.loads(verification_path.read_text(encoding='utf-8'))
    raw_inputs = [source/'daily.pkl', source/'candidates.pkl',
                  Path(__file__).resolve().parents[1]/'src/research_market_states.py',
                  Path(__file__).resolve().parents[1]/'src/research_scoped_actions.py']
    for path in raw_inputs:
        if str(path.resolve()) not in verified['inputs'] or sha256(path) != verified['inputs'][str(path.resolve())]:
            raise ValueError('Original input lacks matching verified provenance: '+str(path))
    output_names = [f'{prefix}_{NAME}.csv' for prefix in ('trades', 'exit_signals', 'daily_nav', 'account')]
    for name in output_names:
        if sha256(source/name) != verified['outputs'][name]:
            raise ValueError('Original account output changed: '+name)
    trades = pd.read_csv(source/f'trades_{NAME}.csv', parse_dates=['date', 'signal_date'])
    signals = pd.read_csv(source/f'exit_signals_{NAME}.csv', parse_dates=['date'])
    nav = pd.read_csv(source/f'daily_nav_{NAME}.csv', parse_dates=['date'])
    daily = pd.read_pickle(source/'daily.pkl')
    candidates = pd.read_pickle(source/'candidates.pkl')
    positions = []
    source_files = set(raw_inputs+[verification_path]+[source/name for name in output_names])
    session_path = ROOT/'boundaries/sessions.json'
    if sha256(session_path) != verified['inputs'].get(str(session_path.resolve())):
        raise ValueError('Original boundary session provenance changed')
    sessions = json.loads(session_path.read_text(encoding='utf-8'))
    boundaries = {pd.Timestamp(item[key]) for item in sessions for key in ('first', 'last')}
    source_files.add(session_path)
    for buy in trades[trades.side.eq('buy')].itertuples():
        sells = trades[trades.code.eq(buy.code)&trades.side.eq('sell')&trades.date.gt(buy.date)]
        sale = sells.iloc[0] if len(sells) else None
        end = sale.date if sale is not None else nav.date.max()+pd.Timedelta(days=1)
        quotes = daily[daily.ts_code.eq(buy.code)&daily.date.ge(buy.date)&daily.date.lt(end)&daily.volume.gt(0)].copy()
        actual_days = nav.loc[nav.date.ge(buy.date)&nav.date.lt(end), 'date']
        if set(quotes.date) != set(actual_days):
            raise ValueError('Held interval has missing quotes; suspension-aware reconstruction required')
        for day in set(quotes.date) & boundaries:
            boundary_path = ROOT/'boundaries'/f'daily_{day:%Y%m%d}.pkl'
            if sha256(boundary_path) != verified['inputs'].get(str(boundary_path.resolve())):
                raise ValueError('Original monthly boundary quote provenance changed')
            boundary = pd.read_pickle(boundary_path)
            selected = boundary[boundary.ts_code.eq(buy.code)]
            if len(selected) != 1 or str(selected.trade_date.iloc[0]) != f'{day:%Y%m%d}':
                raise ValueError('Missing or wrong-date actual boundary quote')
            quotes.loc[quotes.date.eq(day), 'close'] = float(selected.close.iloc[0])
            source_files.add(boundary_path)
        row = candidates[candidates.ts_code.eq(buy.code)&candidates.month.eq(buy.signal_date)]
        if len(row) != 1:
            raise ValueError('Missing exact original signal phase')
        action_path = source/'scoped_actions'/f'{buy.code}.pkl'
        action_meta = action_path.with_suffix('.json')
        for path in (action_path, action_meta):
            if sha256(path) != verified['inputs'].get(str(path.resolve())):
                raise ValueError('Original scoped action provenance changed')
        actions = validate_history(pd.read_pickle(action_path), buy.code, '2026-01-01', '2026-09-24')
        source_files.update((action_path, action_meta))
        result = scan_holding(quotes, actions, buy.price, row.phase.iloc[0])
        recorded = signals[signals.code.eq(buy.code)&signals.date.ge(buy.date)&signals.date.lt(end)]
        expected = None if recorded.empty else dict(date=str(recorded.date.iloc[0].date()), reason=recorded.reason.iloc[0])
        if len(recorded) > 1 or result['original_first_risk_signal'] != expected:
            raise AssertionError('Reconstructed original risk signal differs from original export')
        result.update(code=buy.code, buy_date=str(buy.date.date()), shares=int(buy.shares),
                      actual_exit=None if sale is None else dict(date=str(sale.date.date()), reason=sale.reason, price=float(sale.price)))
        positions.append(result)
    changed = [row for row in positions if row['signal_changed']]
    result = dict(status='complete', source=str(source.resolve()), account_variant=NAME,
                  hypothesis='Only advance trailing distance .15 -> .08; activation +20%, hard stop -8%, entry/exit execution and all selection unchanged',
                  positions=len(positions), activated_positions=sum(row['trailing_activation_date'] is not None for row in positions),
                  original_risk_signals=sum(row['original_first_risk_signal'] is not None for row in positions),
                  changed_first_signals=len(changed),
                  proposed_additional_signals=sum(row['signal_changed'] and row['proposed_first_risk_signal'] is not None for row in positions),
                  conclusion=('Zero signal change: same full-account trading path under this single isolated stop alteration'
                              if not changed else 'Signals change: a complete account replay is needed; no new return is claimed here'),
                  holdings=positions, input_sha256={str(path.resolve()):sha256(path) for path in sorted(source_files)},
                  audit_source_sha256=sha256(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('positions', 'activated_positions', 'original_risk_signals',
                                                'changed_first_signals', 'proposed_additional_signals', 'conclusion')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/dc_member_relative_leader_2026_research'))
    parser.add_argument('--out', type=Path, default=Path('results/dc_leader_capital_review/trail8_applicability_20261003.json'))
    args = parser.parse_args()
    audit(args.source, args.out)
