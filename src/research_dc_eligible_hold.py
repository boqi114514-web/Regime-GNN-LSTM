"""One isolated monthly continuation hypothesis using frozen dated candidates.

Keep an existing position at a monthly rebalance only when it still passes
the original complete eligibility rule and has no pending risk exit. All
ranking, new entries, cash limits, stops and the original engine are retained.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from data_pipeline.execution_data import ROOT
import research_dc_themes as themes
import research_market_states as states
import verify_small_account as monthly_verifier
from verify_market_states import verify_daily


ENGINE_VARIANT = 'dc_member_relative_leader_2026_retry'
VARIANT = 'dc_member_relative_eligible_hold_20261003'
PROTOCOL = dict(
    variant=VARIANT, engine_artifact_alias=ENGINE_VARIANT,
    hypothesis='Continue monthly holdings iff the original frozen signal row has eligible=True and no pending risk exit',
    inputs='All original verification inputs hash-checked; exact frozen candidates/plans, no scoring changes or future label use',
    unchanged='Original mainboard rank-first 100-share purchases, monthly new entries, no topups/fees, 25000 initial cash and capped new allocation',
    risk='Original 8% hard stop; +20% activation and 15% advance trailing stop; completed close then next executable open',
    continuation='Apply the same complete monthly entry eligibility to every existing stock and every month; pending exits always take priority',
    capital='Retained appreciated market value may exceed 25000; new purchase amount <= min(available cash,max(0,min(opening equity,25000)*exposure-retained opening value))',
    research_status='One declared hypothesis on previously inspected 2026 history; not an untouched holdout',
    period='2026-01-01 through 2026-09-24',
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish_manifest(payload, path):
    path = Path(path)
    temporary = path.with_suffix('.staging.json')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def continues(row, entry_phase, has_pending_exit, advance_only=False):
    if row is None or has_pending_exit:
        return False
    value = row.get('eligible', False)
    return isinstance(value, (bool, np.bool_)) and bool(value)


def verify_frozen_inputs(source):
    source = Path(source)
    manifest = json.loads((source/'verification.json').read_text(encoding='utf-8'))
    for filename, expected in manifest['inputs'].items():
        if sha256(filename) != expected:
            raise ValueError('Original verified input changed: '+filename)
    for filename, expected in manifest['outputs'].items():
        if sha256(source/filename) != expected:
            raise ValueError('Original verified account output changed: '+filename)
    for filename in ('candidates.pkl', 'plans.pkl', 'daily.pkl'):
        if str((source/filename).resolve()) not in manifest['inputs']:
            raise ValueError('Original verification omitted required frozen input: '+filename)
    candidates = pd.read_pickle(source/'candidates.pkl')
    plans = pd.read_pickle(source/'plans.pkl')
    if candidates.duplicated(['month', 'ts_code']).any():
        raise ValueError('Ambiguous original monthly candidates')
    if (candidates.date.gt(candidates.signal_date).any()
            or candidates.signal_date.gt(candidates.month).any()
            or not candidates.date.dt.to_period('M').eq(candidates.month.dt.to_period('M')).all()
            or candidates.liquidity_last_day.gt(candidates.signal_date).any()
            or plans.signal_date.gt(plans.date).any()):
        raise ValueError('Frozen monthly signal contains future observations')
    if set(candidates.month.dt.to_period('M')) != set(pd.period_range('2025-12', '2026-08', freq='M')):
        raise ValueError('Frozen candidates do not cover every required signal month')
    return candidates, plans, manifest


@contextmanager
def continuation_engine(out, candidates, plans):
    old = states.OUT, states.HOLD_VARIANTS, states.may_continue_position, themes.prepare
    try:
        states.OUT = Path(out)
        states.HOLD_VARIANTS = states.HOLD_VARIANTS+(ENGINE_VARIANT,)
        states.may_continue_position = continues

        def frozen_prepare(folder, name):
            if Path(folder).resolve() != Path(out).resolve() or name != ENGINE_VARIANT:
                raise ValueError('Frozen monthly continuation used with another output/variant')
            return candidates.loc[candidates.eligible.eq(True)].copy(), plans.copy()

        themes.prepare = frozen_prepare
        yield
    finally:
        states.OUT, states.HOLD_VARIANTS, states.may_continue_position, themes.prepare = old


def check_capital(equity, exposure, cash_after_sales, retained_value, buy_cost):
    """Independent rule for appreciated retention, without relaxing new buys."""
    if not np.isfinite([equity, exposure, cash_after_sales, retained_value, buy_cost]).all():
        raise AssertionError('Nonfinite capital reconstruction')
    if min(equity, cash_after_sales, retained_value, buy_cost) < -.005 or not 0 <= exposure <= 1:
        raise AssertionError('Invalid reconstructed capital or exposure')
    target = min(equity, 25000.)*exposure
    available = min(cash_after_sales, max(0., target-retained_value))
    if buy_cost > available+.005:
        raise AssertionError('New purchases exceed cash/residual capped budget')
    return target, available


def audit_account(out):
    """Independent cash/share reconciliation plus opening-capital reconstruction."""
    out = Path(out)
    old = monthly_verifier.OUT
    monthly_verifier.OUT = out
    try:
        monthly = monthly_verifier.verify(ENGINE_VARIANT, expected_months=9, end_date='2026-09-24')
    finally:
        monthly_verifier.OUT = old
    daily_check = verify_daily(out, ROOT, [ENGINE_VARIANT])[0]
    trades = pd.read_csv(out/f'trades_{ENGINE_VARIANT}.csv', parse_dates=['date', 'signal_date'])
    events = pd.read_csv(out/f'corporate_events_{ENGINE_VARIANT}.csv', parse_dates=['date', 'payment_date'])
    allocations = pd.read_csv(out/f'allocations_{ENGINE_VARIANT}.csv', parse_dates=['date'])
    candidates = pd.read_pickle(out/'candidates.pkl')
    plans = pd.read_pickle(out/'plans.pkl')
    continuations = pd.read_csv(out/f'continuations_{ENGINE_VARIANT}.csv', parse_dates=['date', 'signal_date'])
    signals = pd.read_csv(out/f'exit_signals_{ENGINE_VARIANT}.csv', parse_dates=['date'])
    trades['cashflow'] = trades.shares*trades.price*np.where(trades.side.eq('sell'), 1., -1.)
    trades['shareflow'] = trades.shares*np.where(trades.side.eq('buy'), 1., -1.)
    for column in ('cash_entitlement', 'bonus_shares'):
        events[column] = pd.to_numeric(events[column], errors='raise').astype(float)
    proofs, bound_sources = [], set()
    for allocation in allocations.itertuples():
        day = allocation.date
        prior = trades[trades.date.lt(day)]
        morning_events = events[events.date.le(day)]
        paid = morning_events.loc[morning_events.payment_date.le(day), 'cash_entitlement'].sum()
        receivable = morning_events.loc[morning_events.payment_date.gt(day), 'cash_entitlement'].sum()
        before = prior.groupby('code').shareflow.sum().add(morning_events.groupby('code').bonus_shares.sum(), fill_value=0)
        before = before[before.gt(0)]
        today = trades[trades.date.eq(day)]
        sold = today[today.side.eq('sell')].groupby('code').shares.sum()
        retained = before.subtract(sold, fill_value=0)
        if retained.lt(0).any():
            raise AssertionError('Opening sales exceed available inventory')
        retained = retained[retained.gt(0)]
        quote_path = ROOT/'boundaries'/f'daily_{day:%Y%m%d}.pkl'
        limit_path = ROOT/'boundaries'/f'stk_limit_{day:%Y%m%d}.pkl'
        quotes = pd.read_pickle(quote_path).set_index('ts_code')
        limits = pd.read_pickle(limit_path).set_index('ts_code')
        bound_sources.update((quote_path, limit_path))
        if quotes.index.has_duplicates or limits.index.has_duplicates:
            raise AssertionError('Duplicate opening evidence')
        quote_days = pd.to_datetime(quotes.trade_date.astype(str), format='%Y%m%d', errors='raise')
        limit_days = pd.to_datetime(limits.trade_date.astype(str), format='%Y%m%d', errors='raise')
        if not quote_days.eq(day).all() or not limit_days.eq(day).all():
            raise AssertionError('Wrong-date opening evidence')
        before_prices = quotes.open.reindex(before.index)
        retained_prices = quotes.open.reindex(retained.index)
        if before_prices.isna().any() or retained_prices.isna().any():
            raise AssertionError('Unmarked opening holdings')
        cash_before = 25000.+prior.cashflow.sum()+paid
        equity = cash_before+receivable+(before*before_prices).sum()
        cash_after_sales = cash_before+today.loc[today.side.eq('sell'), 'cashflow'].sum()
        retained_value = float((retained*retained_prices).sum())
        buys = today[today.side.eq('buy')]
        buy_cost = float((buys.shares*buys.price).sum())
        for trade in today.itertuples():
            if abs(float(quotes.loc[trade.code, 'open'])-trade.price) > .005:
                raise AssertionError('Trade does not use actual opening quote')
            if trade.side == 'buy':
                row = limits.loc[trade.code]
                if not float(row.down_limit)+.005 < trade.price < float(row.up_limit)-.005:
                    raise AssertionError('New purchase at official price limit')
        exposure = plans.loc[plans.date.dt.to_period('M').eq(day.to_period('M')-1), 'risk_exposure']
        if exposure.empty or exposure.nunique() != 1:
            raise AssertionError('Missing or ambiguous original monthly exposure')
        target, available = check_capital(equity, float(exposure.iloc[0]), cash_after_sales, retained_value, buy_cost)
        np.testing.assert_allclose([allocation.equity, allocation.target, allocation.invested, allocation.cash],
                                   [equity, target, retained_value+buy_cost, cash_after_sales-buy_cost], atol=1e-6, rtol=0)
        if allocation.retained != len(retained):
            raise AssertionError('Retained position count mismatch')
        proofs.append(dict(date=str(day.date()), opening_equity=float(equity), target=target,
                           retained_value=retained_value, new_buy_cost=buy_cost, available_new_capital=available,
                           appreciated_retention_over_target=bool(retained_value > target+.005)))
    for continued in continuations.itertuples():
        signal = candidates[candidates.month.dt.to_period('M').eq(continued.date.to_period('M')-1)
                            & candidates.ts_code.eq(continued.code)]
        if len(signal) != 1 or not bool(signal.eligible.iloc[0]) or not signal.signal_date.iloc[0] < continued.date:
            raise AssertionError('Retained holding fails original completed-signal eligibility')
        if continued.signal_date != signal.date.iloc[0]:
            raise AssertionError('Continuation uses a different candidate signal')
        for triggered in signals[signals.code.eq(continued.code)&signals.date.lt(continued.date)].itertuples():
            sold_since = trades[trades.code.eq(continued.code)&trades.side.eq('sell')
                                &trades.date.gt(triggered.date)&trades.date.le(continued.date)]
            if sold_since.empty:
                raise AssertionError('Continuation cancelled a pending risk exit')
    account = pd.read_csv(out/f'account_{ENGINE_VARIANT}.csv', parse_dates=['date'])
    daily = pd.read_csv(out/f'daily_nav_{ENGINE_VARIANT}.csv')
    nav = np.r_[25000., daily.equity.to_numpy()]
    metrics = dict(variant=VARIANT, total_profit=float(account.profit.sum()),
                   ending_equity=float(account.equity.iloc[-1]),
                   daily_drawdown_pct=float(100*(1-nav/np.maximum.accumulate(nav)).max()),
                   target_hit_months=int(account.profit.ge(7500.-1e-7).sum()),
                   continuations=len(continuations), months=len(account), days=len(daily))
    source_files = [out/f'{label}_{ENGINE_VARIANT}.csv' for label in
                    ('account', 'daily_nav', 'trades', 'corporate_events', 'allocations', 'continuations', 'exit_signals')]
    source_files += [out/'candidates.pkl', out/'plans.pkl', out/'daily.pkl', out/'eligible_hold_protocol.json',
                     Path(__file__), Path(monthly_verifier.__file__), Path(verify_daily.__code__.co_filename)]
    source_files += list(bound_sources)
    proof = dict(status='passed', monthly_reconciliation=monthly, daily_reconciliation=daily_check,
                 capital_reconstruction=proofs, continuation_eligibility_checked=True,
                 pending_exit_preservation_checked=True, metrics=metrics,
                 input_sha256={str(path.resolve()):sha256(path) for path in source_files})
    publish_manifest(proof, out/'eligible_hold_independent_checks.json')
    account.assign(strategy=VARIANT).to_csv(out/'monthly_results.csv', index=False)
    pd.DataFrame([metrics]).to_csv(out/'metrics.csv', index=False)
    return proof


def run(source, out, offline=True):
    source, out = Path(source), Path(out)
    if source.resolve() == out.resolve() or (out/'eligible_hold_independent_checks.json').exists():
        raise ValueError('Use a separate uncompleted continuation output')
    candidates, plans, original = verify_frozen_inputs(source)
    out.mkdir(parents=True, exist_ok=True)
    for name in ('daily.pkl', 'candidates.pkl', 'plans.pkl', 'dc_theme_protocol.json'):
        shutil.copy2(source/name, out/name)
    for name in ('scoped_actions', 'exit_quotes', 'replacement_limits', 'dividend_repairs', 'suspension_events'):
        if (source/name).exists():
            shutil.copytree(source/name, out/name, dirs_exist_ok=True)
    if (source/'verified_suspensions.json').exists():
        shutil.copy2(source/'verified_suspensions.json', out/'verified_suspensions.json')
    protocol = dict(PROTOCOL, frozen_source=str(source.resolve()),
                    original_verification_sha256=sha256(source/'verification.json'),
                    validated_original_inputs=len(original['inputs']),
                    original_input_sha256=original['inputs'], source_sha256=sha256(__file__))
    publish_manifest(protocol, out/'eligible_hold_protocol.json')
    status_path = out/f'run_status_{ENGINE_VARIANT}.json'
    status = dict(variant=VARIANT, engine_artifact_alias=ENGINE_VARIANT, status='running', account_complete=False)
    publish_manifest(status, status_path)
    # The frozen engine writes its historical hold-protocol description. That
    # description is replaced below by the actual isolated wrapper protocol.
    legacy = out/'trend_holding_protocol.json'
    if legacy.exists():
        legacy.unlink()  # Only this wrapper's own generated retry metadata.
    try:
        with continuation_engine(out, candidates, plans):
            result = states.run(ENGINE_VARIANT, offline=offline)
        status.update(status='completed', account_complete=True, months=len(result),
                      ending_equity=float(result.equity.iloc[-1]))
    except BaseException as exc:
        status.update(status='incomplete', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        publish_manifest(protocol, legacy)
        publish_manifest(status, status_path)
    proof = audit_account(out)
    print(json.dumps(proof['metrics'], ensure_ascii=False), flush=True)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/dc_member_relative_leader_2026_research'))
    parser.add_argument('--out', type=Path, default=Path('results/dc_member_relative_eligible_hold_20261003'))
    parser.add_argument('--online', action='store_true')
    args = parser.parse_args()
    run(args.source, args.out, offline=not args.online)
