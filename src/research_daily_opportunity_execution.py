"""Standalone causal daily opportunity account; historical accounts untouched.

Orders depend on the preceding completed exchange session's predictions. All
fills use the subsequent opening and observed official limits, never that day's
high/low, close, or volume. Corporate actions use the existing validated ledger.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

import small_account_backtest as engine
from data_pipeline.execution_data import ROOT, validate_rows
from data_pipeline.tushare_config import get_pro
from research_scoped_actions import load_scoped_actions
import research_scoped_actions as scoped_actions
from s7_budget_portfolio import is_main_board


@dataclass(frozen=True)
class ExecutionPolicy:
    initial_cash: float = 25000.
    capital_cap: float = 25000.
    max_names: int = 3
    candidate_limit: int = 40
    minimum_holding_sessions: int = 5
    switch_margin: float = .01
    cooldown_sessions: int = 2
    hard_stop: float = .08
    trailing_activation: float = .20
    trailing_distance: float = .15
    risk_aversion: float = .5


def _mainboard(code):
    return is_main_board(str(code).split('.')[0])


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _frame_sha(frame):
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    digest = hashlib.sha256(values.tobytes())
    digest.update(json.dumps([(str(k),str(v)) for k,v in frame.dtypes.items()]).encode('utf-8'))
    return digest.hexdigest()


def allocate_expected_utility(candidates, budget, max_names=3, candidate_limit=40):
    """Exact integer allocation, objective sum(amount * predicted utility).

    The caller supplies utility in expected-return units (e.g. mu20-.5*risk20),
    not a cross-sectional rank or historical momentum. There is no objective for
    using cash. Negative/zero utilities and unaffordable stocks remain unbought.
    Cash, each stock, and total investment are bounded in integer fen.
    """
    columns = ['ts_code', 'shares', 'price', 'amount', 'utility', 'objective']
    if not {'ts_code', 'open', 'utility'}.issubset(candidates):
        raise ValueError('Allocator requires ts_code/open/utility')
    if not np.isfinite(budget) or budget < 0 or max_names < 0:
        raise ValueError('Invalid allocation budget/name limit')
    c = candidates[['ts_code', 'open', 'utility']].copy()
    if c.ts_code.duplicated().any():
        raise ValueError('Duplicate allocation stocks')
    for column in ('open', 'utility'):
        c[column] = pd.to_numeric(c[column], errors='coerce')
    c = c[np.isfinite(c.open) & np.isfinite(c.utility) & c.open.gt(0)
          & c.utility.gt(0) & c.ts_code.map(_mainboard)]
    c = c.sort_values(['utility', 'ts_code'], ascending=[False, True], kind='stable').head(candidate_limit)
    cash_fen = int(np.floor(float(budget)*100+1e-6))
    # Raw A-share prices must be cent-denominated, not adjustment factors.
    if not np.allclose(c.open*100, np.round(c.open*100), atol=1e-6, rtol=0):
        raise ValueError('Execution price is not cent-denominated raw price')
    costs = np.round(c.open.to_numpy()*100).astype(np.int64)*100
    feasible = costs <= cash_fen
    c, costs = c.iloc[np.flatnonzero(feasible)].reset_index(drop=True), costs[feasible]
    if c.empty or max_names == 0:
        return pd.DataFrame(columns=columns)
    n = len(c)
    upper = cash_fen//costs
    values = costs/100*c.utility.to_numpy()
    # x = integer lots, y = active-name indicator.
    objective = np.r_[-values, np.zeros(n)]
    matrix = np.zeros((n+2, 2*n))
    matrix[0, :n] = costs
    matrix[1, n:] = 1
    for i in range(n):
        matrix[i+2, i] = 1
        matrix[i+2, n+i] = -upper[i]
    result = milp(objective, integrality=np.ones(2*n),
                  bounds=Bounds(np.zeros(2*n), np.r_[upper, np.ones(n)]),
                  constraints=LinearConstraint(matrix, np.full(n+2, -np.inf),
                                               np.r_[cash_fen, max_names, np.zeros(n)]),
                  options={'presolve': True})
    if not result.success:
        raise RuntimeError('Whole-lot expected-utility optimization failed: '+str(result.message))
    lots = np.rint(result.x[:n]).astype(int)
    if (lots < 0).any() or int(np.dot(costs, lots)) > cash_fen or np.count_nonzero(lots) > max_names:
        raise AssertionError('Optimizer violated discrete allocation constraints')
    result_frame = c[lots > 0].copy()
    result_frame['shares'] = lots[lots > 0]*100
    result_frame['price'] = result_frame.open
    result_frame['amount'] = result_frame.price*result_frame.shares
    result_frame['objective'] = result_frame.amount*result_frame.utility
    return result_frame[columns].reset_index(drop=True)


def next_open_schedule(predictions, sessions):
    """Map every completed prediction session to its immediate next session."""
    dates = pd.DatetimeIndex(pd.to_datetime(sessions))
    if dates.has_duplicates or not dates.is_monotonic_increasing or dates.hasnans:
        raise ValueError('Sessions must be distinct, sorted exchange sessions')
    if not {'signal_date', 'ts_code'}.issubset(predictions):
        raise ValueError('Missing prediction keys')
    p = predictions.copy()
    p['signal_date'] = pd.to_datetime(p.signal_date)
    if p.duplicated(['signal_date', 'ts_code']).any():
        raise ValueError('Duplicate daily prediction keys')
    if not p.signal_date.isin(dates).all():
        raise ValueError('Prediction date is not an exchange session')
    mapping = dict(zip(dates[:-1], dates[1:]))
    p['execution_date'] = p.signal_date.map(mapping)
    return p


def exit_reason(anchor, close, utility, competitor_utility, session_index, policy):
    """Close-only risk/model exit; returned reason executes no earlier than next open."""
    if not np.isfinite(close) or close <= 0:
        raise ValueError('Invalid held close')
    if close <= anchor['entry']*(1-policy.hard_stop):
        return 'hard_stop'
    if (anchor['peak'] >= anchor['entry']*(1+policy.trailing_activation)
            and close <= anchor['peak']*(1-policy.trailing_distance)):
        return 'trailing_stop'
    if session_index-anchor['entry_index'] < policy.minimum_holding_sessions:
        return None
    if utility is None or not np.isfinite(utility):
        return None  # Missing prediction does not fabricate bearish evidence.
    if utility <= 0:
        return 'nonpositive_expected_utility'
    if (competitor_utility is not None and np.isfinite(competitor_utility)
            and competitor_utility > utility+policy.switch_margin):
        return 'better_predicted_opportunity'
    return None


class OfficialLimits:
    """Exact vendor limits only: no inferred ST status or invented 5/10% bands."""
    def __init__(self, out, pro=None, offline=False, cache_roots=()):
        self.out, self.pro, self.offline = Path(out), pro, offline
        self.folder = self.out/'daily_limits'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.cache_roots = [ROOT/'boundaries', *map(Path, cache_roots)]
        self.frames, self.sources, self.batches, self.batch_attempted = {}, {}, {}, set()

    @staticmethod
    def validate_batch(frame, day):
        required = {'ts_code','trade_date','up_limit','down_limit'}
        if not isinstance(frame,pd.DataFrame) or frame.empty or not required.issubset(frame):
            raise ValueError('Incomplete official daily-limit batch schema')
        if (frame[['ts_code','trade_date']].isna().any().any()
                or not frame.trade_date.astype(str).eq(f'{day:%Y%m%d}').all()
                or frame.duplicated(['ts_code','trade_date']).any()):
            raise ValueError('Wrong/duplicate official daily-limit batch keys')
        values = frame[['up_limit','down_limit']].apply(pd.to_numeric,errors='coerce')
        if not np.isfinite(values).all().all():
            raise ValueError('Nonfinite official daily-limit batch')
        # IPO no-band zero observations may be preserved, but cannot satisfy an
        # executable candidate's subsequent strictly-positive band validation.
        return frame

    def _batch(self, day):
        if day in self.batches:
            return self.batches[day]
        path = self.folder/f'stk_limit_{day:%Y%m%d}.pkl'
        if path.exists():
            raw = self.validate_batch(pd.read_pickle(path),day)
            self.batches[day] = raw
            self.sources[str(path.resolve())] = _sha(path)
            return raw
        if self.offline or self.pro is None or day in self.batch_attempted:
            return None
        self.batch_attempted.add(day)
        for params in (dict(trade_date=f'{day:%Y%m%d}',offset=0,limit=6000,
                            fields='ts_code,trade_date,up_limit,down_limit'),
                       dict(trade_date=f'{day:%Y%m%d}',offset=0,limit=6000)):
            try:
                raw = self.validate_batch(self.pro.query('stk_limit',**params),day)
                raw.to_pickle(path)
                self.batches[day] = raw
                self.sources[str(path.resolve())] = _sha(path)
                return raw
            except Exception:
                pass
        return None

    @staticmethod
    def validate(frame, day, code):
        if not isinstance(frame, pd.DataFrame) or not {'ts_code', 'trade_date'}.issubset(frame):
            raise ValueError('Missing exact-limit schema')
        selected = frame[frame.ts_code.eq(code) & frame.trade_date.astype(str).eq(f'{day:%Y%m%d}')]
        if len(selected) != 1:
            raise ValueError('Missing/conflicting requested exact limit')
        selected = validate_rows(selected, ['ts_code', 'trade_date'], ['up_limit', 'down_limit'], f'{day:%Y%m%d}')
        if len(selected) != 1:
            raise ValueError('Missing/conflicting requested exact limit')
        row = selected.iloc[0]
        if row.down_limit >= row.up_limit:
            raise ValueError('Invalid official limit band')
        return row

    def __call__(self, day, code):
        day = pd.Timestamp(day)
        key = (day, code)
        if key in self.frames:
            return self.frames[key]
        target = self.folder/f'{code}_{day:%Y%m%d}.pkl'
        paths = [target]
        for folder in self.cache_roots:
            paths += [folder/f'stk_limit_{day:%Y%m%d}.pkl', folder/f'{day:%Y%m%d}.pkl',
                      folder/f'stk_limit_{code}_{day:%Y%m%d}.pkl']
        for path in paths:
            if not path.exists():
                continue
            try:
                row = self.validate(pd.read_pickle(path), day, code)
            except ValueError:
                continue
            self.frames[key], self.sources[str(path.resolve())] = row, _sha(path)
            return row
        raw = self._batch(day)
        if raw is not None and raw.ts_code.eq(code).any():
            # A recorded zero/no-band IPO row is not replaced by a guessed
            # ordinary-stock band or treated as a normally executable stock.
            row = self.validate(raw,day,code)
            self.frames[key] = row
            return row
        if self.offline or self.pro is None:
            raise RuntimeError(f'Missing exact stk_limit cache: {code} {day.date()}')
        errors = []
        for params in (dict(ts_code=code, trade_date=f'{day:%Y%m%d}'),
                       dict(ts_code=code, start_date=f'{day:%Y%m%d}', end_date=f'{day:%Y%m%d}')):
            try:
                raw = self.pro.query('stk_limit', **params)
                row = self.validate(raw, day, code)
                raw[raw.ts_code.eq(code) & raw.trade_date.astype(str).eq(f'{day:%Y%m%d}')].to_pickle(target)
                self.frames[key], self.sources[str(target.resolve())] = row, _sha(target)
                return row
            except Exception as exc:
                errors.append(type(exc).__name__)
        raise RuntimeError(f'No exact stk_limit {code} {day.date()}; error types={errors}')


def _prepare_predictions(predictions, sessions, policy):
    p = next_open_schedule(predictions, sessions)
    for name in ('mu5','mu10','mu20','risk5','risk10','risk20','prob5','prob10','prob20'):
        if name not in p:
            continue
        p[name] = pd.to_numeric(p[name], errors='coerce')
        if not np.isfinite(p[name]).all():
            raise ValueError('Nonfinite prediction head: '+name)
        if name.startswith('risk') and p[name].lt(0).any():
            raise ValueError('Predicted adverse excursion cannot be negative')
        if name.startswith('prob') and not p[name].between(0,1).all():
            raise ValueError('Predicted stage probability outside [0,1]')
    if 'utility' not in p:
        if not {'mu20', 'risk20'}.issubset(p):
            raise ValueError('Prediction needs utility or mu20/risk20')
        p['utility'] = p.mu20-policy.risk_aversion*p.risk20
    p['utility'] = pd.to_numeric(p.utility, errors='coerce')
    if not np.isfinite(p.utility).all():
        raise ValueError('Nonfinite predicted opportunity utility')
    return p


def _marks(ledger, quotes, day, field, actions, pro, audit):
    # Shared helper refuses unverified quote gaps and adjusts confirmed suspended
    # last-close marks for booked cash/stock distributions.
    return engine.held_marks(ledger, quotes, day, field, actions, pro, audit)


def run_account(predictions, daily, sessions, out, pro=None, offline=False,
                start='2026-01-01', end='2026-09-24', policy=None,
                limit_provider=None, action_provider=None, cache_roots=()):
    """Run one isolated daily account and export every session/month, not winners.

    ``sessions`` must come from the exchange calendar and include the preceding
    session. Test hooks are pure providers(day,code) and providers(code); normal
    execution validates exact official limits and complete scoped action history.
    """
    policy = policy or ExecutionPolicy()
    if policy.initial_cash != 25000 or policy.capital_cap != 25000:
        raise ValueError('This experiment fixes initial cash and new-investment cap at 25000')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    status_path = out/'daily_run_status.json'
    status_path.write_text(json.dumps(dict(account_complete=False, start=str(start), end=str(end))), encoding='utf-8')
    protocol_path = out/'execution_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8')).get('policy') != asdict(policy):
        raise ValueError('Do not silently change a completed daily-account policy; use a new output directory')
    limits = limit_provider or OfficialLimits(out, pro, offline, cache_roots)
    action_provider = action_provider or (lambda code: load_scoped_actions(
        engine.OfflineClient() if offline else pro, code, out, start, end))
    p = _prepare_predictions(predictions, sessions, policy)
    session_dates = pd.DatetimeIndex(pd.to_datetime(sessions))
    positions = {d: i for i, d in enumerate(session_dates)}
    account_sessions = session_dates[(session_dates >= pd.Timestamp(start)) & (session_dates <= pd.Timestamp(end))]
    if len(account_sessions) == 0 or positions[account_sessions[0]] == 0:
        raise ValueError('Account needs sessions and a preceding exchange session')
    d = daily.copy()
    if 'date' not in d and 'trade_date' in d:
        d['date'] = pd.to_datetime(d.trade_date.astype(str))
    if not {'date', 'ts_code', 'open', 'close'}.issubset(d):
        raise ValueError('Incomplete raw daily quotation schema')
    d['date'] = pd.to_datetime(d.date)
    if d.duplicated(['date', 'ts_code']).any():
        raise ValueError('Duplicate raw daily quotation keys')
    if not np.isfinite(d[['open', 'close']]).all().all() or (d[['open', 'close']] <= 0).any().any():
        raise ValueError('Invalid raw daily opening/closing quotes')
    if 'adj_factor' in d and (not np.isfinite(d.adj_factor).all() or not d.adj_factor.gt(0).all()):
        raise ValueError('Invalid optional daily action cross-check factor')
    d = d[d.date.isin(account_sessions)]
    frames = {day: g.set_index('ts_code') for day, g in d.groupby('date')}
    if set(account_sessions)-set(frames):
        raise ValueError('Missing full-market quotation session')
    p_by_close = {day: g.set_index('ts_code') for day, g in p.groupby('signal_date')}
    p_by_open = {day: g for day, g in p.dropna(subset=['execution_date']).groupby('execution_date')}
    ledger = engine.Ledger()
    actions = pd.DataFrame(columns=['ts_code','record_date','ex_date','pay_date','div_listdate','cash','stock','event'])
    known, anchors, pending, sold_at = set(), {}, {}, {}
    trades, allocation_rows, holds, navs, signals, stale, order_checks, contributions = [], [], [], [], [], [], [], []
    previous_values, previous_nav, last_day = {}, policy.initial_cash, None
    try:
        for day in pd.date_range(pd.Timestamp(start), pd.Timestamp(end)):
            events_before = len(ledger.events)
            ledger.morning(day, actions)
            for code, anchor in anchors.items():
                change = actions[actions.ts_code.eq(code) & actions.ex_date.eq(day)]
                if not change.empty:
                    anchor['entry'] = (anchor['entry']-change.cash.sum())/(1+change.stock.sum())
                    anchor['peak'] = (anchor['peak']-change.cash.sum())/(1+change.stock.sum())
            if day not in frames:
                ledger.close_record(day, actions)
                # Weekend payment transfers a receivable to cash without P&L.
                if len(ledger.events) != events_before:
                    raise ValueError('Corporate ex-date outside supplied exchange calendar')
                continue
            index, q = positions[day], frames[day].copy()
            # Check an already-held factor before a sale too: otherwise an
            # unexplained ex-right gap could be sold before the close guard.
            for code, anchor in anchors.items():
                if 'adj_factor' in q and code in q.index:
                    current_factor = float(q.loc[code,'adj_factor'])
                    if 'factor' in anchor:
                        engine.check_unexplained_actions([code],{code:anchor['factor']},{code:current_factor},
                                                         actions,anchor['factor_date'],day)
                    anchor['factor'],anchor['factor_date'] = current_factor,day
            open_marks = _marks(ledger, q, day, 'open', actions, pro, stale)
            daily_buys, daily_sells = {}, {}
            for code, (reason, signal_day) in list(pending.items()):
                anchor = anchors[code]
                unlocked = ledger.shares[code]-ledger.blocked_shares(code)
                if unlocked <= 0 or code not in q.index or index <= anchor['entry_index']:
                    continue
                official = limits(day, code)
                price = float(q.loc[code, 'open'])
                if not official.down_limit-.005 <= price <= official.up_limit+.005:
                    raise ValueError('Raw opening outside official limits')
                order_checks.append(dict(date=day, code=code, side='sell', price=price,
                                         down_limit=float(official.down_limit), up_limit=float(official.up_limit),
                                         executable=price > official.down_limit+.005))
                if price <= official.down_limit+.005:
                    continue  # Keep a triggered exit until it can actually execute.
                ledger.cash += unlocked*price
                ledger.shares[code] -= unlocked
                daily_sells[code] = daily_sells.get(code, 0.)+unlocked*price
                sold_at[code] = index
                trades.append(dict(date=day, code=code, side='sell', shares=unlocked, price=price,
                                   reason=reason, signal_date=signal_day))
                if ledger.shares[code] == 0:
                    anchors.pop(code)
                    pending.pop(code)
            today = p_by_open.get(day, p.iloc[:0])
            retained = {c for c, qty in ledger.shares.items() if qty > 0}
            retained_value = sum(ledger.shares[c]*open_marks[c] for c in retained)
            equity = ledger.nav(open_marks)
            target = min(equity, policy.capital_cap)
            available = max(0., min(ledger.cash, target-retained_value))
            name_slots = max(0, policy.max_names-len(retained))
            candidates = today[today.utility.gt(0) & today.ts_code.map(_mainboard).astype(bool)
                               & ~today.ts_code.isin(retained)].copy()
            if 'eligible' in candidates:
                candidates = candidates[candidates.eligible.eq(True)]
            candidates = candidates.loc[candidates.ts_code.map(lambda c: index-sold_at.get(c, -1000000) >= policy.cooldown_sessions).astype(bool)]
            candidates = candidates.sort_values(['utility','ts_code'], ascending=[False,True], kind='stable')
            # Cap the pre-fill pool in model-score order. No future realized
            # trading volume, day high/low, or close enters these filters.
            candidates = candidates.head(policy.candidate_limit)
            executable = []
            if available > 0 and name_slots > 0:
                for row in candidates.itertuples():
                    if row.ts_code not in q.index:
                        continue
                    price = float(q.loc[row.ts_code, 'open'])
                    if price*100 > available+1e-8:
                        continue
                    official = limits(day, row.ts_code)
                    if not official.down_limit-.005 <= price <= official.up_limit+.005:
                        raise ValueError('Raw opening outside official limits')
                    eligible = official.down_limit+.005 < price < official.up_limit-.005
                    order_checks.append(dict(date=day, code=row.ts_code, side='buy', price=price,
                                             down_limit=float(official.down_limit), up_limit=float(official.up_limit),
                                             executable=eligible))
                    if eligible:
                        executable.append(dict(ts_code=row.ts_code, open=price, utility=float(row.utility)))
                eligible = pd.DataFrame(executable, columns=['ts_code','open','utility'])
                allocation = allocate_expected_utility(eligible, available, name_slots, policy.candidate_limit)
                # Load/validate action histories before any buy is committed.
                for code in allocation.ts_code:
                    if code not in known:
                        fresh = action_provider(code)
                        if not fresh.empty and not fresh.ts_code.eq(code).all():
                            raise ValueError('Wrong stock corporate actions')
                        actions = pd.concat([actions, fresh], ignore_index=True)
                        known.add(code)
                for row in allocation.itertuples():
                    signal_day = today.loc[today.ts_code.eq(row.ts_code), 'signal_date'].iloc[0]
                    if positions[signal_day]+1 != index:
                        raise AssertionError('Buy is not next session after prediction')
                    ledger.cash -= row.amount
                    ledger.shares[row.ts_code] = row.shares
                    ledger.industry[row.ts_code] = 'stock_opportunity'
                    anchors[row.ts_code] = dict(entry=row.price, peak=row.price, entry_index=index)
                    if 'adj_factor' in q:
                        anchors[row.ts_code]['factor'] = float(q.loc[row.ts_code,'adj_factor'])
                        anchors[row.ts_code]['factor_date'] = day
                    daily_buys[row.ts_code] = row.amount
                    trades.append(dict(date=day, code=row.ts_code, side='buy', shares=row.shares, price=row.price,
                                       reason='positive_expected_utility', signal_date=signal_day))
                    allocation_rows.append(dict(date=day, signal_date=signal_day, code=row.ts_code, shares=row.shares,
                                                price=row.price, amount=row.amount, utility=row.utility,
                                                target_budget=target, available_budget=available))
            if ledger.cash < -1e-6:
                raise AssertionError('Negative cash')
            close_marks = _marks(ledger, q, day, 'close', actions, pro, stale)
            close_predictions = p_by_close.get(day, p.iloc[:0].set_index('ts_code'))
            competitor = close_predictions[close_predictions.index.map(_mainboard).to_numpy(dtype=bool)
                                           & ~close_predictions.index.isin([c for c,n in ledger.shares.items() if n])]
            if 'eligible' in competitor:
                competitor = competitor[competitor.eligible.eq(True)]
            # A known-to-be unaffordable or suspended stock must not force the
            # sale of a winner. This uses the current *completed* close only.
            competitor = competitor[competitor.index.isin(q.index)]
            competitor = competitor[q.loc[competitor.index, 'close'].mul(100).le(min(ledger.nav(close_marks), policy.capital_cap))
                                    & competitor.index.map(lambda c: index-sold_at.get(c, -1000000) >= policy.cooldown_sessions).to_numpy(dtype=bool)]
            best_utility = float(competitor.utility.max()) if len(competitor) else None
            for code, qty in ledger.shares.items():
                if qty <= 0:
                    continue
                anchor = anchors[code]
                anchor['peak'] = max(anchor['peak'], close_marks[code])
                utility = float(close_predictions.loc[code,'utility']) if code in close_predictions.index else None
                if code not in pending:
                    reason = exit_reason(anchor, close_marks[code], utility, best_utility, index, policy)
                    if reason:
                        pending[code] = (reason, day)
                        signals.append(dict(date=day, code=code, reason=reason, close=close_marks[code],
                                            entry=anchor['entry'], peak=anchor['peak'], utility=utility,
                                            competitor_utility=best_utility))
                holds.append(dict(date=day, code=code, shares=qty, price=close_marks[code],
                                  value=qty*close_marks[code], locked_shares=ledger.blocked_shares(code),
                                  entry=anchor['entry'], peak=anchor['peak'], utility=utility,
                                  pending_exit=code in pending))
            ledger.close_record(day, actions)
            stock_values = {c: n*close_marks[c] for c,n in ledger.shares.items() if n}
            new_entitlements = {}
            for event in ledger.events[events_before:]:
                new_entitlements[event['code']] = new_entitlements.get(event['code'],0.)+event['cash_entitlement']
            day_profit = 0.
            for code in set(previous_values) | set(stock_values) | set(daily_buys) | set(daily_sells) | set(new_entitlements):
                contribution = (stock_values.get(code,0.)-previous_values.get(code,0.)
                                -daily_buys.get(code,0.)+daily_sells.get(code,0.)+new_entitlements.get(code,0.))
                contributions.append(dict(date=day, code=code, profit=contribution))
                day_profit += contribution
            nav = ledger.nav(close_marks)
            if abs(nav-previous_nav-day_profit) > 1e-6:
                raise AssertionError('Per-stock contribution does not reconcile daily account P&L')
            navs.append(dict(date=day, cash=ledger.cash, stock_value=sum(stock_values.values()),
                             receivable=sum(v[0] for v in ledger.receivable.values()), equity=nav,
                             profit=nav-previous_nav, return_=nav/previous_nav-1))
            previous_values, previous_nav, last_day = stock_values, nav, day
        daily_nav = pd.DataFrame(navs)
        if len(daily_nav) != len(account_sessions) or last_day != account_sessions[-1]:
            raise AssertionError('Incomplete daily account')
        monthly = daily_nav.groupby(daily_nav.date.dt.to_period('M'), sort=True).tail(1).copy()
        previous = np.r_[policy.initial_cash, monthly.equity.to_numpy()[:-1]]
        monthly['profit'] = monthly.equity-previous
        monthly['return'] = monthly.equity/previous-1
        monthly['budget_return'] = monthly.profit/policy.capital_cap
        monthly['target_hit'] = monthly.profit.ge(7500.-1e-7)
        monthly['month'] = monthly.date.dt.strftime('%Y-%m')
        exports = {
            'daily_nav': daily_nav,
            'account': monthly.drop(columns='return_'),
            'trades': pd.DataFrame(trades, columns=['date','code','side','shares','price','reason','signal_date']),
            'allocations': pd.DataFrame(allocation_rows, columns=['date','signal_date','code','shares','price','amount','utility','target_budget','available_budget']),
            'holdings': pd.DataFrame(holds, columns=['date','code','shares','price','value','locked_shares','entry','peak','utility','pending_exit']),
            'exit_signals': pd.DataFrame(signals, columns=['date','code','reason','close','entry','peak','utility','competitor_utility']),
            'suspension_marks': pd.DataFrame(stale, columns=['date','code','quote_date','mark','field','reason']),
            'execution_checks': pd.DataFrame(order_checks, columns=['date','code','side','price','down_limit','up_limit','executable']),
            'stock_daily_contributions': pd.DataFrame(contributions, columns=['date','code','profit']),
            'corporate_events': pd.DataFrame(ledger.events, columns=['date','event','code','payment_date','cash_entitlement','bonus_shares','fractional_bonus_discarded'])}
        for name, frame in exports.items():
            frame.to_csv(out/f'{name}.csv', index=False)
        if contributions:
            stock_months = pd.DataFrame(contributions)
            stock_months['month'] = stock_months.date.dt.strftime('%Y-%m')
            stock_months.groupby(['month','code'], as_index=False).profit.sum().to_csv(out/'stock_monthly_contributions.csv', index=False)
        nav_array = np.r_[policy.initial_cash, daily_nav.equity.to_numpy()]
        metrics = dict(ending_equity=float(nav_array[-1]), total_profit=float(nav_array[-1]-policy.initial_cash),
                       total_return_pct=float(100*(nav_array[-1]/policy.initial_cash-1)),
                       max_daily_drawdown_pct=float(100*(1-(nav_array/np.maximum.accumulate(nav_array)).min())),
                       target_hit_months=int(monthly.target_hit.sum()), losing_months=int(monthly.profit.lt(0).sum()),
                       trades=len(trades), trading_sessions=len(daily_nav), account_months=len(monthly))
        (out/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        protocol = dict(architecture='Daily stock-level forecast and expected-utility whole-lot account',
                        policy=asdict(policy), fees=0, external_topups=False,
                        timing='Previous completed exchange session predictions; next-session observed open/official limits',
                        monthly_forced_liquidation=False, intraday_fill_model=False,
                        appreciated_positions='No forced trimming; residual new investment max(0,min(equity,25000)-retained_open_value)',
                        stop='Raw close thresholds adjusted only for verified corporate actions; earliest next executable open',
                        cooldown='Per sold stock, including model exits; fresh other stocks allowed on the next daily prediction',
                        actions='Existing scoped validated raw histories and event ledger; no inferred event dates')
        protocol['factor_crosscheck'] = ('All observed held-factor changes must have a verified in-window corporate action'
                                        if 'adj_factor' in d else 'Not supplied; this account uses raw quotes plus validated actions, not factor inference')
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
        sources = getattr(limits, 'sources', {})
        sources[str(Path(__file__).resolve())] = _sha(__file__)
        sources[str(Path(engine.__file__).resolve())] = _sha(engine.__file__)
        sources[str(Path(scoped_actions.__file__).resolve())] = _sha(scoped_actions.__file__)
        for path in (out/'scoped_actions').glob('*'):
            if path.is_file():
                sources[str(path.resolve())] = _sha(path)
        status_path.write_text(json.dumps(dict(account_complete=True, start=str(start), end=str(end),
                                             last_session=str(last_day.date()), metrics=metrics,
                                             source_sha256=sources,
                                             prediction_frame_sha256=_frame_sha(p),
                                             active_daily_frame_sha256=_frame_sha(d),
                                             sessions_sha256=hashlib.sha256(session_dates.asi8.tobytes()).hexdigest()), indent=2), encoding='utf-8')
        return monthly.reset_index(drop=True)
    except Exception as exc:
        status_path.write_text(json.dumps(dict(account_complete=False, error_type=type(exc).__name__,
                                             start=str(start), end=str(end)), indent=2), encoding='utf-8')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--daily', required=True)
    parser.add_argument('--sessions', required=True, help='JSON list of YYYY-MM-DD exchange sessions')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--start', default='2026-01-01')
    parser.add_argument('--end', default='2026-09-24')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    prediction_path, daily_path = Path(args.predictions), Path(args.daily)
    read = lambda path: pd.read_pickle(path) if path.suffix == '.pkl' else pd.read_csv(path)
    result = run_account(read(prediction_path), read(daily_path),
                         json.loads(Path(args.sessions).read_text(encoding='utf-8')),
                         args.output_dir, engine.OfflineClient() if args.offline else get_pro(), args.offline,
                         args.start, args.end)
    print(result[['month','profit','equity','target_hit']].to_string(index=False))


if __name__ == '__main__':
    main()
