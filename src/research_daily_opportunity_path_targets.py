"""Future path supervision aligned with close decisions and next-open exits.

These are supervisory reference labels, not exact executable trade returns:
future official locked-limit prices and pending orders are not simulated here.
The cash-account engine remains responsible for those execution constraints.
The builder never reads or mutates model features, graph edges, or eligibility.
"""
from dataclasses import dataclass
import copy

import numpy as np
import pandas as pd

from research_daily_opportunity_features import normalized_dates


HORIZONS = (5, 10, 20)
UPPER_THRESHOLDS = (.10, .15, .30)
MISSING = -1
EXIT_HORIZON, EXIT_HARD_STOP, EXIT_TRAILING_STOP = 0, 1, 2
PASSAGE_NEITHER, PASSAGE_UPPER, PASSAGE_LOWER = 0, 1, 2
BOUNDARY_RTOL = 1e-12

PATH_PROTOCOL = {
    'version': 'daily_close_trigger_next_open_path_targets_v1',
    'entry': 'Actual factor-adjusted open at market session t+1',
    'stop': 'Close <= entry*(1-.08); trigger at close, exit at next market-session actual open',
    'trailing': 'Peak starts at entry and uses subsequent closes only; activate at +20%, trigger at 8% retreat from that peak',
    'horizon_exit': 'If no earlier trigger, exit at actual open t+h+1; a trigger on close t+h retains its stop reason',
    'stop_priority': 'First trigger wins; if hard and trailing trigger together, hard stop takes priority',
    'downside': 'max(0, 1-min(entry open, intraday lows through last held close, exit open)/entry); exclude all exit-day intraday lows',
    'first_passage': 'First close at upper gain (.10,.15,.30) or at -8% during t+1...t+h, independently of any earlier trailing exit; otherwise neither',
    'passage_ties': 'Inclusive boundaries; positive upper and negative lower are disjoint, so a close cannot hit both; earliest session wins',
    'boundary_rtol': BOUNDARY_RTOL,
    'complete_window': 'Require every actual OHLC/factor quote from t+1 through t+h+1, including after an early reference exit; no fill or compressed calendar',
    'label_maturity': 'Always market session t+h+1, not early exit date; training requires label_end strictly before cutoff',
    'missing_labels': 'NaN payoff/downside, -1 passage/reason/indices, valid=False; label_end remains calendar-based when that session exists',
    'quote_presence': 'Actual supplied OHLC and adjustment-factor quote; this label-only flag never changes feature eligibility',
    'adjustment': 'Raw OHLC times actual same-day adj_factor; no future-normalised qfq series',
    'execution_scope': 'Reference label omits official future locked-limit execution, fees and model-switch exits; exact account execution remains separate',
}


@dataclass(frozen=True)
class PathTargets:
    dates: pd.DatetimeIndex
    stock_codes: tuple
    horizons: tuple
    path_payoff: np.ndarray
    max_adverse_excursion: np.ndarray
    first_passage: np.ndarray
    first_passage_index: np.ndarray
    exit_reason: np.ndarray
    trigger_index: np.ndarray
    exit_index: np.ndarray
    label_end_dates: np.ndarray
    valid: np.ndarray
    quote_presence: np.ndarray
    protocol: dict

    def matured_before(self, cutoff):
        """Per-stock/per-horizon supervision mask; never an inference gate."""
        day = normalized_dates([cutoff], 'Path-label cutoff')[0].to_datetime64().astype('datetime64[D]')
        matured = (~np.isnat(self.label_end_dates)) & (self.label_end_dates < day)
        return self.valid & matured[:, None, :]


def _configuration(horizons, upper_thresholds, hard_stop, trailing_activation, trailing_distance):
    horizons = tuple(horizons)
    if (not horizons or any(isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer)) or h < 1 for h in horizons)
            or tuple(sorted(set(horizons))) != horizons):
        raise ValueError('Horizons must be unique, increasing positive integers')
    upper = np.asarray(upper_thresholds, dtype=float)
    if upper.shape != (len(horizons),) or not np.isfinite(upper).all() or (upper <= 0).any():
        raise ValueError('One finite positive upper threshold per horizon is required')
    controls = np.asarray((hard_stop, trailing_activation, trailing_distance), dtype=float)
    if (not np.isfinite(controls).all() or not 0 < hard_stop < 1
            or not trailing_activation > 0 or not 0 < trailing_distance < 1):
        raise ValueError('Positive activation and stop distances strictly between zero and one are required')
    return tuple(int(h) for h in horizons), upper


def _below(value, boundary):
    return value <= boundary + BOUNDARY_RTOL*np.abs(boundary)


def _above(value, boundary):
    return value >= boundary - BOUNDARY_RTOL*np.abs(boundary)


def _adjusted_panel(daily, sessions):
    required = ('date', 'ts_code', 'open', 'high', 'low', 'close', 'adj_factor')
    if (not isinstance(daily, pd.DataFrame) or not daily.columns.is_unique
            or not set(required).issubset(daily.columns)):
        raise ValueError('Unique raw daily OHLC, code, date and actual adjustment factor columns are required')
    frame = daily.loc[:, required].copy()
    frame['date'] = normalized_dates(frame.date, 'Path-target quote dates')
    if frame.empty or frame.duplicated(['date', 'ts_code']).any():
        raise ValueError('Empty or duplicate path-target quotes')
    if (not frame.ts_code.map(lambda value: isinstance(value, str)).all()
            or not frame.ts_code.str.fullmatch(r'\d{6}\.(SH|SZ)').all()):
        raise ValueError('SH/SZ stock codes are required')
    numeric = frame[['open', 'high', 'low', 'close', 'adj_factor']].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric <= 0).any():
        raise ValueError('Supplied quotes and adjustment factors must be finite and positive; absent rows remain missing')
    if ((frame.high < frame[['open', 'close', 'low']].max(axis=1)-1e-6).any()
            or (frame.low > frame[['open', 'close', 'high']].min(axis=1)+1e-6).any()):
        raise ValueError('Inconsistent path-target OHLC')
    observed = pd.DatetimeIndex(sorted(frame.date.unique()))
    days = observed if sessions is None else normalized_dates(sessions, 'Path-target market calendar')
    if not days.is_unique or not days.is_monotonic_increasing or not observed.isin(days).all():
        raise ValueError('Market calendar must be unique, ordered and contain every observed quote date')
    codes = tuple(sorted(frame.ts_code.unique()))
    panel = frame.pivot(index='date', columns='ts_code', values=['open', 'high', 'low', 'close', 'adj_factor'])
    factor = panel['adj_factor'].reindex(index=days, columns=codes).to_numpy(dtype=float)
    prices = tuple(panel[column].reindex(index=days, columns=codes).to_numpy(dtype=float)*factor
                   for column in ('open', 'high', 'low', 'close'))
    presence = np.logical_and.reduce([np.isfinite(values) for values in prices])
    if any(np.isinf(values).any() for values in prices):
        raise ValueError('Adjustment overflow is invalid')
    return days, codes, prices, presence


def build_path_targets(daily, *, sessions=None, horizons=HORIZONS,
                       upper_thresholds=UPPER_THRESHOLDS, hard_stop=.08,
                       trailing_activation=.20, trailing_distance=.08):
    """Vectorise dates/stocks; loop only over the three bounded future windows.

    Array axes are [signal market date, stock code, horizon]. Event and exit
    indices are absolute positions in ``dates``. ``trigger_index == -1`` is
    also used for a valid horizon expiry (no stop trigger). First-passage
    outcomes concern the complete h-close path, even after a reference exit.
    """
    horizons, upper = _configuration(horizons, upper_thresholds, hard_stop,
                                     trailing_activation, trailing_distance)
    days, codes, (opening, _high, low, close), presence = _adjusted_panel(daily, sessions)
    shape = (len(days), len(codes), len(horizons))
    payoff = np.full(shape, np.nan, dtype=np.float32)
    downside = np.full(shape, np.nan, dtype=np.float32)
    passage = np.full(shape, MISSING, dtype=np.int8)
    reasons = np.full(shape, MISSING, dtype=np.int8)
    passage_index = np.full(shape, MISSING, dtype=np.int32)
    trigger_index = np.full(shape, MISSING, dtype=np.int32)
    exit_index = np.full(shape, MISSING, dtype=np.int32)
    valid = np.zeros(shape, dtype=bool)
    ends = np.full((len(days), len(horizons)), np.datetime64('NaT', 'D'), dtype='datetime64[D]')
    for j, horizon in enumerate(horizons):
        count = len(days)-horizon-1
        if count <= 0:
            continue
        base = np.arange(count, dtype=np.int32)[:, None]
        entry = opening[1:count+1]
        complete = np.ones(entry.shape, dtype=bool)
        for offset in range(1, horizon+2):
            complete &= presence[offset:offset+count]
        valid[:count, :, j] = complete
        ends[:count, j] = days[horizon+1:].to_numpy(dtype='datetime64[D]')
        peak, minimum = entry.copy(), entry.copy()
        active = np.isfinite(entry)
        price = opening[horizon+1:horizon+1+count].copy()
        exit_at = np.broadcast_to(base+horizon+1, entry.shape).copy()
        trigger_at = np.full(entry.shape, MISSING, dtype=np.int32)
        reason = np.full(entry.shape, EXIT_HORIZON, dtype=np.int8)
        first = np.full(entry.shape, PASSAGE_NEITHER, dtype=np.int8)
        first_at = np.full(entry.shape, MISSING, dtype=np.int32)
        lower_price, upper_price = entry*(1-hard_stop), entry*(1+upper[j])
        activation_price = entry*(1+trailing_activation)
        for offset in range(1, horizon+1):
            closing = close[offset:offset+count]
            minimum = np.where(active, np.minimum(minimum, low[offset:offset+count]), minimum)
            peak = np.where(active, np.maximum(peak, closing), peak)
            hard = active & _below(closing, lower_price)
            trailing = active & _above(peak, activation_price) & _below(closing, peak*(1-trailing_distance))
            triggered = hard | trailing
            price = np.where(triggered, opening[offset+1:offset+1+count], price)
            exit_at = np.where(triggered, base+offset+1, exit_at)
            trigger_at = np.where(triggered, base+offset, trigger_at)
            reason = np.where(triggered, np.where(hard, EXIT_HARD_STOP, EXIT_TRAILING_STOP), reason)
            active &= ~triggered
            upper_hit, lower_hit = _above(closing, upper_price), _below(closing, lower_price)
            newly_hit = (first == PASSAGE_NEITHER) & (upper_hit | lower_hit)
            first = np.where(newly_hit, np.where(upper_hit, PASSAGE_UPPER, PASSAGE_LOWER), first)
            first_at = np.where(newly_hit, base+offset, first_at)
        # Exit is at open: its gap belongs to held-period risk, its later low does not.
        minimum = np.minimum(minimum, price)
        payoff[:count, :, j] = np.where(complete, price/entry-1, np.nan)
        downside[:count, :, j] = np.where(complete, np.maximum(0., 1-minimum/entry), np.nan)
        passage[:count, :, j] = np.where(complete, first, MISSING)
        passage_index[:count, :, j] = np.where(complete, first_at, MISSING)
        reasons[:count, :, j] = np.where(complete, reason, MISSING)
        trigger_index[:count, :, j] = np.where(complete, trigger_at, MISSING)
        exit_index[:count, :, j] = np.where(complete, exit_at, MISSING)
    protocol = copy.deepcopy(PATH_PROTOCOL)
    protocol['configuration'] = dict(horizons=list(horizons), upper_thresholds=upper.tolist(),
                                     hard_stop=float(hard_stop), trailing_activation=float(trailing_activation),
                                     trailing_distance=float(trailing_distance))
    return PathTargets(days, codes, horizons, payoff, downside, passage, passage_index,
                       reasons, trigger_index, exit_index, ends, valid, presence, protocol)
