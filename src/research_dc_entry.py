"""Causal daily-close entry features with no industry-classification dependency.

Pressure is the product of each session's close/open ratio, not an adjusted
close-to-close total return or a multi-session price breakout. Rescaling an
entire day's OHLC (an overnight split/dividend price gap) leaves it unchanged.
All 25 local market sessions must have valid OHLC, volume and traded amount.
"""

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    'signal_day', 'ts_code', 'pressure5', 'pressure20', 'location5',
    'amount_ratio5_vs_prior20', 'mean_amount20', 'entry_score',
    'reversal_eligible',
]
PRICE_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'amount']
ENTRY_PROTOCOL = {
    'window': 'Complete 25 local market sessions; invalid or absent quotes invalidate the window',
    'features': '5/20-session products of close/open minus one (intraday pressure, NOT total returns); 5-session mean close location; latest 5-session mean amount / preceding 20-session mean amount; latest 20-session mean amount',
    'score': '35% pressure5 percentile + 35% pressure20 percentile + 15% amount expansion percentile + 15% mean amount20 percentile; valid all-board SH/SZ stocks ranked together',
    'reversal_eligibility': 'pressure5 > .03; pressure20 > .05; location5 >= .60; amount expansion >= 1.10',
    'weekly_timing': 'Last observed session of a completed calendar week -> first actual session of the following observed week; completion requires the following week to be present in the declared calendar',
    'execution': 'Signals are available after close only; never execute on signal day',
}


def _session_calendar(values):
    """Validate actual, unique, date-only sessions rather than invent weekdays."""
    dates = pd.DatetimeIndex(pd.to_datetime(list(values), errors='raise'))
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError('Session calendar must contain timezone-free date-only values')
    if dates.has_duplicates:
        raise ValueError('Duplicate session calendar dates')
    return dates.sort_values()


def weekly_execution_schedule(session_calendar):
    """Return {next_open_session: preceding_completed_week_close_session}.

    The final observed calendar week is deliberately not presumed complete.
    Missing weekdays caused by market holidays do not require special cases.
    """
    dates = _session_calendar(session_calendar)
    if len(dates) < 2:
        return {}
    weeks = dates.to_period('W-SUN')
    transitions = np.flatnonzero(weeks[1:] != weeks[:-1]) + 1
    return {dates[index]: dates[index - 1] for index in transitions}


def entry_features(daily, signal_dates=None, session_calendar=None):
    """Return complete-window daily features and all-board percentile scores.

    Invalid numeric quotes fail closed by excluding their stock's entire
    affected 25-session window. Duplicate dated stock keys and signals outside
    the declared market calendar are input errors. When no calendar is passed,
    the sorted union of all supplied stock quote dates is the local calendar.
    An explicit calendar can detect a session missing from *every* stock.
    """
    required = ['date', 'ts_code'] + PRICE_COLUMNS
    missing = set(required).difference(daily.columns)
    if missing:
        raise ValueError(f'Missing daily columns: {sorted(missing)}')
    frame = daily[required].copy()
    frame['date'] = pd.to_datetime(frame.date, errors='raise')
    if frame.date.isna().any() or not frame.date.eq(frame.date.dt.normalize()).all():
        raise ValueError('Daily quotes must have valid date-only keys')
    if frame.duplicated(['date', 'ts_code']).any():
        raise ValueError('Duplicate daily keys')
    calendar = (_session_calendar(session_calendar) if session_calendar is not None
                else _session_calendar(frame.date.unique()))
    if not frame.date.isin(calendar).all():
        raise ValueError('Daily quote date is outside declared session calendar')
    signals = calendar if signal_dates is None else _session_calendar(signal_dates)
    positions = calendar.get_indexer(signals)
    if np.any(positions < 0):
        raise ValueError('Signal date is outside declared session calendar')
    warm = positions >= 24
    signals, positions = signals[warm], positions[warm]
    if len(signals) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Future observations cannot affect either quote validation or ranking.
    start, end = int(positions.min()) - 24, int(positions.max())
    calendar = calendar[start:end + 1]
    positions = positions - start
    frame = frame[frame.date.between(calendar[0], calendar[-1])
                  & frame.ts_code.astype('string').str.match(
                      r'^(?:0\d{5}\.SZ|3\d{5}\.SZ|6\d{5}\.SH)$', na=False)]
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    codes = pd.Index(sorted(frame.ts_code.unique()))
    date_index = calendar.get_indexer(frame.date)
    code_index = codes.get_indexer(frame.ts_code)
    # Assign only six numeric fields, avoiding six repeated wide pivots.
    values = np.full((6, len(calendar), len(codes)), np.nan, dtype=float)
    for number, name in enumerate(PRICE_COLUMNS):
        values[number, date_index, code_index] = pd.to_numeric(frame[name], errors='coerce').to_numpy(dtype=float)
    opening, high, low, close, volume, amount = values
    valid = (np.isfinite(values).all(axis=0) & (values > 0).all(axis=0)
             & (high >= np.maximum(opening, close))
             & (low <= np.minimum(opening, close)) & (high >= low))
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        log_pressure = np.log(close / opening)
        location = np.where(high > low, (close - low) / (high - low), .5)

    blocks = []
    for day, index in zip(signals, positions):
        window = slice(index - 24, index + 1)
        recent5 = slice(index - 4, index + 1)
        recent20 = slice(index - 19, index + 1)
        prior20 = slice(index - 24, index - 4)
        good = valid[window].all(axis=0)
        if not good.any():
            continue
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            p5 = np.expm1(log_pressure[recent5].sum(axis=0))
            p20 = np.expm1(log_pressure[recent20].sum(axis=0))
            loc5 = location[recent5].mean(axis=0)
            expansion = amount[recent5].mean(axis=0) / amount[prior20].mean(axis=0)
            mean_amount20 = amount[recent20].mean(axis=0)
        calculated = np.column_stack([p5, p20, loc5, expansion, mean_amount20])
        good &= np.isfinite(calculated).all(axis=1)
        if not good.any():
            continue
        block = pd.DataFrame(calculated[good], columns=OUTPUT_COLUMNS[2:7])
        block.insert(0, 'ts_code', codes[good])
        block.insert(0, 'signal_day', day)
        ranks = block[['pressure5', 'pressure20', 'amount_ratio5_vs_prior20', 'mean_amount20']].rank(pct=True)
        block['entry_score'] = (.35 * ranks.pressure5 + .35 * ranks.pressure20
                                + .15 * ranks.amount_ratio5_vs_prior20 + .15 * ranks.mean_amount20)
        block['reversal_eligible'] = (block.pressure5.gt(.03) & block.pressure20.gt(.05)
                                      & block.location5.ge(.60) & block.amount_ratio5_vs_prior20.ge(1.10))
        blocks.append(block)
    return (pd.concat(blocks, ignore_index=True)[OUTPUT_COLUMNS] if blocks
            else pd.DataFrame(columns=OUTPUT_COLUMNS))
