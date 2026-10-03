"""Causal price-pattern cohorts, not industry or historical membership labels.

Each monthly cohort is fitted afresh using only that signal's trailing daily
intraday pressure. Gap-neutral pressure and adjusted monthly total returns are
different quantities; the latter are aggregated only after cohort formation.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits


CONTEXT_COLUMNS = ['peer_mom1', 'peer_mom3', 'peer_mom6', 'peer_positive3', 'peer_count']
OUTPUT_COLUMNS = ['signal_date', 'ts_code'] + CONTEXT_COLUMNS
PRICE_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
CLUSTER_PARAMS = dict(batch_size=1024, n_init=3, random_state=42, max_iter=60)
PEER_PROTOCOL = {
    'version': 1,
    'lookback_sessions': 60,
    'complete_window': 'Every local session has positive finite OHLC/volume and coherent high/low',
    'cohort_input': 'log(close/open), minus each day all-valid-stock median; stock series centered and L2 normalized',
    'zero_norm_tolerance': 1e-12,
    'estimator': 'MiniBatchKMeans',
    'parameters': dict(n_clusters=20, **CLUSTER_PARAMS),
    'small_sample': 'With fewer than 40 valid patterns: max(2, floor(n/2)) clusters; fewer than 2 patterns omitted',
    'context': 'Same-signal adjusted mom1/3/6 medians, positive mom3 fraction and count of complete-momentum cohort members',
    'minimum_complete_momentum_cohort': 10,
    'asof': 'Independent fit at each actual monthly signal, with no quotes or labels after the signal',
    'scope': 'All SH/SZ mainboard, ChiNext and STAR stock patterns; no industry or named-stock rules',
}


def _dates(values, name):
    dates = pd.DatetimeIndex(pd.to_datetime(values, errors='raise'))
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError(f'{name} must be valid timezone-free date-only values')
    return dates


def peer_context(monthly, daily, signal_dates=None):
    """Return complete monthly peer contexts with unique dated stock keys.

    Cluster identifiers are deliberately not returned: they are arbitrary,
    change across refits and are not classifications of any named industry.
    Cohorts with fewer than ten complete adjusted-momentum members fail closed.
    """
    from research_dc_themes import stock_momentum

    required = {'date', 'ts_code', *PRICE_COLUMNS}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f'Missing peer daily columns: {sorted(missing)}')
    stocks = stock_momentum(monthly)
    if stocks.groupby(stocks.date.dt.to_period('M')).date.nunique().gt(1).any():
        raise ValueError('Monthly stock observations must share actual month-end signal dates')
    stocks = stocks.rename(columns={'date': 'signal_date'})
    signals = (_dates(sorted(stocks.loc[stocks.signal_date.ge(pd.Timestamp('2023-01-01')),
                                       'signal_date'].unique()), 'Peer signals')
               if signal_dates is None else _dates(signal_dates, 'Peer signals'))
    if signals.has_duplicates:
        raise ValueError('Duplicate peer signal dates')
    signals = signals.sort_values()
    if signals.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if not signals.isin(stocks.signal_date.unique()).all():
        raise ValueError('Peer signal is not an actual monthly quote date')
    frame = daily[['date', 'ts_code'] + PRICE_COLUMNS].copy()
    frame['date'] = _dates(frame.date, 'Peer daily dates')
    if frame.duplicated(['date', 'ts_code']).any():
        raise ValueError('Duplicate peer daily stock/date keys')
    # A future listing/quote must not alter the current fitting sample/order.
    frame = frame[frame.date.le(signals.max()) & frame.ts_code.astype('string').str.match(
        r'^(?:0\d{5}\.SZ|3\d{5}\.SZ|6\d{5}\.SH)$', na=False)].copy()
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    calendar = _dates(sorted(frame.date.unique()), 'Peer session calendar')
    positions = calendar.get_indexer(signals)
    if (positions < 0).any():
        raise ValueError('Peer signal is absent from the daily market calendar')
    warm = positions >= 59
    signals, positions = signals[warm], positions[warm]
    if signals.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    start, end = int(positions.min()) - 59, int(positions.max())
    calendar = calendar[start:end + 1]
    positions -= start
    frame = frame[frame.date.between(calendar[0], calendar[-1])]
    codes = pd.Index(sorted(frame.ts_code.unique()))
    date_index, code_index = calendar.get_indexer(frame.date), codes.get_indexer(frame.ts_code)
    values = np.full((len(PRICE_COLUMNS), len(calendar), len(codes)), np.nan)
    for number, column in enumerate(PRICE_COLUMNS):
        values[number, date_index, code_index] = pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=float)
    opening, high, low, close, volume = values
    valid = (np.isfinite(values).all(axis=0) & (values > 0).all(axis=0)
             & (high >= np.maximum(opening, close))
             & (low <= np.minimum(opening, close)) & (high >= low))
    with np.errstate(divide='ignore', invalid='ignore'):
        pressure = np.log(close / opening)
    pressure[~valid] = np.nan
    # Invalid or absent per-stock rows never contribute to the daily benchmark.
    day_median = pd.DataFrame(pressure).median(axis=1, skipna=True).to_numpy()
    pressure -= day_median[:, None]
    blocks = []
    momenta = ['mom1', 'mom3', 'mom6']
    for signal, index in zip(signals, positions):
        window = slice(index - 59, index + 1)
        good = valid[window].all(axis=0)
        patterns = pressure[window, :][:, good].T.copy()
        pattern_codes = codes[good]
        if len(patterns) < 2:
            continue
        patterns -= patterns.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(patterns, axis=1)
        nonzero = np.isfinite(patterns).all(axis=1) & np.isfinite(norms) & (norms > 1e-12)
        patterns, pattern_codes, norms = patterns[nonzero], pattern_codes[nonzero], norms[nonzero]
        if len(patterns) < 2:
            continue
        patterns /= norms[:, None]
        clusters = 20 if len(patterns) >= 40 else max(2, len(patterns) // 2)
        estimator = MiniBatchKMeans(n_clusters=clusters, **CLUSTER_PARAMS)
        with threadpool_limits(limits=4):
            labels = estimator.fit_predict(patterns)
        cohort = pd.DataFrame(dict(ts_code=pattern_codes, cohort=labels))
        asof = stocks.loc[stocks.signal_date.eq(signal), ['ts_code'] + momenta]
        cohort = cohort.merge(asof, on='ts_code', how='inner', validate='one_to_one')
        cohort = cohort[np.isfinite(cohort[momenta].to_numpy(dtype=float)).all(axis=1)].copy()
        if cohort.empty:
            continue
        cohort['positive3'] = cohort.mom3.gt(0).astype(float)
        summaries = cohort.groupby('cohort').agg(
            peer_mom1=('mom1', 'median'), peer_mom3=('mom3', 'median'),
            peer_mom6=('mom6', 'median'), peer_positive3=('positive3', 'mean'),
            peer_count=('ts_code', 'size'))
        cohort = cohort.merge(summaries, on='cohort', how='left', validate='many_to_one')
        cohort = cohort[cohort.peer_count.ge(10)]
        if cohort.empty:
            continue
        cohort.insert(0, 'signal_date', signal)
        blocks.append(cohort[OUTPUT_COLUMNS])
    output = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=OUTPUT_COLUMNS)
    return output.sort_values(['signal_date', 'ts_code']).reset_index(drop=True)
