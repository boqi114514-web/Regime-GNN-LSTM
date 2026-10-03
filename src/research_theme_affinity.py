"""Causal price affinity, not evidence of historical theme membership.

Raw close-to-close returns are technical observations, not adjusted investment
returns. Each signal uses 60 local trading-session returns from 61 prices,
clips returns to +/-20%, and subtracts the all-SH/SZ cross-sectional median.
Missing observations are never filled or replaced with a longer return lag.
"""
import re

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = ['signal_date', 'ts_code', 'affinity_theme_code',
                  'affinity_theme_name', 'affinity_theme_score', 'affinity_corr', 'nobs']
WINDOW = 60
MIN_OBSERVATIONS = 40
_VARIANCE_FLOOR = 1e-20


def _empty():
    return pd.DataFrame({column: pd.Series(dtype=(
        'datetime64[ns]' if column == 'signal_date' else 'int64' if column == 'nobs'
        else 'float64' if column in ('affinity_theme_score', 'affinity_corr') else 'object'))
        for column in OUTPUT_COLUMNS})


def _required(frame, columns, label):
    if not isinstance(frame, pd.DataFrame) or not set(columns) <= set(frame.columns):
        raise ValueError(f'{label} lacks required columns: {columns}')
    if not frame.columns.is_unique:
        raise ValueError(f'{label} has duplicate columns')


def _dates(values, label):
    text = pd.Series(values, copy=False).astype('string')
    if not text.str.fullmatch(r'\d{8}|\d{4}-\d{2}-\d{2}(?:[ T]00:00:00(?:\.0+)?)?', na=False).all():
        raise ValueError(f'Invalid {label}: expected midnight calendar dates')
    parsed = pd.to_datetime(text, format='mixed', errors='coerce')
    if parsed.isna().any() or parsed.dt.tz is not None or not parsed.eq(parsed.dt.normalize()).all():
        raise ValueError(f'Invalid {label}: missing, timezone, or intraday date')
    return parsed


def _codes(values, pattern, label):
    codes = pd.unique(values)
    if any(not isinstance(code, str) or re.fullmatch(pattern, code) is None for code in codes):
        raise ValueError(f'Invalid {label} code')
    return codes


def _theme_audit(features, signals):
    _required(features, ['signal_date', 'ts_code', 'name', 'ret20', 'ret60'], 'Theme features')
    dates = _dates(features.signal_date, 'theme signal dates')
    selected = dates.isin(signals)
    f = features.loc[selected].copy()
    f['signal_date'] = dates.loc[selected].to_numpy()
    if f.empty:
        return f
    _codes(f.ts_code, r'BK\d{4}\.DC', 'DC theme')
    if f.duplicated(['signal_date', 'ts_code']).any():
        raise ValueError('Duplicate dated theme keys')
    if not f.name.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError('Invalid dated theme name')
    if 'source_last_date' in f:
        if _dates(f.source_last_date, 'theme source dates').gt(f.signal_date).any():
            raise ValueError('Theme features contain future observations')
    if 'catalogue_date' in f:
        if not _dates(f.catalogue_date, 'catalogue dates').eq(f.signal_date).all():
            raise ValueError('Catalogue date differs from the signal session')
    audit_fields = ['theme_score', 'history_complete', 'theme_comparable']
    if not set(audit_fields) <= set(f.columns):
        from research_dc_themes import rank_themes
        f = rank_themes(f)
    for flag in ('history_complete', 'theme_comparable'):
        if not f[flag].map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError(f'Invalid audit flag: {flag}')
    for column in ('ret20', 'ret60', 'theme_score'):
        f[column] = pd.to_numeric(f[column], errors='coerce')
    eligible = f.history_complete & f.theme_comparable & (f.ret20.gt(0) | f.ret60.gt(0))
    f = f.loc[eligible].copy()
    if (not np.isfinite(f[['ret20', 'ret60', 'theme_score']]).all(axis=None)
            or not f.theme_score.between(0, 1).all()):
        raise ValueError('Invalid qualified theme audit values')
    return f.sort_values(['signal_date', 'ts_code']).reset_index(drop=True)


def _pairwise_pearson(stock, themes):
    """Pairwise complete-observation Pearson using matrix sufficient statistics."""
    mx, my = np.isfinite(stock).astype(float), np.isfinite(themes).astype(float)
    x, y = np.where(mx, stock, 0.), np.where(my, themes, 0.)
    # Center columns first to avoid cancellation and spurious correlation of
    # constant series with residual floating-point round-off after demeaning.
    x = np.where(mx, stock - np.divide(x.sum(axis=0), mx.sum(axis=0),
                 out=np.zeros(stock.shape[1]), where=mx.sum(axis=0) > 0), 0.)
    y = np.where(my, themes - np.divide(y.sum(axis=0), my.sum(axis=0),
                 out=np.zeros(themes.shape[1]), where=my.sum(axis=0) > 0), 0.)
    counts = mx.T @ my
    sx, sy = x.T @ my, mx.T @ y
    inv = np.divide(1., counts, out=np.zeros_like(counts), where=counts > 0)
    vx = (x*x).T @ my - sx*sx*inv
    vy = mx.T @ (y*y) - sy*sy*inv
    covariance = x.T @ y - sx*sy*inv
    valid = ((counts >= MIN_OBSERVATIONS) & (vx > counts*_VARIANCE_FLOOR)
             & (vy > counts*_VARIANCE_FLOOR))
    denominator = np.sqrt(np.maximum(vx, 0.)*np.maximum(vy, 0.))
    correlation = np.divide(covariance, denominator, out=np.full_like(counts, np.nan),
                            where=valid & (denominator > 0))
    return np.clip(correlation, -1., 1.), counts


def affinity_features(features, index_market, daily, signal_dates):
    """Return unique matched signal/stock rows with positive causal affinity.

    ``features`` may be a complete ``rank_themes`` audit (whose fixed scores
    are retained) or its raw feature inputs. All qualified, comparable themes
    participate, including those outside the top five. Unmatched stocks are
    omitted; an empty result retains OUTPUT_COLUMNS. ``daily`` must retain the
    whole SH/SZ stock universe used to compute the market median. A caller may
    filter the resulting stock rows, but must not shrink that median's input.
    """
    if isinstance(signal_dates, (str, bytes)):
        raise ValueError('signal_dates must be a collection')
    signals = pd.DatetimeIndex(_dates(pd.Series(list(signal_dates)), 'signal dates'))
    if signals.has_duplicates or signals.to_period('M').has_duplicates:
        raise ValueError('Duplicate signal dates or multiple signal sessions in one month')
    if signals.empty:
        return _empty()
    signals = signals.sort_values()
    audit = _theme_audit(features, signals)
    _required(daily, ['date', 'ts_code', 'close', 'volume'], 'Stock daily quotes')
    _required(index_market, ['trade_date', 'ts_code', 'close'], 'Index market quotes')
    daily_dates = _dates(daily.date, 'stock dates')
    stock_codes = _codes(daily.ts_code, r'\d{6}\.(?:SH|SZ|BJ)', 'stock')
    stock_codes = sorted(code for code in stock_codes if code.endswith(('.SH', '.SZ')))
    if not stock_codes:
        raise ValueError('No SH/SZ stocks for the market median')
    stock_mask = daily.ts_code.isin(stock_codes) & daily_dates.le(signals[-1])
    calendar = pd.DatetimeIndex(pd.unique(daily_dates.loc[stock_mask])).sort_values()
    positions = calendar.get_indexer(signals)
    if (positions < WINDOW).any():
        raise ValueError('Every signal must have 61 exact local calendar price sessions')
    if audit.empty:
        return _empty()
    calendar = calendar[positions.min()-WINDOW:positions.max()+1]
    stock_mask &= daily_dates.ge(calendar[0])
    stocks = daily.loc[stock_mask, ['ts_code', 'close', 'volume']].copy()
    stocks['date'] = daily_dates.loc[stock_mask].to_numpy()
    if stocks.duplicated(['date', 'ts_code']).any():
        raise ValueError('Duplicate stock/date observations')
    for column in ('close', 'volume'):
        stocks[column] = pd.to_numeric(stocks[column], errors='coerce')
    valid = np.isfinite(stocks[['close', 'volume']]).all(axis=1) & stocks.close.gt(0) & stocks.volume.gt(0)
    stocks.loc[~valid, 'close'] = np.nan
    stock_prices = stocks.pivot(index='date', columns='ts_code', values='close').reindex(calendar)
    stock_returns = (stock_prices/stock_prices.shift(1)-1).clip(-.20, .20)
    market = stock_returns.median(axis=1, skipna=True)
    stock_returns = stock_returns.sub(market, axis=0)

    index_dates = _dates(index_market.trade_date, 'index dates')
    _codes(index_market.ts_code, r'BK\d{4}\.DC', 'DC index')
    index_mask = index_dates.between(calendar[0], calendar[-1])
    indices = index_market.loc[index_mask, ['ts_code', 'close']].copy()
    indices['date'] = index_dates.loc[index_mask].to_numpy()
    if indices.duplicated(['date', 'ts_code']).any():
        raise ValueError('Duplicate index/date observations')
    indices['close'] = pd.to_numeric(indices.close, errors='coerce')
    indices.loc[~np.isfinite(indices.close) | indices.close.le(0), 'close'] = np.nan
    index_prices = indices.pivot(index='date', columns='ts_code', values='close').reindex(calendar)
    index_returns = (index_prices/index_prices.shift(1)-1).clip(-.20, .20).sub(market, axis=0)

    matched = []
    for signal in signals:
        themes = audit.loc[audit.signal_date.eq(signal)].sort_values('ts_code')
        if themes.empty:
            continue
        end = calendar.get_loc(signal)
        sessions = calendar[end-WINDOW+1:end+1]
        x = stock_returns.loc[sessions].to_numpy(dtype=float)
        y = index_returns.reindex(columns=themes.ts_code).loc[sessions].to_numpy(dtype=float)
        # Bound temporary pairwise matrices while preserving the all-stock median.
        for first in range(0, x.shape[1], 512):
            corr, counts = _pairwise_pearson(x[:, first:first+512], y)
            positive = np.isfinite(corr) & (corr > 0)
            maxima = np.where(positive, corr, -np.inf).max(axis=1)
            rows = np.flatnonzero(maxima > 0)
            if not len(rows):
                continue
            ties = positive[rows] & np.isclose(corr[rows], maxima[rows, None], atol=1e-12, rtol=0)
            choices = ties.argmax(axis=1)  # Columns are already ordered by theme code.
            for row, choice in zip(rows, choices):
                theme = themes.iloc[choice]
                matched.append(dict(signal_date=signal, ts_code=stock_prices.columns[first+row],
                    affinity_theme_code=theme.ts_code, affinity_theme_name=theme['name'],
                    affinity_theme_score=float(theme.theme_score), affinity_corr=float(corr[row, choice]),
                    nobs=int(counts[row, choice])))
    if not matched:
        return _empty()
    result = pd.DataFrame(matched, columns=OUTPUT_COLUMNS).sort_values(['signal_date', 'ts_code']).reset_index(drop=True)
    if result.duplicated(['signal_date', 'ts_code']).any():
        raise ValueError('Duplicate affinity output keys')
    return result
