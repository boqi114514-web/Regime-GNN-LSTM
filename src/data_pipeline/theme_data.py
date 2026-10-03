"""Pure, fail-closed validators for dated concept/industry source snapshots.

Sources: Tushare docs 362 (dc_index), 363 (dc_member), 382 (dc_daily),
and 261 (ths_member). In particular, a current THS membership response must
not be backfilled into history. DC/TDX catalog/member adapters must supply
the requested historical trade_date; a current undated catalog is not one.

These functions never fetch, write, deduplicate, fill, or silently filter data.
Passing validation establishes consistency with the request, not source truth.
Callers must pass the endpoint/request row limit when known; reaching it is
ambiguous truncation and is rejected. Row count alone cannot detect a response
silently truncated below its advertised limit. Calendar coverage additionally
requires expected_sessions for index bars.
"""
from datetime import date, datetime
import re

import numpy as np
import pandas as pd


def _day(value, label):
    if isinstance(value, (str, int, np.integer)) and not isinstance(value, bool):
        text = str(value)
        fmt = '%Y%m%d' if re.fullmatch(r'\d{8}', text) else '%Y-%m-%d'
        if fmt == '%Y-%m-%d' and not re.fullmatch(r'\d{4}-\d{2}-\d{2}', text):
            raise ValueError(f'Invalid {label}: expected a calendar date')
        try:
            result = pd.to_datetime(text, format=fmt, errors='raise')
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Invalid {label}') from exc
    elif isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        result = pd.Timestamp(value)
    else:
        raise ValueError(f'Invalid {label}: expected a calendar date')
    if pd.isna(result) or result.tzinfo is not None or result != result.normalize():
        raise ValueError(f'Invalid {label}: missing, timezone, or intraday value')
    return result.strftime('%Y%m%d')


def _checked_frame(frame, required, limit):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError('Response is not a DataFrame')
    if not frame.columns.is_unique:
        raise ValueError('Duplicate response columns')
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if frame.empty:
        raise ValueError('Empty response cannot establish complete coverage')
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or limit <= 0:
            raise ValueError('limit must be a positive integer or None')
        if len(frame) >= limit:
            raise ValueError('Response reached limit; possible truncation requires audited pagination')
    result = frame.copy(deep=True)
    result['trade_date'] = result.trade_date.map(lambda value: _day(value, 'response trade_date'))
    return result


def _strings(frame, columns):
    for column in columns:
        good = frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())
                                 and value == value.strip())
        if not good.all():
            raise ValueError(f'Missing/invalid {column}')


def _requested_code(ts_code):
    if not isinstance(ts_code, str) or not ts_code.strip() or ts_code != ts_code.strip():
        raise ValueError('Invalid requested ts_code')
    return ts_code


def validate_catalog(frame, trade_date, *, limit=None):
    """Return a copy of one dated DC/TDX catalog, rejecting date/filter drift.

    trade_date accepts YYYYMMDD, YYYY-MM-DD, or a midnight date-like scalar.
    Returned trade_date is canonical YYYYMMDD. Extra columns are preserved.
    A duplicate code is rejected even if both rows happen to be identical.
    """
    requested = _day(trade_date, 'requested trade_date')
    result = _checked_frame(frame, ['ts_code', 'name', 'trade_date'], limit)
    _strings(result, ['ts_code', 'name'])
    if not result.trade_date.eq(requested).all():
        raise ValueError('Catalog trade_date differs from requested historical date')
    if result.ts_code.duplicated().any():
        raise ValueError('Duplicate catalog ts_code; conflicting snapshots are not deduplicated')
    return result


def validate_members(frame, trade_date, ts_code, *, limit=None):
    """Return a copy of membership for exactly one historical date and theme.

    SH/SZ/BJ or other source code syntax is not rewritten here, and upstream
    memberships are not restricted to account-tradable boards. The request's
    theme code and date must match every returned row; con_code must be unique.
    """
    requested_date = _day(trade_date, 'requested trade_date')
    requested_code = _requested_code(ts_code)
    result = _checked_frame(frame, ['trade_date', 'ts_code', 'con_code'], limit)
    _strings(result, ['ts_code', 'con_code'])
    if not result.trade_date.eq(requested_date).all():
        raise ValueError('Membership trade_date differs from requested historical date')
    if not result.ts_code.eq(requested_code).all():
        raise ValueError('Membership ts_code differs from requested theme')
    if result.con_code.duplicated().any():
        raise ValueError('Duplicate member con_code')
    return result


def validate_index_bars(frame, ts_code, start_date, end_date, *,
                        expected_sessions=None, limit=None):
    """Validate one concept index's OHLCV, optionally requiring full calendar.

    Extra/out-of-range rows are errors, never silently removed. If provided,
    expected_sessions must contain unique in-range calendar dates and must
    exactly match returned dates. Output preserves row order and extra columns.
    """
    requested_code = _requested_code(ts_code)
    start, end = _day(start_date, 'start_date'), _day(end_date, 'end_date')
    if start > end:
        raise ValueError('start_date must not follow end_date')
    result = _checked_frame(frame, ['ts_code', 'trade_date', 'open', 'high',
                                    'low', 'close', 'vol'], limit)
    _strings(result, ['ts_code'])
    if not result.ts_code.eq(requested_code).all():
        raise ValueError('Index ts_code differs from requested code')
    if not result.trade_date.between(start, end).all():
        raise ValueError('Index bars outside requested date interval')
    if result.trade_date.duplicated().any():
        raise ValueError('Duplicate index trade_date')
    for column in ['open', 'high', 'low', 'close', 'vol']:
        values = pd.to_numeric(result[column], errors='coerce')
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f'Non-finite/invalid index {column}')
        if (values < 0).any() or (column != 'vol' and values.eq(0).any()):
            raise ValueError(f'Invalid sign for index {column}')
        result[column] = values
    if (result.high.lt(result[['open', 'close', 'low']].max(axis=1)).any()
            or result.low.gt(result[['open', 'close', 'high']].min(axis=1)).any()):
        raise ValueError('Invalid index OHLC geometry')
    if expected_sessions is not None:
        if isinstance(expected_sessions, (str, bytes)):
            raise ValueError('expected_sessions must be a collection of dates')
        expected = [_day(value, 'expected session') for value in expected_sessions]
        if len(set(expected)) != len(expected):
            raise ValueError('Duplicate expected sessions')
        if any(value < start or value > end for value in expected):
            raise ValueError('Expected sessions outside requested date interval')
        actual = set(result.trade_date)
        if actual != set(expected):
            missing, extra = sorted(set(expected) - actual), sorted(actual - set(expected))
            raise ValueError(f'Incomplete index calendar: missing={missing}, extra={extra}')
    return result


def reject_latest_membership_for_history(source):
    """Reject THS/current-only membership sources for historical reconstruction.

    A non-THS source is not thereby certified point-in-time: its historical
    snapshots must still pass validate_members for each requested date.
    """
    if not isinstance(source, str) or not source.strip():
        raise ValueError('Membership source must be specified')
    normalized = source.strip().lower()
    if 'ths' in normalized or 'tonghuashun' in normalized or '同花顺' in normalized:
        raise ValueError('THS current membership cannot be used as historical membership or backfilled')
