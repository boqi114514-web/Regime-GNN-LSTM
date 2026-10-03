"""Safely seed only explicitly observed full-day suspension cache dates.

The official suspend_d endpoint describes daily records, not an interval
lasting indefinitely after its last S row. This helper never manufactures S
rows or assumes that an old S event continues because no later R was returned.
Existing shared cache files are never overwritten. A successful full-history
response is saved separately; only dates with their own full-day S record can
be published into the account cache.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import re
import time
import uuid

import numpy as np
import pandas as pd

from data_pipeline.tushare_config import GatewayClient
from research_daily_graph_data import load_adjusted_daily, publish_manifest, sha256


DOCUMENTATION = 'https://tushare.pro/document/2?doc_id=214'
OFFICIAL_ROW_LIMIT = 5000


def validate_history(frame, code, start, end):
    required = {'ts_code', 'trade_date', 'suspend_type', 'suspend_timing'}
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or not required <= set(frame):
        raise ValueError('Missing original suspension-history schema')
    if len(frame) >= OFFICIAL_ROW_LIMIT:
        raise ValueError('Full-history response may be capped; do not assume coverage')
    if not frame.ts_code.eq(code).all() or not frame.suspend_type.isin(['S', 'R']).all():
        raise ValueError('Wrong stock or invalid suspension type')
    dates = pd.to_datetime(frame.trade_date.astype(str), format='%Y%m%d', errors='raise')
    if dates.isna().any() or not dates.between(pd.Timestamp(start), pd.Timestamp(end)).all():
        raise ValueError('Suspension history has missing/out-of-request dates')
    if frame.duplicated(['ts_code', 'trade_date', 'suspend_type', 'suspend_timing']).any():
        raise ValueError('Duplicate suspension history rows')
    if not frame.suspend_timing.map(lambda value: pd.isna(value) or isinstance(value, str)).all():
        raise ValueError('Invalid suspension timing')
    return frame.copy()


def explicit_full_day_dates(history):
    """An absent daily record never creates a publishable suspension date."""
    dates = pd.to_datetime(history.trade_date.astype(str), format='%Y%m%d')
    full_day = history.suspend_timing.isna() | history.suspend_timing.fillna('').str.strip().eq('')
    stops = set(dates[history.suspend_type.eq('S') & full_day])
    resumes = set(dates[history.suspend_type.eq('R')])
    partial = set(dates[history.suspend_type.eq('S') & ~full_day])
    return sorted(stops-resumes-partial)


def slice_for_day(history, last_quote, day):
    day, last_quote = pd.Timestamp(day), pd.Timestamp(last_quote)
    if day not in explicit_full_day_dates(history) or last_quote >= day:
        raise ValueError('No explicit full-day suspension evidence for requested missing-quote date')
    dates = pd.to_datetime(history.trade_date.astype(str), format='%Y%m%d')
    result = history.loc[dates.gt(last_quote) & dates.le(day)].copy()
    if result.empty:
        raise ValueError('No post-quote suspension evidence')
    return result


def publish_exclusive_pickle(frame, target):
    """A hard-link publication is atomic and refuses an existing destination."""
    target = Path(target)
    if target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name+'.'+uuid.uuid4().hex+'.staging')
    try:
        with temporary.open('xb') as handle:
            pickle.dump(frame, handle, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            os.link(temporary, target)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def recent_active_date(cache, codes, *, max_age_seconds=600):
    """Conservative current-run guard; never write its date or the next two sessions."""
    recent = []
    now = time.time()
    for code in codes:
        for path in Path(cache).glob(code+'_*_daily.pkl'):
            match = re.fullmatch(re.escape(code)+r'_(\d{8})_daily\.pkl', path.name)
            if match and now-path.stat().st_mtime <= max_age_seconds:
                recent.append(pd.Timestamp(match.group(1)))
    return max(recent) if recent else None


def fill_observed_dates(history, code, daily, cache, *, after, through, active_codes=()):
    cache = Path(cache)
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(daily.date).unique()))
    quoted = daily.loc[daily.ts_code.eq(code)].copy()
    quoted['date'] = pd.to_datetime(quoted.date)
    if quoted.duplicated('date').any() or not np.isfinite(quoted.close).all() or quoted.close.le(0).any():
        raise ValueError('Invalid original stock quote rows')
    records = []
    for day in explicit_full_day_dates(history):
        if day <= pd.Timestamp(after) or day > pd.Timestamp(through) or day not in sessions:
            continue
        active = recent_active_date(cache, active_codes)
        if active is not None:
            position = sessions.searchsorted(active, side='right')
            if day <= sessions[min(position+1, len(sessions)-1)]:
                continue
        if quoted.date.eq(day).any():
            continue  # This is not a missing-quote holding mark.
        prior = quoted.loc[quoted.date.lt(day)].sort_values('date')
        if prior.empty:
            continue
        last = prior.iloc[-1]
        observed = slice_for_day(history, last.date, day)
        # These are original daily API quote rows, not synthetic forward fills.
        # Retain the actual last quote even if the suspension exceeds 90 days.
        raw = prior.loc[prior.date.ge(day-pd.Timedelta(days=90))].copy()
        if raw.empty:
            raw = prior.tail(1).copy()
        raw['trade_date'] = raw.date.dt.strftime('%Y%m%d')
        for kind, frame in (('daily', raw), ('suspend', observed)):
            target = cache/f'{code}_{day:%Y%m%d}_{kind}.pkl'
            created = publish_exclusive_pickle(frame, target)
            if created:
                records.append(dict(path=str(target.resolve()), sha256=sha256(target),
                                    code=code, day=str(day.date()), kind=kind,
                                    last_original_quote=str(last.date.date())))
    return records


def run(codes, output, cache, inputs, *, after, through):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    gateway = GatewayClient(os.environ['TUSHARE_API_KEY'])
    daily, daily_meta = load_adjusted_daily(inputs)
    reports = []
    for code in codes:
        source = output/f'{code}_full_history.pkl'
        meta_path = output/f'{code}_full_history.json'
        params = dict(ts_code=code, start_date='20000103', end_date=pd.Timestamp(through).strftime('%Y%m%d'))
        if source.exists():
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            if meta['params'] != params or meta['sha256'] != sha256(source):
                raise ValueError('Original full-history artifact scope/hash differs')
            history = pd.read_pickle(source)
        else:
            # Exactly one real API request per stock. Network failures are
            # reported; no old interval is substituted for missing evidence.
            history = gateway.query('suspend_d', **params)
            history = validate_history(history, code, '20000103', through)
            if not publish_exclusive_pickle(history, source):
                raise ValueError('Concurrent full-history collection; retry from existing artifact')
            publish_manifest(dict(api='suspend_d', params=params, sha256=sha256(source),
                                  retrieved_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                                  rows=len(history), documentation=DOCUMENTATION,
                                  policy='Only explicitly observed full-day S dates may seed caches'), meta_path)
        history = validate_history(history, code, '20000103', through)
        created = fill_observed_dates(history, code, daily, cache, after=after, through=through,
                                      active_codes=codes)
        record = dict(code=code, source=str(source.resolve()), source_sha256=sha256(source),
                      rows=len(history), latest_observed_date=(str(pd.to_datetime(history.trade_date).max().date())
                                                            if len(history) else None),
                      explicit_dates_after_boundary=sum(day > pd.Timestamp(after) for day in explicit_full_day_dates(history)),
                      created=created, daily_source_sha256=daily_meta['adjusted_daily']['sha256'])
        reports.append(record)
        print(json.dumps(dict(code=code, rows=record['rows'], latest_observed_date=record['latest_observed_date'],
                              explicit_dates_after_boundary=record['explicit_dates_after_boundary'],
                              created_files=len(created))), flush=True)
        publish_manifest(dict(status='complete_for_finished_requests', reports=reports,
                              no_absence_inference=True, existing_caches_overwritten=False,
                              source_sha256=sha256(__file__)), output/'completion_manifest.json')
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codes', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('data/raw/execution_v1/suspension_checks'))
    parser.add_argument('--inputs', type=Path, default=Path('data/raw/daily_opportunity_20261002'))
    parser.add_argument('--after', required=True)
    parser.add_argument('--through', default='2026-09-24')
    args = parser.parse_args()
    run(args.codes, args.output, args.cache, args.inputs, after=args.after, through=args.through)
