"""Small explicit THS/DC pilot; only validated complete runs are published.

Uses the new GET gateway with TUSHARE_API_KEY only. Index requests are split
by calendar month with explicit limits; DC memberships are historical snapshots
from 2024-12-20 onward. No THS current-membership backfill and no SW fallback.
"""
import argparse
from datetime import date, datetime, timezone
import json
from numbers import Integral
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_pipeline.theme_data import validate_index_bars, validate_members
from probe_concept_gateway import probe_request, safe_text


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'data/raw/theme_pilot'
DC_MEMBER_START = '20241220'


def _day(value):
    try:
        if isinstance(value, (str, Integral)) and not isinstance(value, bool):
            text = str(value)
            if not re.fullmatch(r'\d{8}|\d{4}-\d{2}-\d{2}', text):
                raise ValueError('Expected YYYYMMDD or YYYY-MM-DD')
            result = pd.to_datetime(text, format='%Y%m%d' if len(text) == 8 else '%Y-%m-%d')
        elif isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
            result = pd.Timestamp(value)
        else:
            raise ValueError('Expected a calendar date')
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid calendar date') from exc
    if pd.isna(result) or result.tzinfo is not None or result != result.normalize():
        raise ValueError('Dates must be valid midnight calendar dates')
    return result.strftime('%Y%m%d')


def _codes(values, pattern, label):
    if isinstance(values, (str, bytes)):
        raise ValueError(f'{label} codes must be a collection')
    codes = list(values)
    if any(not isinstance(code, str) or not re.fullmatch(pattern, code) for code in codes):
        raise ValueError(f'{label} requires unique source-specific index codes')
    if len(codes) != len(set(codes)):
        raise ValueError(f'{label} requires unique source-specific index codes')
    return codes


def build_requests(ths_codes, dc_codes, start_date, end_date, snapshot_dates, *,
                   bars_limit=100, members_limit=1000):
    """Build the bounded-by-user-scope plan without making requests."""
    ths = _codes(ths_codes, r'\d{6}\.TI', 'THS')
    dc = _codes(dc_codes, r'BK\d{4}\.DC', 'DC')
    if not ths and not dc:
        raise ValueError('Specify at least one THS or DC index code')
    for limit in (bars_limit, members_limit):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError('Request limits must be positive integers')
    start, end = _day(start_date), _day(end_date)
    if start > end:
        raise ValueError('start_date must not follow end_date')
    snapshots = [_day(value) for value in snapshot_dates]
    if len(snapshots) != len(set(snapshots)):
        raise ValueError('Duplicate snapshot dates')
    if dc and not snapshots:
        raise ValueError('DC pilot requires explicit historical snapshot dates')
    if snapshots and not dc:
        raise ValueError('Snapshot dates require DC codes; THS current members are not historical')
    if any(value < DC_MEMBER_START for value in snapshots):
        raise ValueError('DC historical membership is supported only from 20241220')
    chunks = []
    for month in pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq='M'):
        first = max(start, month.start_time.strftime('%Y%m%d'))
        last = min(end, month.end_time.strftime('%Y%m%d'))
        chunks.append((first, last))
    plan = []
    for source, codes in [('ths', ths), ('dc', dc)]:
        for code in codes:
            for first, last in chunks:
                plan.append(dict(name=f'{source}_daily_{code}_{first}_{last}', api=f'{source}_daily',
                                 kind='bars', source=source,
                                 params=dict(ts_code=code, start_date=first, end_date=last, limit=bars_limit)))
    for code in dc:
        for day in sorted(snapshots):
            plan.append(dict(name=f'dc_member_{code}_{day}', api='dc_member', kind='members', source='dc',
                             params=dict(ts_code=code, trade_date=day, limit=members_limit)))
    return plan


def _retryable(record):
    if record['status'] in {'unexpected_error', 'raw_frame_write_error'}:
        return False
    if record['status'] == 'transport_error':
        return record.get('transport_error_type') in {
            'Timeout', 'ConnectTimeout', 'ReadTimeout', 'ConnectionError', 'ChunkedEncodingError'}
    if record.get('http_status') == 504:
        return True
    diagnostics = ' '.join(str(value) for value in record.get('diagnostics', {}).values()).lower()
    return record.get('http_status') == 503 and 'upstream_pool_exhausted' in diagnostics


def _write_json(path, value):
    """Replace metadata atomically so an interrupted write cannot corrupt it."""
    pending = path.with_name(path.name + '.pending')
    try:
        pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def _write_raw_frame(path, frame):
    pending = path.with_name(path.name + '.pending')
    try:
        frame.to_pickle(pending)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def _publish_aggregates(directory, aggregates):
    """Stage every aggregate first and remove this run's outputs on failure."""
    targets = {api: directory / f'validated_{api}.pkl' for api in aggregates}
    pending = {api: path.with_name(path.name + '.pending') for api, path in targets.items()}
    try:
        for api, frame in aggregates.items():
            frame.to_pickle(pending[api])
        for api, path in targets.items():
            pending[api].replace(path)
    except BaseException:
        for path in targets.values():
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in pending.values():
            path.unlink(missing_ok=True)
    return {api: dict(file=path.name, rows=len(aggregates[api])) for api, path in targets.items()}


def _sessions_for(sessions, first, last):
    return None if sessions is None else [day for day in sessions if first <= day <= last]


def run_pilot(ths_codes, dc_codes, start_date, end_date, snapshot_dates, *,
              output_dir=DEFAULT_OUTPUT, expected_sessions=None, bars_limit=100,
              members_limit=1000, session=None, environ=None):
    """Return the run manifest; each request gets at most two HTTP attempts.

    Failed runs retain attempt-level raw frames and diagnostics, but create no
    validated aggregate. Passing expected_sessions asserts exact date coverage
    within each requested interval; it must come from a trusted local calendar.
    """
    plan = build_requests(ths_codes, dc_codes, start_date, end_date, snapshot_dates,
                          bars_limit=bars_limit, members_limit=members_limit)
    if isinstance(expected_sessions, (str, bytes)):
        raise ValueError('Expected sessions must be a collection of dates')
    sessions = None if expected_sessions is None else [_day(day) for day in expected_sessions]
    if sessions is not None and len(set(sessions)) != len(sessions):
        raise ValueError('Expected sessions must be unique')
    env = os.environ if environ is None else environ
    key = env.get('TUSHARE_API_KEY', '').strip()
    if not key:
        raise ValueError('Missing TUSHARE_API_KEY; legacy credentials are not used')
    secrets = (key, env.get('TUSHARE_TOKEN', '').strip())
    directory = Path(output_dir) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    raw_dir = directory / 'attempts'
    raw_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', transport='gateway', output_dir=str(directory.resolve()),
                    planned_requests=len(plan), calendar_coverage_checked=sessions is not None,
                    units={'ths_daily': {'vol': '手', 'amount': 'not_provided_by_documented_gateway_schema'},
                           'dc_daily': {'vol': '股', 'amount': '元'}},
                    units_policy='documented_source_units; values_are_not_converted_or_unit_certified',
                    membership={'dc_earliest_date': DC_MEMBER_START,
                                'ths': 'current_membership_forbidden_for_history'},
                    no_partial_validated_outputs=True, requests=[], validated_files={})
    def save_manifest():
        _write_json(directory / 'manifest.json', manifest)
    save_manifest()
    owned_session = session is None
    validated = {'ths_daily': [], 'dc_daily': [], 'dc_member': []}
    try:
        if owned_session:
            session = requests.Session()
        for request_number, spec in enumerate(plan, 1):
            request_audit = dict(name=spec['name'], api=spec['api'], params=spec['params'], status='failed', attempts=[])
            manifest['requests'].append(request_audit)
            for attempt in (1, 2):
                try:
                    record, frame = probe_request(session, 'gateway', key, spec, secrets=secrets)
                except Exception as exc:
                    record, frame = dict(name=spec['name'], api=spec['api'], params=spec['params'],
                                         transport='gateway', status='unexpected_error', http_status=None,
                                         diagnostics={'error': safe_text(str(exc), secrets)},
                                         error_type=type(exc).__name__, raw_frame_saved=False), None
                record['attempt'] = attempt
                stem = f'{request_number:03d}_{spec["name"]}_attempt{attempt}'
                if frame is not None:
                    try:
                        _write_raw_frame(raw_dir / f'{stem}.pkl', frame)
                        record.update(raw_frame_saved=True, raw_frame_file=f'attempts/{stem}.pkl')
                    except Exception as exc:
                        record.update(probe_status=record['status'], status='raw_frame_write_error',
                                      raw_frame_save_error=safe_text(str(exc), secrets))
                        frame = None
                result = None
                if record['status'] == 'ok' and frame is not None:
                    try:
                        params = spec['params']
                        if spec['kind'] == 'bars':
                            result = validate_index_bars(frame, params['ts_code'], params['start_date'], params['end_date'],
                                                         expected_sessions=_sessions_for(sessions, params['start_date'], params['end_date']),
                                                         limit=params['limit'])
                        else:
                            result = validate_members(frame, params['trade_date'], params['ts_code'], limit=params['limit'])
                        record['validation_status'] = 'validated'
                    except Exception as exc:
                        record['validation_status'] = 'failed'
                        record['validation_error'] = safe_text(str(exc), secrets)
                        record['validation_error_type'] = type(exc).__name__
                else:
                    record['validation_status'] = 'not_validated'
                _write_json(raw_dir / f'{stem}.json', record)
                request_audit['attempts'].append(dict(attempt=attempt, status=record['status'],
                                                     validation_status=record['validation_status'],
                                                     audit_file=f'attempts/{stem}.json'))
                if result is not None:
                    validated[spec['api']].append(result)
                    request_audit['status'] = 'validated'
                    save_manifest()
                    break
                save_manifest()
                if attempt == 2 or not _retryable(record):
                    break
    except BaseException as exc:
        manifest.update(status='failed', execution_error=safe_text(str(exc), secrets),
                        error_type=type(exc).__name__)
        save_manifest()
        if not isinstance(exc, Exception):
            raise
        return manifest
    finally:
        if owned_session and session is not None:
            try:
                session.close()
            except BaseException as exc:
                manifest.update(status='failed', session_close_error=safe_text(str(exc), secrets),
                                error_type=type(exc).__name__)
                save_manifest()
                if not isinstance(exc, Exception):
                    raise
    if (manifest['status'] == 'failed' or len(manifest['requests']) != len(plan)
            or not all(request['status'] == 'validated' for request in manifest['requests'])):
        manifest['status'] = 'failed'
        save_manifest()
        return manifest
    # Final cross-shard checks precede every aggregate write.
    aggregates = {}
    try:
        for api, frames in validated.items():
            if not frames:
                continue
            combined = pd.concat(frames, ignore_index=True)
            keys = ['ts_code', 'trade_date'] + (['con_code'] if api == 'dc_member' else [])
            if combined.duplicated(keys).any():
                raise ValueError(f'Duplicate cross-shard keys for {api}')
            if api.endswith('_daily'):
                for code, bars in combined.groupby('ts_code'):
                    validate_index_bars(bars, code, _day(start_date), _day(end_date),
                                        expected_sessions=_sessions_for(sessions, _day(start_date), _day(end_date)))
            aggregates[api] = combined.sort_values(keys).reset_index(drop=True)
    except BaseException as exc:
        manifest.update(status='failed', aggregate_error=safe_text(str(exc), secrets))
        save_manifest()
        if not isinstance(exc, Exception):
            raise
        return manifest
    try:
        manifest['validated_files'] = _publish_aggregates(directory, aggregates)
        manifest['status'] = 'complete'
        save_manifest()
    except BaseException as exc:
        for api in aggregates:
            (directory / f'validated_{api}.pkl').unlink(missing_ok=True)
        manifest.update(status='failed', validated_files={}, publication_error=safe_text(str(exc), secrets),
                        error_type=type(exc).__name__)
        save_manifest()
        if not isinstance(exc, Exception):
            raise
    return manifest


def load_expected_sessions(path):
    """Read a user-selected trusted local daily.pkl, never a remote calendar."""
    data = pd.read_pickle(path)
    if isinstance(data, dict):
        data = data.get('df_stock')
    if not isinstance(data, pd.DataFrame):
        raise ValueError('Expected a local daily DataFrame or df_stock cache')
    column = 'date' if 'date' in data else 'trade_date' if 'trade_date' in data else None
    if column is None:
        raise ValueError('Local calendar cache lacks date/trade_date')
    return sorted({_day(day) for day in data[column].unique()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ths-codes', nargs='*', default=[])
    parser.add_argument('--dc-codes', nargs='*', default=[])
    parser.add_argument('--start-date', required=True)
    parser.add_argument('--end-date', required=True)
    parser.add_argument('--snapshot-dates', nargs='*', default=[])
    parser.add_argument('--expected-sessions', type=Path, help='Trusted local daily.pkl providing session dates')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--bars-limit', type=int, default=100)
    parser.add_argument('--members-limit', type=int, default=1000)
    args = parser.parse_args()
    try:
        sessions = load_expected_sessions(args.expected_sessions) if args.expected_sessions else None
        manifest = run_pilot(args.ths_codes, args.dc_codes, args.start_date, args.end_date, args.snapshot_dates,
                             output_dir=args.output_dir, expected_sessions=sessions,
                             bars_limit=args.bars_limit, members_limit=args.members_limit)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({key: manifest[key] for key in ('status', 'output_dir', 'planned_requests', 'validated_files')}, ensure_ascii=False))
    return 0 if manifest['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
