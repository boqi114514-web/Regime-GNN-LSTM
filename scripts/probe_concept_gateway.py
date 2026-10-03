"""Nine read-only THS/DC gateway diagnostics; never a point-in-time data repair.

Run explicitly with --transport gateway or --transport legacy. Credentials are
read only from that transport's distinct environment variable. TLS verification
is on; redirects and retries are off. No headers or unfiltered response bodies
are logged. A matching date is not proof of historical membership completeness.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from urllib.parse import quote, quote_plus

import pandas as pd
import requests


ENDPOINTS = {
    'gateway': ('https://tl.kaixin8.top/tushare/pro', 'TUSHARE_API_KEY'),
    'legacy': ('https://t.xiaodefa.top/', 'TUSHARE_TOKEN'),
}
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'data/raw/concept_gateway_probe'
DAY = '20260331'
_LABEL = re.compile(
    r'''(?i)((?:["']?(?:x[-_]?api[-_]?key|api[-_]?key|access_token|token|key|authorization|secret|password)["']?)\s*[:=]\s*)(?:"[^"]*"|'[^']*'|[^\s,&;}\]]+)''')
_TOKEN = re.compile(r'\b(?:tsr_|sk-)[A-Za-z0-9_-]+\b|\b[A-Fa-f0-9]{32,}\b')
_BEARER = re.compile(r'(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*')
_SENSITIVE_FIELD = re.compile(r'(?i)^(?:x[-_]?api[-_]?key|api[-_]?key|access_token|token|authorization|secret|password)$')


def probe_matrix():
    """Return fresh copies of the bounded request matrix; no SW fallback."""
    return [
        dict(name='daily_control', api='daily', params=dict(ts_code='000001.SZ', trade_date=DAY, limit=3)),
        dict(name='ths_catalog', api='ths_index', params=dict(exchange='A')),
        dict(name='ths_daily_date', api='ths_daily', params=dict(ts_code='885959.TI', trade_date=DAY)),
        dict(name='ths_daily_range', api='ths_daily', params=dict(ts_code='885959.TI', start_date='20260330', end_date=DAY)),
        dict(name='ths_members_current', api='ths_member', params=dict(con_code='002384.SZ')),
        dict(name='dc_catalog_minimal', api='dc_index', params=dict(trade_date=DAY, idx_type='概念板块')),
        dict(name='dc_catalog_explicit_page', api='dc_index', params=dict(trade_date=DAY, idx_type='概念板块', offset=0, limit=5000)),
        dict(name='dc_members_date', api='dc_member', params=dict(con_code='002384.SZ', trade_date=DAY)),
        dict(name='dc_daily_date', api='dc_daily', params=dict(trade_date=DAY, idx_type='概念板块')),
    ]


def safe_text(value, secrets=(), max_length=500):
    """Redact before truncation; never return a raw error/body fragment."""
    text = json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else str(value)
    for secret in sorted({str(s) for s in secrets if s}, key=len, reverse=True):
        for form in {secret, quote(secret, safe=''), quote_plus(secret)}:
            text = text.replace(form, '[REDACTED]')
    text = _BEARER.sub('Bearer [REDACTED]', text)
    text = _LABEL.sub(lambda match: match.group(1) + '[REDACTED]', text)
    text = _TOKEN.sub('[REDACTED]', text)
    text = re.sub(r'[\x00-\x1f\x7f]', ' ', text)
    return text[:max_length]


def _safe_diagnostics(payload, secrets):
    if not isinstance(payload, dict):
        return {}
    return {field: safe_text(payload[field], secrets) for field in ('msg', 'message', 'error', 'detail', 'code')
            if field in payload}


def _frame(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get('data'), dict):
        return None
    data = payload['data']
    fields, items = data.get('fields'), data.get('items')
    if (not isinstance(fields, list) or not fields or not all(isinstance(field, str) for field in fields)
            or len(set(fields)) != len(fields) or not isinstance(items, list)):
        return None
    if any(not isinstance(row, (list, tuple)) or len(row) != len(fields) for row in items):
        return None
    # Nested server error objects are not raw tabular market observations.
    if any(isinstance(value, (dict, list, tuple)) for row in items for value in row):
        return None
    return pd.DataFrame(items, columns=fields)


def _dates(series):
    text = series.astype(str)
    parsed = pd.to_datetime(text, errors='coerce', format='mixed')
    return parsed.dt.strftime('%Y%m%d'), parsed.isna().any()


def classify_frame(frame, spec):
    """Classify ignored filters, duplicate snapshots and empty tabular replies."""
    api, params = spec['api'], spec['params']
    issues = []
    if frame.empty:
        issues.append('empty')
    required = {'ts_code'}
    if api.endswith('_index'):
        required.add('name')
    if api.endswith('_member'):
        required.add('con_code')
    if api not in ('ths_index', 'ths_member'):
        required.add('trade_date')
    if not required <= set(frame.columns):
        issues.append('missing_required_fields')
    for field in ('ts_code', 'con_code'):
        if field in params and field in frame:
            if not frame[field].astype(str).eq(params[field]).all():
                issues.append('wrong_code')
    if 'trade_date' in frame:
        dates, invalid = _dates(frame.trade_date)
        if invalid:
            issues.append('invalid_dates')
        if 'trade_date' in params and not dates.eq(params['trade_date']).all():
            issues.append('wrong_dates')
        if 'start_date' in params and not dates.between(params['start_date'], params['end_date']).all():
            issues.append('wrong_dates')
    keys = ['ts_code'] if api.endswith('_index') else ['ts_code', 'con_code'] if api.endswith('_member') else ['ts_code', 'trade_date']
    if all(key in frame for key in keys) and frame.duplicated(keys).any():
        issues.append('duplicate_keys')
    if 'limit' in params and len(frame) >= params['limit']:
        issues.append('possible_truncation')
    return list(dict.fromkeys(issues))


def _contains_credentials(frame, secrets):
    if any(_SENSITIVE_FIELD.fullmatch(column) for column in frame.columns):
        return True
    strings = list(frame.columns)
    strings.extend(value for value in frame.to_numpy().flat if isinstance(value, str))
    for value in strings:
        if any(secret and secret in value for secret in secrets) or _TOKEN.search(value) or _LABEL.search(value) or _BEARER.search(value):
            return True
    return False


def probe_request(session, transport, credential, spec, *, secrets=()):
    """Make exactly one bounded request and return (sanitized metadata, frame)."""
    if transport not in ENDPOINTS:
        raise ValueError('Unknown transport')
    secrets = tuple(secrets) + (credential,)
    url, _ = ENDPOINTS[transport]
    record = dict(name=spec['name'], api=spec['api'], params=dict(spec['params']),
                  transport=transport, http_status=None, response_type=None, status='pending',
                  issues=[], fields=[], rows=None, returned_dates=[], returned_codes={},
                  diagnostics={}, raw_frame_saved=False, point_in_time_verified=False,
                  membership_policy='current_only' if spec['api'] == 'ths_member' else
                  'current_catalog' if spec['api'] == 'ths_index' else 'unverified_historical_response')
    try:
        options = dict(verify=True, allow_redirects=False, timeout=30)
        if transport == 'gateway':
            response = session.get(url + '/' + spec['api'], params=dict(spec['params']),
                                   headers={'X-API-Key': credential}, **options)
        else:
            response = session.post(url, json=dict(api_name=spec['api'], token=credential,
                                                  params=dict(spec['params']), fields=''), **options)
    except requests.RequestException as exc:
        record.update(status='transport_error', transport_error_type=type(exc).__name__,
                      diagnostics={'error': safe_text(str(exc), secrets)})
        return record, None
    record['http_status'] = int(response.status_code)
    if 'html' in response.headers.get('Content-Type', '').lower():
        record.update(response_type='html', status='html_response')
        if response.status_code >= 400:
            record['status'] = 'http_error'
        if 300 <= response.status_code < 400:
            record['status'] = 'redirect_blocked'
        return record, None
    try:
        payload = response.json()
        record['response_type'] = 'json'
    except ValueError:
        record.update(response_type='non_json', status='invalid_json')
        if response.status_code >= 400:
            record['status'] = 'http_error'
        if 300 <= response.status_code < 400:
            record['status'] = 'redirect_blocked'
        return record, None
    record['diagnostics'] = _safe_diagnostics(payload, secrets)
    frame = _frame(payload)
    if frame is not None:
        record['fields'] = [safe_text(field, secrets, 100) for field in frame.columns]
        record['rows'] = len(frame)
        if 'trade_date' in frame:
            values = frame.trade_date.astype(str).unique()
            record['returned_dates'] = [safe_text(value, secrets, 80) for value in sorted(values)[:20]]
            record['returned_dates_count'] = len(values)
        for field in ('ts_code', 'con_code'):
            if field in frame:
                values = frame[field].astype(str).unique()
                record['returned_codes'][field] = dict(count=len(values), sample=[safe_text(v, secrets, 80) for v in sorted(values)[:20]])
        record['issues'] = classify_frame(frame, spec)
        if _contains_credentials(frame, secrets):
            record['issues'].append('credential_like_tabular_content_not_saved')
            frame = None
    if 300 <= response.status_code < 400:
        record['status'] = 'redirect_blocked'
    elif response.status_code != 200:
        record['status'] = 'http_error'
    elif not isinstance(payload, dict) or payload.get('ok') is False or payload.get('code', 0) not in (0, '0', None):
        record['status'] = 'api_error'
    elif frame is None:
        record['status'] = 'invalid_schema'
    elif record['issues']:
        record['status'] = record['issues'][0]
    else:
        record['status'] = 'ok'
    return record, frame


def run_probe(transport, output_dir=DEFAULT_OUTPUT, *, session=None, environ=None):
    """Run the fixed nine requests; environment overrides cannot change origins."""
    if transport not in ENDPOINTS:
        raise ValueError('Unknown transport')
    env = os.environ if environ is None else environ
    variable = ENDPOINTS[transport][1]
    credential = env.get(variable, '').strip()
    if not credential:
        raise ValueError(f'Missing {variable}; no fallback to another transport credential')
    secrets = tuple(env.get(name, '').strip() for _, name in ENDPOINTS.values())
    directory = Path(output_dir) / transport
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    owned_session = session is None
    session = requests.Session() if owned_session else session
    records = []
    try:
        for number, spec in enumerate(probe_matrix(), 1):
            record, frame = probe_request(session, transport, credential, spec, secrets=secrets)
            stem = f'{stamp}_{number:02d}_{spec["name"]}'
            if frame is not None:
                frame.to_pickle(directory / f'{stem}.pkl')
                record['raw_frame_saved'] = True
                record['raw_frame_file'] = f'{stem}.pkl'
            (directory / f'{stem}.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
            records.append(record)
    finally:
        if owned_session:
            session.close()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport', required=True, choices=tuple(ENDPOINTS))
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT,
                        help='Output root; a transport subdirectory is appended')
    args = parser.parse_args()
    try:
        records = run_probe(args.transport, args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    for record in records:
        print(json.dumps({key: record[key] for key in ('name', 'status', 'http_status', 'rows', 'diagnostics')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
