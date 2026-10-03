"""Credential-free public THS / Eastmoney index history, with strict validation.

These endpoints supply index prices, never historical stock membership. Dated
membership must be acquired separately. Source volume is retained unchanged;
the adapter deliberately does not infer shares versus hands from price levels.
"""
import json
import re

import numpy as np
import pandas as pd
import requests

from data_pipeline.theme_data import validate_index_bars


EM_HISTORY_URL = 'https://91.push2his.eastmoney.com/api/qt/stock/kline/get'
THS_HISTORY_URL = 'https://d.10jqka.com.cn/v4/line/bk_{code}/01/{year}.js'


def _numeric_amount(frame):
    amount = pd.to_numeric(frame.amount, errors='coerce')
    if amount.isna().any() or not np.isfinite(amount).all() or amount.lt(0).any():
        raise ValueError('Invalid public index amount')
    frame['amount'] = amount
    return frame


def parse_eastmoney_history(payload, ts_code, start_date, end_date, *, expected_sessions=None):
    """Validate the returned code and requested interval before returning prices."""
    if not re.fullmatch(r'BK\d{4}\.DC', ts_code):
        raise ValueError('Expected Eastmoney BKxxxx.DC code')
    if not isinstance(payload, dict) or payload.get('rc') != 0:
        raise ValueError('Eastmoney did not return success')
    data = payload.get('data')
    if not isinstance(data, dict) or data.get('code') != ts_code[:-3]:
        raise ValueError('Eastmoney response code mismatch')
    rows = data.get('klines')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Empty Eastmoney history')
    split = [row.split(',') if isinstance(row, str) else [] for row in rows]
    if any(len(row) != 11 for row in split):
        raise ValueError('Unexpected Eastmoney kline schema')
    frame = pd.DataFrame(split, columns=['trade_date', 'open', 'close', 'high', 'low',
                                         'vol', 'amount', 'swing', 'pct_change',
                                         'change', 'turnover_rate'])
    frame['ts_code'] = ts_code
    frame['source_name'] = data.get('name', '')
    frame = validate_index_bars(frame, ts_code, start_date, end_date,
                                expected_sessions=expected_sessions)
    frame = _numeric_amount(frame)
    frame['source_vol'] = frame.vol
    frame.attrs.update(source='eastmoney_public', source_vol_unit='unconfirmed_raw',
                       amount_unit='CNY', historical_membership=False)
    return frame.sort_values('trade_date').reset_index(drop=True)


def parse_ths_year(payload, ts_code, year):
    """Validate a whole yearly file. The URL code is the index inner code."""
    if not re.fullmatch(r'\d{6}\.TI', ts_code) or not isinstance(year, int):
        raise ValueError('Expected THS inner index code and integer year')
    if not isinstance(payload, dict) or not isinstance(payload.get('data'), str):
        raise ValueError('Unexpected THS yearly file schema')
    split = [row.split(',') for row in payload['data'].split(';') if row]
    if not split or any(len(row) not in (11, 12) for row in split):
        raise ValueError('Empty or malformed THS yearly history')
    frame = pd.DataFrame([row[:7] for row in split], columns=[
        'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount'])
    frame['ts_code'] = ts_code
    frame = validate_index_bars(frame, ts_code, f'{year}0101', f'{year}1231')
    frame = _numeric_amount(frame)
    frame['source_vol'] = frame.vol
    frame.attrs.update(source='ths_public', source_vol_unit='unconfirmed_raw',
                       amount_unit='CNY', historical_membership=False)
    return frame.sort_values('trade_date').reset_index(drop=True)


def _get_json(session, url, params=None):
    try:
        response = session.get(url, params=params, verify=True,
                               allow_redirects=False, timeout=30)
    except requests.RequestException as exc:
        raise RuntimeError(f'Public theme transport: {type(exc).__name__}') from None
    if response.status_code != 200:
        raise RuntimeError(f'Public theme HTTP {response.status_code}')
    if 'html' in response.headers.get('Content-Type', '').lower():
        raise ValueError('Public endpoint returned HTML instead of history')
    # THS supplies a JSONP wrapper; never evaluate the response as JavaScript.
    text = response.text
    first, last = text.find('{'), text.rfind('}')
    if first < 0 or last < first:
        raise ValueError('Public endpoint returned no JSON object')
    try:
        return json.loads(text[first:last+1])
    except ValueError:
        raise ValueError('Invalid public theme JSON') from None


def fetch_eastmoney_history(session, ts_code, start_date, end_date, *, expected_sessions=None):
    if not isinstance(ts_code, str) or not re.fullmatch(r'BK\d{4}\.DC', ts_code):
        raise ValueError('Expected Eastmoney BKxxxx.DC code')
    params = dict(secid='90.'+ts_code[:-3], fields1='f1,f2,f3,f4,f5,f6',
                  fields2='f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
                  klt='101', fqt='0', beg=start_date, end=end_date,
                  smplmt='10000', lmt='1000000')
    return parse_eastmoney_history(_get_json(session, EM_HISTORY_URL, params),
                                   ts_code, start_date, end_date,
                                   expected_sessions=expected_sessions)


def fetch_ths_year(session, ts_code, year):
    if not isinstance(ts_code, str) or not re.fullmatch(r'\d{6}\.TI', ts_code):
        raise ValueError('Expected THS inner index code')
    if not isinstance(year, int) or not 1990 <= year <= 2100:
        raise ValueError('Invalid THS history year')
    url = THS_HISTORY_URL.format(code=ts_code[:-3], year=year)
    return parse_ths_year(_get_json(session, url), ts_code, year)
