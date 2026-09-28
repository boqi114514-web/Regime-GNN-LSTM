"""Validated, restartable execution data downloads, isolated from legacy inputs."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import LOCAL_DATA_RAW, STOCK_DAILY_PATH
from data_pipeline.tushare_config import get_pro

ROOT = Path(LOCAL_DATA_RAW)/'execution_v1'


def validate_rows(frame, keys, required, date=None):
    if not set(keys + required) <= set(frame.columns):
        raise ValueError('Missing required data columns')
    frame = frame.drop_duplicates().copy()
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError('Missing or conflicting keys')
    if date is not None and not frame.trade_date.astype(str).eq(date).all():
        raise ValueError('Server returned dates outside requested session')
    for col in required:
        values = pd.to_numeric(frame[col], errors='coerce')
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('Nonpositive/invalid '+col)
        frame[col] = values
    return frame


def fetch(pro, api, add_offset=True, **params):
    # Explicit offset avoids the gateway's incomplete unpaginated responses.
    if add_offset:
        params.setdefault('offset', 0)
    error = None
    for attempt in range(5):
        try:
            return pro.query(api, **params)
        except Exception as exc:
            error = exc
            time.sleep(min(1+attempt, 3))
    raise RuntimeError(f'{api} {params} failed: {error}')


def fetch_pages(pro, api, **params):
    pages, signatures = [], set()
    for offset in range(0, 60000, 6000):
        page = fetch(pro, api, limit=6000, offset=offset, **params)
        signature = int(pd.util.hash_pandas_object(page, index=False).sum())
        if len(page) and signature in signatures:
            raise ValueError('Repeated page; offset appears ignored')
        signatures.add(signature)
        pages.append(page)
        if len(page) < 6000:
            return pd.concat(pages, ignore_index=True).drop_duplicates()
    raise ValueError('Pagination exceeded bounded request count')


def fetch_variants(pro, api, variants, required, date=None):
    """Try equivalent bounded query forms; never accept an empty schema."""
    errors = []
    for _ in range(2):
        for params in variants:
            try:
                frame = pro.query(api, **params)
                if not set(required) <= set(frame) or frame.empty:
                    raise ValueError('Empty/incomplete response')
                if date is not None and not frame.trade_date.astype(str).eq(date).all():
                    raise ValueError('Wrong response dates')
                return frame
            except Exception as exc:
                errors.append(str(exc))
        time.sleep(2.)
    raise RuntimeError(f'{api}: all query variants failed: {errors[-3:]}')


def load_monthly(start='2021-12-01', end='2026-08-31'):
    ROOT.mkdir(parents=True, exist_ok=True)
    cache = ROOT/'stock_month_end.pkl'
    daily = pd.read_pickle(STOCK_DAILY_PATH)['df_stock']
    dates = pd.to_datetime(daily.date)
    calendar = pd.Series(dates.unique()).sort_values()
    calendar = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    ends = calendar.groupby(calendar.dt.to_period('M')).max()
    monthly = daily.loc[dates.isin(ends.values), ['date', 'code', 'close', 'amount']].copy()
    monthly['date'] = pd.to_datetime(monthly.date)
    monthly['code'] = monthly.code.astype(str).str.zfill(6)
    monthly = monthly[monthly.code.str.startswith(('0', '3', '6'))].copy()
    monthly['ts_code'] = monthly.code + np.where(monthly.code.str.startswith('6'), '.SH', '.SZ')
    if monthly.duplicated(['date', 'code']).any():
        raise ValueError('Local daily data has duplicate stock dates')
    monthly.to_pickle(cache)
    return monthly


def monthly_adjustments(pro, monthly):
    directory = ROOT/'adj_month_end'
    directory.mkdir(parents=True, exist_ok=True)
    audit = []
    for date, frame in monthly.groupby('date', sort=True):
        day = date.strftime('%Y%m%d')
        path = directory/f'{day}.pkl'
        try:
            if path.exists():
                adj = pd.read_pickle(path)
            else:
                adj = fetch_pages(pro, 'adj_factor', trade_date=day)
            adj = validate_rows(adj, ['ts_code', 'trade_date'], ['adj_factor'], day)
            missing = sorted(set(frame.ts_code)-set(adj.ts_code))
            if missing:
                raise ValueError(f'Missing factors for {len(missing)} locally quoted stocks: {missing[:5]}')
            adj.to_pickle(path)
            audit.append(dict(date=day, status='ok', rows=len(adj), expected=len(frame)))
        except Exception as exc:
            audit.append(dict(date=day, status='failed', error=str(exc)))
        print(json.dumps(audit[-1]), flush=True)
        (ROOT/'monthly_adjustment_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    if any(row['status'] != 'ok' for row in audit):
        raise RuntimeError('Incomplete adjustment coverage; inspect audit and rerun to resume')
    return audit


def monthly_quotes(pro, monthly):
    directory = ROOT/'daily_month_end'
    directory.mkdir(parents=True, exist_ok=True)
    audit, frames = [], []
    for date, old in monthly.groupby('date', sort=True):
        day = date.strftime('%Y%m%d')
        path = directory/f'{day}.pkl'
        try:
            quotes = pd.read_pickle(path) if path.exists() else fetch_pages(pro, 'daily', trade_date=day)
            # First listing sessions may legitimately have zero pre_close.
            # It is not used for month-end feature construction.
            quotes = validate_rows(quotes, ['ts_code', 'trade_date'], ['open', 'high', 'low', 'close'], day)
            quotes = quotes[quotes.ts_code.str.endswith(('.SH', '.SZ'))].copy()
            # Legacy cache includes zero-amount suspension placeholders. They
            # are not executable quotes; keep this exclusion explicit in audit.
            missing = set(old.loc[old.amount > 0, 'ts_code'])-set(quotes.ts_code)
            if missing:
                raise ValueError(f'Server missing {len(missing)} locally quoted stocks')
            quotes.to_pickle(path)
            compare = old.merge(quotes[['ts_code', 'close']], on='ts_code', suffixes=('_local', '_remote'))
            row = dict(date=day, status='ok', old_rows=len(old), new_rows=len(quotes),
                       absent_zero_amount_rows=len(set(old.loc[old.amount <= 0, 'ts_code'])-set(quotes.ts_code)),
                       close_disagreements=int((abs(compare.close_local-compare.close_remote)>.005).sum()))
            adj = pd.read_pickle(ROOT/'adj_month_end'/f'{day}.pkl')
            quotes = quotes.merge(adj[['ts_code', 'adj_factor']], on='ts_code', how='left', validate='one_to_one')
            if quotes.adj_factor.isna().any():
                raise ValueError('New month-end quotes missing adjustment factors')
            frames.append(quotes)
            audit.append(row)
        except Exception as exc:
            audit.append(dict(date=day, status='failed', error=str(exc)))
        print(json.dumps(audit[-1]), flush=True)
        (ROOT/'monthly_quote_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    if any(row['status'] != 'ok' for row in audit):
        raise RuntimeError('Incomplete quote coverage; resume after checking audit')
    result = pd.concat(frames, ignore_index=True)
    result['date'] = pd.to_datetime(result.trade_date)
    result['code'] = result.ts_code.str[:6]
    result.to_pickle(ROOT/'stock_month_end_verified.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2021-12-01')
    parser.add_argument('--end', default='2026-08-31')
    parser.add_argument('--quotes', action='store_true', help='Also refresh month-end quotes in isolated storage')
    args = parser.parse_args()
    monthly = load_monthly(args.start, args.end)
    monthly_adjustments(get_pro(), monthly)
    if args.quotes:
        monthly_quotes(get_pro(), monthly)
