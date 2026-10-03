"""Point-in-time, multi-label DC graphs and verified daily adjustments.

No SW input, current THS backfill, theme top-N gate, named-stock filter or
future-return selection is used here.  A monthly snapshot is observable at
that day's close and is carried forward, never backwards.  Missing edges
remain unknown.  Incidence message passing does not materialize N x N edges.

DC's documented historical start is 2024-12-20:
https://tushare.pro/document/2?doc_id=363
Adjustment factors: https://tushare.pro/document/2?doc_id=28
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse


EDGE_COLUMNS = ['snapshot_date', 'theme_code', 'ts_code']
DC_FIRST_DATE = pd.Timestamp('2024-12-20')


def _dates(values, label):
    result = pd.to_datetime(values, errors='raise')
    if result.isna().any() or getattr(result.dt, 'tz', None) is not None:
        raise ValueError('Missing or timezone-aware '+label)
    if not result.eq(result.dt.normalize()).all():
        raise ValueError('Intraday '+label)
    return result


def validate_edges(edges):
    if not isinstance(edges, pd.DataFrame) or not edges.columns.is_unique:
        raise ValueError('Edges must have unique columns')
    if not set(EDGE_COLUMNS) <= set(edges):
        raise ValueError('Missing multi-label edge columns')
    result = edges.copy()
    result['snapshot_date'] = _dates(result.snapshot_date, 'snapshot_date')
    if result.snapshot_date.lt(DC_FIRST_DATE).any():
        raise ValueError('DC membership before documented historical coverage')
    if not result.theme_code.map(lambda s: isinstance(s, str)).all():
        raise ValueError('Invalid theme code type')
    if not result.theme_code.str.fullmatch(r'BK\d{4}\.DC').all():
        raise ValueError('Non-DC theme code')
    if not result.ts_code.map(lambda s: isinstance(s, str)).all():
        raise ValueError('Invalid stock code type')
    if not result.ts_code.str.fullmatch(r'\d{6}\.(SH|SZ|BJ)').all():
        raise ValueError('Invalid A-share stock code')
    if result.duplicated(EDGE_COLUMNS).any():
        raise ValueError('Duplicate dated graph edge')
    return result.sort_values(EDGE_COLUMNS).reset_index(drop=True)


def validate_membership_snapshot(frame, day, theme_code, minimum=0, limit=5000):
    from data_pipeline.theme_data import validate_members
    result = validate_members(frame, day, theme_code, limit=limit)
    if isinstance(minimum, bool) or int(minimum) != minimum or minimum < 0:
        raise ValueError('Invalid contemporaneous membership lower bound')
    if len(result) < minimum:
        raise ValueError(f'Membership has {len(result)} rows, below dated lower bound {minimum}')
    validate_edges(pd.DataFrame(dict(snapshot_date=pd.to_datetime(result.trade_date),
                                    theme_code=result.ts_code, ts_code=result.con_code)))
    return result


def fetch_audited_pages(client, api, *, params, keys, page_size=6000, max_pages=30):
    """Reject repeated/overlapping pages; only a short page ends pagination.

    Coverage must still be checked independently by a caller.  Some mirror
    endpoints ignore both limit and offset and return a *short* but incomplete
    response.  In particular an all-theme DC response is not a graph census.
    """
    if 'limit' in params or 'offset' in params:
        raise ValueError('Pagination parameters are owned by this function')
    pages, seen, audits = [], set(), []
    for page_number in range(max_pages):
        offset = page_number*page_size
        page = client.query(api, limit=page_size, offset=offset, **params)
        if not isinstance(page, pd.DataFrame) or not page.columns.is_unique or not set(keys) <= set(page):
            raise ValueError('Malformed paginated response')
        if len(page) > page_size or page[keys].isna().any().any() or page.duplicated(keys).any():
            raise ValueError('Invalid page size or keys')
        page_keys = set(map(tuple, page[keys].to_numpy()))
        if seen & page_keys:
            raise ValueError('Repeated/overlapping page; offset appears ignored')
        seen.update(page_keys)
        audits.append(dict(offset=offset, rows=len(page),
                           sha256=hashlib.sha256(pd.util.hash_pandas_object(page, index=False).values.tobytes()).hexdigest()))
        pages.append(page)
        if len(page) < page_size:
            return pd.concat(pages, ignore_index=True), audits
    raise ValueError('Pagination exceeded bounded request count')


def validate_adjustment_snapshot(frame, day, expected_codes=()):
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError('Invalid factor response')
    required = ['trade_date', 'ts_code', 'adj_factor']
    if not set(required) <= set(frame) or frame.empty:
        raise ValueError('Missing/empty factor response')
    result = frame[required].copy()
    result['trade_date'] = _dates(pd.Series(result.trade_date), 'factor date').dt.strftime('%Y%m%d')
    day_text = pd.Timestamp(day).strftime('%Y%m%d')
    if not result.trade_date.eq(day_text).all() or result.duplicated(['trade_date', 'ts_code']).any():
        raise ValueError('Wrong factor date or duplicate stock')
    if not result.ts_code.map(lambda s: isinstance(s, str)).all() or not result.ts_code.str.fullmatch(r'\d{6}\.(SH|SZ|BJ)').all():
        raise ValueError('Invalid adjustment stock code')
    result['adj_factor'] = pd.to_numeric(result.adj_factor, errors='coerce')
    if not np.isfinite(result.adj_factor).all() or result.adj_factor.le(0).any():
        raise ValueError('Nonpositive/nonfinite adjustment factor')
    missing = set(expected_codes)-set(result.ts_code)
    if missing:
        raise ValueError(f'Missing factors for {len(missing)} actual daily stocks: {sorted(missing)[:4]}')
    return result.sort_values('ts_code').reset_index(drop=True)


def adjusted_prices(daily, adjustments):
    required = ['date', 'ts_code', 'open', 'high', 'low', 'close']
    if not set(required) <= set(daily) or not daily.columns.is_unique:
        raise ValueError('Missing daily price columns')
    raw = daily.copy()
    raw['date'] = _dates(raw.date, 'daily date')
    if raw.duplicated(['date', 'ts_code']).any():
        raise ValueError('Duplicate daily quote')
    if set(['adj_factor', 'adj_open', 'adj_close', 'adj_high', 'adj_low']) & set(raw):
        raise ValueError('Input already has adjusted columns')
    factors = adjustments.copy()
    if not set(['trade_date', 'ts_code', 'adj_factor']) <= set(factors):
        raise ValueError('Missing factors')
    factors['date'] = _dates(factors.trade_date, 'factor date')
    if factors.duplicated(['date', 'ts_code']).any():
        raise ValueError('Duplicate adjustment factor')
    factors['adj_factor'] = pd.to_numeric(factors.adj_factor, errors='coerce')
    if not np.isfinite(factors.adj_factor).all() or factors.adj_factor.le(0).any():
        raise ValueError('Nonpositive/nonfinite adjustment factor')
    result = raw.merge(factors[['date', 'ts_code', 'adj_factor']], how='left', on=['date', 'ts_code'], validate='one_to_one')
    if result.adj_factor.isna().any():
        raise ValueError('Daily adjustment coverage incomplete; no factor filling is permitted')
    for column in ['open', 'high', 'low', 'close']:
        values = pd.to_numeric(result[column], errors='coerce')
        if not np.isfinite(values).all() or values.le(0).any():
            raise ValueError('Nonpositive/nonfinite daily '+column)
        result['adj_'+column] = values*result.adj_factor
    if result.high.lt(result[['open', 'close', 'low']].max(axis=1)).any() or result.low.gt(result[['open', 'close', 'high']].min(axis=1)).any():
        raise ValueError('Invalid daily OHLC geometry')
    # Product prices are factor-adjusted; no end-of-sample qfq normalization.
    return result.sort_values(['date', 'ts_code']).reset_index(drop=True)


@dataclass(frozen=True)
class AsOfGraph:
    date: pd.Timestamp
    snapshot_date: pd.Timestamp | None
    stock_codes: tuple
    theme_codes: tuple
    incidence: sparse.csr_matrix

    @property
    def known(self):
        return np.asarray(self.incidence.sum(axis=1)).ravel() > 0

    def messages(self, features, *, exclude_self=True):
        """Mean features of observable theme peers, then equal mean over labels.

        Self-free messages retain each overlapping theme separately. Stocks
        with unknown graph or no observable peers get zero and a known flag;
        no synthetic graph edge is invented for these stocks.
        """
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[0] != len(self.stock_codes) or not np.isfinite(values).all():
            raise ValueError('Graph features must be finite N x F values')
        b = self.incidence
        counts = np.asarray(b.sum(axis=0)).ravel()
        sums = b.T @ values
        if not exclude_self:
            means = sums/np.maximum(counts[:, None], 1)
            degrees = np.asarray(b.sum(axis=1)).ravel()
            return np.asarray(b @ means)/np.maximum(degrees[:, None], 1)
        usable = counts > 1
        if not usable.any():
            return np.zeros_like(values)
        c = b[:, usable]
        inv = 1/(counts[usable]-1)
        degree = np.asarray(c.sum(axis=1)).ravel()
        self_weight = np.asarray(c @ inv).ravel()
        messages = np.asarray(c @ (sums[usable]*inv[:, None]))-self_weight[:, None]*values
        return messages/np.maximum(degree[:, None], 1)


def asof_graph(edges, day, stock_codes, *, max_age_days=62, validated=False):
    data = edges if validated else validate_edges(edges)
    day = pd.Timestamp(day)
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError('Invalid graph as-of date')
    if max_age_days is not None and (isinstance(max_age_days, bool) or max_age_days < 0):
        raise ValueError('Invalid graph staleness bound')
    codes = tuple(stock_codes)
    if len(set(codes)) != len(codes):
        raise ValueError('Duplicate requested graph stocks')
    dates = data.snapshot_date[data.snapshot_date.le(day)]
    snapshot = dates.max() if len(dates) else None
    if snapshot is None or (max_age_days is not None and (day-snapshot).days > max_age_days):
        return AsOfGraph(day, None, codes, (), sparse.csr_matrix((len(codes), 0), dtype=float))
    # Select latest whole snapshot, not latest edge per stock. Removed members
    # must not persist after a newer snapshot no longer lists them.
    current = data[data.snapshot_date.eq(snapshot) & data.ts_code.isin(codes)]
    themes = tuple(sorted(current.theme_code.unique()))
    stock_index = {code: i for i, code in enumerate(codes)}
    theme_index = {code: i for i, code in enumerate(themes)}
    rows = current.ts_code.map(stock_index).to_numpy(dtype=int)
    columns = current.theme_code.map(theme_index).to_numpy(dtype=int)
    incidence = sparse.csr_matrix((np.ones(len(current)), (rows, columns)), shape=(len(codes), len(themes)))
    return AsOfGraph(day, snapshot, codes, themes, incidence)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish_frame(frame, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.staging.pkl')
    frame.to_pickle(temporary); temporary.replace(path)
    return dict(file=path.name, rows=len(frame), sha256=sha256(path))


def publish_manifest(payload, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.staging.json')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def safe_query(client, api, *, attempts=3, **params):
    errors = []
    for number in range(attempts):
        try:
            return client.query(api, **params)
        except Exception as exc:
            # GatewayClient sanitizes transport and remote response errors.
            errors.append(type(exc).__name__)
            if number+1 < attempts:
                time.sleep(min(number+1, 2))
    raise RuntimeError(api+' query failed: '+','.join(errors))


def load_graph_inputs(directory, *, allow_partial=False, require_finished=True):
    """Load only a fingerprint-bound graph, with explicit partial opt-in.

    A finished partial graph is not promoted to "complete". Its per-date
    failed/missing themes remain in the returned manifest for reporting and
    downstream unknown masks. Live acquisition is rejected by default.
    """
    directory = Path(directory)
    manifest = json.loads((directory/'graph_manifest.json').read_text(encoding='utf-8'))
    status = manifest.get('status')
    if status not in ('complete', 'incomplete', 'running'):
        raise ValueError('Invalid graph acquisition state')
    if require_finished and status == 'running':
        raise ValueError('Graph acquisition still running; freeze inputs before fitting')
    if status == 'complete':
        artifact = manifest.get('aggregate')
    elif allow_partial:
        artifact = manifest.get('partial_aggregate')
    else:
        raise ValueError('Partial graph requires explicit allow_partial=True')
    if not isinstance(artifact, dict):
        raise ValueError('Missing graph aggregate')
    path = directory/artifact['file']
    if sha256(path) != artifact['sha256']:
        raise ValueError('Graph aggregate fingerprint mismatch')
    edges = validate_edges(pd.read_pickle(path))
    if len(edges) != artifact['rows']:
        raise ValueError('Graph row count mismatch')
    if set(edges.snapshot_date.dt.strftime('%Y%m%d')) != set(manifest.get('snapshots', {})):
        raise ValueError('Graph snapshot coverage differs from manifest')
    for day, snapshot in manifest['snapshots'].items():
        shard = directory/snapshot['file']
        if sha256(shard) != snapshot['sha256']:
            raise ValueError('Graph snapshot fingerprint mismatch')
        if require_finished and not snapshot.get('attempts_finished', status == 'complete'):
            raise ValueError('Dated graph snapshot acquisition unfinished')
    return edges, manifest


def load_adjusted_daily(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'adjustment_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete' or manifest.get('failures'):
        raise ValueError('Adjustment acquisition incomplete')
    if manifest.get('completed_dates') != manifest.get('planned_dates'):
        raise ValueError('Adjustment date coverage mismatch')
    artifact = manifest.get('adjusted_daily', {})
    path = directory/artifact.get('file', 'missing')
    if not path.exists() or sha256(path) != artifact.get('sha256'):
        raise ValueError('Adjusted daily fingerprint mismatch')
    frame = pd.read_pickle(path)
    if len(frame) != artifact.get('rows') or len(frame) != manifest.get('expected_daily_rows'):
        raise ValueError('Adjusted daily row coverage mismatch')
    return frame, manifest
