"""Resume point-in-time daily opportunity inputs without changing old caches.

The all-theme dc_member response has been observed to return one row per
theme and ignore offset.  Therefore this collector requests *every* dated
catalogue theme individually, checks its dated count lower bound, and never
calls that one-row-per-theme response a complete graph.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import sys
import threading

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from data_pipeline.tushare_config import GatewayClient
from research_daily_graph_data import (
    adjusted_prices, fetch_audited_pages, publish_frame, publish_manifest,
    safe_query, sha256, validate_adjustment_snapshot, validate_edges,
    validate_membership_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'data/raw/daily_opportunity_20261002'
LOCAL = threading.local()


def client():
    if not hasattr(LOCAL, 'client'):
        key = os.environ.get('TUSHARE_API_KEY', '').strip()
        if not key:
            raise ValueError('Missing TUSHARE_API_KEY environment variable')
        LOCAL.client = GatewayClient(key)
    return LOCAL.client


def selected_daily(path, start, end):
    daily = pd.read_pickle(path)
    daily['date'] = pd.to_datetime(daily.date)
    daily = daily[daily.date.between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    if daily.empty or daily.duplicated(['date', 'ts_code']).any():
        raise ValueError('Missing or duplicate daily prices')
    return daily


def _cached_frame(path, validator):
    manifest_path = path.with_suffix('.json')
    if not path.exists():
        return None
    if not manifest_path.exists():
        # An interrupted publish is not accepted as a verified cache hit.
        # Re-fetch or independently validate the legacy source before publish.
        return None
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('sha256') != sha256(path) or manifest.get('status') != 'complete':
        raise ValueError('Cache shard fingerprint mismatch')
    return validator(pd.read_pickle(path))


def adjustments(args):
    daily = selected_daily(args.daily, args.start, args.end)
    groups = {pd.Timestamp(day): set(g.ts_code) for day, g in daily.groupby('date')}
    directory = args.output_dir/'adjustments'; directory.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir/'adjustment_manifest.json'
    results, failures = {}, {}
    publish_manifest(dict(status='running', planned_dates=len(groups)), manifest_path)

    def one(item):
        day, codes = item; text = day.strftime('%Y%m%d'); path = directory/(text+'.pkl')
        validator = lambda frame: validate_adjustment_snapshot(frame, day, codes)
        try:
            cached = _cached_frame(path, validator)
            if cached is not None:
                return text, dict(file=str(path.relative_to(args.output_dir)), rows=len(cached), sha256=sha256(path)), None
            old = ROOT/'data/raw/execution_v1/adj_month_end'/(text+'.pkl')
            if old.exists():
                frame = validator(pd.read_pickle(old)); origin = dict(source='verified_legacy_date_factors', path=str(old.relative_to(ROOT)), source_sha256=sha256(old))
            elif args.offline:
                raise ValueError('Missing offline adjustment shard')
            else:
                class RetryingClient:
                    def query(self, api, **params):
                        return safe_query(client(), api, attempts=4, **params)
                frame, pages = fetch_audited_pages(RetryingClient(), 'adj_factor',
                    params=dict(trade_date=text, fields='trade_date,ts_code,adj_factor'),
                    keys=['trade_date', 'ts_code'], page_size=6000, max_pages=3)
                frame = validator(frame); origin = dict(source='tushare_gateway', api='adj_factor', trade_date=text, pages=pages)
            artifact = publish_frame(frame, path)
            publish_manifest(dict(status='complete', **artifact, expected_quoted_stocks=len(codes), **origin), path.with_suffix('.json'))
            return text, dict(file=str(path.relative_to(args.output_dir)), rows=len(frame), sha256=artifact['sha256']), None
        except Exception as exc:
            return text, None, str(exc)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, item) for item in sorted(groups.items())]
        for number, future in enumerate(as_completed(futures), 1):
            text, artifact, failure = future.result()
            if failure:
                failures[text] = failure
            else:
                results[text] = artifact
            if number % 25 == 0 or failure:
                publish_manifest(dict(status='running', planned_dates=len(groups), completed_dates=len(results), failures=failures, shards=results), manifest_path)
                print('adjustments', number, '/', len(groups), 'verified', len(results), 'failures', len(failures), flush=True)
    manifest = dict(status='incomplete' if failures else 'complete', requested_start=args.start,
        requested_end=args.end, planned_dates=len(groups), completed_dates=len(results),
        expected_daily_rows=len(daily), failures=failures, shards=results,
        daily_source=dict(path=str(args.daily.resolve()), sha256=sha256(args.daily)),
        collector_sha256=sha256(Path(__file__)), validator_sha256=sha256(ROOT/'src/research_daily_graph_data.py'),
        factor_fill=False, qfq_end_sample_normalization=False)
    if not failures:
        frames = [pd.read_pickle(args.output_dir/results[text]['file']) for text in sorted(results)]
        factors = pd.concat(frames, ignore_index=True)
        manifest['aggregate'] = publish_frame(factors, args.output_dir/'adjustments.pkl')
        adjusted = adjusted_prices(daily, factors)
        manifest['adjusted_daily'] = publish_frame(adjusted, args.output_dir/'adjusted_daily.pkl')
    publish_manifest(manifest, manifest_path)
    if failures:
        raise ValueError('Adjustment coverage incomplete; resume missing dates')
    print('adjustments complete', len(groups), 'dates', len(daily), 'quoted rows', flush=True)


def members(args):
    catalog = pd.read_pickle(args.catalog)
    required = ['trade_date', 'ts_code', 'up_num', 'down_num']
    if not set(required) <= set(catalog) or catalog.duplicated(['trade_date', 'ts_code']).any():
        raise ValueError('Invalid dated concept catalogue')
    catalog['trade_date'] = pd.to_datetime(catalog.trade_date).dt.strftime('%Y%m%d')
    catalog = catalog[catalog.trade_date.between(pd.Timestamp(args.graph_start).strftime('%Y%m%d'), pd.Timestamp(args.graph_end).strftime('%Y%m%d'))].copy()
    if args.graph_dates:
        requested = set(pd.Timestamp(day).strftime('%Y%m%d') for day in args.graph_dates.split(','))
        missing = requested-set(catalog.trade_date)
        if missing:
            raise ValueError('Requested graph dates absent from dated catalogue: '+str(sorted(missing)))
        catalog = catalog[catalog.trade_date.isin(requested)].copy()
    catalog['minimum'] = catalog[['up_num', 'down_num']].apply(pd.to_numeric, errors='raise').sum(axis=1)
    if catalog.empty or not np.isfinite(catalog.minimum).all() or catalog.minimum.lt(0).any() or not catalog.minimum.eq(catalog.minimum.astype(int)).all():
        raise ValueError('Invalid dated member lower bounds')
    if not catalog.ts_code.str.fullmatch(r'BK\d{4}\.DC').all():
        raise ValueError('Graph source contains non-DC concept code')
    directory = args.output_dir/'members'; directory.mkdir(parents=True, exist_ok=True)
    snapshots_dir = args.output_dir/'graphs'; snapshots_dir.mkdir(parents=True, exist_ok=True)
    results, failures, snapshots = {}, {}, {}
    planned = catalog.groupby('trade_date').ts_code.apply(set).to_dict()
    completed = {day: set() for day in planned}
    attempted = {day: set() for day in planned}
    manifest_path = args.output_dir/'graph_manifest.json'
    publish_manifest(dict(status='running', planned_snapshots=len(planned), planned_theme_snapshots=len(catalog)), manifest_path)

    def one(row):
        text, code, minimum = str(row.trade_date), row.ts_code, int(row.minimum)
        key = code+'_'+text; path = directory/(key+'.pkl')
        validator = lambda frame: validate_membership_snapshot(frame, text, code, minimum, limit=5000)
        try:
            cached = _cached_frame(path, validator)
            if cached is not None:
                return text, code, dict(file=str(path.relative_to(args.output_dir)), rows=len(cached), sha256=sha256(path), minimum=minimum), None
            old = ROOT/'data/raw/dc_theme_research/members'/('dc_member_'+key+'.pkl')
            if old.exists():
                frame = validator(pd.read_pickle(old)); origin = dict(source='verified_legacy_date_members', path=str(old.relative_to(ROOT)), source_sha256=sha256(old))
            elif args.offline:
                raise ValueError('Missing offline membership shard')
            else:
                fields = 'trade_date,ts_code,con_code,name'
                variants = [dict(trade_date=text, fields=fields, limit=5000, offset=0),
                    dict(start_date=text, end_date=text, fields=fields, limit=5000, offset=0)]
                errors, audits = [], []
                for params in variants:
                    try:
                        response = safe_query(client(), 'dc_member', attempts=2, ts_code=code, **params)
                        frame = validator(response)
                        audits.append(dict(params=params, rows=len(response), accepted=True))
                        break
                    except Exception as exc:
                        errors.append(str(exc)); audits.append(dict(params=params, accepted=False, error=str(exc)))
                else:
                    publish_manifest(dict(status='rejected', ts_code=code, trade_date=text,
                        minimum=minimum, requests=audits, errors=errors), path.with_suffix('.failed.json'))
                    raise ValueError('No complete dated membership query: '+str(errors))
                origin = dict(source='tushare_gateway', api='dc_member', ts_code=code, trade_date=text, requests=audits)
            artifact = publish_frame(frame, path)
            publish_manifest(dict(status='complete', **artifact, minimum=minimum,
                count_validation='dated catalogue up_num+down_num lower bound, not independent census', **origin), path.with_suffix('.json'))
            return text, code, dict(file=str(path.relative_to(args.output_dir)), rows=len(frame), sha256=artifact['sha256'], minimum=minimum), None
        except Exception as exc:
            return text, code, None, str(exc)

    def checkpoint(status):
        for day in sorted(planned):
            if completed[day] and snapshots.get(day, {}).get('successful_themes') != len(completed[day]):
                pieces = []
                for code in sorted(completed[day]):
                    frame = pd.read_pickle(args.output_dir/results[code+'_'+day]['file'])
                    pieces.append(pd.DataFrame(dict(snapshot_date=pd.to_datetime(frame.trade_date), theme_code=frame.ts_code, ts_code=frame.con_code)))
                edges = validate_edges(pd.concat(pieces, ignore_index=True))
                snapshots[day] = publish_frame(edges, snapshots_dir/(day+'.pkl'))
                snapshots[day]['file'] = 'graphs/'+day+'.pkl'
                stock_degrees = edges.groupby('ts_code').theme_code.nunique()
                snapshots[day].update(successful_themes=len(completed[day]),
                    catalogue_themes=len(planned[day]), attempted_themes=len(attempted[day]),
                    failed_themes=len(attempted[day]-completed[day]), stocks=int(edges.ts_code.nunique()),
                    multi_label_stocks=int(stock_degrees.gt(1).sum()),
                    multi_label_ratio=float(stock_degrees.gt(1).mean()),
                    status='complete' if completed[day] == planned[day] else 'verified_partial',
                    member_count_validation='>= dated catalogue up_num+down_num lower bound; not independent census')
                print('graph snapshot', day, len(completed[day]), '/', len(planned[day]), 'verified themes', len(edges), 'edges', edges.ts_code.nunique(), 'stocks', flush=True)
            if day in snapshots:
                snapshots[day]['attempted_themes'] = len(attempted[day])
                snapshots[day]['failed_themes'] = len(attempted[day]-completed[day])
                snapshots[day]['attempts_finished'] = attempted[day] == planned[day]
        manifest = dict(status=status, source='historical_dc_members',
            graph_start=args.graph_start, graph_end=args.graph_end,
            explicitly_requested_snapshot_dates=sorted(planned),
            planned_snapshots=len(planned), verified_partial_snapshots=len(snapshots),
            completed_snapshots=sum(s['status']=='complete' for s in snapshots.values()),
            planned_theme_snapshots=len(catalog), completed_theme_snapshots=len(results),
            failures=failures, shards=results, snapshots=snapshots,
            catalogue=dict(path=str(args.catalog.resolve()), sha256=sha256(args.catalog)),
            collector_sha256=sha256(Path(__file__)), validator_sha256=sha256(ROOT/'src/research_daily_graph_data.py'),
            theme_selection='ALL contemporaneous dated catalogue concept themes; no top-N or future whitelist',
            multi_label=True, carry_forward='last completed whole snapshot at or before signal close; caller must declare max-age policy',
            partial_graph_policy='verified edges may be used as contextual messages, never candidate eligibility; absent edges are unknown, not non-membership',
            request_order='date round-robin; SHA256(theme_code) order within each date, independent of stock names and future prices',
            snapshot_coverage={day: dict(catalogue_themes=len(planned[day]), successful_themes=len(completed[day]),
                failed_themes=len(attempted[day]-completed[day]), attempted_themes=len(attempted[day]),
                attempts_finished=attempted[day]==planned[day],
                failed_lower_bound_reasons={code: failures[code+'_'+day] for code in sorted(attempted[day]-completed[day])}) for day in sorted(planned)},
            current_ths_backfill=False, all_theme_api_census_accepted=False,
            count_validation='per-theme >= contemporaneous up_num+down_num lower bound; not independent member census')
        if status == 'complete':
            edges = validate_edges(pd.concat([pd.read_pickle(args.output_dir/snapshots[day]['file']) for day in sorted(snapshots)], ignore_index=True))
            manifest['aggregate'] = publish_frame(edges, args.output_dir/'monthly_edges.pkl')
        elif snapshots:
            partial = pd.concat([pd.read_pickle(args.output_dir/snapshots[day]['file']) for day in sorted(snapshots)], ignore_index=True)
            manifest['partial_aggregate'] = publish_frame(partial, args.output_dir/'monthly_edges.partial.pkl')
        publish_manifest(manifest, manifest_path)

    # Publish all already verified local shards before requesting any missing
    # theme. Resume ordering must not temporarily remove known old edges.
    cached_keys = set()
    for row in catalog.itertuples():
        day, code = str(row.trade_date), row.ts_code; key = code+'_'+day
        path = directory/(key+'.pkl')
        try:
            cached = _cached_frame(path, lambda frame: validate_membership_snapshot(
                frame, day, code, int(row.minimum), limit=5000))
        except ValueError:
            cached = None
        if cached is not None:
            results[key] = dict(file=str(path.relative_to(args.output_dir)), rows=len(cached), sha256=sha256(path), minimum=int(row.minimum))
            completed[day].add(code); attempted[day].add(code); cached_keys.add(key)
    checkpoint('running')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        catalog['theme_request_hash'] = catalog.ts_code.map(lambda code: hashlib.sha256(code.encode('ascii')).hexdigest())
        per_date_rows = [list(group.sort_values('theme_request_hash').itertuples())
                         for _, group in catalog.groupby('trade_date', sort=True)]
        request_rows = [row for bundle in zip_longest(*per_date_rows) for row in bundle
                        if row is not None and row.ts_code+'_'+str(row.trade_date) not in cached_keys]
        futures = [pool.submit(one, row) for row in request_rows]
        for number, future in enumerate(as_completed(futures), 1):
            text, code, artifact, failure = future.result(); key = code+'_'+text
            attempted[text].add(code)
            if failure:
                failures[key] = failure
            else:
                results[key] = artifact; completed[text].add(code)
            if number % 100 == 0:
                checkpoint('running')
                print('members', sum(map(len, attempted.values())), '/', len(catalog), 'verified', len(results), 'failures', len(failures), flush=True)
            elif failure:
                print('membership unavailable', key, failure, flush=True)
    checkpoint('incomplete' if failures else 'complete')
    if failures:
        raise ValueError('Historical graph incomplete; resume failed theme/date shards')
    print('graph complete', len(snapshots), 'snapshots', len(results), 'dated themes', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['adjustments', 'members'])
    parser.add_argument('--output-dir', type=Path, default=DEFAULT)
    parser.add_argument('--daily', type=Path, default=ROOT/'results/dc_context_inputs_20261002/daily.pkl')
    parser.add_argument('--catalog', type=Path, default=ROOT/'data/raw/dc_theme_research/catalogs.pkl')
    parser.add_argument('--start', default='2024-09-01')
    parser.add_argument('--end', default='2026-09-24')
    parser.add_argument('--graph-start', default='2024-12-20')
    parser.add_argument('--graph-end', default='2026-08-31')
    parser.add_argument('--graph-dates', help='Explicit generic snapshot-date schedule, comma-separated; every dated theme is still requested')
    parser.add_argument('--workers', type=int, choices=range(1, 13), default=6)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.action == 'adjustments':
        adjustments(args)
    else:
        members(args)


if __name__ == '__main__':
    main()
