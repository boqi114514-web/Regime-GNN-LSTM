"""Dated Eastmoney research inputs, without SW or current-member backfill.

Catalogue/membership and dated market snapshots: user-authorized GET gateway.
The credential-free Eastmoney endpoint is an independent price cross-check.
All caches are source-specific and validated on read.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_pipeline.execution_data import ROOT
from data_pipeline.public_theme_data import fetch_eastmoney_history
from data_pipeline.theme_data import validate_catalog, validate_members, validate_index_bars
from research_dc_themes import rank_themes
from probe_concept_gateway import probe_request


DEFAULT = Path(__file__).resolve().parents[1] / 'data/raw/dc_theme_research'
START, END = '20241001', '20260924'


def publish_frame(frame, path):
    staged = path.with_suffix('.staging.pkl')
    frame.to_pickle(staged)
    staged.replace(path)
    return dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=len(frame))


def publish_manifest(payload, path):
    staged = path.with_suffix('.staging.json')
    staged.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    staged.replace(path)


def signal_dates():
    monthly = pd.read_pickle(ROOT/'stock_month_end_verified.pkl')
    dates = sorted(pd.to_datetime(monthly.date.unique()))
    return [day for day in dates if pd.Timestamp('2024-12-01') <= day <= pd.Timestamp('2026-08-31')]


def gateway_snapshot(directory, api, code, day, validate):
    key = os.environ.get('TUSHARE_API_KEY', '').strip()
    if not key:
        raise ValueError('Missing TUSHARE_API_KEY')
    params = dict(trade_date=day.strftime('%Y%m%d'), limit=1000)
    if code is None:
        params['idx_type'] = '概念板块'
    else:
        params['ts_code'] = code
    name = api+'_'+(code+'_' if code else '')+params['trade_date']
    path = directory/(name+'.pkl')
    if path.exists():
        cached = pd.read_pickle(path)
        try:
            return validate(cached)
        except ValueError:
            # Preserve rejected source evidence while allowing a verified repair.
            cached.to_pickle(directory/(name+'.rejected.pkl'))
    audit_path = directory/(name+'.json')
    audits = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else []
    variants = [params]
    if api=='dc_member':
        explicit = dict(params,fields='trade_date,ts_code,con_code,name')
        ranged = {k:v for k,v in explicit.items() if k!='trade_date'}
        ranged.update(start_date=params['trade_date'],end_date=params['trade_date'])
        # Both forms are officially supported. A successful one-row response
        # remains invalid when it fails the dated catalogue's count lower bound.
        variants = [explicit,ranged,params]
    minimal = {k: v for k, v in params.items() if k != 'limit'}
    variants.append(minimal)
    variants.append(dict(params, offset=0, limit=5000 if code is None else 2000))
    with requests.Session() as session:
        for attempt, request_params in enumerate(variants, 1):
            record, frame = probe_request(session, 'gateway', key,
                dict(name=name, api=api, params=request_params))
            record['attempt'] = attempt
            audits.append(record)
            if record['status'] == 'ok' and frame is not None:
                try:
                    result = validate(frame)
                except ValueError as exc:
                    record['validation_error'] = str(exc)
                else:
                    result.to_pickle(path)
                    audit_path.write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding='utf-8')
                    return result
            audit_path.write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding='utf-8')
            if (record['http_status'] not in (400, 503, 504)
                    and record['status'] not in ('transport_error','invalid_schema')
                    and 'validation_error' not in record):
                break
    raise ValueError('Dated theme snapshot unavailable: '+name)


def catalogs(root):
    directory = root/'catalogs'; directory.mkdir(parents=True, exist_ok=True)
    frames, failures = [], []
    publish_manifest(dict(status='running'), root/'catalog_manifest.json')
    for day in signal_dates():
        try:
            frame = gateway_snapshot(directory, 'dc_index', None, day,
                lambda d: validate_catalog(d, day, limit=1000))
            frames.append(frame)
            print('catalog', day.strftime('%Y%m%d'), len(frame), flush=True)
        except ValueError as exc:
            failures.append(str(exc)); print(str(exc), flush=True)
    manifest = dict(status='complete' if not failures else 'incomplete',
                    snapshots=len(frames), failures=failures)
    if failures:
        publish_manifest(manifest, root/'catalog_manifest.json')
        raise ValueError('Incomplete dated catalogue; resume to retry missing snapshots')
    result = pd.concat(frames, ignore_index=True)
    manifest['aggregate'] = publish_frame(result, root/'catalogs.pkl')
    publish_manifest(manifest, root/'catalog_manifest.json')
    return result


def histories(root, workers):
    catalog = pd.read_pickle(root/'catalogs.pkl')
    directory = root/'histories'; directory.mkdir(parents=True, exist_ok=True)
    codes = sorted(catalog.ts_code.unique())
    def one(code):
        path = directory/(code+'.pkl')
        if path.exists():
            frame = validate_index_bars(pd.read_pickle(path), code, START, END)
            return code, len(frame), None
        errors = []
        for attempt in range(1, 3):
            try:
                with requests.Session() as session:
                    frame = fetch_eastmoney_history(session, code, START, END)
                frame.to_pickle(path)
                (directory/(code+'.json')).write_text(json.dumps(dict(
                    source='eastmoney_public', requested_start=START, requested_end=END,
                    source_name=str(frame.source_name.iloc[0]), rows=len(frame),
                    first=frame.trade_date.min(), last=frame.trade_date.max(),
                    attempts=attempt, previous_errors=errors, volume_unit='unconfirmed_raw',
                    amount_unit='CNY', historical_membership=False), ensure_ascii=False, indent=2), encoding='utf-8')
                return code, len(frame), None
            except (ValueError, RuntimeError) as exc:
                errors.append(str(exc))
        (directory/(code+'.json')).write_text(json.dumps(dict(status='failed', errors=errors), indent=2), encoding='utf-8')
        return code, 0, errors
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, code) for code in codes]
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            if len(results) % 25 == 0 or result[2]:
                print('histories', len(results), '/', len(codes), 'last', result[0], result[1], result[2], flush=True)
    failures = {code: errors for code, _, errors in results if errors}
    manifest = dict(status='complete' if not failures else 'incomplete', indexes=len(codes), failures=failures)
    (root/'history_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if failures:
        raise ValueError('Incomplete index acquisition; resume missing indexes')


def features(root, output, scope='full'):
    output.mkdir(parents=True, exist_ok=True)
    feature_manifest_path = output/'theme_feature_manifest.json'
    publish_manifest(dict(status='running', scope=scope), feature_manifest_path)
    history_manifest = ('daily_market_2026_manifest.json' if scope=='2026' else
                        'daily_market_manifest.json' if (root/'daily_market_manifest.json').exists() else 'history_manifest.json')
    for name in ('catalog_manifest.json', history_manifest):
        manifest = json.loads((root/name).read_text(encoding='utf-8'))
        if manifest.get('status') != 'complete':
            raise ValueError('Input acquisition is incomplete: '+name)
        if 'aggregate' in manifest:
            artifact = manifest['aggregate']
            if hashlib.sha256((root/artifact['file']).read_bytes()).hexdigest() != artifact['sha256']:
                raise ValueError('Acquisition aggregate fingerprint mismatch: '+name)
    catalog = pd.read_pickle(root/'catalogs.pkl')
    if scope=='2026':
        catalog = catalog[catalog.trade_date.between('20251201','20260831')].copy()
    daily = pd.read_pickle(Path(__file__).resolve().parents[1]/'results/liquid_leader_research/daily.pkl')
    calendar = sorted(pd.to_datetime(daily.date.unique()).strftime('%Y%m%d'))
    del daily
    positions = {day: i for i, day in enumerate(calendar)}
    records, unavailable = [], []
    if history_manifest.startswith('daily_market'):
        artifact = json.loads((root/history_manifest).read_text(encoding='utf-8'))['aggregate']
        market = pd.read_pickle(root/artifact['file'])
        by_code = {code: group.sort_values('trade_date') for code, group in market.groupby('ts_code')}
    else:
        by_code = {code: pd.read_pickle(root/'histories'/(code+'.pkl')).sort_values('trade_date')
                   for code in catalog.ts_code.unique()}
    for row in catalog.itertuples():
        day = str(row.trade_date)
        if row.ts_code not in by_code:
            unavailable.append(dict(signal_date=day, ts_code=row.ts_code, observations=0, reason='no_index_observations'))
            continue
        bars = by_code[row.ts_code]
        bars = bars[bars.trade_date.le(day)]
        if len(bars) < 61 or bars.trade_date.iloc[-1] != day:
            unavailable.append(dict(signal_date=day, ts_code=row.ts_code, observations=len(bars),
                                    reason='insufficient_warmup' if len(bars)<61 else 'missing_signal_day'))
            continue
        # Consecutive observations are independently checked against a stock
        # trading calendar, rather than treating a hole as a 20/60-session lag.
        index = positions[day]
        expected = calendar[index-60:index+1]
        if list(bars.trade_date.iloc[-61:]) != expected:
            unavailable.append(dict(signal_date=day, ts_code=row.ts_code, observations=len(bars),
                                    reason='incomplete_61_session_calendar'))
            continue
        records.append(dict(signal_date=pd.Timestamp(day), catalogue_date=pd.Timestamp(day),
            index_obs=len(bars), ts_code=row.ts_code, name=row.name,
            ret20=float(bars.close.iloc[-1]/bars.close.iloc[-21]-1),
            ret60=float(bars.close.iloc[-1]/bars.close.iloc[-61]-1),
            amount_ratio=float(bars.amount.iloc[-20:].mean()/bars.amount.iloc[-60:].mean()),
            source_last_date=pd.Timestamp(bars.trade_date.iloc[-1])))
    frame = pd.DataFrame(records)
    if frame.empty or not np.isfinite(frame[['ret20','ret60','amount_ratio']]).all().all():
        raise ValueError('Invalid theme features')
    artifact = publish_frame(frame, output/'theme_features.pkl')
    (output/'theme_feature_coverage.json').write_text(json.dumps(dict(
        source='dated_dc_catalog_and_dc_index_prices', scope=scope,
        catalog_rows=len(catalog), feature_rows=len(frame), unavailable=unavailable,
        no_current_membership_backfill=True), ensure_ascii=False, indent=2), encoding='utf-8')
    inputs = [root/'catalogs.pkl', root/'catalog_manifest.json', root/history_manifest,
              Path(__file__).resolve()]
    history = json.loads((root/history_manifest).read_text(encoding='utf-8'))
    if 'aggregate' in history:
        inputs.append(root/history['aggregate']['file'])
    else:
        inputs += [root/'histories'/(code+'.pkl') for code in catalog.ts_code.unique()]
    publish_manifest(dict(status='complete', scope=scope, aggregate=artifact,
        source_inputs={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}),
        feature_manifest_path)
    return frame


def validate_market_day(frame, day, expected_codes=None):
    required = ['trade_date','ts_code','open','high','low','close','vol','amount']
    if not set(required) <= set(frame.columns) or frame.empty or len(frame)>=1000:
        raise ValueError('Empty, truncated or malformed DC market-day response')
    d = frame.copy()
    d['trade_date'] = pd.to_datetime(d.trade_date, errors='raise').dt.strftime('%Y%m%d')
    if not d.trade_date.eq(day.strftime('%Y%m%d')).all() or d.ts_code.duplicated().any():
        raise ValueError('Wrong date or duplicate code in DC market-day response')
    if not d.ts_code.str.fullmatch(r'BK\d{4}\.DC').all():
        raise ValueError('Non-DC code in DC market-day response')
    if 'category' not in d or not d.category.eq('概念板块').all():
        raise ValueError('DC market-day response does not exclusively contain concepts')
    if expected_codes is not None and set(expected_codes)-set(d.ts_code):
        raise ValueError('Missing contemporaneous catalogue indexes in market-day response: '+
                         str(sorted(set(expected_codes)-set(d.ts_code))))
    if len(d)<100:
        raise ValueError('Unexpectedly small whole-market concept snapshot')
    for col in ['open','high','low','close','vol','amount']:
        d[col] = pd.to_numeric(d[col], errors='coerce')
        if not np.isfinite(d[col]).all() or d[col].lt(0).any() or (col in ['open','high','low','close'] and d[col].eq(0).any()):
            raise ValueError('Invalid market-day '+col)
    if d.high.lt(d[['open','close','low']].max(axis=1)).any() or d.low.gt(d[['open','close','high']].min(axis=1)).any():
        raise ValueError('Invalid market-day OHLC geometry')
    return d


def daily_market(root, workers=2, scope='full'):
    """Date batches retain discontinued themes, unlike a current public catalog."""
    raw = pd.read_pickle(Path(__file__).resolve().parents[1]/'results/liquid_leader_research/daily.pkl')
    days = sorted(pd.to_datetime(raw.date.unique()))
    del raw
    first = pd.Timestamp('2025-08-01') if scope=='2026' else pd.Timestamp(START)
    days = [day for day in days if first<=day<=pd.Timestamp('2026-08-31')]
    directory = root/'daily_market'; directory.mkdir(parents=True, exist_ok=True)
    catalogue = pd.read_pickle(root/'catalogs.pkl')
    expected = {pd.Timestamp(day):set(group.ts_code) for day,group in catalogue.groupby('trade_date')}
    manifest_name = 'daily_market_2026_manifest.json' if scope=='2026' else 'daily_market_manifest.json'
    aggregate_name = 'daily_market_2026.pkl' if scope=='2026' else 'daily_market.pkl'
    publish_manifest(dict(status='running',scope=scope), root/manifest_name)
    frames, failures = [], []
    consecutive_failures = 0
    batches = [days[i:i+3] for i in range(0,len(days),3)]
    def one(batch):
        cached = []
        for day in batch:
            path = directory/f'dc_daily_{day:%Y%m%d}.pkl'
            if path.exists():
                try:
                    cached.append(validate_market_day(pd.read_pickle(path),day,expected.get(day)))
                except ValueError:
                    # Keep every valid overlap anchor when repairing a bad shard.
                    continue
        if len(cached)==len(batch):
            return cached, None
        key = os.environ.get('TUSHARE_API_KEY','').strip()
        if not key:
            return [], 'Missing TUSHARE_API_KEY'
        name=f'market_range_{batch[0]:%Y%m%d}_{batch[-1]:%Y%m%d}'
        base=dict(start_date=f'{batch[0]:%Y%m%d}',end_date=f'{batch[-1]:%Y%m%d}',idx_type='概念板块',limit=2000)
        variants=[base,dict(base,offset=0),{k:v for k,v in base.items() if k!='limit'}]
        audits=[]
        with requests.Session() as session:
            for attempt,params in enumerate(variants,1):
                record,frame=probe_request(session,'gateway',key,dict(name=name,api='dc_daily',params=params))
                record['attempt']=attempt
                audits.append(record)
                validated=[]
                if record['status']=='ok' and frame is not None:
                    try:
                        wanted={f'{day:%Y%m%d}' for day in batch}
                        if set(frame.trade_date.astype(str))!=wanted or len(frame)>=2000:
                            raise ValueError('Batch dates missing, extra or row limit reached')
                        for day in batch:
                            shard=validate_market_day(frame[frame.trade_date.eq(f'{day:%Y%m%d}')],day,expected.get(day))
                            validated.append(shard)
                        for old in cached:
                            fresh=next(d for d in validated if d.trade_date.iloc[0]==old.trade_date.iloc[0])
                            a=old.set_index('ts_code').sort_index();b=fresh.set_index('ts_code').sort_index()
                            if not a.index.equals(b.index) or not np.allclose(a[['open','close','high','low']],b[['open','close','high','low']],atol=1e-6,rtol=0):
                                raise ValueError('Batch differs from previously verified daily prices')
                    except ValueError as exc:
                        record['validation_error']=str(exc);validated=[]
                publish_manifest(audits,directory/(name+'.json'))
                if validated:
                    publish_frame(frame,directory/(name+'.pkl'))
                    for day,shard in zip(batch,validated):
                        shard.attrs['source_audit']=name+'.json'
                        publish_frame(shard,directory/f'dc_daily_{day:%Y%m%d}.pkl')
                    return validated,None
                if record['http_status'] not in (400,503,504) and record['status']!='invalid_schema' and 'validation_error' not in record:
                    break
        return [],'Unavailable or incomplete market batch: '+name
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = [pool.submit(one, batch) for batch in batches]
    try:
        for number, future in enumerate(as_completed(futures),1):
            frame, failure = future.result()
            if failure:
                failures.append(failure); consecutive_failures+=1
            else:
                frames.extend(frame); consecutive_failures=0
            if number%20==0 or failure:
                print('market_batches',number,'/',len(batches),'complete_days',len(frames),'/',len(days),'failures',len(failures),flush=True)
            if consecutive_failures>=5:
                failures.append('Circuit stop after 5 consecutive failed date batches')
                break
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
    complete = len(frames)==len(days)
    manifest = dict(status='complete' if complete else 'incomplete',
        planned_days=len(days), completed_days=len(frames), failures=failures,
        signal_catalogue_coverage_required=True,scope=scope,
        start_date=f'{first:%Y%m%d}',end_date='20260831')
    if not complete:
        publish_manifest(manifest,root/manifest_name)
        raise ValueError('DC market-day acquisition incomplete; resume missing dates')
    manifest['aggregate'] = publish_frame(pd.concat(frames,ignore_index=True),root/aggregate_name)
    publish_manifest(manifest,root/manifest_name)


def members(root, output, scope='full'):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'theme_membership_manifest.json'
    source_manifest_path = root/('membership_2026_manifest.json' if scope=='2026' else 'membership_manifest.json')
    publish_manifest(dict(status='running',scope=scope), manifest_path)
    feature_manifest = json.loads((output/'theme_feature_manifest.json').read_text(encoding='utf-8'))
    feature_sha = hashlib.sha256((output/'theme_features.pkl').read_bytes()).hexdigest()
    if (feature_manifest.get('status')!='complete' or feature_manifest.get('scope')!=scope
            or feature_manifest.get('aggregate',{}).get('sha256')!=feature_sha):
        raise ValueError('Incomplete, wrong-scope or modified feature input')
    frame = rank_themes(pd.read_pickle(output/'theme_features.pkl'))
    chosen = frame[frame.theme_selected].copy()
    directory = root/'members'; directory.mkdir(parents=True, exist_ok=True)
    catalogue = pd.read_pickle(root/'catalogs.pkl').set_index(['trade_date','ts_code'])
    records, failures = [], []
    publish_manifest(dict(status='running',scope=scope),source_manifest_path)
    for row in chosen.itertuples():
        day = row.signal_date; code = row.ts_code
        try:
            minimum = int(catalogue.loc[(day.strftime('%Y%m%d'), code),['up_num','down_num']].sum())
            def validate(d):
                result = validate_members(d, day, code, limit=1000)
                if len(result)<minimum:
                    raise ValueError(f'Membership has {len(result)} rows, below {minimum} contemporaneous advancing/declining stocks')
                return result
            data = gateway_snapshot(directory, 'dc_member', code, day,
                validate)
            records.append(data)
            print('members', day.strftime('%Y%m%d'), code, row.name, len(data), flush=True)
        except ValueError as exc:
            failures.append(str(exc)); print(str(exc), flush=True)
    selected_artifact = publish_frame(chosen,output/'theme_selected.pkl')
    manifest = dict(status='complete' if not failures else 'incomplete', scope=scope,
                    requested=len(chosen), failures=failures, selected=selected_artifact,
                    feature_sha256=feature_sha)
    if failures:
        publish_manifest(manifest,source_manifest_path)
        publish_manifest(manifest,manifest_path)
        raise ValueError('Incomplete historical memberships; resume missing snapshots')
    manifest['aggregate'] = publish_frame(pd.concat(records, ignore_index=True),output/'theme_members.pkl')
    manifest['aggregate']['directory'] = str(output.resolve())
    inputs = [root/'catalogs.pkl',output/'theme_feature_manifest.json',Path(__file__).resolve(),
              Path(__file__).resolve().parents[1]/'src/research_dc_themes.py']
    manifest['source_inputs'] = {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    manifest['coverage_check'] = 'At least dated catalogue up_num+down_num; this is a lower bound, not an independent membership census'
    publish_manifest(manifest,source_manifest_path)
    publish_manifest(manifest,manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['catalogs','histories','daily-market','features','members'])
    parser.add_argument('--source-dir', type=Path, default=DEFAULT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--workers', type=int, choices=[1,2], default=2)
    parser.add_argument('--scope',choices=['full','2026'],default='full')
    args = parser.parse_args(); args.source_dir.mkdir(parents=True, exist_ok=True)
    if args.output_dir is None:
        args.output_dir = Path('results/dc_theme_2026_research' if args.scope=='2026' else 'results/dc_theme_research')
    if args.action=='catalogs': catalogs(args.source_dir)
    elif args.action=='histories': histories(args.source_dir, args.workers)
    elif args.action=='daily-market': daily_market(args.source_dir, args.workers, args.scope)
    elif args.action=='features': features(args.source_dir, args.output_dir, args.scope)
    else: members(args.source_dir, args.output_dir, args.scope)


if __name__=='__main__':
    main()
