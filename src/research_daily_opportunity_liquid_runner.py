"""Verified liquid-native experiment, reusing exact caches without old-source edits."""
import argparse
import json
import os
from pathlib import Path
import shutil

import pandas as pd

from research_daily_graph_data import load_adjusted_daily, publish_manifest, sha256
from research_daily_opportunity import validate_predictions
from research_daily_opportunity_execution import ExecutionPolicy, OfficialLimits, run_account
from research_daily_opportunity_leadership import leadership_variant
from research_daily_opportunity_liquid_filter import filter_liquid_predictions, validate_liquidity_inputs
from research_daily_opportunity_runner import BackoffGateway, limits_caches, seed_actions
from research_scoped_actions import validate_history


class ExactBatchGateway(BackoffGateway):
    """Use the gateway's full-date official-limit request, validate keys downstream."""
    def query(self, api, **params):
        if api == 'stk_limit' and 'trade_date' in params and 'ts_code' not in params:
            params = {'trade_date': params['trade_date']}
        return super().query(api, **params)


def cache_roots():
    roots = limits_caches()
    for pattern in ('daily_opportunity*/daily_limits', 'daily_opportunity*/account/daily_limits'):
        roots.extend(sorted(Path('results').glob(pattern)))
    return roots


def seed_all_actions(account):
    seed_actions(account)
    for source in sorted(Path('results').glob('daily_opportunity*/account/scoped_actions/*.pkl')):
        target = account/'scoped_actions'/source.name
        meta = source.with_suffix('.json')
        if target.exists() or not meta.exists():
            continue
        manifest = json.loads(meta.read_text(encoding='utf-8'))
        if (manifest.get('start'), manifest.get('end'), manifest.get('raw_sha256')) != (
                '2026-01-01', '2026-09-24', sha256(source)):
            continue
        validate_history(pd.read_pickle(source), source.stem, '2026-01-01', '2026-09-24')
        shutil.copy2(source, target)
        shutil.copy2(meta, target.with_suffix('.json'))


def run_liquid_account(source, out, inputs, offline=False):
    source, out, inputs = Path(source), Path(out), Path(inputs)
    if source.resolve() == out.resolve() or (out/'account/daily_run_status.json').exists():
        raise ValueError('Use a separate output without an already completed account')
    checked = validate_liquidity_inputs(source)
    with leadership_variant():
        predictions = validate_predictions(source)
    out.mkdir(parents=True, exist_ok=True)
    for file in source.iterdir():
        if file.is_file() and file.name != 'predictions.csv':
            shutil.copy2(file, out/file.name)
    shutil.copytree(source/'checkpoints', out/'checkpoints', dirs_exist_ok=True)
    daily, manifest = load_adjusted_daily(inputs)
    feature = json.loads((source/'feature_manifest.json').read_text(encoding='utf-8'))
    if manifest['adjusted_daily']['sha256'] != feature['raw_adjusted_daily']['sha256']:
        raise ValueError('Account raw quotes differ from trained feature source')
    account = out/'account'
    account.mkdir(exist_ok=True)
    seed_all_actions(account)
    pro = None if offline else ExactBatchGateway(os.environ['TUSHARE_API_KEY'])
    limits = OfficialLimits(account, pro=pro, offline=offline, cache_roots=cache_roots())
    result = run_account(predictions, daily, sorted(daily.date.unique()), account,
                         pro=pro, offline=offline, policy=ExecutionPolicy(trailing_distance=.08),
                         limit_provider=limits)
    path = account/'daily_run_status.json'
    status = json.loads(path.read_text(encoding='utf-8'))
    status['source_sha256'].update(checked['band_manifest']['source_sha256'])
    for file in ('prior_band_manifest.json', 'prior_band_checks.csv',
                 'liquidity_manifest.json', 'liquidity_checks.csv'):
        status['source_sha256'][str((source/file).resolve())] = sha256(source/file)
    for name in ('research_daily_opportunity_liquid_filter.py', Path(__file__).name):
        file = Path(__file__).parent/name
        status['source_sha256'][str(file.resolve())] = sha256(file)
    publish_manifest(status, path)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('filter', 'account'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, default=Path('data/raw/daily_opportunity_20261002'))
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.action == 'filter':
        pro = None if args.offline else ExactBatchGateway(os.environ['TUSHARE_API_KEY'])
        filter_liquid_predictions(args.source, args.out, pro, offline=args.offline, cache_roots=cache_roots())
        checked = validate_liquidity_inputs(args.out)
        print('verified liquidity eligibility', checked['manifest']['accepted'], flush=True)
    else:
        run_liquid_account(args.source, args.out, args.inputs, args.offline)
