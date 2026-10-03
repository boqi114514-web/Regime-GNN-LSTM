"""Preserve path-label provenance through the declared band/liquid ablations.

Both accounts retain all frozen forecasts and the same 25,000-yuan whole-lot
engine. Only new-buy eligibility differs. Training targets omit locked-limit
execution; actual accounts use exact official limits and pending exit orders.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from research_daily_graph_data import load_adjusted_daily, publish_manifest, sha256
from research_daily_opportunity_band_filter import filter_buy_risk_band, validate_band_inputs
from research_daily_opportunity_execution import ExecutionPolicy, OfficialLimits, _mainboard, run_account
from research_daily_opportunity_leadership import leadership_variant
from research_daily_opportunity_liquid_filter import filter_liquid_predictions, validate_liquidity_inputs
from research_daily_opportunity_liquid_runner import ExactBatchGateway, cache_roots, seed_all_actions
from research_daily_opportunity_path import FIXED_CONFIGURATION, validate_path_predictions


PATH_ARTIFACTS = ('path_targets.pkl', 'path_targets_manifest.json')
MODES = ('band', 'liquid')


def bind_path_filter_artifacts(source, out, mode):
    """Restore separate target artifacts after old filters copy their fixed file list."""
    source, out = Path(source), Path(out)
    if mode not in MODES or source.resolve() == out.resolve():
        raise ValueError('A separate destination and band/liquid mode are required')
    source_manifest = json.loads((source/'prediction_manifest.json').read_text(encoding='utf-8'))
    for name in PATH_ARTIFACTS:
        expected = source_manifest.get('artifacts', {}).get(name)
        if expected is None or sha256(source/name) != expected:
            raise ValueError('Original prediction manifest does not bind '+name)
        shutil.copy2(source/name, out/name)
    protocol = json.loads((out/'experiment_protocol.json').read_text(encoding='utf-8'))
    protocol['path_account_ablation'] = dict(mode=mode, declared_modes=list(MODES),
                                           targets_and_native_forecasts='Unchanged',
                                           stop_configuration=FIXED_CONFIGURATION,
                                           source_directory=str(source.resolve()))
    for name in (Path(__file__).name, 'research_daily_opportunity_liquid_runner.py'):
        protocol['source_sha256'][name] = sha256(Path(__file__).parent/name)
    publish_manifest(protocol, out/'experiment_protocol.json')
    sources = ('predictions.pkl', 'prediction_manifest.json')+PATH_ARTIFACTS
    publish_manifest(dict(status='complete', mode=mode, declared_modes=list(MODES),
                          source_directory=str(source.resolve()), source_sha256=sha256(__file__),
                          source_artifacts={name:dict(path=str((source/name).resolve()), sha256=sha256(source/name))
                                            for name in sources}), out/'path_filter_manifest.json')
    manifest = json.loads((out/'prediction_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise ValueError('Cannot bind incomplete filtered predictions')
    for name in PATH_ARTIFACTS+('experiment_protocol.json', 'path_filter_manifest.json'):
        manifest['artifacts'][name] = sha256(out/name)
    publish_manifest(manifest, out/'prediction_manifest.json')


def validate_path_filter(out):
    out = Path(out)
    result = validate_path_predictions(out)
    meta = json.loads((out/'path_filter_manifest.json').read_text(encoding='utf-8'))
    if (meta.get('status') != 'complete' or meta.get('mode') not in MODES
            or meta.get('declared_modes') != list(MODES) or meta.get('source_sha256') != sha256(__file__)):
        raise ValueError('Incomplete or changed path-filter protocol')
    for item in meta['source_artifacts'].values():
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Original path-filter input artifact changed')
    original = pd.read_pickle(meta['source_artifacts']['predictions.pkl']['path'])
    if 'eligible' in original or 'eligible' not in result or result.eligible.dtype != bool:
        raise ValueError('Path filtering must start from native predictions and add a boolean eligibility flag')
    try:
        pd.testing.assert_frame_equal(result.drop(columns='eligible'), original, check_exact=True)
    except AssertionError as exc:
        raise ValueError('Path filtering changed native forecasts, order or coverage') from exc
    if meta['mode'] == 'liquid':
        checked = validate_liquidity_inputs(out)
        band = checked['band_manifest']
    else:
        with leadership_variant():
            band = validate_band_inputs(out)
        pool = original[original.utility.gt(0) & original.ts_code.map(_mainboard)]
        pool = pool.sort_values(['signal_date', 'utility', 'ts_code'],
                                ascending=[True, False, True], kind='stable').groupby('signal_date').head(40)
        recorded = pd.read_csv(out/'prior_band_checks.csv', parse_dates=['signal_date'])
        expected_keys = set(zip(pool.signal_date, pool.ts_code))
        recorded_keys = set(zip(recorded.signal_date, recorded.ts_code))
        if recorded_keys != expected_keys or len(recorded) != len(pool):
            raise ValueError('Prior bands do not cover exactly the declared native top-40 pool')
    rows = pd.read_csv(out/'prior_band_checks.csv', parse_dates=['signal_date'])
    accepted = set(zip(rows.loc[rows.eligible, 'signal_date'], rows.loc[rows.eligible, 'ts_code']))
    expected_flags = np.asarray([(day, code) in accepted for day, code in zip(result.signal_date, result.ts_code)])
    if not np.array_equal(result.eligible.to_numpy(), expected_flags):
        raise ValueError('Path eligibility differs from exact prior-band acceptance')
    return dict(predictions=result, filter_manifest=meta, band_manifest=band)


def filter_path_predictions(source, out, mode='band', *, offline=False, pro=None):
    source, out = Path(source), Path(out)
    if mode not in MODES or source.resolve() == out.resolve() or (out/'prediction_manifest.json').exists():
        raise ValueError('Use a separate unfinished band/liquid output')
    original = validate_path_predictions(source)
    if 'eligible' in original:
        raise ValueError('Path ablations must both start from the same unfiltered predictions')
    gateway = None if offline else (pro if pro is not None else ExactBatchGateway(os.environ['TUSHARE_API_KEY']))
    if mode == 'band':
        with leadership_variant():
            result = filter_buy_risk_band(source, out, gateway, cache_roots=cache_roots())
    else:
        result = filter_liquid_predictions(source, out, gateway, offline=offline, cache_roots=cache_roots())
    try:
        pd.testing.assert_frame_equal(result.drop(columns='eligible'), original, check_exact=True)
    except AssertionError as exc:
        raise ValueError('Filter changed the frozen native path predictions') from exc
    # The old band filter emits an empty headerless CSV when every utility is
    # nonpositive. Preserve a valid empty observation schema in that case.
    band_path = out/'prior_band_checks.csv'
    band_meta_path = out/'prior_band_manifest.json'
    band_meta = json.loads(band_meta_path.read_text(encoding='utf-8'))
    if mode == 'band' and band_meta['checks'] == 0:
        pd.DataFrame(columns=['signal_date', 'ts_code', 'up_limit', 'down_limit',
                              'official_band_fraction', 'eligible']).to_csv(band_path, index=False)
        band_meta['checks_sha256'] = sha256(band_path)
        publish_manifest(band_meta, band_meta_path)
        manifest_path = out/'prediction_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for name in ('prior_band_checks.csv', 'prior_band_manifest.json'):
            manifest['artifacts'][name] = sha256(out/name)
        publish_manifest(manifest, manifest_path)
    bind_path_filter_artifacts(source, out, mode)
    checked = validate_path_filter(out)
    print(json.dumps(dict(event='path_filter_verified', mode=mode,
                          accepted=int(checked['predictions'].eligible.sum()))), flush=True)
    return checked['predictions']


def run_path_account(source, out, inputs, *, offline=False):
    source, out, inputs = Path(source), Path(out), Path(inputs)
    previous_status = out/'account/daily_run_status.json'
    completed = (previous_status.exists()
                 and json.loads(previous_status.read_text(encoding='utf-8')).get('account_complete') is True)
    if source.resolve() == out.resolve() or completed:
        raise ValueError('Use a separate output without a completed path account')
    checked = validate_path_filter(source)
    out.mkdir(parents=True, exist_ok=True)
    for file in source.iterdir():
        if file.is_file() and file.name != 'predictions.csv':
            shutil.copy2(file, out/file.name)
    shutil.copytree(source/'checkpoints', out/'checkpoints', dirs_exist_ok=True)
    daily, daily_meta = load_adjusted_daily(inputs)
    feature_meta = json.loads((source/'feature_manifest.json').read_text(encoding='utf-8'))
    if daily_meta['adjusted_daily']['sha256'] != feature_meta['raw_adjusted_daily']['sha256']:
        raise ValueError('Account quotes differ from path model and target raw inputs')
    account = out/'account'
    account.mkdir(exist_ok=True)
    seed_all_actions(account)
    gateway = None if offline else ExactBatchGateway(os.environ['TUSHARE_API_KEY'])
    limits = OfficialLimits(account, pro=gateway, offline=offline, cache_roots=cache_roots())
    policy = ExecutionPolicy(hard_stop=.08, trailing_activation=.20, trailing_distance=.08)
    result = run_account(checked['predictions'], daily, sorted(daily.date.unique()), account,
                         pro=gateway, offline=offline, policy=policy, limit_provider=limits)
    status_path = account/'daily_run_status.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    status['source_sha256'].update(checked['band_manifest']['source_sha256'])
    sidecars = PATH_ARTIFACTS+('path_filter_manifest.json', 'prior_band_manifest.json', 'prior_band_checks.csv',
                              'experiment_protocol.json', 'prediction_manifest.json', 'predictions.pkl',
                              'feature_manifest.json', 'fit_audits.json')
    if checked['filter_manifest']['mode'] == 'liquid':
        sidecars += ('liquidity_manifest.json', 'liquidity_checks.csv')
    for name in sidecars:
        status['source_sha256'][str((out/name).resolve())] = sha256(out/name)
    for file in (out/'checkpoints').glob('*.pt'):
        status['source_sha256'][str(file.resolve())] = sha256(file)
    for name in (Path(__file__).name, 'research_daily_opportunity_path.py',
                 'research_daily_opportunity_path_targets.py', 'research_daily_opportunity_liquid_runner.py',
                 'research_daily_opportunity_band_filter.py', 'research_daily_opportunity_liquid_filter.py'):
        file = Path(__file__).parent/name
        status['source_sha256'][str(file.resolve())] = sha256(file)
    status['path_account_ablation'] = checked['filter_manifest']['mode']
    status['path_supervision_execution_scope'] = 'Reference labels omit locked-limit execution; this account applies official limits and pending orders'
    publish_manifest(status, status_path)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('filter', 'account', 'validate'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--mode', choices=MODES, default='band')
    parser.add_argument('--inputs', type=Path, default=Path('data/raw/daily_opportunity_20261002'))
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.action != 'validate' and args.out is None:
        parser.error('--out is required for filtering and account execution')
    if args.action == 'filter':
        filter_path_predictions(args.source, args.out, args.mode, offline=args.offline)
    elif args.action == 'account':
        run_path_account(args.source, args.out, args.inputs, offline=args.offline)
    else:
        checked = validate_path_filter(args.source)
        print('verified path filter', checked['filter_manifest']['mode'], len(checked['predictions']))
