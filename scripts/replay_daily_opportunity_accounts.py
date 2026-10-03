"""Exact, offline-only daily-account replays into separate output directories.

Original predictions, accounts and raw caches are read-only. The copied raw
corporate-action history comes exclusively from the corresponding source
account. Official limits reuse fingerprinted original vendor observations.
Economic CSV records are compared exactly after canonical row sorting because
the original engine emits contribution rows from unordered Python sets.
"""
import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))

from research_daily_graph_data import load_adjusted_daily, publish_manifest, sha256
from research_daily_opportunity import validate_predictions
from research_daily_opportunity_execution import ExecutionPolicy, OfficialLimits, run_account
from research_scoped_actions import validate_history


DEFAULT_SOURCES = (
    PROJECT/'results/daily_opportunity_trees_20261002',
    PROJECT/'results/daily_opportunity_ranked_band_trail8_20261002',
)
DEFAULT_OUT = PROJECT/'results/daily_opportunity_offline_replays_20261002'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@contextmanager
def prediction_context(source):
    protocol = read_json(Path(source)/'experiment_protocol.json')
    version = protocol.get('features', {}).get('version', '')
    if version.startswith('daily_opportunity_leadership_features'):
        from research_daily_opportunity_leadership import leadership_variant
        with leadership_variant():
            yield
    else:
        yield


class RecordedOfflineLimits(OfficialLimits):
    """Resolve fingerprinted vendor rows even when legacy filenames differ."""
    def __init__(self, out, recorded_paths, cache_roots):
        super().__init__(out, pro=None, offline=True, cache_roots=cache_roots)
        self.recorded = {}
        for path in recorded_paths:
            path = Path(path)
            raw = pd.read_pickle(path)
            if not isinstance(raw, pd.DataFrame) or not {'ts_code','trade_date','up_limit','down_limit'} <= set(raw):
                continue
            dates = pd.to_datetime(raw.trade_date.astype(str), format='%Y%m%d', errors='coerce')
            if dates.isna().any():
                raise ValueError('Malformed recorded official-limit dates: '+path.name)
            for day in dates.unique():
                frame = raw.loc[dates.eq(day)].copy()
                self.validate_batch(frame, pd.Timestamp(day))
                self.recorded.setdefault(pd.Timestamp(day), []).append((path, frame))

    def __call__(self, day, code):
        day = pd.Timestamp(day)
        key = (day, code)
        if key in self.frames:
            return self.frames[key]
        candidates = self.recorded.get(day, ())
        found = []
        for path, frame in candidates:
            if frame.ts_code.eq(code).any():
                # A known zero / invalid IPO band must not become a fabricated
                # executable band through fallback to some other observation.
                row = self.validate(frame, day, code)
                found.append((path, row))
        if found:
            values = {(float(row.down_limit), float(row.up_limit)) for _,row in found}
            if len(values) != 1:
                raise ValueError('Conflicting recorded official-limit observations')
            path, row = found[0]
            self.frames[key] = row
            self.sources[str(path.resolve())] = sha256(path)
            return row
        return super().__call__(day, code)


def copy_own_actions(source_account, target_account, start, end):
    source_folder = Path(source_account)/'scoped_actions'
    target_folder = Path(target_account)/'scoped_actions'
    target_folder.mkdir(parents=True, exist_ok=False)
    copied = {}
    for raw_path in sorted(source_folder.glob('*.pkl')):
        metadata_path = raw_path.with_suffix('.json')
        metadata = read_json(metadata_path)
        if (metadata.get('start'), metadata.get('end'), metadata.get('raw_sha256')) != (start, end, sha256(raw_path)):
            raise ValueError('Invalid original scoped action artifact: '+raw_path.name)
        validate_history(pd.read_pickle(raw_path), raw_path.stem, start, end)
        for path in (raw_path, metadata_path):
            destination = target_folder/path.name
            shutil.copy2(path, destination)
            if sha256(destination) != sha256(path):
                raise AssertionError('Copied action fingerprint differs')
            copied[str(path.resolve())] = sha256(path)
    return copied


def compare_csv(source, replay):
    original = pd.read_csv(source)
    rerun = pd.read_csv(replay)
    if list(original.columns) != list(rerun.columns):
        raise AssertionError('Replay CSV schema differs: '+Path(source).name)
    columns = list(original.columns)
    canonical = lambda frame: frame.sort_values(columns, na_position='last', kind='stable').reset_index(drop=True)
    a, b = canonical(original), canonical(rerun)
    pd.testing.assert_frame_equal(a, b, check_exact=True, check_dtype=True)
    # A canonical serialization hash describes the exact compared records,
    # separate from byte hashes whose row order can be nondeterministic.
    import hashlib
    canonical_sha = hashlib.sha256(a.to_csv(index=False).encode('utf-8')).hexdigest()
    return dict(rows=len(a), columns=columns, exact_cells_equal=True,
                canonical_sha256=canonical_sha,
                source_sha256=sha256(source), replay_sha256=sha256(replay),
                byte_identical=sha256(source) == sha256(replay))


def load_auditor():
    path = PROJECT/'scripts/audit_daily_opportunity_account.py'
    spec = importlib.util.spec_from_file_location('daily_opportunity_independent_replay_auditor', path)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    return auditor


def replay_one(source, out, daily_cache=None):
    source, out = Path(source).resolve(), Path(out).resolve()
    source_account = source/'account'
    target_root = out/source.name
    target_account = target_root/'account'
    if target_root.exists() or source == target_root or source_account == target_account:
        raise ValueError('Replay target must be a new, separate directory: '+str(target_root))
    status_path = source_account/'daily_run_status.json'
    status = read_json(status_path)
    if status.get('account_complete') is not True:
        raise ValueError('Cannot replay incomplete original account')
    original_sources = status.get('source_sha256', {})
    for name, fingerprint in original_sources.items():
        if not Path(name).is_file() or sha256(name) != fingerprint:
            raise ValueError('Original account source fingerprint changed: '+Path(name).name)
    with prediction_context(source):
        predictions = validate_predictions(source)
    feature_manifest = read_json(source/'feature_manifest.json')
    raw_path = Path(feature_manifest['raw_adjusted_daily']['path'])
    raw_path = raw_path if raw_path.is_absolute() else (PROJECT/raw_path)
    raw_path = raw_path.resolve()
    if sha256(raw_path) != feature_manifest['raw_adjusted_daily']['sha256']:
        raise ValueError('Fitted raw daily fingerprint mismatch')
    if daily_cache is not None and raw_path in daily_cache:
        daily, adjustment_manifest = daily_cache[raw_path]
    else:
        daily, adjustment_manifest = load_adjusted_daily(raw_path.parent)
        if daily_cache is not None:
            daily_cache[raw_path] = (daily, adjustment_manifest)
    if adjustment_manifest['adjusted_daily']['sha256'] != feature_manifest['raw_adjusted_daily']['sha256']:
        raise ValueError('Adjustment coverage differs from fitted raw source')
    protocol_path = source_account/'execution_protocol.json'
    execution_protocol = read_json(protocol_path)
    if execution_protocol.get('fees') != 0 or execution_protocol.get('external_topups') is not False:
        raise ValueError('Unsupported financing/fee protocol')
    policy = ExecutionPolicy(**execution_protocol['policy'])
    band = None
    if (source/'prior_band_manifest.json').exists():
        from research_daily_opportunity_band_filter import validate_band_inputs
        band = validate_band_inputs(source)
    target_account.mkdir(parents=True, exist_ok=False)
    copied_actions = copy_own_actions(source_account, target_account, status['start'], status['end'])
    recorded_paths = [Path(name) for name in original_sources
                      if Path(name).suffix == '.pkl' and Path(name).parent.name != 'scoped_actions']
    cache_roots = list(dict.fromkeys([source_account/'daily_limits', *[path.parent for path in recorded_paths]]))
    limits = RecordedOfflineLimits(target_account, recorded_paths, cache_roots)
    sessions = pd.DatetimeIndex(sorted(daily.date.unique()))
    run_account(predictions, daily, sessions, target_account, pro=None, offline=True,
                start=status['start'], end=status['end'], policy=policy, limit_provider=limits)
    # Bind replay status to its actual frozen prediction/raw/policy inputs as
    # well as the execution engine's existing frame-level fingerprints.
    replay_status_path = target_account/'daily_run_status.json'
    replay_status = read_json(replay_status_path)
    binding_files = [source/'predictions.pkl', source/'prediction_manifest.json',
                     source/'feature_manifest.json', source/'experiment_protocol.json',
                     status_path, protocol_path, raw_path, Path(__file__).resolve()]
    for path in binding_files:
        replay_status['source_sha256'][str(path.resolve())] = sha256(path)
    if band:
        replay_status['source_sha256'].update(band['source_sha256'])
        for name in ('prior_band_manifest.json', 'prior_band_checks.csv'):
            replay_status['source_sha256'][str((source/name).resolve())] = sha256(source/name)
    publish_manifest(replay_status, replay_status_path)
    auditor = load_auditor()
    independent = auditor.audit_account(target_account, daily, predictions, sessions,
                                       output_dir=target_account/'independent_audit')
    source_csvs = {path.name:path for path in source_account.glob('*.csv')}
    replay_csvs = {path.name:path for path in target_account.glob('*.csv')}
    if set(source_csvs) != set(replay_csvs):
        raise AssertionError('Replay exported CSV set differs from original')
    comparisons = {name:compare_csv(path, replay_csvs[name]) for name,path in sorted(source_csvs.items())}
    original_metrics = read_json(source_account/'metrics.json')
    replay_metrics = read_json(target_account/'metrics.json')
    if original_metrics != replay_metrics:
        raise AssertionError('Replay metrics are not exact')
    if execution_protocol != read_json(target_account/'execution_protocol.json'):
        raise AssertionError('Replay execution protocol is not exact')
    result = dict(passed=True, offline=True, network_client=None,
                  source=str(source), replay_account=str(target_account),
                  policy=execution_protocol['policy'], exact_metrics=replay_metrics,
                  exact_csv_cells=True, row_order='Canonical all-column sorting; byte equality reported separately',
                  csv_comparisons=comparisons, copied_own_action_source_sha256=copied_actions,
                  replay_status_sha256=sha256(replay_status_path),
                  independent_audit_sha256=sha256(target_account/'independent_audit/independent_checks.json'),
                  independent_book_passed=independent['passed'],
                  input_sha256={str(path.resolve()):sha256(path) for path in binding_files})
    publish_manifest(result, target_root/'replay_checks.json')
    print(json.dumps(dict(event='offline_replay_complete', source=source.name,
                         csvs=len(comparisons), independent_book_passed=True,
                         total_profit=replay_metrics['total_profit']), ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', type=Path,
                        help='Model-output folder containing the original account; may be repeated')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    sources = args.source or DEFAULT_SOURCES
    args.out.mkdir(parents=True, exist_ok=True)
    cache, results, failures = {}, [], []
    for source in sources:
        try:
            results.append(replay_one(source, args.out, cache))
        except Exception as exc:
            failures.append(dict(source=str(source.resolve()), error_type=type(exc).__name__, message=str(exc)))
            print(json.dumps(dict(event='offline_replay_failed', source=source.name,
                                 error_type=type(exc).__name__, message=str(exc)), ensure_ascii=False), flush=True)
    summary = dict(passed=not failures and len(results)==len(sources), offline=True,
                   requested_accounts=len(sources), completed_accounts=len(results),
                   replays=results, failures=failures, script_sha256=sha256(__file__))
    publish_manifest(summary, args.out/'replay_summary.json')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
