"""Reproducible daily stock-opportunity experiment, separate from old accounts.

Prepare actual adjusted inputs, fit quarterly-frozen checkpoints using daily
past-only targets, predict every session, then run the raw-price cash ledger.
None of the 2026 account results determines hyperparameters or a checkpoint.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

from research_daily_graph_data import (load_adjusted_daily, load_graph_inputs,
                                       publish_frame, publish_manifest, sha256, validate_edges)
from research_daily_opportunity_features import (
    FEATURE_PROTOCOL, LazyDailyOpportunityBatch, build_feature_store, expected_utility,
)
from research_daily_opportunity_model import FitConfig, FittedOpportunityModel, fit_walk_forward, PROTOCOL


REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = REPO/'data/raw/daily_opportunity_20261002'
DEFAULT_OUT = REPO/'results/daily_opportunity_20261002'


def _progress(event):
    print(json.dumps(event, ensure_ascii=False, default=str), flush=True)


def prepare(inputs, out, *, use_graph=True, allow_partial_graph=False):
    inputs, out = Path(inputs), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    daily, adjustment_meta = load_adjusted_daily(inputs)
    daily_path = inputs/adjustment_meta['adjusted_daily']['file']
    if use_graph:
        edges, graph_meta = load_graph_inputs(inputs, allow_partial=allow_partial_graph)
        if edges.empty:
            raise ValueError('A verified dated graph is required; do not silently substitute the old top-five graph')
    else:
        edges = pd.DataFrame(columns=['snapshot_date', 'theme_code', 'ts_code'])
        edges['snapshot_date'] = pd.to_datetime(edges.snapshot_date)
        graph_meta = dict(status='disabled', reason='Explicit no-graph ablation; no collection result enters this model')
    frozen_graph = out/'frozen_edges.pkl'
    graph_artifact = publish_frame(edges, frozen_graph)
    # The quote panel supplies the exchange-wide calendar, independently of
    # the eligibility of any one stock. Collector coverage is separately audited.
    store = build_feature_store(daily)
    target = out/'feature_store.pkl'
    temporary = target.with_suffix('.staging.pkl')
    with temporary.open('wb') as handle:
        pickle.dump(store, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(target)
    manifest = dict(feature_protocol=FEATURE_PROTOCOL,
                    raw_adjusted_daily=dict(path=str(daily_path), sha256=sha256(daily_path)),
                    adjustment_coverage=adjustment_meta,
                    edges=dict(path=str(frozen_graph), sha256=graph_artifact['sha256']),
                    use_graph=use_graph,
                    graph_coverage=graph_meta, feature_store_sha256=sha256(target),
                    first_date=str(store.dates.min().date()), last_date=str(store.dates.max().date()),
                    market_sessions=len(store.dates), all_board_stocks=len(store.stock_codes),
                    feature_valid_observations=int(store.valid_features.sum()),
                    graph_freshness='Latest verified dated snapshot only, carried forward; no current backfill',
                    graph_gaps='Unknown edges do not exclude stocks; collection failures are retained in graph coverage',
                    feature_builder_sha256=sha256(Path(__file__).parent/'research_daily_opportunity_features.py'))
    publish_manifest(manifest, out/'feature_manifest.json')
    _progress(dict(event='features_prepared', sessions=len(store.dates), stocks=len(store.stock_codes),
                   valid_rows=int(store.valid_features.sum())))
    return store, edges


def validate_prepared(out, *, use_graph=None):
    out = Path(out)
    metadata = json.loads((out/'feature_manifest.json').read_text(encoding='utf-8'))
    if metadata.get('feature_protocol') != FEATURE_PROTOCOL:
        raise ValueError('Prepared feature protocol mismatch')
    if use_graph is not None and metadata.get('use_graph') != use_graph:
        raise ValueError('Prepared graph/ablation mode differs from requested training')
    if metadata.get('feature_builder_sha256') != sha256(Path(__file__).parent/'research_daily_opportunity_features.py'):
        raise ValueError('Feature builder changed after preparation')
    if sha256(out/'feature_store.pkl') != metadata.get('feature_store_sha256'):
        raise ValueError('Prepared feature-store fingerprint mismatch')
    for name in ('raw_adjusted_daily', 'edges'):
        artifact = metadata.get(name, {})
        if sha256(Path(artifact['path'])) != artifact.get('sha256'):
            raise ValueError('Prepared '+name+' fingerprint mismatch')
    with (out/'feature_store.pkl').open('rb') as handle:
        store = pickle.load(handle)
    edges = validate_edges(pd.read_pickle(metadata['edges']['path']))
    return store, edges, metadata


def validate_predictions(out):
    out = Path(out)
    manifest = json.loads((out/'prediction_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise ValueError('Daily predictions incomplete')
    for file, expected in manifest.get('artifacts', {}).items():
        if sha256(out/file) != expected:
            raise ValueError('Prediction artifact fingerprint mismatch: '+file)
    required = {'predictions.pkl', 'fit_audits.json', 'experiment_protocol.json', 'feature_manifest.json'}
    if not required <= set(manifest.get('artifacts', {})):
        raise ValueError('Missing required prediction provenance')
    protocol = json.loads((out/'experiment_protocol.json').read_text(encoding='utf-8'))
    for name, expected in protocol['source_sha256'].items():
        if sha256(Path(__file__).parent/name) != expected:
            raise ValueError('Prediction source changed after fitting: '+name)
    validate_prepared(out, use_graph=protocol['use_graph'])
    audits = json.loads((out/'fit_audits.json').read_text(encoding='utf-8'))
    for quarter, audit in audits.items():
        if sha256(out/'checkpoints'/f'{quarter}.pt') != audit['checkpoint_sha256']:
            raise ValueError('Checkpoint fingerprint mismatch')
        if not pd.Timestamp(audit['maximum_train_label_end']) < pd.Timestamp(audit['train_cutoff']) <= pd.Timestamp(audit['fit_cutoff']):
            raise ValueError('Unpurged checkpoint audit')
    predictions = pd.read_pickle(out/'predictions.pkl')
    if len(predictions) != manifest['rows'] or predictions.duplicated(['signal_date', 'ts_code']).any():
        raise ValueError('Daily prediction row coverage mismatch')
    if predictions.model_fit_cutoff.gt(predictions.signal_date).any():
        raise ValueError('Prediction uses a future checkpoint')
    return predictions


def quarterly_plan(store, start, end, sequence_length=20):
    """Quarter is determined by next execution, not by preceding signal month."""
    plan = {}
    for index in store.signal_indices(end=end, sequence_length=sequence_length):
        if index+1 >= len(store.dates):
            continue
        execution_date = store.dates[index+1]
        if execution_date < pd.Timestamp(start) or execution_date > pd.Timestamp(end):
            continue
        period = execution_date.to_period('Q')
        first_calendar_day = period.start_time.normalize()
        prior = store.dates[store.dates < first_calendar_day]
        if not len(prior):
            raise ValueError('No prior session for quarterly checkpoint')
        key = str(period)
        plan.setdefault(key, dict(fit_cutoff=prior[-1], indices=[]))['indices'].append(index)
    if not plan:
        raise ValueError('No complete daily prediction sequences in requested account period')
    return plan


def train_and_predict(store, edges, out, *, start='2026-01-01', end='2026-09-24',
                      train_start='2025-01-01', config=None, device='cuda', use_graph=True):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out/'checkpoints'; checkpoint_dir.mkdir(exist_ok=True)
    config = config or FitConfig(epochs=8, validation_sessions=20)
    all_indices = store.signal_indices(start=train_start, end=end, sequence_length=config.sequence_length)
    batches = [LazyDailyOpportunityBatch(store, index, edges, sequence_length=config.sequence_length,
                                        use_graph=use_graph) for index in all_indices]
    by_index = dict(zip(all_indices, batches))
    plan = quarterly_plan(store, start, end, config.sequence_length)
    frames, audits = [], {}
    for quarter, fold in plan.items():
        fit_cutoff = fold['fit_cutoff']
        past_batches = [batch for batch in batches if batch.signal_date < fit_cutoff]
        _progress(dict(event='quarter_start', quarter=quarter, fit_cutoff=str(fit_cutoff.date()),
                       past_sessions=len(past_batches), prediction_sessions=len(fold['indices'])))
        fitted = fit_walk_forward(past_batches, fit_cutoff, config=config, device=device, progress=_progress)
        checkpoint = checkpoint_dir/f'{quarter}.pt'
        fitted.save(checkpoint)
        # Predictions use the saved-and-reloaded checkpoint, not a transient
        # in-memory state. Fail if a checkpoint loses audit or calibration.
        fitted = FittedOpportunityModel.load(checkpoint, device=device)
        audits[quarter] = dict(**fitted.audit, checkpoint_sha256=sha256(checkpoint))
        for number, index in enumerate(fold['indices']):
            batch = by_index[index]
            prediction = fitted.predict(batch, device=device)
            frame = pd.DataFrame(dict(signal_date=batch.signal_date, ts_code=batch.stock_codes,
                                      utility=expected_utility(prediction['return_mu'], prediction['downside']),
                                      model_fit_cutoff=fit_cutoff,
                                      graph_snapshot_date=batch.membership_asof))
            for j, horizon in enumerate((5, 10, 20)):
                frame[f'mu{horizon}'] = prediction['return_mu'][:, j]
                frame[f'risk{horizon}'] = prediction['downside'][:, j]
                frame[f'rally_probability{horizon}'] = prediction['rally_probability'][:, j]
            for j, name in enumerate(('breakout', 'reversal', 'continuation', 'range')):
                frame[f'gate_{name}'] = prediction['gate'][:, j]
            frames.append(frame)
            if number % 20 == 0:
                _progress(dict(event='prediction_progress', quarter=quarter,
                               session=number+1, total=len(fold['indices']), stocks=len(frame)))
        publish_manifest(audits, out/'fit_audits.json')
        publish_frame(pd.concat(frames, ignore_index=True), out/'predictions.pkl')
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(out/'predictions.csv', index=False, encoding='utf-8-sig')
    protocol = dict(model=PROTOCOL, features=FEATURE_PROTOCOL, fit_configuration=asdict(config),
                    train_start=train_start, account_start=start, account_end=end,
                    use_graph=use_graph, graph_update='Dated snapshot schedule, not a newly observed daily graph',
                    score='(.2*mu5+.3*mu10+.5*mu20) - .5*(.2*risk5+.3*risk10+.5*risk20)',
                    checkpoint_fits='Quarterly, strictly purged before fit; daily inference and next-open execution',
                    evaluation_status='2026 is previously inspected research history, not a new untouched test set',
                    source_sha256={name: sha256(Path(__file__).parent/name) for name in (
                        'research_daily_opportunity_features.py', 'research_daily_opportunity_model.py',
                        'research_daily_graph_data.py', 'research_daily_opportunity.py')})
    publish_manifest(protocol, out/'experiment_protocol.json')
    artifacts = {name: sha256(out/name) for name in
                 ('predictions.pkl', 'fit_audits.json', 'experiment_protocol.json')}
    if (out/'feature_manifest.json').exists():
        artifacts['feature_manifest.json'] = sha256(out/'feature_manifest.json')
    publish_manifest(dict(status='complete', rows=len(predictions),
                          sessions=int(predictions.signal_date.nunique()), artifacts=artifacts),
                     out/'prediction_manifest.json')
    _progress(dict(event='predictions_complete', rows=len(predictions), sessions=predictions.signal_date.nunique()))
    return predictions


def run_daily_account(predictions, inputs, out, *, start='2026-01-01', end='2026-09-24', offline=False):
    from data_pipeline.tushare_config import get_pro
    from research_daily_opportunity_execution import run_account
    inputs, out = Path(inputs), Path(out)
    daily, input_manifest = load_adjusted_daily(inputs)
    prepared = json.loads((out/'feature_manifest.json').read_text(encoding='utf-8'))
    if input_manifest['adjusted_daily']['sha256'] != prepared['raw_adjusted_daily']['sha256']:
        raise ValueError('Account raw-price source differs from fitted feature source')
    sessions = pd.DatetimeIndex(sorted(daily.date.unique()))
    base_caches = [REPO/'results/dc_member_relative_leader_2026_research',
                   REPO/'results/dc_member_relative_moderate_2026_research',
                   REPO/'results/dc_structure_replay_20261002']
    caches = [folder/name for folder in base_caches for name in ('exit_quotes', 'replacement_limits')]
    return run_account(predictions, daily, sessions, out/'account',
                       pro=None if offline else get_pro(), offline=offline,
                       start=start, end=end, cache_roots=caches)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'train', 'account', 'all'))
    parser.add_argument('--inputs', type=Path, default=DEFAULT_INPUTS)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--start', default='2026-01-01')
    parser.add_argument('--end', default='2026-09-24')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-graph', action='store_true')
    parser.add_argument('--allow-partial-graph', action='store_true')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args(argv)
    if args.action in ('prepare', 'all'):
        store, edges = prepare(args.inputs, args.out, use_graph=not args.no_graph,
                               allow_partial_graph=args.allow_partial_graph)
    if args.action in ('train', 'all'):
        if args.action == 'train':
            store, edges, _ = validate_prepared(args.out, use_graph=not args.no_graph)
        predictions = train_and_predict(store, edges, args.out, start=args.start, end=args.end,
                                        config=FitConfig(epochs=args.epochs, validation_sessions=20),
                                        device=args.device, use_graph=not args.no_graph)
    if args.action in ('account', 'all'):
        predictions = validate_predictions(args.out)
        run_daily_account(predictions, args.inputs, args.out, start=args.start, end=args.end, offline=args.offline)


if __name__ == '__main__':
    main()
