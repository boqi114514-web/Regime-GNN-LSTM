"""One fixed path-payoff tree experiment with separately bound target artifacts.

V3 causal factors are retained unchanged. This branch predicts native reference
exit payoff, held-period adverse excursion, and three-class first passage.
First-passage probabilities are recorded but do not enter the utility score.
"""
import argparse
import copy
import json
from pathlib import Path
import pickle
import shutil

import lightgbm as lgb
import numpy as np
import pandas as pd

import research_daily_opportunity_features as features
import research_daily_opportunity_leadership as leadership
import research_daily_opportunity_path_targets as path_labels
from research_daily_graph_data import publish_frame, publish_manifest, sha256
from research_daily_opportunity import quarterly_plan, validate_predictions
from research_daily_opportunity_features import expected_utility
from research_daily_opportunity_trees import PARAMS


FIXED_CONFIGURATION = dict(horizons=[5, 10, 20], upper_thresholds=[.10, .15, .30],
                           hard_stop=.08, trailing_activation=.20, trailing_distance=.08)
CLASS_NAMES = ('neither', 'upper', 'lower')


def publish_path_targets(out, targets, raw_artifact):
    out = Path(out)
    if targets.protocol.get('configuration') != FIXED_CONFIGURATION:
        raise ValueError('Only the declared fixed path configuration is allowed')
    if sha256(raw_artifact['path']) != raw_artifact['sha256']:
        raise ValueError('Path-target raw source changed')
    with (out/'path_targets.pkl').open('wb') as handle:
        pickle.dump(targets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta = dict(status='complete', protocol=targets.protocol, raw_adjusted_daily=raw_artifact,
                dates=len(targets.dates), stocks=len(targets.stock_codes),
                valid_labels_by_horizon=targets.valid.sum(axis=(0, 1)).tolist(),
                path_builder_sha256=sha256(path_labels.__file__),
                artifacts={name:sha256(out/name) for name in
                           ('path_targets.pkl', 'feature_store.pkl', 'feature_manifest.json')})
    publish_manifest(meta, out/'path_targets_manifest.json')


def validate_target_artifact(out, store=None):
    """Bind targets to the immutable original feature inputs, not to a relabelled store."""
    out = Path(out)
    meta = json.loads((out/'path_targets_manifest.json').read_text(encoding='utf-8'))
    if meta.get('status') != 'complete' or meta.get('path_builder_sha256') != sha256(path_labels.__file__):
        raise ValueError('Incomplete targets or changed path-label source')
    required = {'path_targets.pkl', 'feature_store.pkl', 'feature_manifest.json'}
    if set(meta.get('artifacts', {})) != required:
        raise ValueError('Incomplete path-target artifact binding')
    for name, expected in meta['artifacts'].items():
        if sha256(out/name) != expected:
            raise ValueError('Changed path-target artifact: '+name)
    feature_meta = json.loads((out/'feature_manifest.json').read_text(encoding='utf-8'))
    raw = meta['raw_adjusted_daily']
    if raw != feature_meta['raw_adjusted_daily'] or sha256(raw['path']) != raw['sha256']:
        raise ValueError('Path targets and feature inputs have different raw provenance')
    with (out/'path_targets.pkl').open('rb') as handle:
        targets = pickle.load(handle)
    expected_protocol = dict(path_labels.PATH_PROTOCOL, configuration=FIXED_CONFIGURATION)
    if (not isinstance(targets, path_labels.PathTargets) or targets.protocol != expected_protocol
            or meta['protocol'] != expected_protocol or targets.horizons != (5, 10, 20)):
        raise ValueError('Path target semantics/configuration mismatch')
    if store is not None and (not targets.dates.equals(store.dates) or targets.stock_codes != store.stock_codes):
        raise ValueError('Path target date/stock axes differ from model features')
    shape = (len(targets.dates), len(targets.stock_codes), 3)
    if (meta['dates'] != shape[0] or meta['stocks'] != shape[1]
            or targets.label_end_dates.shape != (shape[0], 3)
            or targets.quote_presence.shape != shape[:2]):
        raise ValueError('Path target calendar/stock coverage mismatch')
    names = ('path_payoff', 'max_adverse_excursion', 'first_passage', 'first_passage_index',
             'exit_reason', 'trigger_index', 'exit_index', 'valid')
    if any(getattr(targets, name).shape != shape for name in names):
        raise ValueError('Path target array dimensions mismatch')
    if targets.valid.dtype != bool or targets.quote_presence.dtype != bool:
        raise ValueError('Path target observation masks must be boolean')
    if (not np.array_equal(np.isfinite(targets.path_payoff), targets.valid)
            or not np.array_equal(np.isfinite(targets.max_adverse_excursion), targets.valid)
            or (targets.max_adverse_excursion[targets.valid] < 0).any()):
        raise ValueError('Invalid path payoff/downside missingness')
    for name in ('first_passage', 'exit_reason'):
        values = getattr(targets, name)
        if not np.isin(values[targets.valid], (0, 1, 2)).all() or (values[~targets.valid] != -1).any():
            raise ValueError('Invalid path event/reason categories')
    for j, horizon in enumerate(targets.horizons):
        count = max(0, len(targets.dates)-horizon-1)
        expected_end = np.full(len(targets.dates), np.datetime64('NaT', 'D'), dtype='datetime64[D]')
        expected_end[:count] = targets.dates[horizon+1:].to_numpy(dtype='datetime64[D]')
        if not np.array_equal(targets.label_end_dates[:, j].astype('int64'), expected_end.astype('int64')):
            raise ValueError('Path label maturity must be horizon plus one market session')
        complete = np.zeros(shape[:2], dtype=bool)
        if count:
            complete[:count] = True
            for offset in range(1, horizon+2):
                complete[:count] &= targets.quote_presence[offset:offset+count]
        if not np.array_equal(complete, targets.valid[..., j]):
            raise ValueError('Path labels do not require the entire actual future quote window')
        base = np.arange(shape[0])[:, None]
        exits = targets.exit_index[..., j]
        if ((complete & ((exits < base+2) | (exits > base+horizon+1))).any()
                or (exits[~complete] != -1).any()):
            raise ValueError('Path exit indices are not next-open/horizon compatible')
    if meta['valid_labels_by_horizon'] != targets.valid.sum(axis=(0, 1)).tolist():
        raise ValueError('Path target label coverage changed')
    return targets


def path_quarter_split(targets, indices, cutoff):
    cutoff = pd.Timestamp(cutoff)
    completed = [i for i in indices if targets.dates[i] < cutoff
                 and not np.isnat(targets.label_end_dates[i]).any()
                 and (targets.label_end_dates[i] < np.datetime64(cutoff, 'D')).all()]
    if len(completed) < 60:
        raise ValueError('Insufficient completed path-label days')
    validation = completed[-20:]
    train_cutoff = targets.dates[validation[0]]
    training = [i for i in completed
                if (targets.label_end_dates[i] < np.datetime64(train_cutoff, 'D')).all()]
    if len(training) < 20:
        raise ValueError('Insufficient purged path-training days')
    return training, validation, train_cutoff


def path_training_arrays(days, factors, store, targets):
    xs, returns, risks, events = [], [], [], []
    for i in days:
        codes, x, _old_returns, _old_risks, _snapshot = factors[i]
        columns = np.flatnonzero(store.valid_features[i])
        if codes != tuple(store.stock_codes[j] for j in columns) or len(x) != len(columns):
            raise ValueError('Path supervision rows must match original full feature-valid graph order')
        good = targets.valid[i, columns].all(axis=1)
        xs.append(x[good])
        returns.append(targets.path_payoff[i, columns][good])
        risks.append(targets.max_adverse_excursion[i, columns][good])
        events.append(targets.first_passage[i, columns][good])
    arrays = tuple(np.concatenate(parts) for parts in (xs, returns, risks, events))
    if not len(arrays[0]) or not all(np.isfinite(value).all() for value in arrays):
        raise ValueError('Empty/nonfinite completed path supervision')
    return arrays


def fit_path_heads(train_arrays, validation_arrays, quarter):
    """Fixed 120-round, past-validation-stopped native regressors and 3-class heads."""
    x, y, risk, event = train_arrays
    vx, vy, vrisk, vevent = validation_arrays
    heads, iterations = {}, {}
    tasks = [('return', y, vy, PARAMS), ('risk', risk, vrisk, PARAMS),
             ('passage', event, vevent, dict(PARAMS, objective='multiclass',
                                            metric='multi_logloss', num_class=3))]
    for kind, target, validation, params in tasks:
        for j, horizon in enumerate(path_labels.HORIZONS):
            name = f'{kind}{horizon}'
            training = lgb.Dataset(x, label=np.ascontiguousarray(target[:, j]), free_raw_data=False)
            held_out = lgb.Dataset(vx, label=np.ascontiguousarray(validation[:, j]), reference=training, free_raw_data=False)
            fitted = lgb.train(params, training, num_boost_round=120, valid_sets=[held_out],
                               callbacks=[lgb.early_stopping(12, verbose=False)])
            heads[name], iterations[name] = fitted, fitted.best_iteration
            print(json.dumps(dict(event='path_head_complete', quarter=quarter, head=name,
                                  iterations=fitted.best_iteration)), flush=True)
    return heads, iterations


def predict_path_heads(heads, x):
    mu = np.column_stack([heads[f'return{h}'].predict(x) for h in path_labels.HORIZONS])
    risk = np.maximum(0., np.column_stack([heads[f'risk{h}'].predict(x) for h in path_labels.HORIZONS]))
    event = np.stack([heads[f'passage{h}'].predict(x) for h in path_labels.HORIZONS], axis=1)
    if (event.shape != (len(x), 3, 3) or not np.isfinite(event).all()
            or (event < 0).any() or (event > 1).any()
            or not np.allclose(event.sum(axis=-1), 1., rtol=1e-6, atol=1e-6)):
        raise ValueError('Invalid first-passage three-class probabilities')
    return mu, risk, event


def validate_path_predictions(out):
    out = Path(out)
    with leadership.leadership_variant():
        store, _edges = leadership.validate_leadership(out)
        targets = validate_target_artifact(out, store)
        predictions = validate_predictions(out)
    protocol = json.loads((out/'experiment_protocol.json').read_text(encoding='utf-8'))
    if protocol.get('path_target_protocol') != targets.protocol:
        raise ValueError('Predictions and path targets use different semantics')
    audits = json.loads((out/'fit_audits.json').read_text(encoding='utf-8'))
    positions = {str(day.date()): i for i, day in enumerate(targets.dates)}
    for audit in audits.values():
        for kind, boundary in (('train', audit['train_cutoff']), ('validation', audit['fit_cutoff'])):
            rows = [positions[day] for day in audit[f'{kind}_signal_dates']]
            ends = targets.label_end_dates[rows]
            if np.isnat(ends).any() or not (ends < np.datetime64(boundary, 'D')).all():
                raise ValueError('Path labels have not matured before the '+kind+' cutoff')
            if str(ends.max()) != audit[f'maximum_{kind}_label_end']:
                raise ValueError('Path target maturity differs from recorded fit audit')
    return predictions


def run_path_trees(source, out, start='2026-01-01', end='2026-09-24'):
    source, out = Path(source), Path(out)
    if source.resolve() == out.resolve() or out.exists():
        raise ValueError('Use a new output directory for the declared path-target experiment')
    with leadership.leadership_variant():
        store, edges = leadership.validate_leadership(source)
        feature_protocol = copy.deepcopy(features.FEATURE_PROTOCOL)
    out.mkdir(parents=True, exist_ok=False)
    for name in ('feature_store.pkl', 'feature_manifest.json', 'frozen_edges.pkl'):
        shutil.copy2(source/name, out/name)
    checkpoint_dir = out/'checkpoints'
    checkpoint_dir.mkdir()
    feature_meta = json.loads((out/'feature_manifest.json').read_text(encoding='utf-8'))
    raw = feature_meta['raw_adjusted_daily']
    targets = path_labels.build_path_targets(pd.read_pickle(raw['path']), sessions=store.dates)
    if not targets.dates.equals(store.dates) or targets.stock_codes != store.stock_codes:
        raise ValueError('Actual path-target axes differ from frozen model features')
    publish_path_targets(out, targets, raw)
    print(json.dumps(dict(event='path_targets_ready', labels=targets.valid.sum(axis=(0, 1)).tolist(),
                          target_shape=list(targets.path_payoff.shape))), flush=True)
    indices = store.signal_indices(start='2025-01-01', end=end)
    factors = {i:leadership.masked_tree_rows(store, edges, i) for i in indices}
    forecasts, audits = [], {}
    for quarter, plan in quarterly_plan(store, start, end).items():
        cutoff = plan['fit_cutoff']
        training, validation, train_cutoff = path_quarter_split(targets, indices, cutoff)
        training_arrays = path_training_arrays(training, factors, store, targets)
        validation_arrays = path_training_arrays(validation, factors, store, targets)
        print(json.dumps(dict(event='path_fit_start', quarter=quarter, train_rows=len(training_arrays[0]),
                              validation_rows=len(validation_arrays[0]), train_sessions=len(training))), flush=True)
        heads, iterations = fit_path_heads(training_arrays, validation_arrays, quarter)
        audit = dict(fit_cutoff=str(cutoff.date()), train_cutoff=str(train_cutoff.date()),
                     maximum_train_label_end=str(targets.label_end_dates[training].max()),
                     maximum_validation_label_end=str(targets.label_end_dates[validation].max()),
                     train_signal_dates=[str(store.dates[i].date()) for i in training],
                     validation_signal_dates=[str(store.dates[i].date()) for i in validation],
                     train_rows=len(training_arrays[0]), validation_rows=len(validation_arrays[0]),
                     best_iterations=iterations, path_target_protocol=targets.protocol,
                     configuration=dict(params=PARAMS, max_rounds=120, validation_sessions=20),
                     source_sha256=sha256(__file__))
        checkpoint = checkpoint_dir/f'{quarter}.pt'
        with checkpoint.open('wb') as handle:
            pickle.dump(dict(heads=heads, audit=audit, path_target_protocol=targets.protocol),
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
        with checkpoint.open('rb') as handle:
            saved = pickle.load(handle)
        if saved['path_target_protocol'] != targets.protocol:
            raise ValueError('Reloaded path checkpoint protocol mismatch')
        heads = saved['heads']
        audits[quarter] = dict(**audit, checkpoint_sha256=sha256(checkpoint))
        for i in plan['indices']:
            codes, x, _, _, snapshot = factors[i]
            mu, risk, event = predict_path_heads(heads, x)
            frame = pd.DataFrame(dict(signal_date=store.dates[i], ts_code=codes,
                                      utility=expected_utility(mu, risk), model_fit_cutoff=cutoff,
                                      graph_snapshot_date=snapshot))
            for j, horizon in enumerate(path_labels.HORIZONS):
                frame[f'mu{horizon}'], frame[f'risk{horizon}'] = mu[:, j], risk[:, j]
                for category, name in enumerate(CLASS_NAMES):
                    frame[f'passage_probability_{name}{horizon}'] = event[:, j, category]
            forecasts.append(frame)
        publish_manifest(audits, out/'fit_audits.json')
        publish_frame(pd.concat(forecasts, ignore_index=True), out/'predictions.pkl')
        del training_arrays, validation_arrays
    predictions = pd.concat(forecasts, ignore_index=True)
    predictions.to_csv(out/'predictions.csv', index=False, encoding='utf-8-sig')
    sources = ('research_daily_opportunity_features.py', 'research_daily_graph_data.py',
               'research_daily_opportunity.py', 'research_daily_opportunity_trees.py',
               'research_daily_opportunity_leadership.py', 'research_daily_opportunity_ranked.py',
               'research_daily_opportunity_masked_features.py', 'research_daily_opportunity_path_targets.py',
               Path(__file__).name)
    protocol = dict(model='daily_native_path_payoff_lightgbm_v1', features=feature_protocol, use_graph=True,
                    checkpoint_format='Pickled LightGBM boosters; .pt is artifact name only',
                    train_start='2025-01-01', account_start=start, account_end=end,
                    path_target_protocol=targets.protocol, target_artifact='path_targets.pkl',
                    original_feature_store='Unchanged V3 source artifact, including its unused original future labels',
                    factor_inputs='Unchanged finite/masked V3 daily tree factors',
                    heads='Three native path-payoff, three held-period MAE and three first-passage 3-class trees',
                    score='(.2*mu5+.3*mu10+.5*mu20) - .5*(.2*risk5+.3*risk10+.5*risk20)',
                    events_in_utility=False,
                    validation='Twenty past eligible sessions; both train and validation purged using t+h+1 label maturity',
                    hyperparameters_selected_from='Single fixed path-target hypothesis after native-close and residual failures; no sweep',
                    evaluation_status='Previously inspected 2026 research history, not an untouched holdout',
                    diagnostics='Use path_targets.pkl for forecast errors; generic original feature-store forward-return quality is not this target',
                    source_sha256={name:sha256(Path(__file__).parent/name) for name in sources})
    publish_manifest(protocol, out/'experiment_protocol.json')
    artifacts = {name:sha256(out/name) for name in
                 ('predictions.pkl', 'fit_audits.json', 'experiment_protocol.json', 'feature_manifest.json',
                  'path_targets.pkl', 'path_targets_manifest.json')}
    publish_manifest(dict(status='complete', rows=len(predictions), sessions=int(predictions.signal_date.nunique()),
                          artifacts=artifacts), out/'prediction_manifest.json')
    print(json.dumps(dict(event='path_predictions_complete', rows=len(predictions),
                          sessions=int(predictions.signal_date.nunique()))), flush=True)
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/daily_opportunity_leadership_20261002'))
    parser.add_argument('--out', type=Path, default=Path('results/daily_opportunity_path_20261003'))
    args = parser.parse_args()
    run_path_trees(args.source, args.out)
