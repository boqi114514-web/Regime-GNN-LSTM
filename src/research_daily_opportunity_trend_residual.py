"""One fixed causal trend-reference + learned native-future-residual hypothesis.

The supervised future-return target is unchanged. Return trees learn the
future return minus an observable own/theme trend reference; that reference
is added back at prediction. Downside and rally-event heads retain the true
future targets, not residual targets. This is a separate, explicitly recorded
research branch, not a renaming of past momentum as a predicted return.
"""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import shutil

import lightgbm as lgb
import numpy as np
import pandas as pd

import research_daily_opportunity_features as features
import research_daily_opportunity_leadership as leadership
from research_daily_graph_data import publish_frame, publish_manifest, sha256
from research_daily_opportunity import quarterly_plan
from research_daily_opportunity_features import expected_utility
from research_daily_opportunity_trees import PARAMS


HORIZONS = (5, 10, 20)
RALLY_THRESHOLDS = (.10, .15, .30)
REFERENCE_PROTOCOL = {
    'version': 'daily_native_trend_residual_v1',
    'reference': 'Equal mean of available current own trailing return_h and selected self-free theme peer trailing return_h; one if only one exists, zero if neither',
    'availability': 'Explicit encoded availability masks; unknown selected theme context is unavailable, not an observed zero return',
    'return_training_target': 'Actual next-open future native return_h minus the causal signal-day reference_h',
    'return_inference': 'Saved learned residual_h plus current signal-day reference_h; negative residual corrections retained without clipping',
    'downside_target': 'Unchanged true future maximum adverse excursion',
    'rally_target': 'Unchanged true future return >= (.10,.15,.30); never threshold residual return',
    'hyperparameters': 'Same V3 masked factors and fixed 120-round tree ceiling, early stop12, purged quarters; no parameter sweep or stock/theme/month filter',
}


def causal_reference(own, peer, own_available=None, peer_available=None):
    """Pure native-unit reference; missing inputs never become observed trends."""
    own, peer = np.asarray(own, dtype=float), np.asarray(peer, dtype=float)
    if own.shape != peer.shape or own.ndim != 2 or own.shape[1] != 3:
        raise ValueError('Own/peer returns must have equal [stocks,3] shapes')
    if np.isinf(own).any() or np.isinf(peer).any():
        raise ValueError('Infinite trailing returns are invalid')
    mask_own = np.isfinite(own)
    mask_peer = np.isfinite(peer)
    for supplied, shape in ((own_available, own.shape), (peer_available, peer.shape)):
        if supplied is not None and np.asarray(supplied).shape != shape:
            raise ValueError('Return availability shape mismatch')
    if own_available is not None:
        mask_own &= np.asarray(own_available, dtype=bool)
    if peer_available is not None:
        mask_peer &= np.asarray(peer_available, dtype=bool)
    count = mask_own.astype(int)+mask_peer.astype(int)
    total = np.where(mask_own, own, 0.)+np.where(mask_peer, peer, 0.)
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def reference_for_rows(store, index):
    """Same current-valid stock order as V3 masked tree rows; no labels read."""
    indices = np.flatnonzero(store.valid_features[index])
    names = {name:i for i,name in enumerate(store.feature_names)}
    required = [name for h in HORIZONS for name in (
        f'return{h}', f'theme_peer_return{h}',
        f'available__return{h}', f'available__theme_peer_return{h}')]
    required += ['known_theme_context', 'available__known_theme_context']
    if not set(required).issubset(names):
        raise ValueError('Masked V3 own/theme factors and their availability masks are required')
    x = store.features[index, indices]
    if not np.isfinite(x).all():
        raise ValueError('Encoded current factors must be finite')
    own = np.column_stack([x[:,names[f'return{h}']] for h in HORIZONS])
    peer = np.column_stack([x[:,names[f'theme_peer_return{h}']] for h in HORIZONS])
    mask_own = np.column_stack([x[:,names[f'available__return{h}']].eq(1)
                               if isinstance(x[:,names[f'available__return{h}']], pd.Series)
                               else x[:,names[f'available__return{h}']] == 1 for h in HORIZONS])
    known = (x[:,names['known_theme_context']] == 1) & (x[:,names['available__known_theme_context']] == 1)
    mask_peer = np.column_stack([(x[:,names[f'available__theme_peer_return{h}']] == 1) & known
                                for h in HORIZONS])
    return causal_reference(own, peer, mask_own, mask_peer)


def residual_targets(native_returns, reference):
    native_returns, reference = np.asarray(native_returns, dtype=float), np.asarray(reference, dtype=float)
    if native_returns.shape != reference.shape or native_returns.ndim != 2 or native_returns.shape[1] != 3:
        raise ValueError('Native future returns/reference must have equal [stocks,3] shapes')
    if np.isinf(native_returns).any() or not np.isfinite(reference).all():
        raise ValueError('Finite references and finite-or-missing true future labels required')
    return native_returns-reference


def reconstruct_native_returns(predicted_residual, reference):
    predicted_residual, reference = np.asarray(predicted_residual, dtype=float), np.asarray(reference, dtype=float)
    if predicted_residual.shape != reference.shape or predicted_residual.ndim != 2 or predicted_residual.shape[1] != 3:
        raise ValueError('Residual/reference must have equal [stocks,3] shapes')
    if not np.isfinite(predicted_residual).all() or not np.isfinite(reference).all():
        raise ValueError('Native inference components must be finite')
    return predicted_residual+reference


def native_rally_events(native_returns):
    native_returns = np.asarray(native_returns, dtype=float)
    if native_returns.ndim != 2 or native_returns.shape[1] != 3 or np.isinf(native_returns).any():
        raise ValueError('True native future return [stocks,3] labels required')
    return np.where(np.isfinite(native_returns),
                    (native_returns >= np.asarray(RALLY_THRESHOLDS)).astype(float), np.nan)


def run_trend_residual(source, out, rounds=120, start='2026-01-01', end='2026-09-24'):
    source, out = Path(source), Path(out)
    if source.resolve() == out.resolve() or out.exists():
        raise ValueError('Use a new separate output folder; do not overwrite earlier experiments')
    if rounds != 120:
        raise ValueError('This fixed hypothesis uses the unchanged 120-round ceiling')
    with leadership.leadership_variant():
        store, edges = leadership.validate_leadership(source)
        feature_protocol = copy.deepcopy(features.FEATURE_PROTOCOL)
    out.mkdir(parents=True, exist_ok=False)
    for name in ('feature_store.pkl', 'feature_manifest.json', 'frozen_edges.pkl'):
        shutil.copy2(source/name, out/name)
    checkpoint_dir = out/'checkpoints'
    checkpoint_dir.mkdir()
    indices = store.signal_indices(start='2025-01-01', end=end)
    factors = {i:leadership.masked_tree_rows(store, edges, i) for i in indices}
    references = {i:reference_for_rows(store, i) for i in indices}
    forecasts, audits = [], {}
    for quarter, plan in quarterly_plan(store, start, end).items():
        cutoff = plan['fit_cutoff']
        completed = [i for i in indices if store.dates[i] < cutoff
                     and not np.isnat(store.label_end_dates[i]).any()
                     and (store.label_end_dates[i] < np.datetime64(cutoff, 'D')).all()]
        if len(completed) < 60:
            raise ValueError('Insufficient complete historical labels')
        validation = completed[-20:]
        train_cutoff = store.dates[validation[0]]
        training = [i for i in completed if (store.label_end_dates[i] < np.datetime64(train_cutoff, 'D')).all()]
        if len(training) < 20:
            raise ValueError('Insufficient purged training days')

        def arrays(days):
            xs, ys, risks, refs = [], [], [], []
            for i in days:
                _, x, y, risk, _ = factors[i]
                good = np.isfinite(y).all(axis=1) & np.isfinite(risk).all(axis=1)
                xs.append(x[good]); ys.append(y[good]); risks.append(risk[good]); refs.append(references[i][good])
            return np.concatenate(xs), np.concatenate(ys), np.concatenate(risks), np.concatenate(refs)

        x, y, risk, ref = arrays(training)
        vx, vy, vrisk, vref = arrays(validation)
        residual_y, residual_vy = residual_targets(y, ref), residual_targets(vy, vref)
        print(json.dumps(dict(event='trend_residual_fit_start', quarter=quarter, train_rows=len(x),
                              validation_rows=len(vx), train_sessions=len(training))), flush=True)
        heads, iterations = {}, {}
        for kind, target, vtarget in [('residual_return', residual_y, residual_vy), ('risk', risk, vrisk)]:
            for j, h in enumerate(HORIZONS):
                name = f'{kind}{h}'
                train = lgb.Dataset(x, label=target[:,j], free_raw_data=False)
                valid = lgb.Dataset(vx, label=vtarget[:,j], reference=train, free_raw_data=False)
                model = lgb.train(PARAMS, train, num_boost_round=rounds, valid_sets=[valid],
                                  callbacks=[lgb.early_stopping(12, verbose=False)])
                heads[name], iterations[name] = model, model.best_iteration
                print(json.dumps(dict(event='trend_residual_head_complete', quarter=quarter,
                                      head=name, iterations=model.best_iteration)), flush=True)
        events, vevents = native_rally_events(y), native_rally_events(vy)
        for j, h in enumerate(HORIZONS):
            train = lgb.Dataset(x, label=events[:,j], free_raw_data=False)
            valid = lgb.Dataset(vx, label=vevents[:,j], reference=train, free_raw_data=False)
            model = lgb.train(dict(PARAMS, objective='binary', metric='binary_logloss'), train,
                              num_boost_round=rounds, valid_sets=[valid],
                              callbacks=[lgb.early_stopping(12, verbose=False)])
            heads[f'rally{h}'], iterations[f'rally{h}'] = model, model.best_iteration
        audit = dict(fit_cutoff=str(cutoff.date()), train_cutoff=str(train_cutoff.date()),
                     maximum_train_label_end=str(max(store.label_end_dates[i].max() for i in training)),
                     train_signal_dates=[str(store.dates[i].date()) for i in training],
                     validation_signal_dates=[str(store.dates[i].date()) for i in validation],
                     train_rows=len(x), validation_rows=len(vx), best_iterations=iterations,
                     configuration=dict(params=PARAMS, max_rounds=rounds, validation_sessions=20),
                     reference_protocol=REFERENCE_PROTOCOL, source_sha256=sha256(__file__))
        checkpoint = checkpoint_dir/f'{quarter}.pt'
        with checkpoint.open('wb') as handle:
            pickle.dump(dict(heads=heads, audit=audit, reference_protocol=REFERENCE_PROTOCOL),
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
        with checkpoint.open('rb') as handle:
            saved = pickle.load(handle)
        if saved['reference_protocol'] != REFERENCE_PROTOCOL:
            raise ValueError('Saved residual/reference protocol mismatch')
        heads = saved['heads']
        audits[quarter] = dict(**audit, checkpoint_sha256=sha256(checkpoint))
        for i in plan['indices']:
            codes, xday, _, _, snapshot = factors[i]
            residual = np.column_stack([heads[f'residual_return{h}'].predict(xday) for h in HORIZONS])
            mu = reconstruct_native_returns(residual, references[i])
            downside = np.maximum(0., np.column_stack([heads[f'risk{h}'].predict(xday) for h in HORIZONS]))
            frame = pd.DataFrame(dict(signal_date=store.dates[i], ts_code=codes,
                                      utility=expected_utility(mu, downside), model_fit_cutoff=cutoff,
                                      graph_snapshot_date=snapshot))
            for j, h in enumerate(HORIZONS):
                frame[f'mu{h}'], frame[f'risk{h}'] = mu[:,j], downside[:,j]
                frame[f'rally_probability{h}'] = heads[f'rally{h}'].predict(xday)
                frame[f'trend_reference{h}'], frame[f'learned_residual{h}'] = references[i][:,j], residual[:,j]
            forecasts.append(frame)
        publish_manifest(audits, out/'fit_audits.json')
        publish_frame(pd.concat(forecasts, ignore_index=True), out/'predictions.pkl')
    predictions = pd.concat(forecasts, ignore_index=True)
    predictions.to_csv(out/'predictions.csv', index=False, encoding='utf-8-sig')
    source_names = ('research_daily_opportunity_features.py', 'research_daily_graph_data.py',
                    'research_daily_opportunity.py', 'research_daily_opportunity_trees.py',
                    'research_daily_opportunity_leadership.py', 'research_daily_opportunity_ranked.py',
                    'research_daily_opportunity_masked_features.py', Path(__file__).name)
    protocol = dict(model='daily_native_trend_residual_lightgbm_v1', features=feature_protocol,
                    use_graph=True, checkpoint_format='Pickled LightGBM boosters; .pt is artifact name only',
                    train_start='2025-01-01', account_start=start, account_end=end,
                    reference_protocol=REFERENCE_PROTOCOL, factor_inputs='Unchanged finite/masked V3 daily tree factors',
                    experts='Separate residual-return, true-risk and true-event tree heads; not neural experts',
                    score='(.2*mu5+.3*mu10+.5*mu20) - .5*(.2*risk5+.3*risk10+.5*risk20)',
                    validation='Twenty past eligible daily sessions; longest training label end strictly before first validation signal',
                    hyperparameters_selected_from='One declared residual hypothesis after V3 failure; fixed parameters, no sweep',
                    evaluation_status='Previously inspected 2026 research history, not a new untouched test set',
                    source_sha256={name:sha256(Path(__file__).parent/name) for name in source_names})
    publish_manifest(protocol, out/'experiment_protocol.json')
    artifacts = {name:sha256(out/name) for name in ('predictions.pkl','fit_audits.json','experiment_protocol.json','feature_manifest.json')}
    publish_manifest(dict(status='complete', rows=len(predictions), sessions=int(predictions.signal_date.nunique()),
                          artifacts=artifacts), out/'prediction_manifest.json')
    print(json.dumps(dict(event='trend_residual_predictions_complete', rows=len(predictions),
                          sessions=int(predictions.signal_date.nunique()))), flush=True)
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/daily_opportunity_leadership_20261002'))
    parser.add_argument('--out', type=Path, default=Path('results/daily_opportunity_trend_residual_20261002'))
    args = parser.parse_args()
    run_trend_residual(args.source, args.out)
