"""Purged classifier for next-month adjusted returns of at least 30 percent.

``forecast_return`` is kept solely as an account-adapter column alias. Its
values are class-balanced classifier ranking scores, NOT forecast returns and
NOT calibrated natural-frequency probabilities. Do not interpret score times
invested cash as expected profit.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from data_pipeline.execution_data import ROOT
from research_dc_forecast import FEATURE_COLUMNS as BASE_FEATURE_COLUMNS
from research_dc_forecast import OUTPUT_COLUMNS, _dates, _frame_sha256, _sha256


FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ['peer_mom1', 'peer_mom3', 'peer_mom6', 'peer_positive3']
TARGET_THRESHOLD = .30
MODEL_PARAMS = dict(loss='log_loss', max_iter=100, max_leaf_nodes=15,
                    l2_regularization=10, learning_rate=.05, random_state=42)
PROTOCOL = {
    'version': 'rally_classifier_v1',
    'target_kind': 'rally_classifier',
    'features': FEATURE_COLUMNS,
    'estimator': 'HistGradientBoostingClassifier',
    'parameters': MODEL_PARAMS,
    'target_threshold': TARGET_THRESHOLD,
    'target': 'Next calendar-month adjusted close-to-close return >= .30, without clipping labels',
    'sample_weights': 'Each training class receives half the training weight: N/(2*n_class); counts use strictly purged training rows only',
    'fit_schedule': 'First requested monthly signal in each calendar quarter; frozen within quarter',
    'purge': 'Training label_date strictly precedes the quarterly fit signal_date',
    'minimum_training': '12 distinct completed label months; at least 10 positive and 10 negative training labels',
    'ranking': 'Per-signal all-board percentile of class-balanced classifier ranking score',
    'forecast_return_alias': 'Uncalibrated balanced-class ranking score; NOT expected return, NOT natural-frequency probability, NOT expected profit per yuan',
    'feature_source': 'Verified cached 19-feature price/amount and causal peer-context observations; no feature regeneration or label/stock/theme whitelist',
}
MONTHLY_HASH_COLUMNS = ['date', 'ts_code', 'close', 'adj_factor']
DAILY_HASH_COLUMNS = ['date', 'ts_code', 'open', 'high', 'low', 'close', 'volume', 'amount']
FEATURE_HELPERS = ['research_dc_peer_forecast.py', 'research_dc_peer_context.py',
                   'research_dc_forecast.py', 'research_dc_themes.py', 'research_dc_entry.py']


def walk_forward_rally_scores(features, signal_dates):
    """Return quarter-frozen ranking scores, with model/audit objects in attrs."""
    required = {'signal_date', 'ts_code', 'label_return', 'label_date', *FEATURE_COLUMNS}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f'Missing rally feature columns: {sorted(missing)}')
    frame = features.copy()
    frame['signal_date'] = _dates(frame.signal_date, 'Rally feature dates')
    frame['label_date'] = pd.to_datetime(frame.label_date, errors='raise')
    if frame.duplicated(['signal_date', 'ts_code']).any():
        raise ValueError('Duplicate rally stock/signal keys')
    if (frame.label_date.notna() & frame.label_date.le(frame.signal_date)).any():
        raise ValueError('Rally labels must follow their feature dates')
    frame = frame[frame.ts_code.astype('string').str.match(
        r'^(?:0\d{5}\.SZ|3\d{5}\.SZ|6\d{5}\.SH)$', na=False)].copy()
    feature_values = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    frame = frame[np.isfinite(feature_values).all(axis=1)].sort_values(['signal_date', 'ts_code'])
    frame['label_return'] = pd.to_numeric(frame.label_return, errors='coerce')
    signals = _dates(signal_dates, 'Rally prediction signals')
    if signals.has_duplicates:
        raise ValueError('Duplicate rally prediction signals')
    signals = signals.sort_values()
    models, audits, blocks = {}, [], []
    current_quarter, model, metadata = None, None, None
    for signal in signals:
        quarter = str(signal.to_period('Q'))
        if quarter != current_quarter:
            current_quarter = quarter
            train = frame[frame.signal_date.lt(signal) & frame.label_date.lt(signal)
                          & np.isfinite(frame.label_return)].copy()
            target = train.label_return.ge(TARGET_THRESHOLD).to_numpy(dtype=int)
            positives, negatives = int(target.sum()), int(len(target) - target.sum())
            label_months = train.label_date.dt.to_period('M').nunique()
            metadata = dict(quarter=quarter, model_fit_cutoff=signal.strftime('%Y-%m-%d'),
                train_label_end=(train.label_date.max().strftime('%Y-%m-%d') if len(train) else None),
                train_rows=len(train), train_label_months=int(label_months),
                train_positive_count=positives, train_negative_count=negatives,
                target_threshold=TARGET_THRESHOLD, target_kind='rally_classifier')
            model = None
            if label_months < 12:
                metadata['status'] = 'insufficient_label_months'
            elif positives < 10 or negatives < 10:
                metadata['status'] = 'insufficient_class_count'
            else:
                weights = np.where(target == 1, len(target) / (2. * positives),
                                   len(target) / (2. * negatives))
                model = HistGradientBoostingClassifier(**MODEL_PARAMS)
                with threadpool_limits(limits=4):
                    model.fit(train[FEATURE_COLUMNS].to_numpy(dtype=float), target,
                              sample_weight=weights)
                metadata['status'] = 'fitted'
                models[quarter] = dict(model=model, **metadata)
            audits.append(metadata.copy())
        current = frame[frame.signal_date.eq(signal)]
        if model is None or current.empty:
            continue
        classes = np.asarray(model.classes_)
        positive_index = np.flatnonzero(classes == 1)
        if len(positive_index) != 1:
            raise ValueError('Rally model does not contain exactly one positive class')
        with threadpool_limits(limits=4):
            scores = model.predict_proba(current[FEATURE_COLUMNS].to_numpy(dtype=float))[:, positive_index[0]]
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError('Invalid rally classifier ranking scores')
        block = current[['signal_date', 'ts_code']].copy()
        block['forecast_return'] = scores
        block['forecast_percentile'] = block.forecast_return.rank(pct=True)
        block['model_fit_cutoff'] = pd.Timestamp(metadata['model_fit_cutoff'])
        block['train_label_end'] = pd.Timestamp(metadata['train_label_end'])
        block['train_rows'] = metadata['train_rows']
        blocks.append(block[OUTPUT_COLUMNS])
    output = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=OUTPUT_COLUMNS)
    output.attrs['models'] = models
    output.attrs['model_audit'] = audits
    return output


def _validated_feature_cache(out, monthly_hash, daily_hash):
    """Validate only source feature/context artifacts, not overwritten models."""
    manifest_path = out / 'forecast_feature_manifest.json'
    if not manifest_path.is_file():
        raise ValueError('Rally classifier requires a verified peer feature manifest')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    specification = manifest.get('specification', {})
    if manifest.get('status') != 'complete' or specification.get('protocol', {}).get('features') != FEATURE_COLUMNS:
        raise ValueError('Rally classifier requires complete exact 19-feature peer inputs')
    if (specification.get('monthly_frame_sha256') != monthly_hash
            or specification.get('daily_frame_sha256') != daily_hash):
        raise ValueError('Rally in-memory source frames do not match verified peer inputs')
    inputs = specification.get('inputs', {})
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError('Peer feature source provenance is missing')
    resolved_inputs = {Path(path).resolve(): digest for path, digest in inputs.items()}
    for path, expected in resolved_inputs.items():
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f'Peer feature source hash mismatch: {path.name}')
    for name in FEATURE_HELPERS:
        if Path(__file__).with_name(name).resolve() not in resolved_inputs:
            raise ValueError(f'Peer feature helper provenance is missing: {name}')
    monthly_path = (ROOT / 'stock_month_end_verified.pkl').resolve()
    if monthly_path not in resolved_inputs:
        raise ValueError('Peer feature monthly disk provenance is missing')
    daily_sources = [digest for path, digest in resolved_inputs.items() if path.name == 'daily.pkl']
    daily_path = out / 'daily.pkl'
    if len(daily_sources) != 1 or not daily_path.is_file() or _sha256(daily_path) != daily_sources[0]:
        raise ValueError('Rally daily disk data do not match verified peer inputs')
    artifact_inputs = {}
    for name in ('forecast_features.pkl', 'peer_context.pkl'):
        path = out / name
        if not path.is_file() or manifest.get('artifacts', {}).get(name) != _sha256(path):
            raise ValueError(f'Peer feature artifact hash mismatch: {name}')
        artifact_inputs[str(path.resolve())] = _sha256(path)
    artifact_inputs[str(manifest_path.resolve())] = _sha256(manifest_path)
    return manifest, {str(path): digest for path, digest in resolved_inputs.items()}, artifact_inputs


def prepare_rally_scores(out, monthly, daily, signal_dates):
    """Reuse source-bound peer features without regenerating or overwriting them."""
    out = Path(out)
    signals = _dates(signal_dates, 'Rally prediction signals').sort_values()
    if signals.has_duplicates:
        raise ValueError('Duplicate rally prediction signals')
    monthly_hash = _frame_sha256(monthly, MONTHLY_HASH_COLUMNS)
    daily_hash = _frame_sha256(daily, DAILY_HASH_COLUMNS)
    feature_manifest, peer_inputs, artifacts = _validated_feature_cache(out, monthly_hash, daily_hash)
    own_inputs = {**peer_inputs, **artifacts,
                  str(Path(__file__).resolve()): _sha256(Path(__file__)),
                  str((out / 'daily.pkl').resolve()): _sha256(out / 'daily.pkl')}
    specification = dict(protocol=PROTOCOL, sklearn_version=sklearn.__version__, inputs=own_inputs,
        monthly_frame_sha256=monthly_hash, daily_frame_sha256=daily_hash,
        signals=signals.strftime('%Y-%m-%d').tolist(),
        peer_feature_protocol=feature_manifest['specification']['protocol'])
    output_names = ['forecast_predictions.pkl', 'forecast_models.pkl', 'forecast_model_audit.json']
    manifest_path = out / 'rally_feature_manifest.json'
    if manifest_path.is_file() and all((out / name).is_file() for name in output_names):
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (old.get('status') == 'complete' and old.get('specification') == specification
                and all(old.get('artifacts', {}).get(name) == _sha256(out / name) for name in output_names)):
            return pd.read_pickle(out / 'forecast_predictions.pkl')
    features = pd.read_pickle(out / 'forecast_features.pkl')
    if 'peer_count' not in features or not pd.to_numeric(features.peer_count, errors='coerce').ge(10).all():
        raise ValueError('Rally feature cache has invalid peer-count eligibility')
    predictions = walk_forward_rally_scores(features, signals)
    models = predictions.attrs.pop('models')
    audit = predictions.attrs.pop('model_audit')
    predictions.to_pickle(out / 'forecast_predictions.pkl')
    pd.to_pickle(models, out / 'forecast_models.pkl')
    audit_payload = dict(protocol=PROTOCOL, quarterly_models=audit, feature_rows=len(features),
        prediction_rows=len(predictions), train_label_purge_verified=bool(predictions.empty or
            predictions.train_label_end.lt(predictions.model_fit_cutoff).all()))
    (out / 'forecast_model_audit.json').write_text(json.dumps(audit_payload, indent=2), encoding='utf-8')
    payload = dict(status='complete', specification=specification, feature_rows=len(features),
        prediction_rows=len(predictions), artifacts={name: _sha256(out / name) for name in output_names},
        immutable_feature_artifacts={name: _sha256(out / name) for name in
                                    ('forecast_features.pkl', 'peer_context.pkl')})
    manifest_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return predictions
