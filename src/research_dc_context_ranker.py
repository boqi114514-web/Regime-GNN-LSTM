"""Purged monthly learning-to-rank with causal price-cohort context.

Each completed signal month is a separate ranking query. The supervised
target is that query's next-month adjusted-return relevance, not a regression
of every stock's mean return. The compatibility column ``forecast_return``
contains a raw LambdaRank score and MUST NOT be read as expected return,
probability or expected profit. Raw scores may be negative.

The 19 base features are accepted only from source-bound peer feature caches.
No current DC membership is backfilled into historical training observations.
"""

import json
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from data_pipeline.execution_data import ROOT
from research_dc_forecast import FEATURE_COLUMNS as BASE_FEATURE_COLUMNS
from research_dc_forecast import OUTPUT_COLUMNS, _dates, _frame_sha256, _sha256


PEER_COLUMNS = ['peer_mom1', 'peer_mom3', 'peer_mom6', 'peer_positive3']
SOURCE_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + PEER_COLUMNS
EXCESS_COLUMNS = ['stock_peer_excess1', 'stock_peer_excess3',
                  'stock_peer_excess6', 'stock_peer_acceleration']
FEATURE_COLUMNS = SOURCE_FEATURE_COLUMNS + EXCESS_COLUMNS
LABEL_GAIN = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511]
MODEL_PARAMS = dict(objective='lambdarank', metric='ndcg', n_estimators=120,
    num_leaves=15, learning_rate=.05, reg_lambda=10., min_child_samples=50,
    random_state=42, n_jobs=4, deterministic=True, force_col_wise=True,
    lambdarank_truncation_level=6, label_gain=LABEL_GAIN, verbosity=-1)
PROTOCOL = {
    'version': 'context_ranker_v1',
    'target_kind': 'context_ranker',
    'features': FEATURE_COLUMNS,
    'estimator': 'LGBMRanker',
    'parameters': MODEL_PARAMS,
    'query': 'One signal_date per query; all eligible SH/SZ boards included',
    'label': 'Unclipped next-month adjusted returns ranked within each purged training query; floor(10*(average_rank-1)/query_count), clipped to 0..9; tied returns receive tied relevance',
    'evaluation_cutoffs': [1, 5],
    'optimization': 'LambdaRank NDCG with fixed exponential relevance gain and truncation level 6; no account-outcome parameter search or future validation set',
    'fit_schedule': 'First requested monthly signal in each calendar quarter; frozen within quarter',
    'purge': 'Training label_date strictly precedes the quarterly fit signal_date',
    'minimum_training': '12 distinct completed label months with at least two complete-feature stocks per query and at least one nonconstant relevance query',
    'ranking': 'Per-signal all-board percentile of raw LambdaRank score',
    'forecast_return_alias': 'Raw ranking score, possibly negative; NOT expected return, NOT probability, NOT expected profit per yuan',
    'feature_source': 'Immutable source-bound 19-feature price/amount and causal peer context plus stock-minus-peer mom1/3/6 and excess1-excess3/3; no DC historical backfill, stock identifiers, theme whitelist or future labels as features',
    'research_sources': [
        'https://arxiv.org/abs/2012.07149',
        'https://arxiv.org/abs/2105.10019',
        'https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMRanker.html',
    ],
}
MONTHLY_HASH_COLUMNS = ['date', 'ts_code', 'close', 'adj_factor']
DAILY_HASH_COLUMNS = ['date', 'ts_code', 'open', 'high', 'low', 'close', 'volume', 'amount']
FEATURE_HELPERS = ['research_dc_peer_forecast.py', 'research_dc_peer_context.py',
                   'research_dc_forecast.py', 'research_dc_themes.py', 'research_dc_entry.py']


def protocol_for_scope(scope):
    """Return the unchanged original protocol or explicit cohort-query copy.

    Cohorts are identified by exact same-signal contemporaneous peer summary
    statistics. Identical statistics can collide across distinct clusters;
    this is a price-cohort proxy, not economic sector membership.
    """
    if scope == 'monthly':
        return PROTOCOL
    if scope != 'cohort':
        raise ValueError('Context query_scope must be monthly or cohort')
    return {**PROTOCOL,
        'version': 'context_ranker_cohort_v1',
        'query_scope': 'cohort',
        'query': 'Exact (signal_date, peer_mom1, peer_mom3, peer_mom6, peer_positive3) price-cohort-statistics proxy; all eligible SH/SZ boards included; identical statistics may collide across distinct cohorts; not economic membership',
        'label': 'Unclipped next-month adjusted returns ranked within each purged training price-cohort proxy query; floor(10*(average_rank-1)/query_count), clipped to 0..9; tied returns receive tied relevance',
        'minimum_training': '12 distinct completed label months with at least two complete-feature stocks per price-cohort proxy query and at least one nonconstant relevance query',
        'query_key_columns': ['signal_date'] + PEER_COLUMNS,
        'cross_query_score_limitation': 'Shared raw ranking scores are not calibrated cross-cohort returns or probabilities; strict external theme eligibility remains separate',
    }


def _query_columns(scope):
    protocol_for_scope(scope)
    return ['signal_date'] if scope == 'monthly' else ['signal_date'] + PEER_COLUMNS


def context_features(features):
    """Add same-signal stock-versus-price-cohort contrasts, without labels."""
    missing = set(SOURCE_FEATURE_COLUMNS).difference(features.columns)
    if missing:
        raise ValueError(f'Missing context source features: {sorted(missing)}')
    result = features.copy()
    for horizon in (1, 3, 6):
        result[f'stock_peer_excess{horizon}'] = (
            pd.to_numeric(result[f'mom{horizon}'], errors='coerce')
            - pd.to_numeric(result[f'peer_mom{horizon}'], errors='coerce'))
    result['stock_peer_acceleration'] = result.stock_peer_excess1 - result.stock_peer_excess3 / 3.
    return result


def query_relevance(train, query_scope='monthly'):
    """Compute integer relevance independently within each training query.

    The default query identity is signal_date. The explicit cohort mode adds
    same-signal peer statistics. Average ranks avoid granting an arbitrary
    identifier-order advantage to tied returns.
    """
    columns = _query_columns(query_scope)
    rank = train.groupby(columns, sort=False).label_return.rank(method='average')
    count = train.groupby(columns, sort=False).label_return.transform('size')
    return np.floor(10. * (rank - 1.) / count).clip(0, 9).astype(np.int32)


def walk_forward_context_scores(features, signal_dates, query_scope='monthly'):
    """Train monthly-query rankers only on strictly earlier complete labels."""
    query_columns = _query_columns(query_scope)
    frame = context_features(features)
    required = {'signal_date', 'ts_code', 'label_return', 'label_date'}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f'Missing context ranking columns: {sorted(missing)}')
    frame['signal_date'] = _dates(frame.signal_date, 'Context feature dates')
    frame['label_date'] = pd.to_datetime(frame.label_date, errors='raise')
    _dates(frame.label_date.dropna(), 'Context label dates')
    if frame.duplicated(['signal_date', 'ts_code']).any():
        raise ValueError('Duplicate context stock/signal keys')
    observed = frame.label_date.notna()
    if (observed & frame.label_date.le(frame.signal_date)).any():
        raise ValueError('Context labels must follow their feature dates')
    if (frame.loc[observed, 'label_date'].dt.to_period('M').to_numpy()
            != (frame.loc[observed, 'signal_date'].dt.to_period('M') + 1).to_numpy()).any():
        raise ValueError('Context labels must belong to the next calendar month')
    if frame.loc[observed].groupby('signal_date').label_date.nunique().gt(1).any():
        raise ValueError('A context query must share one label date')
    frame = frame[frame.ts_code.astype('string').str.match(
        r'^(?:0\d{5}\.SZ|3\d{5}\.SZ|6\d{5}\.SH)$', na=False)].copy()
    numeric = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors='coerce')
    frame[FEATURE_COLUMNS] = numeric
    frame = frame[np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)].copy()
    frame['label_return'] = pd.to_numeric(frame.label_return, errors='coerce')
    frame = frame.sort_values(['signal_date', 'ts_code'])
    signals = _dates(signal_dates, 'Context prediction signals').sort_values()
    if signals.has_duplicates:
        raise ValueError('Duplicate context prediction signals')
    models, audits, blocks = {}, [], []
    current_quarter, model, metadata = None, None, None
    for signal in signals:
        quarter = str(signal.to_period('Q'))
        if quarter != current_quarter:
            current_quarter = quarter
            train = frame[frame.signal_date.lt(signal) & frame.label_date.lt(signal)
                          & np.isfinite(frame.label_return)].copy()
            sizes = train.groupby(query_columns, sort=False).ts_code.transform('size')
            train = train[sizes.ge(2)].copy()
            if query_scope == 'cohort':
                train = train.sort_values(query_columns + ['ts_code'])
            groups = train.groupby(query_columns, sort=False).size()
            relevance = query_relevance(train, query_scope=query_scope)
            label_months = train.label_date.dt.to_period('M').nunique()
            varying_queries = (train.assign(relevance=relevance).groupby(query_columns)
                               .relevance.nunique().gt(1).sum())
            query_dates = (groups.index.get_level_values('signal_date')
                           if query_scope == 'cohort' else groups.index)
            metadata = dict(quarter=quarter, target_kind='context_ranker',
                model_fit_cutoff=signal.strftime('%Y-%m-%d'),
                train_label_end=(train.label_date.max().strftime('%Y-%m-%d') if len(train) else None),
                train_rows=int(len(train)), train_label_months=int(label_months),
                train_query_count=int(len(groups)), train_query_sizes=groups.astype(int).tolist(),
                train_query_dates=query_dates.strftime('%Y-%m-%d').tolist(),
                train_varying_queries=int(varying_queries),
                train_relevance_counts={str(i): int(relevance.eq(i).sum()) for i in range(10)})
            if query_scope == 'cohort':
                metadata['query_scope'] = 'cohort'
                metadata['train_query_keys'] = [
                    [pd.Timestamp(key[0]).strftime('%Y-%m-%d')]
                    + [float(value) for value in key[1:]] for key in groups.index]
            model = None
            if label_months < 12:
                metadata['status'] = 'insufficient_label_months'
            elif not varying_queries:
                metadata['status'] = 'insufficient_relevance_variation'
            else:
                model = LGBMRanker(**MODEL_PARAMS)
                model.fit(train[FEATURE_COLUMNS].to_numpy(dtype=float),
                          relevance.to_numpy(dtype=np.int32),
                          group=groups.to_numpy(dtype=np.int32), eval_at=(1, 5))
                metadata['status'] = 'fitted'
                models[quarter] = dict(model=model, **metadata)
            audits.append(metadata.copy())
        current = frame[frame.signal_date.eq(signal)]
        if model is None or current.empty:
            continue
        scores = model.predict(current[FEATURE_COLUMNS].to_numpy(dtype=float))
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite context ranking scores')
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
    """Check immutable feature provenance while ignoring old model outputs."""
    manifest_path = out / 'forecast_feature_manifest.json'
    if not manifest_path.is_file():
        raise ValueError('Context ranker requires a verified peer feature manifest')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    spec = manifest.get('specification', {})
    if manifest.get('status') != 'complete' or spec.get('protocol', {}).get('features') != SOURCE_FEATURE_COLUMNS:
        raise ValueError('Context ranker requires complete exact 19-feature peer inputs')
    if spec.get('monthly_frame_sha256') != monthly_hash or spec.get('daily_frame_sha256') != daily_hash:
        raise ValueError('Context in-memory source frames do not match verified peer inputs')
    inputs = spec.get('inputs', {})
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError('Peer feature source provenance is missing')
    resolved = {Path(path).resolve(): digest for path, digest in inputs.items()}
    for path, expected in resolved.items():
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f'Peer feature source hash mismatch: {path.name}')
    for name in FEATURE_HELPERS:
        if Path(__file__).with_name(name).resolve() not in resolved:
            raise ValueError(f'Peer feature helper provenance is missing: {name}')
    if (ROOT / 'stock_month_end_verified.pkl').resolve() not in resolved:
        raise ValueError('Peer feature monthly disk provenance is missing')
    source_daily = [digest for path, digest in resolved.items() if path.name == 'daily.pkl']
    daily_path = out / 'daily.pkl'
    if len(source_daily) != 1 or not daily_path.is_file() or _sha256(daily_path) != source_daily[0]:
        raise ValueError('Context daily disk data do not match verified peer inputs')
    artifact_inputs = {}
    for name in ('forecast_features.pkl', 'peer_context.pkl'):
        path = out / name
        if not path.is_file() or manifest.get('artifacts', {}).get(name) != _sha256(path):
            raise ValueError(f'Peer feature artifact hash mismatch: {name}')
        artifact_inputs[str(path.resolve())] = _sha256(path)
    artifact_inputs[str(manifest_path.resolve())] = _sha256(manifest_path)
    return manifest, {str(path): digest for path, digest in resolved.items()}, artifact_inputs


def prepare_context_scores(out, monthly, daily, signal_dates, query_scope='monthly'):
    """Fit/cache the ranker in an isolated account directory with bound inputs."""
    protocol = protocol_for_scope(query_scope)
    out = Path(out)
    signals = _dates(signal_dates, 'Context prediction signals').sort_values()
    if signals.has_duplicates:
        raise ValueError('Duplicate context prediction signals')
    monthly_hash = _frame_sha256(monthly, MONTHLY_HASH_COLUMNS)
    daily_hash = _frame_sha256(daily, DAILY_HASH_COLUMNS)
    feature_manifest, inputs, artifacts = _validated_feature_cache(out, monthly_hash, daily_hash)
    own_inputs = {**inputs, **artifacts,
        str(Path(__file__).resolve()): _sha256(Path(__file__)),
        str((out / 'daily.pkl').resolve()): _sha256(out / 'daily.pkl')}
    specification = dict(protocol=protocol, lightgbm_version=lightgbm.__version__,
        inputs=own_inputs, monthly_frame_sha256=monthly_hash, daily_frame_sha256=daily_hash,
        signals=signals.strftime('%Y-%m-%d').tolist(),
        peer_feature_protocol=feature_manifest['specification']['protocol'])
    output_names = ['context_ranker_features.pkl', 'forecast_predictions.pkl',
                    'forecast_models.pkl', 'forecast_model_audit.json']
    manifest_path = out / 'context_ranker_feature_manifest.json'
    if manifest_path.is_file() and all((out / name).is_file() for name in output_names):
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (old.get('status') == 'complete' and old.get('specification') == specification
                and all(old.get('artifacts', {}).get(name) == _sha256(out / name) for name in output_names)):
            return pd.read_pickle(out / 'forecast_predictions.pkl')
    base = pd.read_pickle(out / 'forecast_features.pkl')
    if 'peer_count' not in base or not pd.to_numeric(base.peer_count, errors='coerce').ge(10).all():
        raise ValueError('Context feature cache has invalid peer-count eligibility')
    features = context_features(base)
    predictions = walk_forward_context_scores(features, signals, query_scope=query_scope)
    models = predictions.attrs.pop('models')
    audit = predictions.attrs.pop('model_audit')
    features.to_pickle(out / 'context_ranker_features.pkl')
    predictions.to_pickle(out / 'forecast_predictions.pkl')
    pd.to_pickle(models, out / 'forecast_models.pkl')
    payload = dict(protocol=protocol, quarterly_models=audit, feature_rows=len(features),
        prediction_rows=len(predictions), train_label_purge_verified=bool(predictions.empty or
            predictions.train_label_end.lt(predictions.model_fit_cutoff).all()))
    (out / 'forecast_model_audit.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    manifest = dict(status='complete', specification=specification,
        feature_rows=len(features), prediction_rows=len(predictions),
        artifacts={name: _sha256(out / name) for name in output_names},
        immutable_feature_artifacts={name: _sha256(out / name) for name in
                                    ('forecast_features.pkl', 'peer_context.pkl')})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return predictions
