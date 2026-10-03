"""Purged, quarterly-frozen stock forecasts from causal price/amount features.

The supervised target is next calendar month's adjusted close-to-close return.
It is not the return on an opening-auction execution, nor a promised profit.
Industry names, stock identifiers and account outcomes are never model inputs.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from research_dc_entry import entry_features


FEATURE_COLUMNS = [
    'mom1', 'mom3', 'mom6', 'volatility3', 'volatility6',
    'positive_fraction6', 'momentum_acceleration', 'mom1_percentile',
    'mom3_percentile', 'mom6_percentile', 'amount_percentile',
    'pressure5', 'pressure20', 'location5', 'amount_ratio5_vs_prior20',
]
OUTPUT_COLUMNS = [
    'signal_date', 'ts_code', 'forecast_return', 'forecast_percentile',
    'model_fit_cutoff', 'train_label_end', 'train_rows',
]
MODEL_PARAMS = dict(loss='squared_error', max_iter=100, max_leaf_nodes=15,
                    max_depth=None, l2_regularization=10,
                    learning_rate=.05, random_state=42)
FORECAST_PROTOCOL = {
    'version': 1,
    'features': FEATURE_COLUMNS,
    'estimator': 'HistGradientBoostingRegressor',
    'parameters': MODEL_PARAMS,
    'fit_schedule': 'First requested monthly signal in each calendar quarter; frozen within quarter',
    'training_window': 'Expanding complete-feature stock observations; at least 12 distinct label months',
    'purge': 'Training label_date strictly precedes the quarterly fit signal_date',
    'label': 'Next calendar-month adjusted close / signal adjusted close - 1; clipped to [-.50,.50] only in fitting',
    'ranking': 'Per-signal forecast percentile among all predicted SH/SZ mainboard, ChiNext and STAR stocks',
    'daily_features': 'Complete 25-session gap-invariant intraday pressure and amount windows, not close-to-close total return',
    'first_feature_month': '2023-01',
    'minimum_liquidity_days': 10,
}


def _dates(values, name):
    dates = pd.DatetimeIndex(pd.to_datetime(values, errors='raise'))
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError(f'{name} must be valid timezone-free date-only values')
    return dates


def forecast_features(monthly, daily):
    """Return complete as-of monthly features and separately dated future labels.

    Calendar-month reindexing prevents a missing month from shortening a lag.
    Ranks use only the same signal month's all-board observations. Labels are
    retained for later purged fitting, never used to calculate the features.
    """
    # Runtime imports avoid a cycle when the DC account module calls this adapter.
    from research_dc_themes import stock_liquidity, stock_momentum

    stocks = stock_momentum(monthly)
    stocks['period'] = stocks.date.dt.to_period('M')
    if stocks.groupby('period').date.nunique().gt(1).any():
        raise ValueError('Monthly stock observations must share the actual month-end signal date')
    periods = pd.period_range(stocks.period.min(), stocks.period.max(), freq='M')
    prices = stocks.assign(adjusted_close=stocks.close * stocks.adj_factor).pivot(
        index='period', columns='ts_code', values='adjusted_close').reindex(periods)
    prices.index.name = 'period'
    returns = prices.div(prices.shift(1)).sub(1)
    derived = {
        'volatility3': returns.rolling(3, min_periods=3).std(ddof=0),
        'volatility6': returns.rolling(6, min_periods=6).std(ddof=0),
        'positive_fraction6': returns.gt(0).where(returns.notna()).rolling(6, min_periods=6).mean(),
        'label_return': prices.shift(-1).div(prices).sub(1),
    }
    for name, matrix in derived.items():
        values = matrix.stack().rename(name).reset_index()
        stocks = stocks.merge(values, on=['period', 'ts_code'], how='left', validate='one_to_one')
    month_dates = stocks.groupby('period').date.first()
    stocks['label_date'] = stocks.period.map(month_dates.shift(-1))
    # Do not accidentally map the next observed month across a calendar hole.
    next_dates = pd.Series(month_dates.reindex(periods).shift(-1).to_numpy(), index=periods)
    stocks['label_date'] = pd.to_datetime(stocks.period.map(next_dates))
    stocks['signal_date'] = stocks.date
    stocks = stocks[stocks.period.ge(pd.Period('2023-01', freq='M'))].copy()
    if stocks.empty:
        return pd.DataFrame(columns=['signal_date', 'ts_code'] + FEATURE_COLUMNS + ['label_return', 'label_date'])
    signals = _dates(sorted(stocks.signal_date.unique()), 'Feature signals')
    relevant_daily = daily.copy()
    relevant_daily['date'] = _dates(relevant_daily.date, 'Daily dates')
    relevant_daily = relevant_daily[relevant_daily.date.le(signals.max())]
    stocks = stocks.merge(stock_liquidity(stocks, relevant_daily), on=['month', 'ts_code'],
                          how='left', validate='one_to_one')
    for horizon in (1, 3, 6):
        stocks[f'mom{horizon}_percentile'] = stocks.groupby('signal_date')[f'mom{horizon}'].rank(pct=True)
    valid_amount = stocks.signal_amount.where(stocks.liquidity_days.ge(10))
    stocks['amount_percentile'] = valid_amount.groupby(stocks.signal_date).rank(pct=True)
    stocks['momentum_acceleration'] = stocks.mom1 - stocks.mom3 / 3
    pressure = entry_features(relevant_daily, signals).rename(columns={'signal_day': 'signal_date'})
    pressure_cols = ['signal_date', 'ts_code', 'pressure5', 'pressure20', 'location5',
                     'amount_ratio5_vs_prior20']
    stocks = stocks.merge(pressure[pressure_cols], on=['signal_date', 'ts_code'],
                          how='left', validate='one_to_one')
    good = np.isfinite(stocks[FEATURE_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    result = stocks.loc[good, ['signal_date', 'ts_code'] + FEATURE_COLUMNS
                        + ['label_return', 'label_date']].copy()
    return result.sort_values(['signal_date', 'ts_code']).reset_index(drop=True)


def walk_forward_predictions(features, signal_dates, feature_columns=None):
    """Fit only strictly earlier labels, keeping each quarter's model frozen.

    Model objects and JSON-safe audits are attached to the returned frame's
    attrs for the cache adapter; the public columns are always OUTPUT_COLUMNS.
    """
    feature_columns = FEATURE_COLUMNS if feature_columns is None else list(feature_columns)
    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ValueError('Invalid forecast feature columns')
    required = {'signal_date', 'ts_code', 'label_return', 'label_date', *feature_columns}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f'Missing forecast feature columns: {sorted(missing)}')
    frame = features.copy()
    frame['signal_date'] = _dates(frame.signal_date, 'Feature dates')
    frame['label_date'] = pd.to_datetime(frame.label_date, errors='raise')
    if frame.duplicated(['signal_date', 'ts_code']).any():
        raise ValueError('Duplicate forecast stock/signal keys')
    if (frame.label_date.notna() & frame.label_date.le(frame.signal_date)).any():
        raise ValueError('Forecast labels must follow their feature dates')
    frame = frame[frame.ts_code.astype('string').str.match(
        r'^(?:0\d{5}\.SZ|3\d{5}\.SZ|6\d{5}\.SH)$', na=False)].copy()
    values = frame[feature_columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    frame = frame[np.isfinite(values).all(axis=1)].sort_values(['signal_date', 'ts_code'])
    frame['label_return'] = pd.to_numeric(frame.label_return, errors='coerce')
    signals = _dates(signal_dates, 'Prediction signals')
    if signals.has_duplicates:
        raise ValueError('Duplicate forecast prediction signals')
    signals = signals.sort_values()
    models, audits, blocks = {}, [], []
    current_quarter, model, metadata = None, None, None
    for signal in signals:
        quarter = str(signal.to_period('Q'))
        if quarter != current_quarter:
            current_quarter = quarter
            train = frame[frame.signal_date.lt(signal) & frame.label_date.lt(signal)
                          & np.isfinite(frame.label_return)].copy()
            label_months = train.label_date.dt.to_period('M').nunique()
            metadata = dict(quarter=quarter, model_fit_cutoff=signal.strftime('%Y-%m-%d'),
                            train_label_end=(train.label_date.max().strftime('%Y-%m-%d') if len(train) else None),
                            train_rows=len(train), train_label_months=int(label_months))
            model = None
            if label_months >= 12:
                model = HistGradientBoostingRegressor(**MODEL_PARAMS)
                model.fit(train[feature_columns].to_numpy(dtype=float),
                          train.label_return.clip(-.50, .50).to_numpy(dtype=float))
                models[quarter] = dict(model=model, **metadata)
                metadata['status'] = 'fitted'
            else:
                metadata['status'] = 'insufficient_label_months'
            audits.append(metadata.copy())
        current = frame[frame.signal_date.eq(signal)]
        if model is None or current.empty:
            continue
        prediction = model.predict(current[feature_columns].to_numpy(dtype=float))
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite forecast prediction')
        block = current[['signal_date', 'ts_code']].copy()
        block['forecast_return'] = prediction
        block['forecast_percentile'] = block.forecast_return.rank(pct=True)
        block['model_fit_cutoff'] = pd.Timestamp(metadata['model_fit_cutoff'])
        block['train_label_end'] = pd.Timestamp(metadata['train_label_end'])
        block['train_rows'] = metadata['train_rows']
        blocks.append(block[OUTPUT_COLUMNS])
    output = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=OUTPUT_COLUMNS)
    output.attrs['models'] = models
    output.attrs['model_audit'] = audits
    return output


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sha256(frame, columns):
    ordered = frame[columns].sort_values(columns[:2]).reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes()).hexdigest()


def prepare_forecasts(out, monthly, daily, signal_dates):
    """Cache model/features/predictions only when inputs and source hashes match."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    signals = _dates(signal_dates, 'Prediction signals').sort_values()
    if signals.has_duplicates:
        raise ValueError('Duplicate forecast prediction signals')
    source_paths = [Path(__file__), Path(__file__).with_name('research_dc_entry.py'),
                    Path(__file__).with_name('research_dc_themes.py')]
    inputs = {str(path.resolve()): _sha256(path) for path in source_paths}
    # Bound the in-memory inputs too: a caller cannot get a stale cache merely
    # by passing an altered frame while leaving the disk filename unchanged.
    monthly_cols = ['date', 'ts_code', 'close', 'adj_factor']
    daily_cols = ['date', 'ts_code', 'open', 'high', 'low', 'close', 'volume', 'amount']
    monthly_hash = _frame_sha256(monthly, monthly_cols)
    daily_hash = _frame_sha256(daily, daily_cols)
    specification = dict(protocol=FORECAST_PROTOCOL, sklearn_version=sklearn.__version__,
        source_inputs=inputs, monthly_frame_sha256=monthly_hash, daily_frame_sha256=daily_hash,
        signal_dates=signals.strftime('%Y-%m-%d').tolist())
    paths = {name: out / name for name in (
        'forecast_features.pkl', 'forecast_predictions.pkl', 'forecast_models.pkl',
        'forecast_model_audit.json')}
    manifest_path = out / 'forecast_feature_manifest.json'
    if manifest_path.is_file() and all(path.is_file() for path in paths.values()):
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (manifest.get('specification') == specification
                and all(manifest.get('artifacts', {}).get(name) == _sha256(path)
                        for name, path in paths.items())):
            return pd.read_pickle(paths['forecast_predictions.pkl'])
    features = forecast_features(monthly, daily)
    predictions = walk_forward_predictions(features, signals)
    models = predictions.attrs.pop('models')
    audit = predictions.attrs.pop('model_audit')
    features.to_pickle(paths['forecast_features.pkl'])
    predictions.to_pickle(paths['forecast_predictions.pkl'])
    pd.to_pickle(models, paths['forecast_models.pkl'])
    audit_payload = dict(protocol=FORECAST_PROTOCOL, quarterly_models=audit,
                         feature_rows=len(features), prediction_rows=len(predictions),
                         train_label_purge_verified=bool(predictions.empty or
                             predictions.train_label_end.lt(predictions.model_fit_cutoff).all()))
    paths['forecast_model_audit.json'].write_text(json.dumps(audit_payload, indent=2), encoding='utf-8')
    manifest = dict(status='complete', specification=specification,
                    feature_rows=len(features), prediction_rows=len(predictions),
                    artifacts={name: _sha256(path) for name, path in paths.items()})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return predictions
