"""Fixed signal-day top-five-percent amount hypothesis for native forecasts.

This is a research eligibility hypothesis, not a capital-capacity requirement.
Training nodes and all native predictions are preserved. Current observable
amount eligibility precedes positive-utility/mainboard top-40 selection and
the exact official signal-day risk-band check. No execution-day factor enters.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from research_daily_graph_data import publish_frame, publish_manifest, sha256
from research_daily_opportunity import validate_predictions
from research_daily_opportunity_band_filter import validate_band_inputs
from research_daily_opportunity_execution import OfficialLimits, _mainboard
from research_daily_opportunity_leadership import leadership_variant, validate_leadership


AMOUNT_THRESHOLD = .95
CANDIDATE_LIMIT = 40
MINIMUM_BAND_FRACTION = .075
PROTOCOL = dict(
    hypothesis='Fixed current signal-day all-market amount percentile >= .95',
    purpose='Research stock leadership/liquidity selection; not a capital-capacity necessity',
    inputs='V3 frozen amount_percentile, explicit availability and current feature eligibility only',
    order='Current known amount gate, then positive native utility and mainboard top 40, then exact signal-day official band >= .075',
    precision='Threshold represented in the stored factor dtype; no decimal rounding of percentiles',
    retained_positions='Unchanged; eligibility controls new purchases, never a forced-sale rule',
    predictions='All source rows and all native forecast columns retained without changes',
)


def liquidity_checks(predictions, store):
    """Recompute eligibility using only each row's exact signal-day features."""
    required = {'signal_date', 'ts_code', 'utility'}
    if not required <= set(predictions) or predictions.duplicated(['signal_date', 'ts_code']).any():
        raise ValueError('Unique signal-date stock predictions required')
    names = {name:i for i, name in enumerate(store.feature_names)}
    if not {'amount_percentile', 'available__amount_percentile'} <= set(names):
        raise ValueError('Explicit current amount percentile and availability required')
    dates = pd.DatetimeIndex(store.dates)
    codes = pd.Index(store.stock_codes)
    if dates.has_duplicates or codes.has_duplicates:
        raise ValueError('Feature store has duplicate dates or stocks')
    rows = dates.get_indexer(pd.to_datetime(predictions.signal_date))
    columns = codes.get_indexer(predictions.ts_code)
    if (rows < 0).any() or (columns < 0).any():
        raise ValueError('Prediction has no exact signal-day feature observation')
    values = store.features[rows, columns, names['amount_percentile']]
    available = store.features[rows, columns, names['available__amount_percentile']]
    current = np.asarray(store.valid_features[rows, columns], dtype=bool)
    if not np.isin(available, (0., 1.)).all():
        raise ValueError('Amount availability must be an explicit binary mask')
    known = available == 1
    if not np.isfinite(values[known]).all() or ((values[known] <= 0) | (values[known] > 1)).any():
        raise ValueError('Observed amount percentile must be finite in (0,1]')
    utility = pd.to_numeric(predictions.utility, errors='raise').to_numpy()
    if not np.isfinite(utility).all():
        raise ValueError('Native model utility must be finite')
    threshold = np.asarray(AMOUNT_THRESHOLD, dtype=values.dtype)
    amount_pass = current & known & (values >= threshold)
    checks = pd.DataFrame(dict(
        signal_date=pd.to_datetime(predictions.signal_date).to_numpy(),
        ts_code=predictions.ts_code.to_numpy(),
        amount_percentile=values, amount_available=known,
        current_feature_eligible=current, amount_eligible=amount_pass,
        mainboard=predictions.ts_code.map(_mainboard).to_numpy(),
        positive_utility=utility > 0, native_utility=utility,
    ))
    checks['in_model_pool'] = False
    candidates = checks[checks.amount_eligible & checks.mainboard & checks.positive_utility]
    selected = candidates.sort_values(
        ['signal_date', 'native_utility', 'ts_code'], ascending=[True, False, True], kind='stable'
    ).groupby('signal_date', sort=False).head(CANDIDATE_LIMIT).index
    checks.loc[selected, 'in_model_pool'] = True
    return checks


def filter_liquid_predictions(source, out, pro=None, *, offline=False, cache_roots=()):
    source, out = Path(source), Path(out)
    if source.resolve() == out.resolve() or (out/'prediction_manifest.json').exists():
        raise ValueError('Use a separate output folder without completed predictions')
    with leadership_variant():
        predictions = validate_predictions(source)
        store, _ = validate_leadership(source)
    if 'eligible' in predictions:
        raise ValueError('Start from unfiltered native model predictions')
    checks = liquidity_checks(predictions, store)
    del store
    out.mkdir(parents=True, exist_ok=True)
    for name in ('feature_store.pkl', 'feature_manifest.json', 'frozen_edges.pkl', 'fit_audits.json'):
        shutil.copy2(source/name, out/name)
    shutil.copytree(source/'checkpoints', out/'checkpoints', dirs_exist_ok=True)
    limits = OfficialLimits(out, pro=pro, offline=offline, cache_roots=cache_roots)
    result = predictions.copy()
    eligibility = np.zeros(len(result), dtype=bool)
    band_rows = []
    for number, (day, pool) in enumerate(checks[checks.in_model_pool].groupby('signal_date', sort=True)):
        for index, row in pool.iterrows():
            official = limits(day, row.ts_code)
            up, down = float(official.up_limit), float(official.down_limit)
            if not np.isfinite([up, down]).all() or not 0 < down < up:
                raise ValueError('Invalid exact official signal-day band')
            fraction = (up-down)/(up+down)
            eligible = fraction >= MINIMUM_BAND_FRACTION
            eligibility[index] = eligible
            band_rows.append(dict(signal_date=day, ts_code=row.ts_code, up_limit=up,
                                  down_limit=down, official_band_fraction=fraction, eligible=eligible))
        if number % 20 == 0:
            print('liquid native filter', number+1, 'sessions', 'band checks', len(band_rows), flush=True)
    result['eligible'] = eligibility
    checks['eligible'] = eligibility
    band = pd.DataFrame(band_rows, columns=['signal_date', 'ts_code', 'up_limit', 'down_limit',
                                          'official_band_fraction', 'eligible'])
    band.to_csv(out/'prior_band_checks.csv', index=False)
    checks.to_csv(out/'liquidity_checks.csv', index=False)
    publish_frame(result, out/'predictions.pkl')
    result.to_csv(out/'predictions.csv', index=False, encoding='utf-8-sig')
    protocol = json.loads((source/'experiment_protocol.json').read_text(encoding='utf-8'))
    protocol['buy_filter'] = dict(PROTOCOL, amount_threshold=AMOUNT_THRESHOLD,
                                candidate_limit=CANDIDATE_LIMIT, minimum_band_fraction=MINIMUM_BAND_FRACTION)
    protocol['source_sha256'][Path(__file__).name] = sha256(__file__)
    publish_manifest(protocol, out/'experiment_protocol.json')
    publish_manifest(dict(status='complete', checks=len(band), accepted=int(eligibility.sum()),
                          minimum_fraction=MINIMUM_BAND_FRACTION, source_sha256=limits.sources,
                          checks_sha256=sha256(out/'prior_band_checks.csv')), out/'prior_band_manifest.json')
    source_artifacts = {name:dict(path=str((source/name).resolve()), sha256=sha256(source/name))
                        for name in ('predictions.pkl', 'prediction_manifest.json', 'feature_store.pkl',
                                     'feature_manifest.json')}
    publish_manifest(dict(status='complete', protocol=PROTOCOL, rows=len(result),
                          amount_threshold=AMOUNT_THRESHOLD, candidate_limit=CANDIDATE_LIMIT,
                          minimum_band_fraction=MINIMUM_BAND_FRACTION,
                          source_directory=str(source.resolve()), source_artifacts=source_artifacts,
                          feature_store_sha256=sha256(out/'feature_store.pkl'),
                          checks_sha256=sha256(out/'liquidity_checks.csv'), source_sha256=sha256(__file__),
                          amount_eligible=int(checks.amount_eligible.sum()),
                          model_pool=int(checks.in_model_pool.sum()), accepted=int(eligibility.sum())),
                     out/'liquidity_manifest.json')
    artifacts = {name:sha256(out/name) for name in (
        'predictions.pkl', 'fit_audits.json', 'experiment_protocol.json', 'feature_manifest.json',
        'prior_band_checks.csv', 'prior_band_manifest.json', 'liquidity_checks.csv', 'liquidity_manifest.json')}
    publish_manifest(dict(status='complete', rows=len(result), sessions=int(result.signal_date.nunique()),
                          artifacts=artifacts), out/'prediction_manifest.json')
    return result


def validate_liquidity_inputs(out):
    """Independently recalculate complete row preservation and pre-pool gating."""
    out = Path(out)
    manifest = json.loads((out/'liquidity_manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('status') != 'complete' or manifest.get('protocol') != PROTOCOL
            or manifest.get('amount_threshold') != AMOUNT_THRESHOLD
            or manifest.get('candidate_limit') != CANDIDATE_LIMIT
            or manifest.get('minimum_band_fraction') != MINIMUM_BAND_FRACTION
            or manifest.get('source_sha256') != sha256(__file__)
            or manifest.get('checks_sha256') != sha256(out/'liquidity_checks.csv')
            or manifest.get('feature_store_sha256') != sha256(out/'feature_store.pkl')):
        raise ValueError('Incomplete or changed liquidity provenance')
    for item in manifest['source_artifacts'].values():
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Original liquidity-input artifact changed')
    with leadership_variant():
        source = validate_predictions(Path(manifest['source_directory']))
        result = validate_predictions(out)
        store, _ = validate_leadership(out)
        band_manifest = validate_band_inputs(out)
    if 'eligible' in source or len(result) != manifest['rows']:
        raise ValueError('Source is already filtered or output coverage changed')
    try:
        pd.testing.assert_frame_equal(result.drop(columns='eligible'), source, check_exact=True)
    except AssertionError as exc:
        raise ValueError('Native forecast rows or values changed') from exc
    expected = liquidity_checks(source, store)
    bands = pd.read_csv(out/'prior_band_checks.csv', parse_dates=['signal_date'])
    actual_pool = bands[['signal_date', 'ts_code']].sort_values(['signal_date', 'ts_code']).reset_index(drop=True)
    expected_pool = expected.loc[expected.in_model_pool, ['signal_date', 'ts_code']].sort_values(
        ['signal_date', 'ts_code']).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(actual_pool, expected_pool, check_exact=True)
    except AssertionError as exc:
        raise ValueError('Official bands did not cover exactly the liquidity-first model pool') from exc
    accepted = set(map(tuple, bands.loc[bands.eligible, ['signal_date', 'ts_code']].to_numpy()))
    expected['eligible'] = [(day, code) in accepted for day, code in
                            zip(expected.signal_date, expected.ts_code)]
    recorded = pd.read_csv(out/'liquidity_checks.csv', parse_dates=['signal_date'])
    try:
        pd.testing.assert_frame_equal(recorded, expected, check_dtype=False, check_exact=False,
                                      rtol=1e-7, atol=1e-9)
        np.testing.assert_array_equal(result.eligible.to_numpy(), expected.eligible.to_numpy())
    except AssertionError as exc:
        raise ValueError('Liquidity checks or final eligibility differ from causal recomputation') from exc
    if (manifest['amount_eligible'] != int(expected.amount_eligible.sum())
            or manifest['model_pool'] != int(expected.in_model_pool.sum())
            or manifest['accepted'] != int(expected.eligible.sum())):
        raise ValueError('Liquidity manifest coverage counts differ')
    return dict(manifest=manifest, band_manifest=band_manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    from research_daily_opportunity_runner import BackoffGateway, limits_caches
    gateway = None if args.offline else BackoffGateway(os.environ['TUSHARE_API_KEY'])
    filter_liquid_predictions(args.source, args.out, gateway, offline=args.offline,
                              cache_roots=limits_caches()+[args.source/'account/daily_limits',
                                                           args.source/'daily_limits'])
    validate_liquidity_inputs(args.out)
