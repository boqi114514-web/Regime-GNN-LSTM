"""One fixed extension veto layered on the frozen signal-day amount filter.

Known complete-calendar return20 above 50% blocks a new purchase. Unknown
return20 is not treated as zero or rejected: the prior missingness policy is
preserved. The threshold is inherited as a hypothesis from the older monthly
moderate-extension experiment, not claimed equivalent to calendar-month mom1.
"""
import argparse
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from research_daily_graph_data import publish_manifest, sha256
import research_daily_opportunity_liquid_filter as liquid


MAXIMUM_KNOWN_RETURN20 = .50
BASE_CHECKS = liquid.liquidity_checks
PROTOCOL = dict(copy.deepcopy(liquid.PROTOCOL),
    hypothesis='Current amount percentile >= .95 plus a fixed known return20 <= .50 extension veto',
    extension_input='Signal-day complete-market-calendar adjusted return20 and its explicit availability mask',
    extension_unknown='Unknown return20 remains unknown and is not vetoed; no future quote or compressed-stock-calendar fallback',
    extension_threshold_origin='Reuse the prior monthly moderate-extension .50 hypothesis; twenty sessions is not calendar-month mom1; no threshold scan',
    maximum_known_return20=MAXIMUM_KNOWN_RETURN20,
    order='Current known amount gate and extension veto, then positive native utility and mainboard top 40, then exact signal-day official band >= .075',
)


def extension_checks(predictions, store):
    checks = BASE_CHECKS(predictions, store)
    names = {name:i for i, name in enumerate(store.feature_names)}
    if not {'return20', 'available__return20'} <= set(names):
        raise ValueError('Explicit signal-day return20 and availability required')
    rows = pd.DatetimeIndex(store.dates).get_indexer(pd.to_datetime(predictions.signal_date))
    cols = pd.Index(store.stock_codes).get_indexer(predictions.ts_code)
    values = store.features[rows, cols, names['return20']]
    mask = store.features[rows, cols, names['available__return20']]
    if not np.isin(mask, (0., 1.)).all():
        raise ValueError('Return20 availability must be an explicit binary mask')
    known = mask == 1
    if not np.isfinite(values[known]).all():
        raise ValueError('Known return20 must be finite')
    checks['known_return20'] = known
    checks['return20'] = np.where(known, values, np.nan)
    threshold = np.asarray(MAXIMUM_KNOWN_RETURN20, dtype=values.dtype)
    checks['not_extended'] = ~known | (values <= threshold)
    checks['in_model_pool'] = False
    candidates = checks[checks.amount_eligible & checks.not_extended
                        & checks.mainboard & checks.positive_utility]
    selected = candidates.sort_values(
        ['signal_date', 'native_utility', 'ts_code'], ascending=[True, False, True], kind='stable'
    ).groupby('signal_date', sort=False).head(liquid.CANDIDATE_LIMIT).index
    checks.loc[selected, 'in_model_pool'] = True
    return checks


@contextmanager
def extension_variant():
    previous_checks, previous_protocol = liquid.liquidity_checks, liquid.PROTOCOL
    liquid.liquidity_checks, liquid.PROTOCOL = extension_checks, PROTOCOL
    try:
        yield
    finally:
        liquid.liquidity_checks, liquid.PROTOCOL = previous_checks, previous_protocol


def filter_extension_predictions(source, out, pro=None, *, offline=False, cache_roots=()):
    out = Path(out)
    with extension_variant():
        result = liquid.filter_liquid_predictions(source, out, pro, offline=offline,
                                                  cache_roots=cache_roots)
    protocol_path = out/'experiment_protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    protocol['source_sha256'][Path(__file__).name] = sha256(__file__)
    publish_manifest(protocol, protocol_path)
    checks = pd.read_csv(out/'liquidity_checks.csv')
    publish_manifest(dict(status='complete', protocol=PROTOCOL,
                          maximum_known_return20=MAXIMUM_KNOWN_RETURN20,
                          source_sha256=sha256(__file__),
                          liquidity_manifest_sha256=sha256(out/'liquidity_manifest.json'),
                          checks_sha256=sha256(out/'liquidity_checks.csv'),
                          feature_store_sha256=sha256(out/'feature_store.pkl'),
                          rows=len(result), known_return20=int(checks.known_return20.sum()),
                          vetoed=int((checks.known_return20 & ~checks.not_extended).sum())),
                     out/'extension_manifest.json')
    prediction_path = out/'prediction_manifest.json'
    manifest = json.loads(prediction_path.read_text(encoding='utf-8'))
    for name in ('experiment_protocol.json', 'extension_manifest.json'):
        manifest['artifacts'][name] = sha256(out/name)
    publish_manifest(manifest, prediction_path)
    return result


def validate_extension_inputs(out):
    out = Path(out)
    manifest = json.loads((out/'extension_manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('status') != 'complete' or manifest.get('protocol') != PROTOCOL
            or manifest.get('maximum_known_return20') != MAXIMUM_KNOWN_RETURN20
            or manifest.get('source_sha256') != sha256(__file__)
            or manifest.get('liquidity_manifest_sha256') != sha256(out/'liquidity_manifest.json')
            or manifest.get('checks_sha256') != sha256(out/'liquidity_checks.csv')
            or manifest.get('feature_store_sha256') != sha256(out/'feature_store.pkl')):
        raise ValueError('Incomplete or changed extension provenance')
    protocol = json.loads((out/'experiment_protocol.json').read_text(encoding='utf-8'))
    prediction = json.loads((out/'prediction_manifest.json').read_text(encoding='utf-8'))
    if (protocol.get('source_sha256', {}).get(Path(__file__).name) != sha256(__file__)
            or prediction.get('artifacts', {}).get('extension_manifest.json') != sha256(out/'extension_manifest.json')):
        raise ValueError('Extension source or prediction-manifest binding missing')
    with extension_variant():
        checked = liquid.validate_liquidity_inputs(out)
    checks = pd.read_csv(out/'liquidity_checks.csv')
    if (manifest.get('rows') != len(checks)
            or manifest.get('known_return20') != int(checks.known_return20.sum())
            or manifest.get('vetoed') != int((checks.known_return20 & ~checks.not_extended).sum())):
        raise ValueError('Extension manifest coverage counts differ')
    return dict(checked, extension_manifest=manifest)


def run_extension_account(source, out, inputs, offline=False):
    from research_daily_opportunity_liquid_runner import run_liquid_account
    source, out = Path(source), Path(out)
    validate_extension_inputs(source)
    with extension_variant():
        result = run_liquid_account(source, out, inputs, offline)
    path = out/'account/daily_run_status.json'
    status = json.loads(path.read_text(encoding='utf-8'))
    for file in (Path(__file__), source/'extension_manifest.json', source/'prediction_manifest.json'):
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
        from research_daily_opportunity_liquid_runner import ExactBatchGateway, cache_roots
        pro = None if args.offline else ExactBatchGateway(os.environ['TUSHARE_API_KEY'])
        filter_extension_predictions(args.source, args.out, pro, offline=args.offline,
                                     cache_roots=cache_roots())
        checked = validate_extension_inputs(args.out)
        print('verified extension eligibility', checked['manifest']['accepted'], flush=True)
    else:
        run_extension_account(args.source, args.out, args.inputs, args.offline)
