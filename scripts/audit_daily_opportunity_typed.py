"""Typed artifact adapter for the unchanged independent account auditor.

Future-label tensors are fingerprinted model evidence, not vendor limit tables.
All original status hashes remain checked by the original auditor. Only its
official-limit discovery receives the subset of DataFrame artifacts.
"""
import json
from pathlib import Path

import pandas as pd

import audit_daily_opportunity_account as original


def dataframe_limit_status(status):
    filtered = dict(status)
    retained = {}
    for name, expected in status.get('source_sha256', {}).items():
        path = Path(name)
        if path.suffix != '.pkl':
            continue
        original.check(path.exists() and original.sha(path) == expected,
                       'Source fingerprint changed before typed discovery: '+path.name)
        raw = pd.read_pickle(path)
        if isinstance(raw, pd.DataFrame):
            retained[name] = expected
    filtered['source_sha256'] = retained
    return filtered


def audit_account(out, daily, predictions, sessions, output_dir=None):
    out = Path(out)
    status = json.loads((out/'daily_run_status.json').read_text(encoding='utf-8'))
    filtered = dataframe_limit_status(status)
    needed = set()
    for name in ('trades', 'execution_checks'):
        records = pd.read_csv(out/f'{name}.csv')
        needed.update(zip(pd.to_datetime(records.date), records.code))
    limits, fingerprints = original.load_exact_limits(out, filtered, needed)
    checks = original.audit_account(out, daily, predictions, sessions,
                                    output_dir=output_dir, limit_frames=limits)
    checks['source_sha256'].update(fingerprints)
    checks['source_sha256'][str(Path(__file__).resolve())] = original.sha(__file__)
    checks['typed_model_artifacts_separated_from_official_limits'] = True
    destination = Path(output_dir) if output_dir else out/'independent_audit'
    (destination/'independent_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    return checks
