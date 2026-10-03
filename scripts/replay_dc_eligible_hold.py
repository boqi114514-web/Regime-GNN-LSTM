"""Exact offline replay of the isolated monthly eligible-hold account."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from research_dc_eligible_hold import run, audit_account, publish_manifest, sha256


def compare(source, replay):
    if source.stat().st_size <= 3 or replay.stat().st_size <= 3:
        if source.read_bytes() != replay.read_bytes():
            raise AssertionError('Empty output mismatch: '+source.name)
        return dict(rows=0, exact_cells_equal=True, source_sha256=sha256(source), replay_sha256=sha256(replay))
    a, b = pd.read_csv(source), pd.read_csv(replay)
    if list(a.columns) != list(b.columns):
        raise AssertionError('CSV schema mismatch: '+source.name)
    columns = list(a.columns)
    a = a.sort_values(columns, kind='stable', na_position='last').reset_index(drop=True)
    b = b.sort_values(columns, kind='stable', na_position='last').reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=True, check_dtype=True)
    return dict(rows=len(a), exact_cells_equal=True,
                canonical_sha256=hashlib.sha256(a.to_csv(index=False).encode('utf-8')).hexdigest(),
                source_sha256=sha256(source), replay_sha256=sha256(replay))


def replay(source, out, verify_only=False):
    source, out = Path(source), Path(out)
    if source.resolve() == out.resolve():
        raise ValueError('Replay needs a separate destination')
    original_protocol = json.loads((source/'eligible_hold_protocol.json').read_text(encoding='utf-8'))
    if verify_only:
        proof = audit_account(out)
    else:
        proof = run(Path(original_protocol['frozen_source']), out, offline=True)
    original = {path.name:path for path in source.glob('*.csv')}
    repeated = {path.name:path for path in out.glob('*.csv')}
    if set(original) != set(repeated):
        raise AssertionError('Replay output CSV set changed')
    checks = {name:compare(path, repeated[name]) for name,path in original.items()}
    result = dict(passed=True, offline=True, metrics=proof['metrics'], csv_comparisons=checks,
                  source=str(source.resolve()), replay=str(out.resolve()),
                  script_sha256=sha256(__file__),
                  source_audit_sha256=sha256(source/'eligible_hold_independent_checks.json'),
                  replay_audit_sha256=sha256(out/'eligible_hold_independent_checks.json'))
    publish_manifest(result, out/'exact_replay_checks.json')
    print('Monthly exact offline replay passed:', len(checks), 'CSV files')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/dc_member_relative_eligible_hold_20261003'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    replay(args.source, args.out, args.verify_only)
