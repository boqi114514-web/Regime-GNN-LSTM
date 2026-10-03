"""Explicit research-only Git publication inventory; never stages files itself."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MONTHLY_SCRIPTS = {
    'audit_dc_capital_priority.py', 'audit_dc_forecasts.py', 'audit_dc_theme_accounts.py',
    'audit_examples_20260930.py', 'fetch_dc_theme_research.py', 'fetch_theme_pilot.py',
    'probe_concept_gateway.py', 'probe_dragon_tiger_ma20.py',
    'audit_monthly_trail_applicability.py', 'replay_dc_eligible_hold.py',
    'build_research_release_manifest.py',
}
MONTHLY_TESTS = {
    'test_concept_gateway_probe.py', 'test_empty_action_verifiers.py',
    'test_expected_profit_optimizer.py', 'test_factor_cache_publication.py',
    'test_liquid_leaders.py', 'test_liquid_phase_router.py', 'test_public_theme_data.py',
    'test_scoped_actions.py', 'test_monthly_trail_applicability.py',
}
SECRET_PATTERNS = (
    re.compile(r'tsr_[A-Za-z0-9_-]{20,}'),
    re.compile(r'''(?i)(?:token|api_key|api-key)\s*["']?\s*[:=]\s*["'][a-f0-9]{32,}["']'''),
)


def inventory(kind):
    files = {ROOT/'README.md', ROOT/'.gitignore'}
    if kind == 'monthly':
        files.update(path for path in (ROOT/'src').glob('research_*.py')
                     if not path.name.startswith('research_daily'))
        files.update(ROOT/'src'/name for name in ('small_account_backtest.py', 'verify_small_account.py', 'verify_market_states.py'))
        files.update(ROOT/'src/data_pipeline'/name for name in ('theme_data.py', 'public_theme_data.py'))
        files.update(ROOT/'scripts'/name for name in MONTHLY_SCRIPTS)
        files.update(path for path in (ROOT/'tests').glob('*.py')
                     if path.name.startswith(('test_dc_', 'test_theme_')) or path.name in MONTHLY_TESTS)
        files.update((ROOT/'reports').glob('market_state_research_*.md'))
        files.update((ROOT/'reports').glob('monthly_*.md'))
        roots = list((ROOT/'results').glob('dc_*'))+[ROOT/'results/liquid_leader_research']
        for folder in roots:
            if folder.is_dir():
                files.update(path for path in folder.iterdir() if path.is_file()
                             and path.suffix in ('.json', '.csv', '.md', '.txt')
                             and path.name != 'capital_priority.csv')
    else:
        files.update(ROOT/'scripts'/name for name in ('build_research_release_manifest.py', 'complete_suspension_checks.py'))
        files.add(ROOT/'tests/test_complete_suspension_checks.py')
        files.update((ROOT/'src').glob('research_daily*.py'))
        files.update((ROOT/'tests').glob('test_daily*.py'))
        files.update((ROOT/'scripts').glob('*daily_opportunity*.py'))
        files.update((ROOT/'reports').glob('daily_opportunity*.md'))
        files.update((ROOT/'reports').glob('monthly_daily_architecture*.md'))
        for folder in (ROOT/'results').glob('daily_opportunity*'):
            for path in folder.rglob('*'):
                if not path.is_file() or path.suffix not in ('.json', '.csv', '.md', '.pt'):
                    continue
                if any(part in ('scoped_actions', 'daily_limits', 'suspension_events') for part in path.parts):
                    continue
                if path.name in ('predictions.csv', 'liquidity_checks.csv'):
                    continue
                files.add(path)
    missing = [str(path.relative_to(ROOT)) for path in files if not path.is_file()]
    if missing:
        raise ValueError('Missing selected release files: '+str(missing))
    return sorted(files)


def build(kind):
    files = inventory(kind)
    entries, suspect = [], []
    for path in files:
        payload = path.read_bytes()
        relative = path.relative_to(ROOT).as_posix()
        if len(payload) > 48*1024*1024:
            raise ValueError('Unexpected large publication file: '+relative)
        if path.suffix != '.pt':
            text = payload.decode('utf-8-sig')
            # This explicit, nonfunctional unit-test fixture exercises redaction.
            # No production credential or general test-file exemption is allowed.
            if relative == 'tests/test_concept_gateway_probe.py':
                text = text.replace('tsr_'+'fake_secret_for_unit_test_1234', 'KNOWN_NONFUNCTIONAL_FIXTURE')
            if any(pattern.search(text) for pattern in SECRET_PATTERNS):
                suspect.append(relative)
        entries.append(dict(path=relative, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
    if suspect:
        raise ValueError('Potential credential detected; content suppressed: '+str(suspect))
    destination = ROOT/'reports'/f'research_release_20261003_{kind}.json'
    record = dict(kind=kind, files=entries, secret_scan='passed known-key-shape and token-assignment checks',
                  exclusions=['raw market/API caches and all pkl', 'large full-stock forecast/eligibility CSVs',
                              'regenerable capital_priority.csv diagnostics', 'unrelated personal documents/scripts'],
                  local_verification='Exact recorded hashes use original local paths; published outputs do not replace omitted vendor caches')
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    specs = [item['path'] for item in entries]+[destination.relative_to(ROOT).as_posix()]
    pathspec = ROOT/f'.research-release-{kind}.paths'
    pathspec.write_text('\n'.join(specs)+'\n', encoding='utf-8')
    print(json.dumps(dict(kind=kind, files=len(specs), megabytes=round(sum(x['bytes'] for x in entries)/1024**2, 2),
                          secret_scan='passed', pathspec=pathspec.name), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('monthly', 'daily'))
    build(parser.parse_args().kind)
