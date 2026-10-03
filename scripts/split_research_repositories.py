"""Copy three scoped research projects without moving or deleting the archive.

Only Git-tracked source and published artifacts enter the publication inventory.
Raw data and matching local caches are copied separately and never staged here.
Run once into absent sibling directories; original audit manifests stay unchanged.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
NAMES = {'daily': 'Stock-Opportunity-Daily', 'monthly_ml': 'Stock-Selection-Monthly-ML',
         'monthly_rules': 'Stock-Selection-Monthly-Rules'}
ML_DC = ('dc_context_', 'dc_local_', 'dc_forecast_', 'dc_expected_profit_',
         'dc_peer_forecast_', 'dc_rally_')
ML_MODULES = {'research_dc_forecast', 'research_dc_peer_forecast',
              'research_dc_context_ranker', 'research_dc_rally_classifier',
              'research_learned_leaders', 'price_risk_pipeline', 'research_improvements',
              'verify_improvement_research', 'stock_execution_research'}
COMMON_TESTS = {'test_budget_portfolio.py', 'test_execution_data.py',
                'test_scoped_actions.py', 'test_small_account_backtest.py',
                'test_tushare_gateway.py'}
RULE_TESTS = {'test_dc_eligible_hold.py', 'test_dc_specialist.py', 'test_dc_leader_allocation.py',
              'test_dc_capital_priority.py', 'test_dc_theme_collection.py', 'test_dc_themes.py',
              'test_dc_entry.py', 'test_dc_entry_policy.py', 'test_dc_weekly_execution.py',
              'test_monthly_trail_applicability.py', 'test_theme_affinity.py', 'test_theme_data.py',
              'test_public_theme_data.py', 'test_theme_pilot.py', 'test_concept_gateway_probe.py'}
RULE_SCRIPTS = {'audit_dc_capital_priority.py', 'audit_dc_theme_accounts.py',
                'audit_monthly_trail_applicability.py', 'fetch_dc_theme_research.py',
                'fetch_theme_pilot.py', 'probe_concept_gateway.py', 'replay_dc_eligible_hold.py'}
ML_TESTS = {'test_original_architecture.py', 'test_fixed_weight_market_factors.py',
            'test_industry_monthly_integrity.py', 'test_market_return_alignment.py',
            'test_regime_and_signal_timing.py', 'test_research_improvements.py',
            'test_price_risk_pipeline.py', 'test_stock_execution_research.py',
            'test_learned_leaders.py', 'test_dc_forecast.py', 'test_dc_context_ranker.py',
            'test_dc_peer_context.py', 'test_dc_peer_forecast_adapter.py', 'test_dc_rally_classifier.py',
            'test_market_state_research.py', 'test_structural_router.py', 'test_fine_industry.py',
            'test_peer_graph.py', 'test_daily_reentry.py', 'test_quality_floor.py',
            'test_trend_features.py', 'test_leadership_research.py', 'test_liquid_leaders.py',
            'test_liquid_phase_router.py', 'test_expected_profit_optimizer.py',
            'test_factor_cache_publication.py', 'test_verify_market_states.py', 'test_empty_action_verifiers.py'}


def tracked_paths():
    result = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, check=True, capture_output=True)
    return {Path(value.decode('utf-8')) for value in result.stdout.split(b'\0') if value}


def source_seeds(kind, tracked):
    selected = set()
    for path in tracked:
        name = path.name
        if path.parts[0] == 'tests':
            if name in COMMON_TESTS or (kind == 'daily' and (name.startswith('test_daily_opportunity')
                    or name in {'test_daily_graph_data.py', 'test_complete_suspension_checks.py'})) \
                    or (kind == 'monthly_rules' and name in RULE_TESTS) \
                    or (kind == 'monthly_ml' and name in ML_TESTS):
                selected.add(path)
        if path.parts[0] == 'scripts':
            if (kind == 'daily' and ('daily_opportunity' in name or name == 'complete_suspension_checks.py')) \
                    or (kind == 'monthly_rules' and name in RULE_SCRIPTS) \
                    or (kind == 'monthly_ml' and name in {'audit_dc_forecasts.py', 'fetch_dc_theme_research.py'}):
                selected.add(path)
        if path.parts[0] == 'src':
            if kind == 'daily' and (name.startswith('research_daily_opportunity') or name == 'research_daily_graph_data.py'):
                selected.add(path)
            if kind == 'monthly_rules' and name in {'research_dc_eligible_hold.py', 'research_dc_themes.py',
                    'research_dc_specialist.py', 'research_dc_leader_allocation.py'}:
                selected.add(path)
            if kind == 'monthly_ml' and len(path.parts) == 2 and (name.startswith(('s0_', 's1_', 's2_',
                    's3_', 's4_', 's5_', 's6_', 's7_')) or path.stem in ML_MODULES
                    or name in {'run_all.py', 'run_original_architecture.py', 'compare_fixed_weights.py',
                                'research_market_states.py', 'research_dc_themes.py', 'verify_small_account.py'}):
                selected.add(path)
            if kind == 'monthly_ml' and path.parts[:2] == ('src', 'data_pipeline'):
                selected.add(path)
    return selected


def dependencies(seeds, tracked, kind):
    modules = {}
    for path in tracked:
        if path.suffix == '.py' and path.parts[0] in {'src', 'scripts'}:
            parts = path.with_suffix('').parts[1:]
            if parts[-1] == '__init__':
                parts = parts[:-1]
            modules['.'.join(parts)] = path
    selected, todo = set(seeds), list(seeds)
    while todo:
        path = todo.pop()
        if path.suffix != '.py':
            continue
        tree = ast.parse((ROOT/path).read_text(encoding='utf-8-sig'))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    prefix = list(path.with_suffix('').parts[1:-node.level])
                    base = '.'.join(prefix + ([base] if base else []))
                imports.append(base)
                imports += [base+'.'+alias.name for alias in node.names if alias.name != '*']
        for module in imports:
            if kind != 'monthly_ml' and module in ML_MODULES:
                continue  # New copies are decoupled explicitly after materialization.
            dependency = modules.get(module)
            if dependency is not None and dependency not in selected:
                selected.add(dependency)
                todo.append(dependency)
        for parent in path.parents:
            if str(parent) == '.':
                break
            init = parent/'__init__.py'
            if init in tracked and init not in selected:
                selected.add(init)
                todo.append(init)
    return selected


def result_owner(folder):
    if folder.startswith('daily_opportunity'):
        return 'daily'
    if folder.startswith(ML_DC):
        return 'monthly_ml'
    if folder.startswith('dc_'):
        return 'monthly_rules'
    if folder in {'original_architecture', 'price_risk_pipeline', 'stock_execution_research',
                  'improvement_research', 'leadership_research', 'market_state_research',
                  'market_state_board_complete', 'liquid_leader_research', 'l2'}:
        return 'monthly_ml'
    return None


def selected_public(kind, tracked):
    seeds = source_seeds(kind, tracked)
    files = dependencies(seeds, tracked, kind)
    for path in tracked:
        if path.parts[0] == 'results':
            owner = 'monthly_ml' if len(path.parts) == 2 else result_owner(path.parts[1])
            if owner == kind:
                files.add(path)
        if path.parts[0] == 'reports':
            name = path.name
            if (kind == 'daily' and name.startswith('daily_opportunity')) \
                    or (kind == 'monthly_rules' and (name.startswith('monthly_eligible')
                    or name.startswith('market_state_research_2026-10-'))) \
                    or (kind == 'monthly_ml' and not name.startswith(('daily_', 'monthly_', 'research_release_',
                          'market_state_research_2026-10-'))):
                files.add(path)
    return seeds, files


def copy_file(relative, destination):
    target = destination/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT/relative, target)


def build(kind):
    destination = ROOT.parent/NAMES[kind]
    if destination.exists():
        raise ValueError('Refusing to overwrite existing destination: '+str(destination))
    destination.mkdir()
    tracked = tracked_paths()
    seeds, files = selected_public(kind, tracked)
    for relative in sorted(files):
        copy_file(relative, destination)
    # Local raw/processed data are independent copies, not links into another repo.
    local_count, local_bytes = 0, 0
    for folder in ('data/raw', 'data/processed', 'data/cache'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and not path.is_symlink():
                copy_file(path.relative_to(ROOT), destination)
                local_count += 1
                local_bytes += path.stat().st_size
    if kind == 'monthly_ml':
        for path in (ROOT/'models').rglob('*'):
            if path.is_file() and not path.is_symlink():
                copy_file(path.relative_to(ROOT), destination)
                local_count += 1
                local_bytes += path.stat().st_size
    for folder in (ROOT/'results').iterdir():
        if folder.is_dir() and result_owner(folder.name) == kind:
            for path in folder.rglob('*'):
                if path.is_file() and not path.is_symlink() and path.relative_to(ROOT) not in files:
                    copy_file(path.relative_to(ROOT), destination)
                    local_count += 1
                    local_bytes += path.stat().st_size
        elif kind == 'monthly_ml' and folder.is_file() and folder.suffix == '.pkl' \
                and not folder.name.startswith(('_gat', '_mispricing', '_ml_valuation')):
            copy_file(folder.relative_to(ROOT), destination)
            local_count += 1
            local_bytes += folder.stat().st_size
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest = dict(project=NAMES[kind], kind=kind, original_repository='https://github.com/boqi114514-web/Regime-GNN-LSTM',
                    original_commit=revision, original_preserved=True,
                    source_seeds=sorted(p.as_posix() for p in seeds),
                    copied_public_files=[dict(path=p.as_posix(), original_sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in sorted(files)],
                    independent_local_cache_files=local_count, independent_local_cache_bytes=local_bytes,
                    provenance='Original result manifests copied verbatim, including historical absolute paths. No audit proof is rewritten or re-issued by a move.',
                    publication='Only explicit public files plus new project documentation/config/tests are eligible; raw data and local caches remain private to disk.')
    (destination/'migration_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    public = [p.as_posix() for p in sorted(files)] + ['migration_manifest.json']
    (destination/'.split-publication.paths').write_text('\n'.join(public)+'\n', encoding='utf-8')
    print(json.dumps(dict(project=NAMES[kind], public_files=len(files), local_cache_files=local_count,
                          local_cache_gib=round(local_bytes/1024**3, 2)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=list(NAMES))
    build(parser.parse_args().kind)
