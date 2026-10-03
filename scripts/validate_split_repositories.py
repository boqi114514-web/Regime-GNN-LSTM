"""Validate and explicitly inventory the three new repositories for publication."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('Stock-Opportunity-Daily', 'Stock-Selection-Monthly-ML', 'Stock-Selection-Monthly-Rules')
PATTERNS = (re.compile(r'tsr_[A-Za-z0-9_-]{20,}'),
            re.compile(r'''(?i)(?:token|api_key|api-key)\s*["']?\s*[:=]\s*["'][a-f0-9]{32,}["']'''))


def run(project, action):
    process = subprocess.run([sys.executable, 'run.py', action], cwd=project, capture_output=True)
    stdout = process.stdout.decode('utf-8', errors='replace')
    stderr = process.stderr.decode('utf-8', errors='replace')
    if process.returncode:
        print((stdout+'\n'+stderr)[-8000:], flush=True)
        raise RuntimeError(project.name+' '+action+' failed')
    count = re.search(r'Ran (\d+) tests', stderr)
    record = dict(command='python run.py '+action, exit_code=process.returncode)
    if count:
        record['tests_passed'] = int(count.group(1))
    else:
        record['output'] = stdout.strip()
    return record


def validate(name):
    project = ROOT.parent/name
    if project.name not in NAMES or not project.is_dir():
        raise ValueError('Unknown project')
    checks = [run(project, action) for action in ('check', 'test', 'verify-archive')]
    migration = json.loads((project/'migration_manifest.json').read_text(encoding='utf-8'))
    originals = {entry['path']: entry['original_sha256'] for entry in migration['copied_public_files']}
    removed = sorted(path for path in originals if not (project/path).is_file())
    modified = sorted(path for path, fingerprint in originals.items() if (project/path).is_file()
                      and hashlib.sha256((project/path).read_bytes()).hexdigest() != fingerprint)
    if any(path.startswith('results/') for path in removed+modified):
        raise ValueError('Historical account artifact changed')
    manifest_paths = [line for line in (project/'.split-publication.paths').read_text(encoding='utf-8').splitlines()
                      if (project/line).is_file()]
    manifest_paths += ['README.md', 'run.py', 'project.json', 'requirements.txt', '.gitignore', '.gitattributes',
                       'repository_validation.json', 'publication_manifest.json']
    manifest_paths += [path.relative_to(project).as_posix() for path in (project/'tests').glob('test_project*.py')]
    for folder in ('docs',):
        manifest_paths += [path.relative_to(project).as_posix() for path in (project/folder).glob('*.md')]
    allowed = sorted(set(manifest_paths))
    # New README is the maintained entry point; old dated reports preserve historical context.
    broken = []
    for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', (project/'README.md').read_text(encoding='utf-8')):
        if '://' in link or link.startswith('#'):
            continue
        target = link.split('#')[0]
        if target in {'repository_validation.json', 'publication_manifest.json'}:
            continue
        if not (project/target).exists():
            broken.append(link)
    if broken:
        raise ValueError('Broken README links: '+str(broken))
    validation = dict(date='2026-10-03', project=name, checks=checks, original_commit=migration['original_commit'],
                      modified_source_or_tests=modified, removed_unused_copied_files=removed,
                      readme_local_links='passed',
                      research_scope='Repository split, import isolation, portable roots, tests and byte-preserved account exports. No new return or training claim.',
                      original_repository_deleted=False, archival_absolute_references='Preserved verbatim; original archive still retained')
    (project/'repository_validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    entries, suspects = [], []
    for relative in allowed:
        if relative == 'publication_manifest.json':
            continue
        path = project/relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(project.resolve()):
            raise ValueError('Invalid publication path: '+relative)
        data = path.read_bytes()
        if len(data) > 48*1024**2:
            raise ValueError('Unexpected large publication file: '+relative)
        if path.suffix not in {'.pt', '.png', '.jpg'}:
            text = data.decode('utf-8-sig')
            if relative == 'tests/test_concept_gateway_probe.py':
                text = text.replace('tsr_'+'fake_secret_for_unit_test_1234', 'NONFUNCTIONAL_TEST_FIXTURE')
            if any(pattern.search(text) for pattern in PATTERNS):
                suspects.append(relative)
        entries.append(dict(path=relative, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    if suspects:
        raise ValueError('Potential credentials; content not printed: '+str(suspects))
    publication = dict(project=name, files=entries, secret_scan='passed known credential patterns',
                       excludes=['raw data', 'models directory', 'private environment files', 'pkl and large full-market intermediate tables'],
                       preserved_history='Original commit and frozen report metadata remain available in the overview repository')
    (project/'publication_manifest.json').write_text(json.dumps(publication, ensure_ascii=False, indent=2), encoding='utf-8')
    (project/'.split-publication.paths').write_text('\n'.join(allowed)+'\n', encoding='utf-8')
    print(json.dumps(dict(project=name, tests=checks[1].get('tests_passed'), publication_files=len(allowed),
                          source_changes=len(modified), removed_copied_helpers=len(removed), secrets='passed'), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', choices=NAMES)
    validate(parser.parse_args().project)
