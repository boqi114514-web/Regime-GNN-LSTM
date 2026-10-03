"""Scoped entry point: info/check/test are safe; module is an explicit action."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def settings():
    return json.loads((ROOT/'project.json').read_text(encoding='utf-8'))


def environment():
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT/'src')+os.pathsep+str(ROOT/'scripts')
    env.setdefault('OMP_NUM_THREADS', '4')
    return env


def archive_check():
    migration = json.loads((ROOT/'migration_manifest.json').read_text(encoding='utf-8'))
    checked, failures = 0, []
    for item in migration['copied_public_files']:
        if not item['path'].startswith('results/'):
            continue
        path = ROOT/item['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['original_sha256']:
            failures.append(item['path'])
        checked += 1
    print(json.dumps(dict(archived_result_files_checked=checked, failed_files=failures,
                         scope='Byte preservation only, not a new training or account replay'), ensure_ascii=False))
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('info', 'check', 'test', 'verify-archive', 'module'), nargs='?', default='info')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    project = settings()
    if args.action == 'info':
        print(json.dumps(project, ensure_ascii=False, indent=2))
        return 0
    if args.action == 'verify-archive':
        return archive_check()
    if args.action == 'check':
        snippet = "import config, importlib, json; from pathlib import Path; "
        snippet += "[importlib.import_module(x) for x in "+repr(project['smoke_modules'])+"]; "
        snippet += "print(json.dumps({'project_root': config.PROJECT_DIR, 'output_root': config.OUTPUT_DIR, 'imports': 'passed'}, ensure_ascii=False)); "
        snippet += "assert Path(config.PROJECT_DIR).resolve() == Path.cwd().resolve()"
        process = subprocess.run([sys.executable, '-c', snippet], cwd=ROOT, env=environment())
        if process.returncode:
            return process.returncode
        missing = [value for value in project['local_data_checks'] if not (ROOT/value).exists()]
        print(json.dumps(dict(missing_local_inputs=missing,
                             meaning='Missing vendor data are not installed by pip; see README'), ensure_ascii=False))
        return 1 if missing else 0
    if args.action == 'test':
        return subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-q', *args.arguments],
                              cwd=ROOT, env=environment()).returncode
    if not args.arguments or args.arguments[0] not in project['allowed_modules']:
        parser.error('Choose a module listed by `python run.py info`; other project architectures are not exposed here')
    # No implicit training, download or archived output overwrite occurs in info/check/test.
    return subprocess.run([sys.executable, '-m', *args.arguments], cwd=ROOT, env=environment()).returncode


if __name__ == '__main__':
    raise SystemExit(main())
