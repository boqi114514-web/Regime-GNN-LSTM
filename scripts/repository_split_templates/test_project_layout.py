"""Portable repository entry point and preserved archive contract."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ProjectLayoutTests(unittest.TestCase):
    def test_scoped_entry_modules_exist(self):
        project = json.loads((ROOT/'project.json').read_text(encoding='utf-8'))
        for module in project['allowed_modules']+project['smoke_modules']:
            self.assertTrue((ROOT/'src'/Path(*module.split('.')).with_suffix('.py')).is_file(), module)

    def test_no_old_project_directory_in_config(self):
        text = (ROOT/'src/config.py').read_text(encoding='utf-8')
        self.assertNotIn('PROJECT_DIR = r"D:', text)
        self.assertIn('os.path.dirname(os.path.dirname(os.path.abspath(__file__)))', text)

    def test_scoped_entry_rejects_unlisted_module(self):
        result = subprocess.run([sys.executable, str(ROOT/'run.py'), 'module', 'not_a_project_module'],
                                cwd=ROOT, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'not exposed here', result.stderr)

    def test_archive_provenance_is_not_a_new_backtest_claim(self):
        migration = json.loads((ROOT/'migration_manifest.json').read_text(encoding='utf-8'))
        self.assertTrue(migration['original_preserved'])
        self.assertIn('No audit proof is rewritten', migration['provenance'])


if __name__ == '__main__':
    unittest.main()
