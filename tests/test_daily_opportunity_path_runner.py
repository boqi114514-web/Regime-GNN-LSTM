import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from research_daily_graph_data import sha256
from research_daily_opportunity_path_runner import (
    PATH_ARTIFACTS, bind_path_filter_artifacts, filter_path_predictions, run_path_account,
    validate_path_filter,
)


class PathRunnerTests(unittest.TestCase):
    def fixture(self, source, out):
        source.mkdir(); out.mkdir()
        native = pd.DataFrame(dict(signal_date=pd.to_datetime(['2026-01-05']), ts_code=['600001.SH'],
                                   utility=[.1], mu5=[.15], passage_probability_upper5=[.2]))
        native.to_pickle(source/'predictions.pkl')
        for name in PATH_ARTIFACTS:
            (source/name).write_bytes(('unchanged '+name).encode())
        (source/'prediction_manifest.json').write_text(json.dumps(dict(status='complete', artifacts={
            name:sha256(source/name) for name in PATH_ARTIFACTS})), encoding='utf-8')
        (out/'experiment_protocol.json').write_text(json.dumps(dict(source_sha256={})), encoding='utf-8')
        (out/'prediction_manifest.json').write_text(json.dumps(dict(status='complete', artifacts={})), encoding='utf-8')
        return native

    def test_binding_keeps_both_target_sidecars_and_original_input_fingerprints(self):
        with tempfile.TemporaryDirectory() as folder:
            source, out = Path(folder)/'source', Path(folder)/'out'
            self.fixture(source, out)
            bind_path_filter_artifacts(source, out, 'liquid')
            manifest = json.loads((out/'prediction_manifest.json').read_text(encoding='utf-8'))
            for name in PATH_ARTIFACTS:
                self.assertEqual(sha256(out/name), sha256(source/name))
                self.assertEqual(manifest['artifacts'][name], sha256(out/name))
            self.assertIn('path_filter_manifest.json', manifest['artifacts'])
            protocol = json.loads((out/'experiment_protocol.json').read_text(encoding='utf-8'))
            self.assertEqual(protocol['path_account_ablation']['mode'], 'liquid')
            self.assertIn('research_daily_opportunity_path_runner.py', protocol['source_sha256'])

    def test_unbound_target_input_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            source, out = Path(folder)/'source', Path(folder)/'out'
            self.fixture(source, out)
            (source/'path_targets.pkl').write_bytes(b'tampered targets')
            with self.assertRaisesRegex(ValueError, 'does not bind path_targets.pkl'):
                bind_path_filter_artifacts(source, out, 'band')

    def test_filter_verifier_rejects_changed_forecast_even_with_unchanged_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            source, out = Path(folder)/'source', Path(folder)/'out'
            native = self.fixture(source, out)
            bind_path_filter_artifacts(source, out, 'band')
            changed = native.assign(eligible=True)
            changed.loc[0, 'mu5'] = .99
            with patch('research_daily_opportunity_path_runner.validate_path_predictions', return_value=changed):
                with self.assertRaisesRegex(ValueError, 'changed native forecasts'):
                    validate_path_filter(out)

    def test_old_or_same_output_cannot_be_reused_as_another_ablation(self):
        with tempfile.TemporaryDirectory() as folder:
            source, out = Path(folder)/'source', Path(folder)/'out'
            self.fixture(source, out)
            with self.assertRaises(ValueError):
                filter_path_predictions(source, out, offline=True)
            with self.assertRaises(ValueError):
                filter_path_predictions(source, source, offline=True)
            with self.assertRaises(ValueError):
                run_path_account(source, source, Path(folder)/'raw', offline=True)

    def test_failed_account_can_retry_but_completed_account_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            source, out = Path(folder)/'source', Path(folder)/'out'
            self.fixture(source, out)
            account = out/'account'
            account.mkdir()
            status = account/'daily_run_status.json'
            status.write_text(json.dumps(dict(account_complete=False, error_type='RuntimeError')), encoding='utf-8')
            # Reaching input validation proves a failed transport run may reuse
            # its own caches. No account engine or network operation is invoked.
            with patch('research_daily_opportunity_path_runner.validate_path_filter', side_effect=LookupError('validation reached')) as checked:
                with self.assertRaisesRegex(LookupError, 'validation reached'):
                    run_path_account(source, out, Path(folder)/'raw', offline=True)
                checked.assert_called_once_with(source)
            status.write_text(json.dumps(dict(account_complete=True)), encoding='utf-8')
            with patch('research_daily_opportunity_path_runner.validate_path_filter') as checked:
                with self.assertRaisesRegex(ValueError, 'without a completed path account'):
                    run_path_account(source, out, Path(folder)/'raw', offline=True)
                checked.assert_not_called()


if __name__ == '__main__':
    unittest.main()
