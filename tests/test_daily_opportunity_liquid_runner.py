from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import research_daily_opportunity_liquid_runner as module


@contextmanager
def temporary_workspace():
    with tempfile.TemporaryDirectory() as folder:
        previous = Path.cwd()
        try:
            os.chdir(folder)
            yield Path(folder)
        finally:
            os.chdir(previous)


class LiquidRunnerTests(unittest.TestCase):
    def test_batch_gateway_only_simplifies_exact_all_market_date_request(self):
        gateway = object.__new__(module.ExactBatchGateway)
        response = object()
        with patch.object(module.BackoffGateway, 'query', return_value=response) as request:
            result = gateway.query('stk_limit', trade_date='20260106', offset=0,
                                   limit=6000, fields='ts_code,trade_date,up_limit,down_limit')
            self.assertIs(result, response)
            request.assert_called_once_with('stk_limit', trade_date='20260106')
            request.reset_mock()
            single = dict(ts_code='600001.SH', trade_date='20260106', limit=2,
                          fields='ts_code,trade_date,up_limit,down_limit')
            gateway.query('stk_limit', **single)
            request.assert_called_once_with('stk_limit', **single)
            request.reset_mock()
            ranged = dict(start_date='20260101', end_date='20260106', limit=6000, offset=0)
            gateway.query('stk_limit', **ranged)
            request.assert_called_once_with('stk_limit', **ranged)
            request.reset_mock()
            other = dict(trade_date='20260106', offset=0, limit=6000)
            gateway.query('daily', **other)
            request.assert_called_once_with('daily', **other)

    def history(self, code):
        return pd.DataFrame(dict(ts_code=[code]*2, end_date=['20251231', '20241231'],
                                 div_proc=['预案']*2, record_date=['']*2, ex_date=['']*2,
                                 pay_date=['']*2, div_listdate=['']*2,
                                 cash_div_tax=[0., 0.], stk_div=[0., 0.]))

    def write_cache(self, root, code, *, bad_hash=False, history=None, start='2026-01-01'):
        folder = root/'results/daily_opportunity_fixture/account/scoped_actions'
        folder.mkdir(parents=True, exist_ok=True)
        source = folder/f'{code}.pkl'
        (self.history(code) if history is None else history).to_pickle(source)
        metadata = dict(start=start, end='2026-09-24',
                        raw_sha256='wrong' if bad_hash else module.sha256(source))
        source.with_suffix('.json').write_text(json.dumps(metadata), encoding='utf-8')
        return source

    @staticmethod
    def seed_empty(account):
        (account/'scoped_actions').mkdir(parents=True, exist_ok=True)

    def test_seed_actions_validates_history_and_skips_bad_hash_and_wrong_period(self):
        with temporary_workspace() as root:
            good = self.write_cache(root, '600001.SH')
            self.write_cache(root, '600002.SH', bad_hash=True)
            self.write_cache(root, '600003.SH', start='2025-01-01')
            account = root/'new_account'
            with patch.object(module, 'seed_actions', side_effect=self.seed_empty), \
                    patch.object(module, 'validate_history', wraps=module.validate_history) as check:
                module.seed_all_actions(account)
            check.assert_called_once()
            self.assertEqual(check.call_args.args[1:], ('600001.SH', '2026-01-01', '2026-09-24'))
            target = account/'scoped_actions/600001.SH.pkl'
            self.assertEqual(module.sha256(target), module.sha256(good))
            self.assertEqual(target.with_suffix('.json').read_bytes(), good.with_suffix('.json').read_bytes())
            self.assertFalse((account/'scoped_actions/600002.SH.pkl').exists())
            self.assertFalse((account/'scoped_actions/600003.SH.pkl').exists())

    def test_invalid_history_with_matching_hash_is_rejected_before_copy(self):
        with temporary_workspace() as root:
            self.write_cache(root, '600001.SH', history=self.history('600009.SH'))
            account = root/'new_account'
            with patch.object(module, 'seed_actions', side_effect=self.seed_empty):
                with self.assertRaisesRegex(ValueError, 'Wrong stock'):
                    module.seed_all_actions(account)
            self.assertFalse((account/'scoped_actions/600001.SH.pkl').exists())

    def test_existing_target_action_history_is_not_overwritten_by_other_experiment(self):
        with temporary_workspace() as root:
            self.write_cache(root, '600001.SH')
            account = root/'new_account'
            self.seed_empty(account)
            existing = account/'scoped_actions/600001.SH.pkl'
            original = b'pre-existing account cache must be validated by its loader'
            existing.write_bytes(original)
            with patch.object(module, 'seed_actions'), patch.object(module, 'validate_history') as check:
                module.seed_all_actions(account)
            self.assertEqual(existing.read_bytes(), original)
            check.assert_not_called()

    def test_cache_roots_include_new_filter_and_account_batches(self):
        with temporary_workspace() as root:
            first = Path('results/daily_opportunity_new/daily_limits')
            second = Path('results/daily_opportunity_new/account/daily_limits')
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            with patch.object(module, 'limits_caches', return_value=[Path('old/cache')]):
                roots = module.cache_roots()
            self.assertEqual(set(roots), {Path('old/cache'), first, second})

    def account_fixture(self, root):
        source, out = root/'source', root/'target'
        source.mkdir()
        (source/'checkpoints').mkdir()
        (source/'feature_manifest.json').write_text(
            json.dumps(dict(raw_adjusted_daily=dict(sha256='raw-quotes'))), encoding='utf-8')
        for name in ('prior_band_manifest.json', 'prior_band_checks.csv',
                     'liquidity_manifest.json', 'liquidity_checks.csv'):
            (source/name).write_text(name, encoding='utf-8')
        daily = pd.DataFrame(dict(date=pd.to_datetime(['2026-01-05', '2026-01-06'])))
        predictions = pd.DataFrame(dict(signal_date=[pd.Timestamp('2026-01-05')],
                                        ts_code=['600001.SH'], utility=[.1], eligible=[True]))
        return source, out, daily, predictions

    def test_account_binds_all_additional_sidecars_and_both_new_sources(self):
        with temporary_workspace() as root:
            source, out, daily, predictions = self.account_fixture(root)
            checked = dict(band_manifest=dict(source_sha256={'official-source.pkl': 'official-hash'}))
            sentinel = object()

            def run_account(p, raw, sessions, account, **kwargs):
                pd.testing.assert_frame_equal(p, predictions)
                pd.testing.assert_frame_equal(raw, daily)
                self.assertEqual(sessions, list(daily.date))
                self.assertTrue(kwargs['offline'])
                self.assertIsNone(kwargs['pro'])
                self.assertEqual(kwargs['policy'].trailing_distance, .08)
                (account/'daily_run_status.json').write_text(
                    json.dumps(dict(source_sha256={'base-source.py': 'base-hash'}, status='complete')),
                    encoding='utf-8')
                return sentinel

            with patch.object(module, 'validate_liquidity_inputs', return_value=checked) as liquidity, \
                    patch.object(module, 'leadership_variant', side_effect=nullcontext), \
                    patch.object(module, 'validate_predictions', return_value=predictions), \
                    patch.object(module, 'load_adjusted_daily', return_value=(daily, dict(adjusted_daily=dict(sha256='raw-quotes')))), \
                    patch.object(module, 'seed_all_actions'), patch.object(module, 'cache_roots', return_value=[]), \
                    patch.object(module, 'OfficialLimits'), patch.object(module, 'run_account', side_effect=run_account):
                result = module.run_liquid_account(source, out, root/'inputs', offline=True)
            self.assertIs(result, sentinel)
            liquidity.assert_called_once_with(source)
            status = json.loads((out/'account/daily_run_status.json').read_text(encoding='utf-8'))
            hashes = status['source_sha256']
            self.assertEqual(hashes['base-source.py'], 'base-hash')
            self.assertEqual(hashes['official-source.pkl'], 'official-hash')
            for name in ('prior_band_manifest.json', 'prior_band_checks.csv',
                         'liquidity_manifest.json', 'liquidity_checks.csv'):
                self.assertEqual(hashes[str((source/name).resolve())], module.sha256(source/name))
                self.assertEqual((out/name).read_bytes(), (source/name).read_bytes())
            for name in ('research_daily_opportunity_liquid_filter.py', 'research_daily_opportunity_liquid_runner.py'):
                path = Path(module.__file__).parent/name
                self.assertEqual(hashes[str(path.resolve())], module.sha256(path))

    def test_account_raw_source_mismatch_stops_before_execution(self):
        with temporary_workspace() as root:
            source, out, daily, predictions = self.account_fixture(root)
            with patch.object(module, 'validate_liquidity_inputs', return_value={}), \
                    patch.object(module, 'leadership_variant', side_effect=nullcontext), \
                    patch.object(module, 'validate_predictions', return_value=predictions), \
                    patch.object(module, 'load_adjusted_daily', return_value=(daily, dict(adjusted_daily=dict(sha256='different-quotes')))), \
                    patch.object(module, 'run_account') as execute:
                with self.assertRaisesRegex(ValueError, 'raw quotes differ'):
                    module.run_liquid_account(source, out, root/'inputs', offline=True)
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
