"""Offline regressions for acquisition coverage and publication certification."""
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fetch_dc_theme_research as collection


class DCThemeCollectionTests(unittest.TestCase):
    day = pd.Timestamp('2026-03-31')

    def setUp(self):
        guard = patch('requests.sessions.Session.request',
                      side_effect=AssertionError('Network access is prohibited in collection tests'))
        guard.start()
        self.addCleanup(guard.stop)

    def market_frame(self, rows=105):
        return pd.DataFrame([
            dict(trade_date=self.day.strftime('%Y%m%d'), ts_code=f'BK{number:04d}.DC',
                 open=10., high=12., low=9., close=11., vol=200., amount=22000.,
                 category='概念板块')
            for number in range(rows)
        ])

    def catalogue(self, rows=105):
        return pd.DataFrame([
            dict(trade_date=self.day.strftime('%Y%m%d'), ts_code=f'BK{number:04d}.DC',
                 name=f'Theme {number}', up_num=3, down_num=2)
            for number in range(rows)
        ])

    def snapshot_failure(self, *args, **kwargs):
        return dict(status='invalid_schema', http_status=200, diagnostics={}), None

    def member_frame(self, rows=3, trade_date='20260331'):
        return pd.DataFrame([
            dict(trade_date=trade_date, ts_code='BK0877.DC',
                 con_code=f'{number + 1000:06d}.SZ', name=f'Stock {number}')
            for number in range(rows)
        ])

    def test_member_default_request_includes_explicit_fields(self):
        frame = self.member_frame()

        def response(session, transport, key, spec):
            self.assertEqual(transport, 'gateway')
            self.assertEqual(key, 'unit-key')
            self.assertEqual(spec['api'], 'dc_member')
            self.assertEqual(spec['params'], dict(trade_date='20260331', limit=1000,
                                                 ts_code='BK0877.DC', fields='trade_date,ts_code,con_code,name'))
            return dict(status='ok', http_status=200, params=dict(spec['params'])), frame.copy()

        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.dict(os.environ, {'TUSHARE_API_KEY': 'unit-key'}), \
                    patch.object(collection.requests, 'Session'), \
                    patch.object(collection, 'probe_request', side_effect=response) as probe:
                result = collection.gateway_snapshot(root, 'dc_member', 'BK0877.DC', self.day,
                    lambda data: collection.validate_members(data, self.day, 'BK0877.DC', limit=1000))
            self.assertEqual(probe.call_count, 1)
            pd.testing.assert_frame_equal(result, frame)
            pd.testing.assert_frame_equal(pd.read_pickle(root / 'dc_member_BK0877.DC_20260331.pkl'), frame)
            audits = json.loads((root / 'dc_member_BK0877.DC_20260331.json').read_text(encoding='utf-8'))
            self.assertEqual(len(audits), 1)
            self.assertEqual(audits[0]['attempt'], 1)
            self.assertEqual(audits[0]['params']['fields'], 'trade_date,ts_code,con_code,name')

    def test_member_short_explicit_response_can_only_be_repaired_by_exact_date_range(self):
        for returned_date in ('20260331', '20260401'):
            with self.subTest(returned_date=returned_date), TemporaryDirectory() as temp:
                root = Path(temp)
                calls = []

                def response(session, transport, key, spec):
                    calls.append(dict(spec['params']))
                    record = dict(status='ok', http_status=200, params=dict(spec['params']))
                    if len(calls) == 1:
                        return record, self.member_frame(rows=1)
                    if len(calls) == 2:
                        self.assertEqual(spec['params'], dict(start_date='20260331', end_date='20260331',
                            limit=1000, ts_code='BK0877.DC', fields='trade_date,ts_code,con_code,name'))
                        return record, self.member_frame(trade_date=returned_date)
                    return dict(status='invalid_schema', http_status=200, params=dict(spec['params'])), None

                def validate(data):
                    result = collection.validate_members(data, self.day, 'BK0877.DC', limit=1000)
                    if len(result) < 3:
                        raise ValueError('Membership below contemporaneous count lower bound')
                    return result

                with patch.dict(os.environ, {'TUSHARE_API_KEY': 'unit-key'}), \
                        patch.object(collection.requests, 'Session'), \
                        patch.object(collection, 'probe_request', side_effect=response):
                    if returned_date == '20260331':
                        result = collection.gateway_snapshot(root, 'dc_member', 'BK0877.DC', self.day, validate)
                        self.assertEqual(len(calls), 2)
                        self.assertEqual(len(result), 3)
                        self.assertTrue(result.trade_date.eq('20260331').all())
                    else:
                        with self.assertRaisesRegex(ValueError, 'snapshot unavailable'):
                            collection.gateway_snapshot(root, 'dc_member', 'BK0877.DC', self.day, validate)
                cache = root / 'dc_member_BK0877.DC_20260331.pkl'
                audits = json.loads((root / 'dc_member_BK0877.DC_20260331.json').read_text(encoding='utf-8'))
                self.assertEqual(calls[0]['trade_date'], '20260331')
                self.assertEqual(calls[0]['fields'], 'trade_date,ts_code,con_code,name')
                self.assertIn('count lower bound', audits[0]['validation_error'])
                self.assertEqual([record['attempt'] for record in audits[:2]], [1, 2])
                if returned_date == '20260331':
                    self.assertNotIn('validation_error', audits[1])
                    pd.testing.assert_frame_equal(pd.read_pickle(cache), self.member_frame())
                else:
                    self.assertFalse(cache.exists())
                    self.assertIn('differs from requested historical date', audits[1]['validation_error'])

    def assert_cached_frame_rejected(self, cached, expected_codes):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            name = 'dc_daily_' + self.day.strftime('%Y%m%d')
            cached.to_pickle(directory / (name + '.pkl'))
            with patch.dict(os.environ, {'TUSHARE_API_KEY': 'unit-key'}), \
                    patch.object(collection.requests, 'Session') as session, \
                    patch.object(collection, 'probe_request', side_effect=self.snapshot_failure) as probe:
                with self.assertRaisesRegex(ValueError, 'snapshot unavailable'):
                    collection.gateway_snapshot(
                        directory, 'dc_daily', None, self.day,
                        lambda frame: collection.validate_market_day(frame, self.day, expected_codes))
                self.assertTrue(session.called)
                self.assertTrue(probe.called)
            rejected = directory / (name + '.rejected.pkl')
            self.assertTrue(rejected.exists())
            pd.testing.assert_frame_equal(pd.read_pickle(rejected), cached)
            audits = json.loads((directory / (name + '.json')).read_text(encoding='utf-8'))
            self.assertTrue(audits)
            self.assertFalse(any(record['status'] == 'ok' for record in audits))

    def test_nonempty_whole_market_subset_is_not_certified(self):
        for rows in (1, 99):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, 'small whole-market'):
                collection.validate_market_day(self.market_frame(rows), self.day)

    def test_large_snapshot_still_requires_every_contemporaneous_catalogue_code(self):
        partial = self.market_frame(104)
        expected = set(self.catalogue().ts_code)
        with self.assertRaisesRegex(ValueError, 'Missing contemporaneous catalogue indexes'):
            collection.validate_market_day(partial, self.day, expected)
        complete = collection.validate_market_day(self.market_frame(), self.day, expected)
        self.assertEqual(set(complete.ts_code), expected)
        self.assertTrue(complete.trade_date.eq('20260331').all())

    def test_valid_complete_cache_returns_without_gateway_attempt(self):
        cached = self.market_frame()
        expected = set(self.catalogue().ts_code)
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            cached.to_pickle(directory / 'dc_daily_20260331.pkl')
            with patch.dict(os.environ, {'TUSHARE_API_KEY': 'unit-key'}), \
                    patch.object(collection.requests, 'Session') as session, \
                    patch.object(collection, 'probe_request') as probe:
                result = collection.gateway_snapshot(
                    directory, 'dc_daily', None, self.day,
                    lambda frame: collection.validate_market_day(frame, self.day, expected))
                session.assert_not_called()
                probe.assert_not_called()
            pd.testing.assert_frame_equal(result, cached)

    def test_partial_cache_cannot_bypass_signal_catalogue_coverage(self):
        self.assert_cached_frame_rejected(self.market_frame(104), set(self.catalogue().ts_code))

    def test_integer_epoch_interpreted_dates_in_old_cache_are_not_certified(self):
        cached = self.market_frame()
        cached['trade_date'] = 20260331
        self.assert_cached_frame_rejected(cached, set(self.catalogue().ts_code))

    def test_catalogue_publication_failure_overwrites_prior_complete_with_running(self):
        original_pickle = pd.DataFrame.to_pickle

        def fail_publication(frame, path, *args, **kwargs):
            if Path(path).name == 'catalogs.staging.pkl':
                raise OSError('simulated catalogue publication failure')
            return original_pickle(frame, path, *args, **kwargs)

        with TemporaryDirectory() as temp:
            root = Path(temp)
            old = self.catalogue(1)
            old_artifact = collection.publish_frame(old, root / 'catalogs.pkl')
            collection.publish_manifest(dict(status='complete', aggregate=old_artifact),
                                        root / 'catalog_manifest.json')
            old_bytes = (root / 'catalogs.pkl').read_bytes()
            with patch.object(collection, 'signal_dates', return_value=[self.day]), \
                    patch.object(collection, 'gateway_snapshot', return_value=self.catalogue()), \
                    patch('pandas.DataFrame.to_pickle', new=fail_publication), \
                    redirect_stdout(StringIO()), self.assertRaisesRegex(OSError, 'publication failure'):
                collection.catalogs(root)
            manifest = json.loads((root / 'catalog_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'running')
            self.assertNotIn('aggregate', manifest)
            self.assertEqual((root / 'catalogs.pkl').read_bytes(), old_bytes)
            with self.assertRaisesRegex(ValueError, 'acquisition is incomplete'):
                collection.features(root, root / 'features')
            self.assertFalse((root / 'features' / 'theme_features.pkl').exists())

    def run_market_days(self, root, days=None, scope='full', response_frame=None):
        days = [self.day] if days is None else list(pd.to_datetime(days))
        original_read = pd.read_pickle

        def read_local_fixture(path, *args, **kwargs):
            if Path(path).name == 'daily.pkl':
                return pd.DataFrame({'date': days})
            return original_read(path, *args, **kwargs)

        def batch_response(session, transport, key, spec):
            self.assertEqual(transport, 'gateway')
            self.assertEqual(key, 'unit-key')
            self.assertEqual(spec['api'], 'dc_daily')
            params = spec['params']
            self.assertEqual(params['idx_type'], '概念板块')
            if 'limit' in params:
                self.assertEqual(params['limit'], 2000)
            if response_frame is None:
                batch_days = [day for day in days if params['start_date'] <= day.strftime('%Y%m%d') <= params['end_date']]
                response = pd.concat([self.market_frame().assign(trade_date=day.strftime('%Y%m%d'))
                                      for day in batch_days], ignore_index=True)
            else:
                response = response_frame.copy()
            return dict(status='ok', http_status=200, diagnostics={}), response

        with patch.object(collection.pd, 'read_pickle', side_effect=read_local_fixture), \
                patch.dict(os.environ, {'TUSHARE_API_KEY': 'unit-key'}), \
                patch.object(collection.requests, 'Session'), \
                patch.object(collection, 'probe_request', side_effect=batch_response) as probe, \
                redirect_stdout(StringIO()):
            collection.daily_market(root, workers=2, scope=scope)
        return probe

    def run_one_market_day(self, root):
        return self.run_market_days(root)

    def test_daily_market_publication_failure_cannot_authorize_reading_old_aggregate(self):
        original_pickle = pd.DataFrame.to_pickle

        def fail_publication(frame, path, *args, **kwargs):
            if Path(path).name == 'daily_market.staging.pkl':
                raise OSError('simulated market publication failure')
            return original_pickle(frame, path, *args, **kwargs)

        with TemporaryDirectory() as temp:
            root = Path(temp)
            catalog_artifact = collection.publish_frame(self.catalogue(), root / 'catalogs.pkl')
            collection.publish_manifest(dict(status='complete', aggregate=catalog_artifact),
                                        root / 'catalog_manifest.json')
            old_artifact = collection.publish_frame(self.market_frame(100), root / 'daily_market.pkl')
            collection.publish_manifest(dict(status='complete', aggregate=old_artifact),
                                        root / 'daily_market_manifest.json')
            old_bytes = (root / 'daily_market.pkl').read_bytes()
            with patch('pandas.DataFrame.to_pickle', new=fail_publication), \
                    self.assertRaisesRegex(OSError, 'publication failure'):
                self.run_one_market_day(root)
            manifest = json.loads((root / 'daily_market_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'running')
            self.assertNotIn('aggregate', manifest)
            self.assertEqual((root / 'daily_market.pkl').read_bytes(), old_bytes)
            with self.assertRaisesRegex(ValueError, 'acquisition is incomplete'):
                collection.features(root, root / 'features')
            self.assertFalse((root / 'features' / 'theme_features.pkl').exists())

    def test_successful_daily_market_manifest_binds_actual_aggregate_and_coverage(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.catalogue().to_pickle(root / 'catalogs.pkl')
            self.run_one_market_day(root)
            manifest = json.loads((root / 'daily_market_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['planned_days'], 1)
            self.assertEqual(manifest['completed_days'], 1)
            self.assertTrue(manifest['signal_catalogue_coverage_required'])
            artifact = manifest['aggregate']
            self.assertEqual(artifact['rows'], 105)
            self.assertEqual(artifact['sha256'], hashlib.sha256((root / artifact['file']).read_bytes()).hexdigest())
            pd.testing.assert_frame_equal(pd.read_pickle(root / artifact['file']), self.market_frame())

    def test_features_refuse_hash_mismatch_before_reading_any_input_frame(self):
        for scope in ('full', '2026'):
            market_name = 'daily_market_2026' if scope == '2026' else 'daily_market'
            for bad_name in ('catalog_manifest.json', market_name + '_manifest.json'):
                with self.subTest(scope=scope, manifest=bad_name), TemporaryDirectory() as temp:
                    root = Path(temp)
                    artifacts = {
                        'catalog_manifest.json': collection.publish_frame(self.catalogue(), root / 'catalogs.pkl'),
                        market_name + '_manifest.json': collection.publish_frame(self.market_frame(), root / (market_name + '.pkl'))}
                    for name, artifact in artifacts.items():
                        if name == bad_name:
                            artifact['sha256'] = '0' * 64
                        collection.publish_manifest(dict(status='complete', aggregate=artifact, scope=scope), root / name)
                    with patch.object(collection.pd, 'read_pickle', side_effect=AssertionError('Unexpected unverified input read')):
                        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
                            collection.features(root, root / 'features', scope=scope)
                    self.assertFalse((root / 'features' / 'theme_features.pkl').exists())
                    feature_manifest = json.loads((root / 'features' / 'theme_feature_manifest.json').read_text(encoding='utf-8'))
                    self.assertEqual(feature_manifest['status'], 'running')
                    self.assertEqual(feature_manifest['scope'], scope)
                    self.assertNotIn('aggregate', feature_manifest)

    def test_batch_missing_a_requested_date_is_not_published_or_certified(self):
        days = list(pd.to_datetime(['2026-03-27', '2026-03-30', '2026-03-31']))
        partial = pd.concat([self.market_frame().assign(trade_date=day.strftime('%Y%m%d'))
                             for day in (days[0], days[2])], ignore_index=True)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.catalogue().to_pickle(root / 'catalogs.pkl')
            with self.assertRaisesRegex(ValueError, 'acquisition incomplete'):
                self.run_market_days(root, days, response_frame=partial)
            manifest = json.loads((root / 'daily_market_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'incomplete')
            self.assertEqual(manifest['planned_days'], 3)
            self.assertEqual(manifest['completed_days'], 0)
            self.assertNotIn('aggregate', manifest)
            self.assertFalse((root / 'daily_market.pkl').exists())
            self.assertFalse(list((root / 'daily_market').glob('*.pkl')))
            audits = json.loads(next((root / 'daily_market').glob('*.json')).read_text(encoding='utf-8'))
            self.assertTrue(audits)
            self.assertTrue(all('Batch dates missing' in record.get('validation_error', '') for record in audits))

    def test_successful_batch_certifies_all_three_dates(self):
        days = list(pd.to_datetime(['2026-03-27', '2026-03-30', '2026-03-31']))
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.catalogue().to_pickle(root / 'catalogs.pkl')
            probe = self.run_market_days(root, days)
            self.assertEqual(probe.call_count, 1)
            manifest = json.loads((root / 'daily_market_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['planned_days'], 3)
            self.assertEqual(manifest['completed_days'], 3)
            aggregate = pd.read_pickle(root / manifest['aggregate']['file'])
            self.assertEqual(set(aggregate.trade_date), {day.strftime('%Y%m%d') for day in days})
            self.assertEqual(len(aggregate), 315)

    def test_full_and_2026_market_scopes_publish_separate_artifacts(self):
        days = list(pd.to_datetime(['2025-07-31', '2025-08-01', '2025-08-04', '2026-08-31', '2026-09-01']))
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.catalogue().to_pickle(root / 'catalogs.pkl')
            self.run_market_days(root, days, scope='full')
            full_bytes = (root / 'daily_market.pkl').read_bytes()
            full_manifest = json.loads((root / 'daily_market_manifest.json').read_text(encoding='utf-8'))
            probe = self.run_market_days(root, days, scope='2026')
            probe.assert_not_called()
            scoped = json.loads((root / 'daily_market_2026_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(full_manifest['scope'], 'full')
            self.assertEqual(full_manifest['planned_days'], 4)
            self.assertEqual(scoped['scope'], '2026')
            self.assertEqual(scoped['start_date'], '20250801')
            self.assertEqual(scoped['end_date'], '20260831')
            self.assertEqual(scoped['planned_days'], 3)
            self.assertEqual(scoped['completed_days'], 3)
            self.assertEqual(scoped['aggregate']['file'], 'daily_market_2026.pkl')
            self.assertEqual((root / 'daily_market.pkl').read_bytes(), full_bytes)
            self.assertEqual(json.loads((root / 'daily_market_manifest.json').read_text(encoding='utf-8')), full_manifest)
            aggregate = pd.read_pickle(root / scoped['aggregate']['file'])
            self.assertEqual(set(aggregate.trade_date), {'20250801', '20250804', '20260831'})

    def test_cli_default_output_and_members_scope_are_isolated(self):
        for scope, output in (('full', 'results/dc_theme_research'), ('2026', 'results/dc_theme_2026_research')):
            for action in ('features', 'members'):
                with self.subTest(scope=scope, action=action), TemporaryDirectory() as temp:
                    argv = ['fetch_dc_theme_research.py', action, '--scope', scope, '--source-dir', temp]
                    with patch.object(sys, 'argv', argv), patch.object(collection, action) as run:
                        collection.main()
                    self.assertEqual(run.call_args.args, (Path(temp), Path(output), scope))

    def test_batch_cannot_rewrite_previously_verified_overlap_prices(self):
        self.assert_overlap_rejected(invalid_second_cache=False)

    def test_invalid_shard_does_not_erase_another_days_verified_overlap(self):
        self.assert_overlap_rejected(invalid_second_cache=True)

    def assert_overlap_rejected(self, invalid_second_cache):
        days = list(pd.to_datetime(['2026-03-27', '2026-03-30', '2026-03-31']))
        fresh = pd.concat([self.market_frame().assign(trade_date=day.strftime('%Y%m%d'))
                           for day in days], ignore_index=True)
        fresh.loc[fresh.trade_date.eq('20260327'), 'close'] = 11.5
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.catalogue().to_pickle(root / 'catalogs.pkl')
            directory = root / 'daily_market'
            directory.mkdir()
            verified = self.market_frame().assign(trade_date='20260327')
            path = directory / 'dc_daily_20260327.pkl'
            verified.to_pickle(path)
            old_bytes = path.read_bytes()
            if invalid_second_cache:
                self.market_frame().assign(trade_date=20260330).to_pickle(directory / 'dc_daily_20260330.pkl')
            with self.assertRaisesRegex(ValueError, 'acquisition incomplete'):
                self.run_market_days(root, days, response_frame=fresh)
            self.assertEqual(path.read_bytes(), old_bytes)
            manifest = json.loads((root / 'daily_market_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'incomplete')
            self.assertNotIn('aggregate', manifest)
            audits = json.loads(next(directory.glob('market_range_*.json')).read_text(encoding='utf-8'))
            self.assertTrue(any('previously verified daily prices' in record.get('validation_error', '')
                                for record in audits))


if __name__ == '__main__':
    unittest.main()
