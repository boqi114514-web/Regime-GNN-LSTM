from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fetch_theme_pilot import build_requests, load_expected_sessions, main, run_pilot


class Response:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status
        self.headers = {'Content-Type': 'application/json'}

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, first_responses=(), missing_day=False, wrong_code=False):
        self.responses = list(first_responses)
        self.calls = []
        self.missing_day, self.wrong_code = missing_day, wrong_code

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        params = kwargs['params']
        code = 'wrong-source-code' if self.wrong_code else params['ts_code']
        if url.endswith('/dc_member'):
            fields = ['ts_code', 'trade_date', 'con_code']
            items = [[code, params['trade_date'], '002384.SZ'], [code, params['trade_date'], '600183.SH']]
        else:
            dates = pd.bdate_range(params['start_date'], params['end_date'])
            if self.missing_day:
                dates = dates[:1]
            fields = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']
            items = [[code, day.strftime('%Y%m%d'), 10., 12., 9., 11., 100., 1000.] for day in dates]
        return Response({'code': 0, 'data': {'fields': fields, 'items': items}})


class ThemePilotTests(unittest.TestCase):
    def run_one(self, temp, session, **kwargs):
        return run_pilot(['885959.TI'], [], '20260330', '20260331', [], output_dir=temp,
                         session=session, environ={'TUSHARE_API_KEY': 'unit-key', 'TUSHARE_TOKEN': 'never-send-legacy'}, **kwargs)

    def test_plan_monthly_explicit_limits_and_historical_dc_only(self):
        plan = build_requests(['885959.TI'], ['BK0877.DC'], '20260315', '20260515', ['20260331'])
        self.assertEqual(len(plan), 7)
        self.assertEqual(plan[0]['params'], dict(ts_code='885959.TI', start_date='20260315', end_date='20260331', limit=100))
        self.assertEqual(plan[2]['params']['end_date'], '20260515')
        self.assertEqual(plan[-1]['params'], dict(ts_code='BK0877.DC', trade_date='20260331', limit=1000))
        self.assertEqual({request['api'] for request in plan}, {'ths_daily', 'dc_daily', 'dc_member'})

    def test_complete_run_publishes_validated_aggregates_and_preserves_units(self):
        with TemporaryDirectory() as temp:
            session = FakeSession()
            manifest = run_pilot(['885959.TI'], ['BK0877.DC'], '20260330', '20260331', ['20260331'],
                                 output_dir=temp, expected_sessions=['20260330', '20260331'], session=session,
                                 environ={'TUSHARE_API_KEY': 'unit-key', 'TUSHARE_TOKEN': 'never-send-legacy'})
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(len(session.calls), 3)
            self.assertEqual(set(manifest['validated_files']), {'ths_daily', 'dc_daily', 'dc_member'})
            self.assertEqual(manifest['units']['ths_daily']['vol'], '手')
            self.assertEqual(manifest['units']['ths_daily']['amount'], 'not_provided_by_documented_gateway_schema')
            self.assertEqual(manifest['units']['dc_daily']['vol'], '股')
            self.assertEqual(manifest['units']['dc_daily']['amount'], '元')
            self.assertIn('not_converted_or_unit_certified', manifest['units_policy'])
            directory = Path(manifest['output_dir'])
            for api, artifact in manifest['validated_files'].items():
                self.assertEqual(len(pd.read_pickle(directory / artifact['file'])), 2)
            for url, kwargs in session.calls:
                self.assertTrue(url.startswith('https://tl.kaixin8.top/tushare/pro/'))
                self.assertEqual(kwargs['headers'], {'X-API-Key': 'unit-key'})
                self.assertTrue(kwargs['verify'])
                self.assertFalse(kwargs['allow_redirects'])
                self.assertEqual(kwargs['timeout'], 30)
                self.assertNotIn('never-send-legacy', json.dumps(kwargs))
            self.assertNotIn('unit-key', (directory / 'manifest.json').read_text(encoding='utf-8'))

    def test_retry_only_transient_pool_503_504_or_transport(self):
        transient = [Response({'error': 'upstream_pool_exhausted'}, 503),
                     Response({'message': 'gateway timeout'}, 504), requests.Timeout('timeout'),
                     requests.ConnectionError('connection reset'), requests.exceptions.ChunkedEncodingError('incomplete response')]
        for failure in transient:
            with self.subTest(failure=type(failure).__name__), TemporaryDirectory() as temp:
                session = FakeSession([failure])
                manifest = self.run_one(temp, session)
                self.assertEqual(manifest['status'], 'complete')
                self.assertEqual(len(session.calls), 2)
                self.assertEqual(len(manifest['requests'][0]['attempts']), 2)
                self.assertEqual(len(list((Path(manifest['output_dir']) / 'attempts').glob('*.json'))), 2)

    def test_permanent_transport_errors_are_audited_without_retry(self):
        failures = [requests.exceptions.SSLError('invalid certificate'),
                    requests.exceptions.InvalidURL('invalid unit-key URL'),
                    requests.exceptions.InvalidHeader('invalid header'),
                    requests.RequestException('unknown failure')]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__), TemporaryDirectory() as temp:
                session = FakeSession([failure])
                manifest = self.run_one(temp, session)
                self.assertEqual(manifest['status'], 'failed')
                self.assertEqual(len(session.calls), 1)
                self.assertEqual(manifest['validated_files'], {})
                directory = Path(manifest['output_dir'])
                record = json.loads(next((directory / 'attempts').glob('*.json')).read_text(encoding='utf-8'))
                self.assertEqual(record['transport_error_type'], type(failure).__name__)
                self.assertNotIn('unit-key', json.dumps(record))

    def test_400_and_non_pool_503_are_not_retried(self):
        for status in [400, 503]:
            with self.subTest(status=status), TemporaryDirectory() as temp:
                session = FakeSession([Response({'error': 'not_retryable'}, status)])
                manifest = self.run_one(temp, session)
                self.assertEqual(manifest['status'], 'failed')
                self.assertEqual(len(session.calls), 1)
                self.assertFalse(list(Path(manifest['output_dir']).glob('validated_*.pkl')))

    def test_two_attempt_cap_and_error_redaction(self):
        failure = Response({'error': 'upstream_pool_exhausted', 'message': 'X-API-Key=unit-key'}, 503)
        with TemporaryDirectory() as temp:
            session = FakeSession([failure, failure, failure])
            manifest = self.run_one(temp, session)
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(len(session.calls), 2)
            for path in Path(manifest['output_dir']).rglob('*.json'):
                self.assertNotIn('unit-key', path.read_text(encoding='utf-8'))

    def test_reached_row_limit_rejected_without_retry(self):
        with TemporaryDirectory() as temp:
            session = FakeSession()
            manifest = self.run_one(temp, session, bars_limit=2)
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(len(session.calls), 1)
            self.assertEqual(manifest['validated_files'], {})
            directory = Path(manifest['output_dir'])
            self.assertEqual(len(list((directory / 'attempts').glob('*.pkl'))), 1)
            record = json.loads(next((directory / 'attempts').glob('*.json')).read_text(encoding='utf-8'))
            self.assertIn('possible_truncation', record['issues'])

    def test_missing_session_blocks_all_validated_outputs_not_just_bad_shard(self):
        with TemporaryDirectory() as temp:
            session = FakeSession(missing_day=True)
            manifest = run_pilot(['885959.TI'], ['BK0877.DC'], '20260330', '20260331', ['20260331'],
                                 output_dir=temp, expected_sessions=['20260330', '20260331'],
                                 session=session, environ={'TUSHARE_API_KEY': 'unit-key'})
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(len(session.calls), 3)
            self.assertEqual(manifest['requests'][-1]['status'], 'validated')
            self.assertEqual(manifest['validated_files'], {})
            self.assertFalse(list(Path(manifest['output_dir']).glob('validated_*.pkl')))

    def test_wrong_source_response_rejected_and_cannot_fallback(self):
        with TemporaryDirectory() as temp:
            session = FakeSession(wrong_code=True)
            manifest = self.run_one(temp, session)
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(len(session.calls), 1)
            self.assertEqual(manifest['validated_files'], {})
        with TemporaryDirectory() as temp, self.assertRaisesRegex(ValueError, 'legacy'):
            run_pilot(['885959.TI'], [], '20260330', '20260331', [], output_dir=temp,
                       session=FakeSession(), environ={'TUSHARE_TOKEN': 'legacy-only'})

    def test_preflight_rejects_mixed_sources_and_unsupported_membership_history(self):
        for ths, dc, dates in [(['BK0877.DC'], [], []), ([], ['885959.TI'], ['20260331']),
                               ([], ['BK0877.DC'], ['20241219']), ([], ['BK0877.DC'], []),
                               (['885959.TI'], [], ['20260331'])]:
            with self.subTest(ths=ths, dc=dc, dates=dates), self.assertRaises(ValueError):
                build_requests(ths, dc, '20260330', '20260331', dates)

    def test_local_calendar_deduplicates_stock_rows_without_network(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'daily.pkl'
            pd.DataFrame({'date': pd.to_datetime(['2026-03-30', '2026-03-30', '2026-03-31'])}).to_pickle(path)
            self.assertEqual(load_expected_sessions(path), ['20260330', '20260331'])

    def test_date_preflight_rejects_ambiguous_or_intraday_values(self):
        for value in ['2026', 'today', '03/30/2026', '2026-03-30T00:00:00',
                      '20260230', 0, 0.0, True, pd.Timestamp('2026-03-30T10:00:00')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_requests(['885959.TI'], [], value, '20260331', [])
        for code in [None, 885959, ['885959.TI']]:
            with self.subTest(code=code), self.assertRaises(ValueError):
                build_requests([code], [], '20260330', '20260331', [])
        with self.assertRaises(ValueError):
            build_requests('885959.TI', [], '20260330', '20260331', [])

    def test_failed_run_has_its_own_directory_and_cannot_reuse_prior_validated_files(self):
        with TemporaryDirectory() as temp:
            complete = self.run_one(temp, FakeSession())
            failed = self.run_one(temp, FakeSession(wrong_code=True))
            self.assertNotEqual(complete['output_dir'], failed['output_dir'])
            self.assertTrue(list(Path(complete['output_dir']).glob('validated_*.pkl')))
            self.assertEqual(failed['status'], 'failed')
            self.assertEqual(failed['validated_files'], {})
            self.assertFalse(list(Path(failed['output_dir']).glob('validated_*.pkl')))
            saved = json.loads((Path(failed['output_dir']) / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(saved, failed)

    def test_unexpected_probe_failure_is_redacted_and_audited(self):
        with TemporaryDirectory() as temp, patch('fetch_theme_pilot.probe_request',
                                                side_effect=RuntimeError('unit-key never-send-legacy')):
            manifest = self.run_one(temp, FakeSession())
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(len(manifest['requests'][0]['attempts']), 1)
            directory = Path(manifest['output_dir'])
            record = json.loads(next((directory / 'attempts').glob('*.json')).read_text(encoding='utf-8'))
            self.assertEqual(record['status'], 'unexpected_error')
            self.assertNotIn('unit-key', json.dumps(record))
            self.assertNotIn('never-send-legacy', json.dumps(record))
            self.assertFalse(list(directory.glob('validated_*.pkl')))

    def test_raw_frame_write_failure_is_audited_and_blocks_publication(self):
        with TemporaryDirectory() as temp, patch('pandas.DataFrame.to_pickle',
                                                side_effect=OSError('unit-key disk error')):
            manifest = self.run_one(temp, FakeSession())
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(manifest['validated_files'], {})
            directory = Path(manifest['output_dir'])
            record = json.loads(next((directory / 'attempts').glob('*.json')).read_text(encoding='utf-8'))
            self.assertEqual(record['status'], 'raw_frame_write_error')
            self.assertFalse(record['raw_frame_saved'])
            self.assertNotIn('unit-key', json.dumps(record))
            self.assertFalse(list(directory.rglob('*.pending')))

    def test_publication_write_failure_removes_all_staged_aggregates(self):
        original = pd.DataFrame.to_pickle

        def fail_second_aggregate(frame, path, *args, **kwargs):
            if Path(path).name == 'validated_dc_daily.pkl.pending':
                raise OSError('unit-key publication failed')
            return original(frame, path, *args, **kwargs)

        with TemporaryDirectory() as temp, patch('pandas.DataFrame.to_pickle', new=fail_second_aggregate):
            manifest = run_pilot(['885959.TI'], ['BK0877.DC'], '20260330', '20260331', ['20260331'],
                                 output_dir=temp, session=FakeSession(), environ={'TUSHARE_API_KEY': 'unit-key'})
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(manifest['validated_files'], {})
            directory = Path(manifest['output_dir'])
            self.assertEqual(len(list((directory / 'attempts').glob('*.pkl'))), 3)
            self.assertFalse(list(directory.glob('validated_*.pkl')))
            self.assertFalse(list(directory.rglob('*.pending')))
            saved = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(saved, manifest)
            self.assertNotIn('unit-key', json.dumps(saved))

    def test_publication_rename_failure_removes_already_published_aggregate(self):
        original = Path.replace

        def fail_second_rename(path, target):
            if path.name == 'validated_dc_daily.pkl.pending':
                raise OSError('publication rename failed')
            return original(path, target)

        with TemporaryDirectory() as temp, patch('pathlib.Path.replace', new=fail_second_rename):
            manifest = run_pilot(['885959.TI'], ['BK0877.DC'], '20260330', '20260331', ['20260331'],
                                 output_dir=temp, session=FakeSession(), environ={'TUSHARE_API_KEY': 'unit-key'})
            self.assertEqual(manifest['status'], 'failed')
            directory = Path(manifest['output_dir'])
            self.assertEqual(manifest['validated_files'], {})
            self.assertFalse(list(directory.glob('validated_*.pkl')))
            self.assertFalse(list(directory.rglob('*.pending')))

    def test_unexpected_validator_error_keeps_attempt_audit(self):
        with TemporaryDirectory() as temp, patch('fetch_theme_pilot.validate_index_bars',
                                                side_effect=RuntimeError('unit-key unexpected validator error')):
            manifest = self.run_one(temp, FakeSession())
            self.assertEqual(manifest['status'], 'failed')
            directory = Path(manifest['output_dir'])
            record = json.loads(next((directory / 'attempts').glob('*.json')).read_text(encoding='utf-8'))
            self.assertEqual(record['validation_status'], 'failed')
            self.assertEqual(record['validation_error_type'], 'RuntimeError')
            self.assertTrue(record['raw_frame_saved'])
            self.assertNotIn('unit-key', json.dumps(record))

    def test_session_close_failure_produces_failed_manifest(self):
        session = FakeSession()

        def fail_close():
            raise OSError('unit-key close failed')

        session.close = fail_close
        with TemporaryDirectory() as temp, patch('fetch_theme_pilot.requests.Session', return_value=session):
            manifest = run_pilot(['885959.TI'], [], '20260330', '20260331', [], output_dir=temp,
                                 environ={'TUSHARE_API_KEY': 'unit-key'})
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(manifest['validated_files'], {})
            directory = Path(manifest['output_dir'])
            self.assertFalse(list(directory.glob('validated_*.pkl')))
            self.assertNotIn('unit-key', (directory / 'manifest.json').read_text(encoding='utf-8'))

    def test_cli_parses_explicit_scope_and_returns_failure_exit_code(self):
        argv = ['fetch_theme_pilot.py', '--ths-codes', '885959.TI', '--dc-codes', 'BK0877.DC',
                '--start-date', '20260330', '--end-date', '20260331', '--snapshot-dates', '20260331',
                '--bars-limit', '90', '--members-limit', '900', '--output-dir', 'pilot-output']
        manifest = dict(status='failed', output_dir='pilot-output/run', planned_requests=3, validated_files={})
        with patch.object(sys, 'argv', argv), patch('fetch_theme_pilot.run_pilot', return_value=manifest) as run:
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(), 1)
            self.assertEqual(run.call_args.args, (['885959.TI'], ['BK0877.DC'], '20260330', '20260331', ['20260331']))
            self.assertEqual(run.call_args.kwargs['bars_limit'], 90)
            self.assertEqual(run.call_args.kwargs['members_limit'], 900)
            self.assertEqual(run.call_args.kwargs['output_dir'], Path('pilot-output'))
            self.assertEqual(json.loads(output.getvalue()), manifest)


if __name__ == '__main__':
    unittest.main()
