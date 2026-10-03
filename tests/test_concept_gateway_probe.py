import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from probe_concept_gateway import probe_matrix, probe_request, run_probe, safe_text


class Response:
    def __init__(self, payload=None, status=200, content_type='application/json'):
        self.payload, self.status_code = payload, status
        self.headers = {'Content-Type': content_type}

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeSession:
    def __init__(self, response):
        self.response, self.calls = response, []

    def get(self, url, **kwargs):
        return self._call('GET', url, kwargs)

    def post(self, url, **kwargs):
        return self._call('POST', url, kwargs)

    def _call(self, method, url, kwargs):
        self.calls.append((method, url, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def table(fields, items):
    return {'code': 0, 'data': {'fields': fields, 'items': items}}


class ConceptGatewayProbeTests(unittest.TestCase):
    def test_matrix_is_bounded_and_only_requested_ths_dc_and_control(self):
        matrix = probe_matrix()
        self.assertEqual(len(matrix), 9)
        self.assertEqual({s['api'] for s in matrix}, {'daily', 'ths_index', 'ths_daily', 'ths_member', 'dc_index', 'dc_member', 'dc_daily'})
        self.assertEqual(matrix[5]['params']['idx_type'], '概念板块')
        self.assertNotIn('offset', matrix[5]['params'])
        self.assertEqual(matrix[6]['params']['offset'], 0)
        matrix[0]['params']['limit'] = 999
        self.assertEqual(probe_matrix()[0]['params']['limit'], 3)

    def test_transports_use_only_their_own_credential_and_cannot_override_origin(self):
        environment = dict(TUSHARE_API_KEY='gateway-secret-example', TUSHARE_TOKEN='legacy-secret-example',
                           TUSHARE_GATEWAY_URL='https://invalid.example', TUSHARE_URL='https://invalid.example')
        for transport, method, origin in [('gateway', 'GET', 'https://tl.kaixin8.top/tushare/pro/'),
                                          ('legacy', 'POST', 'https://t.xiaodefa.top/')]:
            with self.subTest(transport=transport), TemporaryDirectory() as temp:
                session = FakeSession(Response({'code': 503, 'message': 'temporarily unavailable'}, status=503))
                records = run_probe(transport, temp, session=session, environ=environment)
                self.assertEqual(len(session.calls), 9)
                self.assertEqual(len(records), 9)
                self.assertEqual(len(list((Path(temp) / transport).glob('*.json'))), 9)
                for actual_method, url, kwargs in session.calls:
                    self.assertEqual(actual_method, method)
                    self.assertTrue(url.startswith(origin))
                    self.assertTrue(kwargs['verify'])
                    self.assertFalse(kwargs['allow_redirects'])
                    self.assertEqual(kwargs['timeout'], 30)
                    if transport == 'gateway':
                        self.assertEqual(kwargs['headers'], {'X-API-Key': environment['TUSHARE_API_KEY']})
                        self.assertNotIn('json', kwargs)
                        self.assertNotIn(environment['TUSHARE_TOKEN'], json.dumps(kwargs))
                    else:
                        self.assertEqual(url, origin)
                        self.assertEqual(kwargs['json']['token'], environment['TUSHARE_TOKEN'])
                        self.assertEqual(set(kwargs['json']), {'api_name', 'token', 'params', 'fields'})
                        self.assertNotIn('headers', kwargs)
                        self.assertNotIn(environment['TUSHARE_API_KEY'], json.dumps(kwargs))

    def test_missing_selected_key_does_not_fallback_to_other_origin(self):
        for transport, environment in [('gateway', {'TUSHARE_TOKEN': 'legacy-only'}),
                                       ('legacy', {'TUSHARE_API_KEY': 'gateway-only'})]:
            session = FakeSession(Response())
            with self.subTest(transport=transport), TemporaryDirectory() as temp, self.assertRaisesRegex(ValueError, 'no fallback'):
                run_probe(transport, temp, session=session, environ=environment)
            self.assertEqual(session.calls, [])

    def test_http_errors_preserve_safe_diagnostics_without_credentials(self):
        key = 'tsr_fake_secret_for_unit_test_1234'
        other = 'abcdef0123456789abcdef0123456789abcd'
        for status in [400, 503]:
            payload = dict(code=status, msg=f'invalid key: {key}', message='upstream unavailable',
                           error={'token': other}, detail=f'X-API-Key={key}; token=other-unknown-secret')
            session = FakeSession(Response(payload, status=status))
            record, frame = probe_request(session, 'gateway', key, probe_matrix()[0], secrets=(other,))
            serialized = json.dumps(record)
            self.assertEqual(record['http_status'], status)
            self.assertEqual(record['status'], 'http_error')
            self.assertEqual(set(record['diagnostics']), {'msg', 'message', 'error', 'detail', 'code'})
            self.assertIn('upstream unavailable', serialized)
            for secret in [key, other, 'other-unknown-secret']:
                self.assertNotIn(secret, serialized)
            self.assertIsNone(frame)

    def test_html_is_only_classified_and_redirect_is_not_followed(self):
        for status, expected in [(200, 'html_response'), (503, 'http_error'), (302, 'redirect_blocked')]:
            session = FakeSession(Response('<html>secret body</html>', status=status, content_type='text/html'))
            record, frame = probe_request(session, 'gateway', 'unit-key', probe_matrix()[0])
            self.assertEqual(record['status'], expected)
            self.assertEqual(record['response_type'], 'html')
            self.assertNotIn('secret body', json.dumps(record))
            self.assertEqual(len(session.calls), 1)
            self.assertFalse(session.calls[0][2]['allow_redirects'])
            self.assertIsNone(frame)

    def test_transport_error_is_redacted_and_not_retried(self):
        session = FakeSession(requests.Timeout('timeout; X-API-Key=unit-key'))
        record, frame = probe_request(session, 'gateway', 'unit-key', probe_matrix()[0])
        self.assertEqual(record['status'], 'transport_error')
        self.assertNotIn('unit-key', json.dumps(record))
        self.assertEqual(len(session.calls), 1)
        self.assertIsNone(frame)

    def test_wrong_dates_wrong_code_empty_and_duplicate_catalog_are_not_ok(self):
        cases = [
            (probe_matrix()[2], table(['ts_code', 'trade_date'], [['885959.TI', '20260401']]), 'wrong_dates'),
            (probe_matrix()[2], table(['ts_code', 'trade_date'], [['885001.TI', '20260331']]), 'wrong_code'),
            (probe_matrix()[3], table(['ts_code', 'trade_date'], [['885959.TI', '20260930']]), 'wrong_dates'),
            (probe_matrix()[2], table(['ts_code', 'trade_date'], []), 'empty'),
            (probe_matrix()[1], table(['ts_code', 'name', 'count'], [['885959.TI', 'PCB', 1], ['885959.TI', 'PCB', 2]]), 'duplicate_keys'),
            (probe_matrix()[5], table(['ts_code', 'name', 'trade_date'], [['BK001', 'PCB', '20260930']]), 'wrong_dates'),
            (probe_matrix()[7], table(['ts_code', 'con_code', 'trade_date'], [['BK001', '600183.SH', '20260331']]), 'wrong_code'),
        ]
        for spec, payload, issue in cases:
            with self.subTest(name=spec['name'], issue=issue):
                record, frame = probe_request(FakeSession(Response(payload)), 'gateway', 'unit-key', spec)
                self.assertIn(issue, record['issues'])
                self.assertNotEqual(record['status'], 'ok')
                self.assertFalse(record['point_in_time_verified'])
                self.assertIsNotNone(frame)

    def test_ths_members_always_current_only_even_when_server_adds_old_date(self):
        payload = table(['ts_code', 'con_code', 'trade_date'], [['885959.TI', '002384.SZ', '20260331']])
        record, frame = probe_request(FakeSession(Response(payload)), 'gateway', 'unit-key', probe_matrix()[4])
        self.assertEqual(record['membership_policy'], 'current_only')
        self.assertFalse(record['point_in_time_verified'])
        self.assertEqual(record['status'], 'ok')
        self.assertEqual(len(frame), 1)

    def test_normal_raw_frames_saved_but_bad_schema_or_secret_frame_not_saved(self):
        payload = table(['ts_code', 'trade_date'], [['000001.SZ', '20260331']])
        with TemporaryDirectory() as temp:
            records = run_probe('gateway', temp, session=FakeSession(Response(payload)),
                                environ={'TUSHARE_API_KEY': 'unit-key'})
            for record in records:
                self.assertTrue(record['raw_frame_saved'])
                frame = pd.read_pickle(Path(temp) / 'gateway' / record['raw_frame_file'])
                self.assertEqual(frame.ts_code.iloc[0], '000001.SZ')
                self.assertNotIn('unit-key', json.dumps(record))
        for payload in [table(['ts_code'], [['unit-key']]),
                        table(['token'], [['unknown-secret']]),
                        table(['ts_code', 'trade_date'], [['x']]),
                        {'code': 0, 'data': 'invalid'}]:
            record, frame = probe_request(FakeSession(Response(payload)), 'gateway', 'unit-key', probe_matrix()[0])
            self.assertIsNone(frame)
            self.assertNotEqual(record['status'], 'ok')
            self.assertNotIn('unit-key', json.dumps(record))

    def test_redaction_occurs_before_short_error_truncation(self):
        value = 'prefix ' + 'secret-value' * 100
        self.assertNotIn('secret', safe_text(value, ('secret-value',), 12))
        self.assertNotIn('another-secret', safe_text('Authorization: Bearer another-secret'))


if __name__ == '__main__':
    unittest.main()
