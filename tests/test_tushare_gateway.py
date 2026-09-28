import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from data_pipeline import tushare_config as tc


class GatewayTests(unittest.TestCase):
    def client(self, payload, status=200):
        response = Mock(status_code=status)
        response.json.return_value = payload
        session = Mock()
        session.get.return_value = response
        return tc.GatewayClient('test-secret', session=session), session

    def test_dataframe_and_verified_tls_header(self):
        client, session = self.client({'code': 0, 'data': {'fields': ['ts_code', 'close'], 'items': [['000001.SZ', 12.]]}})
        df = client.daily(ts_code='000001.SZ', fields='ts_code,close')
        self.assertEqual(df.close.iloc[0], 12.)
        kwargs = session.get.call_args.kwargs
        self.assertTrue(kwargs['verify'])
        self.assertEqual(kwargs['headers']['X-API-Key'], 'test-secret')
        self.assertNotIn('test-secret', str(kwargs['params']))

    def test_error_does_not_become_empty_data_or_leak_body(self):
        client, _ = self.client({'error': 'test-secret'}, 503)
        with self.assertRaisesRegex(RuntimeError, '^daily: HTTP 503$'):
            client.daily()

    def test_bad_schema_rejected(self):
        for payload in ({'code': -1}, {'ok': False}, {'code': 0, 'data': None}):
            client, _ = self.client(payload)
            with self.assertRaises(RuntimeError):
                client.adj_factor()

    def test_query_and_empty_valid_table(self):
        client, _ = self.client({'code': 0, 'data': {'fields': ['ts_code'], 'items': []}})
        self.assertEqual(list(client.query('dividend').columns), ['ts_code'])

    def test_environment_selects_gateway_without_sdk_token(self):
        with patch.dict('os.environ', {'TUSHARE_API_KEY': 'test-secret'}, clear=True), patch.object(tc, '_pro', None):
            self.assertIsInstance(tc.get_pro(), tc.GatewayClient)


if __name__ == '__main__':
    unittest.main()
