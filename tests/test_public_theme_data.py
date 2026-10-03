import unittest

from data_pipeline.public_theme_data import parse_eastmoney_history, parse_ths_year


class PublicThemeTest(unittest.TestCase):
    def em(self):
        return dict(rc=0, data=dict(code='BK0877', name='PCB', klines=[
            '2026-03-31,10,11,12,9,200,22000,30,10,1,2']))

    def test_response_identity_and_calendar_must_match(self):
        frame = parse_eastmoney_history(self.em(), 'BK0877.DC', '20260331',
                                       '20260331', expected_sessions=['20260331'])
        self.assertEqual(frame.iloc[0].amount, 22000)
        self.assertEqual(frame.iloc[0].source_vol, 200)
        wrong = self.em(); wrong['data']['code'] = 'BK1136'
        with self.assertRaisesRegex(ValueError, 'code mismatch'):
            parse_eastmoney_history(wrong, 'BK0877.DC', '20260331', '20260331')
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            parse_eastmoney_history(self.em(), 'BK0877.DC', '20260330',
                                   '20260331', expected_sessions=['20260330', '20260331'])

    def test_malformed_geometry_amount_and_duplicate_are_rejected(self):
        for row in ['2026-03-31,10,11,8,9,200,22000,30,10,1,2',
                    '2026-03-31,10,11,12,9,200,nan,30,10,1,2']:
            obj = self.em(); obj['data']['klines'] = [row]
            with self.assertRaises(ValueError):
                parse_eastmoney_history(obj, 'BK0877.DC', '20260331', '20260331')
        obj = self.em(); obj['data']['klines'] *= 2
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            parse_eastmoney_history(obj, 'BK0877.DC', '20260331', '20260331')

    def test_ths_year_and_volume_are_not_silently_rewritten(self):
        obj = {'data': '20260331,10,12,9,11,20000,22000,,,,0'}
        frame = parse_ths_year(obj, '885959.TI', 2026)
        self.assertEqual(frame.iloc[0].vol, 20000)
        self.assertEqual(frame.attrs['source_vol_unit'], 'unconfirmed_raw')
        with self.assertRaisesRegex(ValueError, 'outside'):
            parse_ths_year(obj, '885959.TI', 2025)

    def test_no_current_membership_is_implied(self):
        frame = parse_eastmoney_history(self.em(), 'BK0877.DC', '20260331', '20260331')
        self.assertFalse(frame.attrs['historical_membership'])


if __name__ == '__main__':
    unittest.main()
