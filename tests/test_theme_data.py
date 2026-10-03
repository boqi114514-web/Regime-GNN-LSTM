import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_pipeline.theme_data import (
    reject_latest_membership_for_history,
    validate_catalog,
    validate_index_bars,
    validate_members,
)


class ThemeDataTests(unittest.TestCase):
    def catalog(self):
        return pd.DataFrame(dict(ts_code=['BK001', 'BK002'], name=['PCB', 'Optical'],
                                 trade_date=['20260401'] * 2, vendor_field=[1, 2]))

    def members(self):
        return pd.DataFrame(dict(trade_date=['20260401'] * 3, ts_code=['BK001'] * 3,
                                 con_code=['600183.SH', '300308.SZ', '688001.SH']))

    def bars(self):
        return pd.DataFrame(dict(ts_code=['BK001'] * 3,
                                 trade_date=['20260401', '20260402', '20260403'],
                                 open=[10., 11., 12.], high=[12., 12., 13.],
                                 low=[9., 10., 11.], close=[11., 11.5, 12.5],
                                 vol=[1., 0., 3.], amount=[10., 0., 36.]))

    def test_catalog_accepts_exact_date_without_modifying_input(self):
        frame = self.catalog()
        original = frame.copy(deep=True)
        result = validate_catalog(frame, pd.Timestamp('2026-04-01'), limit=5000)
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_frame_equal(result, original)
        self.assertIsNot(result, frame)
        result.loc[0, 'name'] = 'changed'
        pd.testing.assert_frame_equal(frame, original)

    def test_catalog_rejects_ignored_date_and_duplicate_current_snapshots(self):
        for returned in ['20260930', '20260331']:
            with self.subTest(returned=returned), self.assertRaisesRegex(ValueError, 'trade_date'):
                validate_catalog(self.catalog().assign(trade_date=returned), '20260401')
        for duplicate in [self.catalog().iloc[:1], self.catalog().iloc[:1].assign(name='renamed')]:
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                validate_catalog(pd.concat([self.catalog(), duplicate]), '20260401')

    def test_catalog_empty_missing_or_invalid_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Empty'):
            validate_catalog(self.catalog().iloc[:0], '20260401')
        for column in ['ts_code', 'name', 'trade_date']:
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_catalog(self.catalog().drop(columns=column), '20260401')
        for column in ['ts_code', 'name']:
            for value in [None, '', '   ']:
                frame = self.catalog()
                frame.loc[0, column] = value
                with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                    validate_catalog(frame, '20260401')

    def test_catalog_limit_reached_or_exceeded_is_not_silently_accepted(self):
        for limit in [1, 2]:
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, 'truncation'):
                validate_catalog(self.catalog(), '20260401', limit=limit)
        self.assertEqual(len(validate_catalog(self.catalog(), '20260401', limit=3)), 2)
        for limit in [0, -1, 2.5, True]:
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, 'limit'):
                validate_catalog(self.catalog(), '20260401', limit=limit)

    def test_members_include_all_boards_and_preserve_inputs(self):
        frame = self.members()
        original = frame.copy(deep=True)
        result = validate_members(frame, '2026-04-01', 'BK001', limit=5000)
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_frame_equal(result, original)
        self.assertEqual(len(result), 3)
        self.assertIsNot(result, frame)

    def test_members_reject_ignored_theme_filter_and_future_or_wrong_date(self):
        frame = self.members()
        frame.loc[1, 'ts_code'] = 'BK002'
        with self.assertRaisesRegex(ValueError, 'theme'):
            validate_members(frame, '20260401', 'BK001')
        for returned in ['20260930', '20260331']:
            frame = self.members()
            frame.loc[0, 'trade_date'] = returned
            with self.subTest(returned=returned), self.assertRaisesRegex(ValueError, 'historical'):
                validate_members(frame, '20260401', 'BK001')

    def test_members_reject_duplicate_empty_missing_and_truncation(self):
        frame = self.members()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_members(pd.concat([frame, frame.iloc[:1]]), '20260401', 'BK001')
        with self.assertRaisesRegex(ValueError, 'Empty'):
            validate_members(frame.iloc[:0], '20260401', 'BK001')
        with self.assertRaisesRegex(ValueError, 'truncation'):
            validate_members(frame, '20260401', 'BK001', limit=3)
        for column in ['trade_date', 'ts_code', 'con_code']:
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_members(frame.drop(columns=column), '20260401', 'BK001')
        frame.loc[0, 'con_code'] = None
        with self.assertRaisesRegex(ValueError, 'con_code'):
            validate_members(frame, '20260401', 'BK001')

    def test_current_ths_membership_cannot_be_backfilled_into_history(self):
        for source in ['ths', 'ths_member', 'THS_MEMBER', '同花顺', 'tonghuashun']:
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'historical'):
                reject_latest_membership_for_history(source)
        self.assertIsNone(reject_latest_membership_for_history('dc_member'))
        self.assertIsNone(reject_latest_membership_for_history('tdx_member'))
        with self.assertRaises(ValueError):
            reject_latest_membership_for_history('')

    def test_bars_accept_complete_calendar_zero_volume_and_preserve_input(self):
        frame = self.bars()
        original = frame.copy(deep=True)
        result = validate_index_bars(frame, 'BK001', '20260401', '20260403',
                                     expected_sessions=pd.bdate_range('2026-04-01', '2026-04-03'),
                                     limit=2000)
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_frame_equal(result, original)
        self.assertIsNot(result, frame)
        result.loc[0, 'close'] = 999.
        pd.testing.assert_frame_equal(frame, original)

    def test_bars_reject_ignored_code_date_filters_and_duplicate_dates(self):
        frame = self.bars()
        frame.loc[1, 'ts_code'] = 'BK002'
        with self.assertRaisesRegex(ValueError, 'code'):
            validate_index_bars(frame, 'BK001', '20260401', '20260403')
        with self.assertRaisesRegex(ValueError, 'interval'):
            validate_index_bars(self.bars(), 'BK001', '20260401', '20260402')
        with self.assertRaisesRegex(ValueError, 'interval'):
            validate_index_bars(self.bars(), 'BK001', '20260402', '20260403')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_index_bars(pd.concat([self.bars(), self.bars().iloc[:1]]),
                                'BK001', '20260401', '20260403')

    def test_bars_reject_nonfinite_nonpositive_prices_and_negative_volume(self):
        for column in ['open', 'high', 'low', 'close']:
            for value in [0., -1., np.inf, np.nan, 'bad']:
                frame = self.bars().astype({column: object})
                frame.loc[0, column] = value
                with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                    validate_index_bars(frame, 'BK001', '20260401', '20260403')
        for value in [-1., np.inf, np.nan]:
            with self.subTest(vol=value), self.assertRaises(ValueError):
                validate_index_bars(self.bars().assign(vol=value), 'BK001', '20260401', '20260403')

    def test_bars_reject_impossible_ohlc_geometry(self):
        for column, value in [('open', 13.), ('close', 8.), ('high', 10.), ('low', 11.)]:
            frame = self.bars()
            frame.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, 'geometry'):
                validate_index_bars(frame, 'BK001', '20260401', '20260403')

    def test_nullable_numeric_missing_values_are_not_skipped_by_finite_check(self):
        for column in ['open', 'high', 'low', 'close', 'vol']:
            frame = self.bars().astype({column: 'Float64'})
            frame.loc[0, column] = pd.NA
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, 'invalid'):
                validate_index_bars(frame, 'BK001', '20260401', '20260403')

    def test_optional_calendar_is_exact_and_does_not_invent_missing_sessions(self):
        frame = self.bars().iloc[[0, 2]]
        self.assertEqual(len(validate_index_bars(frame, 'BK001', '20260401', '20260403')), 2)
        with self.assertRaisesRegex(ValueError, 'missing'):
            validate_index_bars(frame, 'BK001', '20260401', '20260403',
                                expected_sessions=['20260401', '20260402', '20260403'])
        with self.assertRaisesRegex(ValueError, 'extra'):
            validate_index_bars(self.bars(), 'BK001', '20260401', '20260403',
                                expected_sessions=['20260401', '20260403'])
        for expected in [['20260401', '20260401'], ['20260331'], '20260401']:
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                validate_index_bars(frame, 'BK001', '20260401', '20260403', expected_sessions=expected)

    def test_bars_empty_missing_fields_and_endpoint_limit_are_rejected(self):
        frame = self.bars()
        with self.assertRaisesRegex(ValueError, 'Empty'):
            validate_index_bars(frame.iloc[:0], 'BK001', '20260401', '20260403')
        with self.assertRaisesRegex(ValueError, 'truncation'):
            validate_index_bars(frame, 'BK001', '20260401', '20260403', limit=3)
        for column in ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']:
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_index_bars(frame.drop(columns=column), 'BK001', '20260401', '20260403')

    def test_dates_are_valid_calendar_days_and_intervals_are_not_reversed(self):
        for value in ['20260230', 'today', None, pd.NaT, pd.Timestamp('2026-04-01 09:30')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_catalog(self.catalog(), value)
        with self.assertRaisesRegex(ValueError, 'start_date'):
            validate_index_bars(self.bars(), 'BK001', '20260403', '20260401')
        with self.assertRaisesRegex(ValueError, 'trade_date'):
            validate_catalog(self.catalog().assign(trade_date=None), '20260401')


if __name__ == '__main__':
    unittest.main()
