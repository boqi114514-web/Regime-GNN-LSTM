from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.complete_suspension_checks import (
    explicit_full_day_dates, fill_observed_dates, publish_exclusive_pickle,
    slice_for_day, validate_history,
)
from research_daily_graph_data import sha256


class SuspensionCacheCompletionTests(unittest.TestCase):
    def history(self):
        return pd.DataFrame(dict(ts_code=['600193.SH']*4,
                                 trade_date=['20260630', '20260701', '20260702', '20260703'],
                                 suspend_type=['S', 'S', 'S', 'R'],
                                 suspend_timing=[None, None, '09:30-10:00', None]))

    def test_old_s_event_cannot_create_an_unobserved_future_suspension(self):
        history = validate_history(self.history(), '600193.SH', '20000103', '20260924')
        self.assertEqual(explicit_full_day_dates(history), [pd.Timestamp('2026-06-30'), pd.Timestamp('2026-07-01')])
        with self.assertRaisesRegex(ValueError, 'No explicit full-day'):
            slice_for_day(history, '2026-06-29', '2026-08-17')
        with self.assertRaises(ValueError):
            slice_for_day(history, '2026-06-29', '2026-07-02')

    def test_slice_cannot_expose_future_resume_or_past_prequote_event(self):
        part = slice_for_day(self.history(), '2026-06-29', '2026-07-01')
        self.assertEqual(part.trade_date.tolist(), ['20260630', '20260701'])
        self.assertFalse(part.suspend_type.eq('R').any())

    def test_exclusive_publication_never_overwrites_existing_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'observed.pkl'
            self.assertTrue(publish_exclusive_pickle(self.history(), path))
            before = sha256(path)
            self.assertFalse(publish_exclusive_pickle(pd.DataFrame({'changed':[1]}), path))
            self.assertEqual(sha256(path), before)
            self.assertEqual(list(Path(folder).glob('*.staging')), [])

    def test_fill_publishes_only_actual_s_dates_and_keeps_original_quotes(self):
        daily = pd.DataFrame(dict(ts_code=['600193.SH', '600001.SH', '600001.SH', '600001.SH'],
                                  date=pd.to_datetime(['20260629', '20260630', '20260701', '20260817']),
                                  close=[.16, 10., 10., 10.]))
        with tempfile.TemporaryDirectory() as folder:
            records = fill_observed_dates(self.history(), '600193.SH', daily, folder,
                                          after='2026-06-29', through='2026-09-24')
            self.assertEqual(len(records), 4)
            self.assertFalse((Path(folder)/'600193.SH_20260817_suspend.pkl').exists())
            mark = pd.read_pickle(Path(folder)/'600193.SH_20260701_daily.pkl')
            self.assertEqual(mark.trade_date.tolist(), ['20260629'])
            self.assertEqual(mark.close.tolist(), [.16])

    def test_wrong_stock_dates_duplicates_and_full_page_are_rejected(self):
        history = self.history()
        with self.assertRaises(ValueError):
            validate_history(history, '600421.SH', '20000103', '20260924')
        with self.assertRaises(ValueError):
            validate_history(history, '600193.SH', '20000103', '20260701')
        with self.assertRaises(ValueError):
            validate_history(pd.concat((history, history.iloc[:1])), '600193.SH', '20000103', '20260924')
        with self.assertRaisesRegex(ValueError, 'may be capped'):
            validate_history(pd.concat([history]*1250), '600193.SH', '20000103', '20260924')


if __name__ == '__main__':
    unittest.main()
