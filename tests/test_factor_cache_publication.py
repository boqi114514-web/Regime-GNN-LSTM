import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import small_account_backtest as engine


class FactorCachePublicationTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(ts_code=['000001.SZ'],trade_date=['20260130'],adj_factor=[1.25]))

    def test_valid_existing_snapshot_is_read_only(self):
        with tempfile.TemporaryDirectory() as task:
            root=Path(task); folder=root/'adj_month_end';folder.mkdir()
            path=folder/'20260130.pkl';self.frame().to_pickle(path)
            before=path.read_bytes()
            with patch.object(engine,'ROOT',root),patch.object(pd.DataFrame,'to_pickle',side_effect=AssertionError('unexpected rewrite')):
                self.assertEqual(engine.factor_snapshot(None,pd.Timestamp('2026-01-30')),{'000001.SZ':1.25})
            self.assertEqual(path.read_bytes(),before)

    def test_new_snapshot_atomically_published_and_stage_removed(self):
        with tempfile.TemporaryDirectory() as task:
            root=Path(task)
            with patch.object(engine,'ROOT',root),patch.object(engine,'fetch_pages',return_value=self.frame()):
                self.assertEqual(engine.factor_snapshot(None,pd.Timestamp('2026-01-30')),{'000001.SZ':1.25})
            self.assertEqual([p.name for p in (root/'adj_month_end').iterdir()],['20260130.pkl'])
            self.assertEqual(pd.read_pickle(root/'adj_month_end/20260130.pkl').adj_factor.tolist(),[1.25])

    def test_invalid_new_snapshot_never_published(self):
        with tempfile.TemporaryDirectory() as task:
            root=Path(task)
            with patch.object(engine,'ROOT',root),patch.object(engine,'fetch_pages',return_value=self.frame().assign(adj_factor=-1)):
                with self.assertRaises(ValueError):engine.factor_snapshot(None,pd.Timestamp('2026-01-30'))
            self.assertFalse((root/'adj_month_end/20260130.pkl').exists())


if __name__=='__main__':unittest.main()
