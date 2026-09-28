import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from data_pipeline.execution_data import validate_rows


class ExecutionDataTests(unittest.TestCase):
    def test_identical_duplicate_removed(self):
        frame = pd.DataFrame({'ts_code': ['x', 'x'], 'trade_date': ['20230131']*2, 'adj_factor': [1., 1.]})
        self.assertEqual(len(validate_rows(frame, ['ts_code', 'trade_date'], ['adj_factor'], '20230131')), 1)

    def test_conflict_rejected(self):
        frame = pd.DataFrame({'ts_code': ['x', 'x'], 'trade_date': ['20230131']*2, 'adj_factor': [1., 2.]})
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            validate_rows(frame, ['ts_code', 'trade_date'], ['adj_factor'])

    def test_wrong_date_rejected(self):
        frame = pd.DataFrame({'ts_code': ['x'], 'trade_date': ['20230130'], 'adj_factor': [1.]})
        with self.assertRaisesRegex(ValueError, 'outside'):
            validate_rows(frame, ['ts_code', 'trade_date'], ['adj_factor'], '20230131')
