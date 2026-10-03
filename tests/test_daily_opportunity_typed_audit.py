import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_daily_opportunity_typed import dataframe_limit_status, original


class TypedAuditTests(unittest.TestCase):
    def test_non_dataframe_targets_are_not_mistaken_for_official_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, vendor = root/'targets.pkl', root/'official.pkl'
            with target.open('wb') as handle:
                pickle.dump(SimpleNamespace(path_payoff=[1., 2.]), handle)
            pd.DataFrame(dict(ts_code=['600001.SH'], trade_date=['20260105'],
                              up_limit=[11.], down_limit=[9.])).to_pickle(vendor)
            status = dict(account_complete=True, source_sha256={str(p):original.sha(p) for p in (target, vendor)})
            filtered = dataframe_limit_status(status)
            self.assertEqual(set(filtered['source_sha256']), {str(vendor)})
            self.assertIn(str(target), status['source_sha256'])
            limits, fingerprints = original.load_exact_limits(root, filtered, [(pd.Timestamp('2026-01-05'), '600001.SH')])
            self.assertEqual(limits[(pd.Timestamp('2026-01-05'), '600001.SH')], (9., 11.))
            self.assertIn(str(vendor.resolve()), fingerprints)

    def test_target_hash_is_checked_even_when_it_is_not_a_vendor_table(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'targets.pkl'
            pd.to_pickle(SimpleNamespace(value=1), path)
            status = dict(source_sha256={str(path):'wrong-hash'})
            with self.assertRaisesRegex(AssertionError, 'fingerprint'):
                dataframe_limit_status(status)


if __name__ == '__main__':
    unittest.main()
