from contextlib import nullcontext
import copy
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import research_daily_opportunity_liquid_filter as module


class LiquidDailyFilterTests(unittest.TestCase):
    def store(self, codes, percentiles):
        n = len(codes)
        values = np.zeros((2, n, 2), dtype=np.float32)
        values[..., 0] = np.asarray(percentiles, dtype=np.float32)
        values[..., 1] = 1
        return SimpleNamespace(dates=pd.to_datetime(['2026-04-01', '2026-04-02']),
                               stock_codes=tuple(codes), features=values,
                               feature_names=('amount_percentile', 'available__amount_percentile'),
                               valid_features=np.ones((2, n), dtype=bool))

    def predictions(self, codes, utility=None):
        n = len(codes)
        return pd.DataFrame(dict(signal_date=[pd.Timestamp('2026-04-01')]*n, ts_code=codes,
                                 utility=np.arange(n, 0, -1)/100 if utility is None else utility,
                                 mu20=np.arange(n)/100, risk20=np.full(n, .05)),
                            index=np.arange(n)*2+9)

    def test_liquidity_applies_before_model_top40_not_after(self):
        codes = [f'{600001+i}.SH' for i in range(43)]
        store = self.store(codes, [.50]*40+[.95, .96, 1.])
        checks = module.liquidity_checks(self.predictions(codes), store)
        self.assertEqual(checks.loc[checks.in_model_pool, 'ts_code'].tolist(), codes[-3:])
        self.assertTrue(checks.amount_eligible.iloc[40])  # Exact stored float32 boundary.

    def test_current_masks_and_board_permission_control_only_new_pool(self):
        codes = ['600001.SH', '600002.SH', '600003.SH', '300001.SZ', '688001.SH', '000001.SZ']
        store = self.store(codes, [1.]*6)
        store.features[0, 0, 1] = 0  # Unknown, even if encoded value looks strong.
        store.valid_features[0, 1] = False
        checks = module.liquidity_checks(self.predictions(codes, [.1, .1, -.1, .9, .8, .05]), store)
        self.assertEqual(checks.in_model_pool.tolist(), [False, False, False, False, False, True])
        self.assertEqual(len(checks), len(codes))

    def test_future_factor_mutation_does_not_change_current_eligibility(self):
        codes = ['600001.SH', '600002.SH']
        store = self.store(codes, [.9, .98])
        predictions = self.predictions(codes)
        expected = module.liquidity_checks(predictions, store)
        store.features[1] = np.nan
        store.valid_features[1] = False
        pd.testing.assert_frame_equal(module.liquidity_checks(predictions, store), expected)
        store.features[0, 0, 0] = .99
        self.assertTrue(module.liquidity_checks(predictions, store).in_model_pool.iloc[0])

    def fixture(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        source, out = root/'source', root/'target'
        source.mkdir()
        (source/'checkpoints').mkdir()
        codes = ['600001.SH', '600002.SH', '300001.SZ', '000001.SZ']
        p = self.predictions(codes)
        store = self.store(codes, [1., .99, 1., .97])
        p.to_pickle(source/'predictions.pkl')
        with (source/'feature_store.pkl').open('wb') as handle:
            pickle.dump(store, handle)
        for name in ('feature_manifest.json', 'frozen_edges.pkl', 'fit_audits.json', 'prediction_manifest.json'):
            (source/name).write_bytes(b'fixture')
        (source/'experiment_protocol.json').write_text('{"source_sha256":{}}', encoding='utf-8')
        seen = []

        class Limits:
            def __init__(self, folder, **kwargs):
                self.sources = {}
                self.folder = Path(folder)/'daily_limits'
                self.folder.mkdir(exist_ok=True)

            def __call__(self, day, code):
                seen.append((day, code))
                raw = pd.DataFrame(dict(ts_code=[code], trade_date=[f'{day:%Y%m%d}'],
                                        up_limit=[10.5 if code == '600001.SH' else 11.],
                                        down_limit=[9.5 if code == '600001.SH' else 9.]))
                path = self.folder/f'{code}_{day:%Y%m%d}.pkl'
                raw.to_pickle(path)
                self.sources[str(path.resolve())] = module.sha256(path)
                return raw.iloc[0]

        return source, out, p, store, Limits, seen

    def test_all_native_rows_preserved_and_band_manifest_recomputes(self):
        source, out, p, store, limits, seen = self.fixture()
        reader = lambda folder: pd.read_pickle(Path(folder)/'predictions.pkl')
        with patch.object(module, 'leadership_variant', side_effect=nullcontext), \
                patch.object(module, 'validate_predictions', side_effect=reader), \
                patch.object(module, 'validate_leadership', return_value=(store, None)), \
                patch('research_daily_opportunity_band_filter.validate_predictions', side_effect=reader), \
                patch.object(module, 'OfficialLimits', limits):
            actual = module.filter_liquid_predictions(source, out, offline=True)
            pd.testing.assert_frame_equal(actual.drop(columns='eligible'), p, check_exact=True)
            self.assertEqual(actual.eligible.tolist(), [False, True, False, True])
            self.assertEqual([code for _, code in seen], ['600001.SH', '600002.SH', '000001.SZ'])
            self.assertTrue(all(day == pd.Timestamp('2026-04-01') for day, _ in seen))
            checked = module.validate_liquidity_inputs(out)
            self.assertEqual(checked['manifest']['model_pool'], 3)
            self.assertEqual(checked['manifest']['accepted'], 2)

    def test_recalculation_rejects_flag_outside_verified_pool(self):
        source, out, p, store, limits, _ = self.fixture()
        reader = lambda folder: pd.read_pickle(Path(folder)/'predictions.pkl')
        with patch.object(module, 'leadership_variant', side_effect=nullcontext), \
                patch.object(module, 'validate_predictions', side_effect=reader), \
                patch.object(module, 'validate_leadership', return_value=(store, None)), \
                patch('research_daily_opportunity_band_filter.validate_predictions', side_effect=reader), \
                patch.object(module, 'OfficialLimits', limits):
            actual = module.filter_liquid_predictions(source, out, offline=True)
            actual.loc[actual.ts_code.eq('300001.SZ'), 'eligible'] = True
            actual.to_pickle(out/'predictions.pkl')
            with self.assertRaisesRegex(ValueError, 'final eligibility'):
                module.validate_liquidity_inputs(out)

    def test_missing_official_band_cannot_publish_complete_filter(self):
        source, out, p, store, _, _ = self.fixture()

        class Missing:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, *args): raise RuntimeError('Missing exact official limits')

        with patch.object(module, 'leadership_variant', side_effect=nullcontext), \
                patch.object(module, 'validate_predictions', return_value=p), \
                patch.object(module, 'validate_leadership', return_value=(store, None)), \
                patch.object(module, 'OfficialLimits', Missing):
            with self.assertRaisesRegex(RuntimeError, 'Missing exact'):
                module.filter_liquid_predictions(source, out, offline=True)
        self.assertFalse((out/'prediction_manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
