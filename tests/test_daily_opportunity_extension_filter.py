from contextlib import nullcontext
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
import research_daily_opportunity_extension_filter as module
import research_daily_opportunity_liquid_filter as liquid


class ExtensionFilterTests(unittest.TestCase):
    def inputs(self, returns):
        n = len(returns)
        codes = [f'{600001+i}.SH' for i in range(n)]
        values = np.ones((2, n, 4), dtype=np.float32)
        values[..., 2] = np.asarray(returns, dtype=np.float32)
        store = SimpleNamespace(dates=pd.to_datetime(['2026-04-01', '2026-04-02']),
                                stock_codes=tuple(codes), features=values,
                                feature_names=('amount_percentile', 'available__amount_percentile',
                                               'return20', 'available__return20'),
                                valid_features=np.ones((2, n), dtype=bool))
        p = pd.DataFrame(dict(signal_date=[pd.Timestamp('2026-04-01')]*n, ts_code=codes,
                              utility=np.arange(n, 0, -1)/100, mu20=np.arange(n)/100),
                         index=np.arange(n)*3+2)
        return p, store

    def test_boundary_and_unknown_preserve_amount_semantics(self):
        p, store = self.inputs([.50, .5001, 2.])
        store.features[0, 2, 3] = 0
        checks = module.extension_checks(p, store)
        self.assertEqual(checks.amount_eligible.tolist(), [True, True, True])
        self.assertEqual(checks.not_extended.tolist(), [True, False, True])
        self.assertEqual(checks.in_model_pool.tolist(), [True, False, True])
        self.assertTrue(np.isnan(checks.return20.iloc[2]))

    def test_extension_before_top40_keeps_next_candidates(self):
        p, store = self.inputs([.8]*40+[.3, .4])
        checks = module.extension_checks(p, store)
        self.assertEqual(checks.loc[checks.in_model_pool, 'ts_code'].tolist(), p.ts_code.tolist()[-2:])
        self.assertTrue(checks.amount_eligible.all())

    def test_future_changes_do_not_affect_current_gate(self):
        p, store = self.inputs([.4, .6])
        expected = module.extension_checks(p, store)
        store.features[1] = np.nan
        store.valid_features[1] = False
        pd.testing.assert_frame_equal(module.extension_checks(p, store), expected)

    def test_bad_availability_or_observed_return_rejected(self):
        p, store = self.inputs([.4])
        store.features[0, 0, 3] = .5
        with self.assertRaisesRegex(ValueError, 'binary'):
            module.extension_checks(p, store)
        store.features[0, 0, 3] = 1
        store.features[0, 0, 2] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            module.extension_checks(p, store)

    def test_scoped_variant_restores_original_after_nested_error(self):
        original = liquid.liquidity_checks, liquid.PROTOCOL
        with self.assertRaisesRegex(RuntimeError, 'stop'):
            with module.extension_variant():
                with module.extension_variant():
                    self.assertIs(liquid.liquidity_checks, module.extension_checks)
                    raise RuntimeError('stop')
        self.assertIs(liquid.liquidity_checks, original[0])
        self.assertIs(liquid.PROTOCOL, original[1])

    def fixture(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        source, out = Path(temp.name)/'source', Path(temp.name)/'out'
        source.mkdir()
        (source/'checkpoints').mkdir()
        p, store = self.inputs([.3, .8, .4])
        p.to_pickle(source/'predictions.pkl')
        with (source/'feature_store.pkl').open('wb') as handle:
            pickle.dump(store, handle)
        for name in ('feature_manifest.json', 'frozen_edges.pkl', 'fit_audits.json', 'prediction_manifest.json'):
            (source/name).write_bytes(b'fixture')
        (source/'experiment_protocol.json').write_text('{"source_sha256":{}}', encoding='utf-8')

        class Limits:
            def __init__(self, folder, **kwargs):
                self.sources = {}
                self.folder = Path(folder)/'daily_limits'
                self.folder.mkdir(exist_ok=True)

            def __call__(self, day, code):
                raw = pd.DataFrame(dict(ts_code=[code], trade_date=[f'{day:%Y%m%d}'],
                                        up_limit=[11.], down_limit=[9.]))
                path = self.folder/f'{code}_{day:%Y%m%d}.pkl'
                raw.to_pickle(path)
                self.sources[str(path.resolve())] = module.sha256(path)
                return raw.iloc[0]

        return source, out, p, store, Limits

    def complete_fixture(self):
        source, out, p, store, limits = self.fixture()
        reader = lambda folder: pd.read_pickle(Path(folder)/'predictions.pkl')
        for target, replacement in (
            (patch.object(liquid, 'leadership_variant', side_effect=nullcontext), None),
            (patch.object(liquid, 'validate_predictions', side_effect=reader), None),
            (patch.object(liquid, 'validate_leadership', return_value=(store, None)), None),
            (patch('research_daily_opportunity_band_filter.validate_predictions', side_effect=reader), None),
            (patch.object(liquid, 'OfficialLimits', limits), None),
        ):
            target.start()
            self.addCleanup(target.stop)
        actual = module.filter_extension_predictions(source, out, offline=True)
        return source, out, p, actual

    def test_native_rows_manifest_and_recomputed_extension_agree(self):
        source, out, p, actual = self.complete_fixture()
        pd.testing.assert_frame_equal(actual.drop(columns='eligible'), p, check_exact=True)
        self.assertEqual(actual.eligible.tolist(), [True, False, True])
        result = module.validate_extension_inputs(out)
        self.assertEqual(result['extension_manifest']['vetoed'], 1)
        self.assertEqual(result['manifest']['accepted'], 2)

    def test_source_hash_tampering_rejected(self):
        _, out, _, _ = self.complete_fixture()
        path = out/'extension_manifest.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        data['source_sha256'] = 'tampered'
        path.write_text(json.dumps(data), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'extension provenance'):
            module.validate_extension_inputs(out)

    def test_missing_prediction_manifest_binding_rejected(self):
        _, out, _, _ = self.complete_fixture()
        path = out/'prediction_manifest.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        del data['artifacts']['extension_manifest.json']
        path.write_text(json.dumps(data), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'binding missing'):
            module.validate_extension_inputs(out)


if __name__ == '__main__':
    unittest.main()
