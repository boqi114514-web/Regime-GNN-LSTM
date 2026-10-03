import contextlib
import io
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from research_daily_graph_data import sha256
from research_daily_opportunity_path import (
    fit_path_heads, path_quarter_split, path_training_arrays, predict_path_heads,
    publish_path_targets, validate_target_artifact,
)
from research_daily_opportunity_path_targets import build_path_targets
from test_daily_opportunity_path_targets import quotes


class PathTrainingTests(unittest.TestCase):
    def test_quarter_training_and_validation_use_horizon_plus_one_maturity(self):
        targets = build_path_targets(quotes(size=140))
        training, validation, cutoff = path_quarter_split(targets, list(range(140)), targets.dates[110])
        self.assertEqual(training[-1], 47)
        self.assertEqual(validation, list(range(69, 89)))
        self.assertEqual(cutoff, targets.dates[69])
        self.assertTrue((targets.label_end_dates[training] < cutoff.to_datetime64()).all())
        self.assertTrue((targets.label_end_dates[validation] < targets.dates[110].to_datetime64()).all())

    def test_path_targets_replace_old_labels_only_at_loss_boundary(self):
        daily = quotes(size=140)
        second = daily.copy()
        second['ts_code'] = '300001.SZ'
        second = second.drop(index=78)
        target = build_path_targets(pd.concat((daily, second)), sessions=daily.date)
        store = SimpleNamespace(stock_codes=target.stock_codes,
                                valid_features=np.ones((140, 2), dtype=bool))
        features = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
        factors = {70:(target.stock_codes, features, np.full((2, 3), 999.), np.full((2, 3), 999.), None)}
        x, y, risk, event = path_training_arrays([70], factors, store, target)
        self.assertEqual(len(factors[70][0]), 2)  # Full input graph still contains both nodes.
        self.assertEqual(len(x), 1)
        surviving = target.stock_codes.index('600001.SH')
        np.testing.assert_array_equal(x[0], features[surviving])
        np.testing.assert_array_equal(y[0], target.path_payoff[70, surviving])
        np.testing.assert_array_equal(risk[0], target.max_adverse_excursion[70, surviving])
        np.testing.assert_array_equal(event[0], target.first_passage[70, surviving])
        np.testing.assert_array_equal(factors[70][2], 999.)

    def test_native_regression_and_three_class_checkpoint_predictions(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=(1200, 4)).astype(np.float32)
        y = rng.normal(0, .1, size=(1200, 3)).astype(np.float32)
        risk = np.abs(rng.normal(0, .1, size=(1200, 3))).astype(np.float32)
        events = rng.integers(0, 3, size=(1200, 3), dtype=np.int8)
        with contextlib.redirect_stdout(io.StringIO()):
            heads, _ = fit_path_heads((x, y, risk, events), (x[:120], y[:120], risk[:120], events[:120]), 'fixture')
        mu, downside, probability = predict_path_heads(heads, x[:8])
        self.assertEqual(mu.shape, (8, 3))
        self.assertTrue(np.isfinite(mu).all())
        self.assertTrue((downside >= 0).all())
        self.assertEqual(probability.shape, (8, 3, 3))
        np.testing.assert_allclose(probability.sum(axis=-1), 1.)
        reloaded = pickle.loads(pickle.dumps(heads))
        for original, restored in zip((mu, downside, probability), predict_path_heads(reloaded, x[:8])):
            np.testing.assert_array_equal(original, restored)

    def test_separate_target_artifact_binding_and_tamper_rejection(self):
        daily = quotes()
        target = build_path_targets(daily)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            raw_path = out/'raw.pkl'
            daily.to_pickle(raw_path)
            raw = dict(path=str(raw_path), sha256=sha256(raw_path))
            # A minimal immutable artifact is sufficient for this binding test.
            with (out/'feature_store.pkl').open('wb') as handle:
                pickle.dump(dict(original_labels='not overwritten'), handle)
            (out/'feature_manifest.json').write_text(json.dumps(dict(raw_adjusted_daily=raw)), encoding='utf-8')
            publish_path_targets(out, target, raw)
            loaded = validate_target_artifact(out)
            np.testing.assert_array_equal(loaded.path_payoff, target.path_payoff)
            with (out/'path_targets.pkl').open('ab') as handle:
                handle.write(b'tamper')
            with self.assertRaisesRegex(ValueError, 'Changed path-target artifact'):
                validate_target_artifact(out)

    def test_artifact_validator_rejects_early_exit_date_as_label_maturity(self):
        daily = quotes()
        daily.loc[1, ['close', 'low']] = [91., 90.]
        target = build_path_targets(daily)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            raw_path = out/'raw.pkl'
            daily.to_pickle(raw_path)
            raw = dict(path=str(raw_path), sha256=sha256(raw_path))
            with (out/'feature_store.pkl').open('wb') as handle:
                pickle.dump('unchanged original feature store', handle)
            (out/'feature_manifest.json').write_text(json.dumps(dict(raw_adjusted_daily=raw)), encoding='utf-8')
            target.label_end_dates[0, 0] = target.dates[2].to_datetime64().astype('datetime64[D]')
            publish_path_targets(out, target, raw)
            with self.assertRaisesRegex(ValueError, 'maturity must be horizon plus one'):
                validate_target_artifact(out)


if __name__ == '__main__':
    unittest.main()
