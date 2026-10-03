import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import research_dc_rally_classifier as rally


class DummyClassifier:
    fits = []

    def __init__(self, **parameters):
        self.parameters = parameters
        self.classes_ = np.array([0, 1])

    def fit(self, x, y, sample_weight):
        self.mean = float(np.average(y, weights=sample_weight))
        self.fits.append((np.asarray(x).copy(), np.asarray(y).copy(),
                          np.asarray(sample_weight).copy(), self.parameters))
        return self

    def predict_proba(self, x):
        score = 1 / (1 + np.exp(-np.asarray(x)[:, 0] - self.mean))
        return np.column_stack([1 - score, score])


class DCRallyClassifierTests(unittest.TestCase):
    def setUp(self):
        DummyClassifier.fits = []

    def feature_fixture(self, periods=30):
        dates = pd.date_range('2023-01-31', periods=periods, freq='ME')
        rows = []
        for index, day in enumerate(dates):
            for number, code in enumerate(('000001.SZ', '300001.SZ', '688001.SH')):
                row = dict(signal_date=day, ts_code=code, peer_count=10,
                    label_date=dates[index + 1] if index + 1 < len(dates) else pd.NaT,
                    label_return=.30 if number == 0 else .2999)
                row.update({name: .01 * (number + 1) + .0001 * index
                            for name in rally.FEATURE_COLUMNS})
                rows.append(row)
        return pd.DataFrame(rows), dates

    def source_fixture(self, directory):
        out = Path(directory) / 'out'
        raw = Path(directory) / 'raw'
        out.mkdir()
        raw.mkdir()
        features, dates = self.feature_fixture()
        monthly = features[['signal_date', 'ts_code']].rename(columns={'signal_date': 'date'})
        monthly = monthly.assign(close=10., adj_factor=1.)
        daily = monthly[['date', 'ts_code']].assign(open=10., high=10.1, low=9.9,
                                                   close=10., volume=100., amount=1000.)
        monthly.to_pickle(raw / 'stock_month_end_verified.pkl')
        daily.to_pickle(out / 'daily.pkl')
        features.to_pickle(out / 'forecast_features.pkl')
        features[['signal_date', 'ts_code', 'peer_count']].to_pickle(out / 'peer_context.pkl')
        # Deliberately store old regression artifacts: after the classifier
        # overwrites them, only feature/context hashes remain relevant.
        pd.DataFrame({'old': [1]}).to_pickle(out / 'forecast_predictions.pkl')
        pd.to_pickle({'old': 'regressor'}, out / 'forecast_models.pkl')
        (out / 'forecast_model_audit.json').write_text('{}', encoding='utf-8')
        source_paths = [Path(rally.__file__).with_name(name) for name in rally.FEATURE_HELPERS]
        source_paths += [out / 'daily.pkl', raw / 'stock_month_end_verified.pkl']
        manifest = dict(status='complete', specification=dict(
            protocol={'features': rally.FEATURE_COLUMNS},
            inputs={str(path.resolve()): rally._sha256(path) for path in source_paths},
            monthly_frame_sha256=rally._frame_sha256(monthly, rally.MONTHLY_HASH_COLUMNS),
            daily_frame_sha256=rally._frame_sha256(daily, rally.DAILY_HASH_COLUMNS)),
            artifacts={name: rally._sha256(out / name) for name in (
                'forecast_features.pkl', 'peer_context.pkl', 'forecast_predictions.pkl',
                'forecast_models.pkl', 'forecast_model_audit.json')})
        (out / 'forecast_feature_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        return out, raw, monthly, daily, dates

    @patch.object(rally, 'HistGradientBoostingClassifier', DummyClassifier)
    def test_strict_purge_and_exact_threshold_with_train_only_balancing(self):
        features, dates = self.feature_fixture()
        actual = rally.walk_forward_rally_scores(features, [dates[13]])
        self.assertEqual(len(actual), 3)
        self.assertTrue(actual.train_label_end.lt(actual.model_fit_cutoff).all())
        self.assertEqual(actual.train_rows.iloc[0], 36)
        _, target, weights, parameters = DummyClassifier.fits[0]
        self.assertEqual(int(target.sum()), 12)
        self.assertEqual(parameters, rally.MODEL_PARAMS)
        self.assertAlmostEqual(weights[target == 1].sum(), 18.)
        self.assertAlmostEqual(weights[target == 0].sum(), 18.)
        self.assertAlmostEqual(weights.sum(), len(weights))
        self.assertTrue((weights[target == 1] == 1.5).all())
        self.assertTrue((weights[target == 0] == .75).all())

    @patch.object(rally, 'HistGradientBoostingClassifier', DummyClassifier)
    def test_minimum_months_and_minimum_class_counts_fail_closed(self):
        features, dates = self.feature_fixture()
        self.assertTrue(rally.walk_forward_rally_scores(features, [dates[12]]).empty)
        features['label_return'] = .31
        actual = rally.walk_forward_rally_scores(features, [dates[13]])
        self.assertTrue(actual.empty)
        self.assertEqual(actual.attrs['model_audit'][0]['status'], 'insufficient_class_count')
        self.assertEqual(DummyClassifier.fits, [])

    @patch.object(rally, 'HistGradientBoostingClassifier', DummyClassifier)
    def test_future_labels_and_features_cannot_change_earlier_score(self):
        features, dates = self.feature_fixture()
        signal = dates[13]
        expected = rally.walk_forward_rally_scores(features, [signal])
        changed = features.copy()
        changed.loc[changed.label_date.ge(signal), 'label_return'] = -9999.
        changed.loc[changed.signal_date.gt(signal), rally.FEATURE_COLUMNS] = 10000.
        actual = rally.walk_forward_rally_scores(changed, [signal])
        expected.attrs = actual.attrs = {}
        pd.testing.assert_frame_equal(expected, actual)

    @patch.object(rally, 'HistGradientBoostingClassifier', DummyClassifier)
    def test_quarter_freeze_and_all_board_percentiles(self):
        features, dates = self.feature_fixture()
        actual = rally.walk_forward_rally_scores(features, dates[13:15])
        self.assertEqual(len(DummyClassifier.fits), 1)
        self.assertEqual(actual.model_fit_cutoff.nunique(), 1)
        self.assertEqual(set(actual.ts_code), {'000001.SZ', '300001.SZ', '688001.SH'})
        for _, rows in actual.groupby('signal_date'):
            np.testing.assert_allclose(rows.forecast_percentile, rows.forecast_return.rank(pct=True))
        self.assertEqual(rally.PROTOCOL['target_kind'], 'rally_classifier')
        self.assertIn('NOT expected return', rally.PROTOCOL['forecast_return_alias'])

    @patch.object(rally, 'HistGradientBoostingClassifier', DummyClassifier)
    def test_cache_keeps_features_immutable_and_ignores_old_regressor_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            out, raw, monthly, daily, dates = self.source_fixture(temporary)
            feature_hash = rally._sha256(out / 'forecast_features.pkl')
            context_hash = rally._sha256(out / 'peer_context.pkl')
            with patch.object(rally, 'ROOT', raw):
                expected = rally.prepare_rally_scores(out, monthly, daily, dates[13:15])
                fit_count = len(DummyClassifier.fits)
                actual = rally.prepare_rally_scores(out, monthly, daily, dates[13:15])
            self.assertEqual(len(DummyClassifier.fits), fit_count)
            pd.testing.assert_frame_equal(expected, actual)
            self.assertEqual(feature_hash, rally._sha256(out / 'forecast_features.pkl'))
            self.assertEqual(context_hash, rally._sha256(out / 'peer_context.pkl'))
            manifest = json.loads((out / 'rally_feature_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['specification']['protocol']['target_kind'], 'rally_classifier')
            self.assertEqual(set(manifest['artifacts']), {'forecast_predictions.pkl',
                'forecast_models.pkl', 'forecast_model_audit.json'})
            audit = json.loads((out / 'forecast_model_audit.json').read_text(encoding='utf-8'))
            self.assertTrue(audit['train_label_purge_verified'])

    @patch.object(rally, 'HistGradientBoostingClassifier', DummyClassifier)
    def test_changed_in_memory_source_rejects_stale_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            out, raw, monthly, daily, dates = self.source_fixture(temporary)
            with patch.object(rally, 'ROOT', raw):
                rally.prepare_rally_scores(out, monthly, daily, [dates[13]])
                changed = daily.copy()
                changed.loc[0, 'amount'] *= 2
                with self.assertRaisesRegex(ValueError, 'in-memory source'):
                    rally.prepare_rally_scores(out, monthly, changed, [dates[13]])
                changed = monthly.copy()
                changed.loc[0, 'close'] *= 2
                with self.assertRaisesRegex(ValueError, 'in-memory source'):
                    rally.prepare_rally_scores(out, changed, daily, [dates[13]])

    @patch.object(rally, 'HistGradientBoostingClassifier', DummyClassifier)
    def test_changed_prediction_artifact_is_regenerated_not_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            out, raw, monthly, daily, dates = self.source_fixture(temporary)
            with patch.object(rally, 'ROOT', raw):
                expected = rally.prepare_rally_scores(out, monthly, daily, [dates[13]])
                fit_count = len(DummyClassifier.fits)
                damaged = expected.copy()
                damaged['forecast_return'] = 0.
                damaged.to_pickle(out / 'forecast_predictions.pkl')
                actual = rally.prepare_rally_scores(out, monthly, daily, [dates[13]])
            self.assertGreater(len(DummyClassifier.fits), fit_count)
            pd.testing.assert_frame_equal(expected, actual)

    @patch.object(rally, 'HistGradientBoostingClassifier', DummyClassifier)
    def test_changed_cached_feature_or_source_disk_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            out, raw, monthly, daily, dates = self.source_fixture(temporary)
            features = pd.read_pickle(out / 'forecast_features.pkl')
            features.loc[0, 'mom1'] = 10.
            features.to_pickle(out / 'forecast_features.pkl')
            with patch.object(rally, 'ROOT', raw):
                with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
                    rally.prepare_rally_scores(out, monthly, daily, [dates[13]])
        with tempfile.TemporaryDirectory() as temporary:
            out, raw, monthly, daily, dates = self.source_fixture(temporary)
            changed = daily.copy()
            changed.loc[0, 'amount'] = 9.
            changed.to_pickle(out / 'daily.pkl')
            with patch.object(rally, 'ROOT', raw):
                with self.assertRaisesRegex(ValueError, 'source hash mismatch'):
                    rally.prepare_rally_scores(out, monthly, daily, [dates[13]])

    def test_duplicate_keys_and_nonfuture_label_dates_are_rejected(self):
        features, dates = self.feature_fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate rally stock'):
            rally.walk_forward_rally_scores(pd.concat([features, features.iloc[:1]]), [dates[13]])
        features.loc[0, 'label_date'] = features.loc[0, 'signal_date']
        with self.assertRaisesRegex(ValueError, 'labels must follow'):
            rally.walk_forward_rally_scores(features, [dates[13]])

    def test_real_weighted_histogram_classifier_accepts_fixed_protocol(self):
        features, dates = self.feature_fixture()
        actual = rally.walk_forward_rally_scores(features, [dates[13]])
        self.assertEqual(len(actual), 3)
        self.assertTrue(actual.forecast_return.between(0., 1.).all())
        self.assertTrue(actual.train_label_end.lt(actual.model_fit_cutoff).all())
        model = actual.attrs['models'][str(dates[13].to_period('Q'))]['model']
        self.assertEqual(list(model.classes_), [0, 1])
        self.assertEqual(model.get_params()['loss'], 'log_loss')


if __name__ == '__main__':
    unittest.main()
