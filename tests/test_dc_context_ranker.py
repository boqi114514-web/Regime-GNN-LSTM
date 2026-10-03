import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import research_dc_context_ranker as ranker


class DummyRanker:
    fits = []

    def __init__(self, **parameters):
        self.parameters = parameters

    def fit(self, x, y, group, eval_at):
        self.offset = float(np.average(y))
        self.fits.append((np.asarray(x).copy(), np.asarray(y).copy(),
                          np.asarray(group).copy(), eval_at, self.parameters))
        return self

    def predict(self, x):
        return np.asarray(x)[:, 0] - self.offset


class DCContextRankerTests(unittest.TestCase):
    def setUp(self):
        DummyRanker.fits = []

    def feature_fixture(self, periods=30):
        dates = pd.date_range('2023-01-31', periods=periods, freq='ME')
        rows = []
        for index, day in enumerate(dates):
            for number, code in enumerate(('000001.SZ', '300001.SZ', '688001.SH')):
                row = dict(signal_date=day, ts_code=code, peer_count=20,
                    label_date=dates[index + 1] if index + 1 < len(dates) else pd.NaT,
                    label_return=(number - 1) * .1)
                row.update({name: .01 * (number + 1) + .0001 * index
                            for name in ranker.SOURCE_FEATURE_COLUMNS})
                row.update(peer_mom1=.005, peer_mom3=.015, peer_mom6=.025,
                           peer_positive3=.7)
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
        pd.DataFrame({'old': [1]}).to_pickle(out / 'forecast_predictions.pkl')
        pd.to_pickle({'old': 'regressor'}, out / 'forecast_models.pkl')
        (out / 'forecast_model_audit.json').write_text('{}', encoding='utf-8')
        source_paths = [Path(ranker.__file__).with_name(name) for name in ranker.FEATURE_HELPERS]
        source_paths += [out / 'daily.pkl', raw / 'stock_month_end_verified.pkl']
        manifest = dict(status='complete', specification=dict(
            protocol={'features': ranker.SOURCE_FEATURE_COLUMNS},
            inputs={str(path.resolve()): ranker._sha256(path) for path in source_paths},
            monthly_frame_sha256=ranker._frame_sha256(monthly, ranker.MONTHLY_HASH_COLUMNS),
            daily_frame_sha256=ranker._frame_sha256(daily, ranker.DAILY_HASH_COLUMNS)),
            artifacts={name: ranker._sha256(out / name) for name in (
                'forecast_features.pkl', 'peer_context.pkl', 'forecast_predictions.pkl',
                'forecast_models.pkl', 'forecast_model_audit.json')})
        (out / 'forecast_feature_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        return out, raw, monthly, daily, dates

    def cohort_fixture(self):
        base, dates = self.feature_fixture()
        rows = []
        for day in dates:
            template = base[base.signal_date.eq(day)].iloc[0].to_dict()
            for number, code in enumerate(('000001.SZ', '300001.SZ', '688001.SH',
                                          '600001.SH', '000002.SZ')):
                row = {**template, 'ts_code': code,
                       'label_return': [.1, .2, -.1, -.2, 5.][number]}
                cohort = 1 if number < 2 else 2 if number < 4 else 3
                row.update(peer_mom1=.01 * cohort, peer_mom3=.02 * cohort,
                           peer_mom6=.03 * cohort, peer_positive3=.6 + .1 * cohort,
                           mom1=.03 * (number + 1))
                rows.append(row)
        return pd.DataFrame(rows), dates

    def test_context_contrasts_are_local_and_do_not_use_labels(self):
        features, _ = self.feature_fixture()
        original = features.copy()
        actual = ranker.context_features(features)
        pd.testing.assert_frame_equal(features, original)
        for horizon in (1, 3, 6):
            np.testing.assert_allclose(actual[f'stock_peer_excess{horizon}'],
                                       features[f'mom{horizon}'] - features[f'peer_mom{horizon}'])
        np.testing.assert_allclose(actual.stock_peer_acceleration,
            (features.mom1 - features.peer_mom1) - (features.mom3 - features.peer_mom3) / 3.)
        changed = features.copy()
        changed['label_return'] = 9999.
        pd.testing.assert_frame_equal(actual[ranker.FEATURE_COLUMNS],
                                      ranker.context_features(changed)[ranker.FEATURE_COLUMNS])

    def test_query_labels_are_local_tie_preserving_and_not_clipped_returns(self):
        days = pd.to_datetime(['2023-01-31'] * 4 + ['2023-02-28'] * 4)
        train = pd.DataFrame(dict(signal_date=days,
            label_return=[-999., .1, .1, 999., -1999., .1, .1, 1999.]))
        actual = ranker.query_relevance(train)
        np.testing.assert_array_equal(actual, [0, 3, 3, 7, 0, 3, 3, 7])
        self.assertTrue(np.issubdtype(actual.dtype, np.integer))

    @patch.object(ranker, 'LGBMRanker', DummyRanker)
    def test_training_is_monthly_grouped_and_strictly_purged(self):
        features, dates = self.feature_fixture()
        actual = ranker.walk_forward_context_scores(features.sample(frac=1, random_state=8), [dates[13]])
        self.assertEqual(len(actual), 3)
        self.assertTrue(actual.train_label_end.lt(actual.model_fit_cutoff).all())
        self.assertEqual(actual.train_rows.iloc[0], 36)
        x, relevance, groups, eval_at, parameters = DummyRanker.fits[0]
        self.assertEqual(x.shape, (36, 23))
        np.testing.assert_array_equal(groups, [3] * 12)
        np.testing.assert_array_equal(relevance, [0, 3, 6] * 12)
        self.assertEqual(int(groups.sum()), len(relevance))
        self.assertEqual(eval_at, (1, 5))
        self.assertEqual(parameters, ranker.MODEL_PARAMS)
        self.assertEqual(actual.attrs['model_audit'][0]['train_query_count'], 12)

    @patch.object(ranker, 'LGBMRanker', DummyRanker)
    def test_future_observations_and_labels_cannot_change_earlier_prediction(self):
        features, dates = self.feature_fixture()
        signal = dates[13]
        expected = ranker.walk_forward_context_scores(features, [signal])
        changed = features.copy()
        changed.loc[changed.label_date.ge(signal), 'label_return'] = -9999.
        changed.loc[changed.signal_date.gt(signal), ranker.SOURCE_FEATURE_COLUMNS] = 10000.
        actual = ranker.walk_forward_context_scores(changed, [signal])
        expected.attrs = actual.attrs = {}
        pd.testing.assert_frame_equal(expected, actual)

    @patch.object(ranker, 'LGBMRanker', DummyRanker)
    def test_minimum_months_and_uninformative_queries_fail_closed(self):
        features, dates = self.feature_fixture()
        self.assertTrue(ranker.walk_forward_context_scores(features, [dates[12]]).empty)
        features['label_return'] = .1
        actual = ranker.walk_forward_context_scores(features, [dates[13]])
        self.assertTrue(actual.empty)
        self.assertEqual(actual.attrs['model_audit'][0]['status'], 'insufficient_relevance_variation')
        self.assertEqual(DummyRanker.fits, [])

    @patch.object(ranker, 'LGBMRanker', DummyRanker)
    def test_quarter_freeze_all_boards_and_negative_raw_scores_are_preserved(self):
        features, dates = self.feature_fixture()
        actual = ranker.walk_forward_context_scores(features, dates[13:15])
        self.assertEqual(len(DummyRanker.fits), 1)
        self.assertEqual(actual.model_fit_cutoff.nunique(), 1)
        self.assertEqual(set(actual.ts_code), {'000001.SZ', '300001.SZ', '688001.SH'})
        self.assertTrue(actual.forecast_return.lt(0.).all())
        for _, rows in actual.groupby('signal_date'):
            np.testing.assert_allclose(rows.forecast_percentile, rows.forecast_return.rank(pct=True))
        self.assertEqual(ranker.PROTOCOL['target_kind'], 'context_ranker')
        self.assertIn('NOT expected return', ranker.PROTOCOL['forecast_return_alias'])

    @patch.object(ranker, 'LGBMRanker', DummyRanker)
    def test_cache_preserves_original_features_and_binds_new_23_feature_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            out, raw, monthly, daily, dates = self.source_fixture(temporary)
            immutable = {name: ranker._sha256(out / name)
                         for name in ('forecast_features.pkl', 'peer_context.pkl')}
            with patch.object(ranker, 'ROOT', raw):
                expected = ranker.prepare_context_scores(out, monthly, daily, dates[13:15])
                fit_count = len(DummyRanker.fits)
                actual = ranker.prepare_context_scores(out, monthly, daily, dates[13:15])
            self.assertEqual(len(DummyRanker.fits), fit_count)
            pd.testing.assert_frame_equal(expected, actual)
            for name, digest in immutable.items():
                self.assertEqual(digest, ranker._sha256(out / name))
            context = pd.read_pickle(out / 'context_ranker_features.pkl')
            self.assertTrue(set(ranker.FEATURE_COLUMNS).issubset(context.columns))
            manifest = json.loads((out / 'context_ranker_feature_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['specification']['protocol']['target_kind'], 'context_ranker')
            self.assertIn('context_ranker_features.pkl', manifest['artifacts'])
            audit = json.loads((out / 'forecast_model_audit.json').read_text(encoding='utf-8'))
            self.assertTrue(audit['train_label_purge_verified'])

    @patch.object(ranker, 'LGBMRanker', DummyRanker)
    def test_mutated_in_memory_frames_reject_stale_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            out, raw, monthly, daily, dates = self.source_fixture(temporary)
            with patch.object(ranker, 'ROOT', raw):
                changed = daily.copy()
                changed.loc[0, 'amount'] *= 2
                with self.assertRaisesRegex(ValueError, 'in-memory source'):
                    ranker.prepare_context_scores(out, monthly, changed, [dates[13]])
                changed = monthly.copy()
                changed.loc[0, 'close'] *= 2
                with self.assertRaisesRegex(ValueError, 'in-memory source'):
                    ranker.prepare_context_scores(out, changed, daily, [dates[13]])

    @patch.object(ranker, 'LGBMRanker', DummyRanker)
    def test_modified_prediction_or_derived_features_regenerate_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            out, raw, monthly, daily, dates = self.source_fixture(temporary)
            with patch.object(ranker, 'ROOT', raw):
                expected = ranker.prepare_context_scores(out, monthly, daily, [dates[13]])
                fit_count = len(DummyRanker.fits)
                changed = expected.copy()
                changed['forecast_return'] = 9999.
                changed.to_pickle(out / 'forecast_predictions.pkl')
                actual = ranker.prepare_context_scores(out, monthly, daily, [dates[13]])
                self.assertGreater(len(DummyRanker.fits), fit_count)
                pd.testing.assert_frame_equal(expected, actual)
                derived = pd.read_pickle(out / 'context_ranker_features.pkl')
                derived['stock_peer_excess1'] = 9999.
                derived.to_pickle(out / 'context_ranker_features.pkl')
                actual = ranker.prepare_context_scores(out, monthly, daily, [dates[13]])
                pd.testing.assert_frame_equal(expected, actual)
                self.assertFalse(pd.read_pickle(out / 'context_ranker_features.pkl').stock_peer_excess1.eq(9999.).any())

    @patch.object(ranker, 'LGBMRanker', DummyRanker)
    def test_modified_base_features_or_disk_sources_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            out, raw, monthly, daily, dates = self.source_fixture(temporary)
            base = pd.read_pickle(out / 'forecast_features.pkl')
            base.loc[0, 'mom1'] = 10.
            base.to_pickle(out / 'forecast_features.pkl')
            with patch.object(ranker, 'ROOT', raw):
                with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
                    ranker.prepare_context_scores(out, monthly, daily, [dates[13]])
        with tempfile.TemporaryDirectory() as temporary:
            out, raw, monthly, daily, dates = self.source_fixture(temporary)
            changed = daily.copy()
            changed.loc[0, 'amount'] = 9.
            changed.to_pickle(out / 'daily.pkl')
            with patch.object(ranker, 'ROOT', raw):
                with self.assertRaisesRegex(ValueError, 'source hash mismatch'):
                    ranker.prepare_context_scores(out, monthly, daily, [dates[13]])

    def test_duplicates_nonfuture_labels_and_calendar_holes_are_rejected(self):
        features, dates = self.feature_fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate context stock'):
            ranker.walk_forward_context_scores(pd.concat([features, features.iloc[:1]]), [dates[13]])
        changed = features.copy()
        changed.loc[0, 'label_date'] = changed.loc[0, 'signal_date']
        with self.assertRaisesRegex(ValueError, 'labels must follow'):
            ranker.walk_forward_context_scores(changed, [dates[13]])
        changed = features.copy()
        changed.loc[0, 'label_date'] = dates[2]
        with self.assertRaisesRegex(ValueError, 'next calendar month'):
            ranker.walk_forward_context_scores(changed, [dates[13]])
        with self.assertRaisesRegex(ValueError, 'Duplicate context prediction'):
            ranker.walk_forward_context_scores(features, [dates[13], dates[13]])

    def test_real_lightgbm_ranker_accepts_fixed_query_protocol(self):
        features, dates = self.feature_fixture()
        actual = ranker.walk_forward_context_scores(features, [dates[13]])
        self.assertEqual(len(actual), 3)
        self.assertTrue(np.isfinite(actual.forecast_return).all())
        self.assertTrue(actual.train_label_end.lt(actual.model_fit_cutoff).all())
        model = actual.attrs['models'][str(dates[13].to_period('Q'))]['model']
        self.assertEqual(model.get_params()['objective'], 'lambdarank')
        self.assertEqual(model.get_params()['label_gain'], ranker.LABEL_GAIN)
        self.assertEqual(model.n_features_in_, 23)

    def test_default_protocol_is_unchanged_and_cohort_scope_is_explicit(self):
        self.assertIs(ranker.protocol_for_scope('monthly'), ranker.PROTOCOL)
        self.assertEqual(ranker.PROTOCOL['version'], 'context_ranker_v1')
        self.assertNotIn('query_scope', ranker.PROTOCOL)
        protocol = ranker.protocol_for_scope('cohort')
        self.assertEqual(protocol['query_scope'], 'cohort')
        self.assertEqual(protocol['version'], 'context_ranker_cohort_v1')
        self.assertEqual(protocol['parameters'], ranker.PROTOCOL['parameters'])
        self.assertEqual(protocol['query_key_columns'], ['signal_date'] + ranker.PEER_COLUMNS)
        self.assertIn('collide', protocol['query'])
        with self.assertRaisesRegex(ValueError, 'query_scope'):
            ranker.protocol_for_scope('sector')

    def test_cohort_relevance_is_local_instead_of_market_rank(self):
        features, dates = self.cohort_fixture()
        train = features[features.signal_date.eq(dates[0])]
        actual = ranker.query_relevance(train, query_scope='cohort')
        np.testing.assert_array_equal(actual, [0, 5, 5, 0, 0])
        market = ranker.query_relevance(train)
        self.assertFalse(actual.equals(market))

    @patch.object(ranker, 'LGBMRanker', DummyRanker)
    def test_cohort_query_groups_are_contiguous_reconstructable_and_drop_singletons(self):
        features, dates = self.cohort_fixture()
        actual = ranker.walk_forward_context_scores(
            features.sample(frac=1, random_state=9), [dates[13]], query_scope='cohort')
        self.assertEqual(len(actual), 5)
        self.assertEqual(actual.train_rows.iloc[0], 48)
        x, relevance, groups, eval_at, parameters = DummyRanker.fits[0]
        self.assertEqual(x.shape, (48, 23))
        np.testing.assert_array_equal(groups, [2] * 24)
        np.testing.assert_array_equal(relevance, [0, 5, 0, 5] * 12)
        self.assertEqual(parameters, ranker.MODEL_PARAMS)
        self.assertEqual(eval_at, (1, 5))
        audit = actual.attrs['model_audit'][0]
        self.assertEqual(audit['train_query_count'], 24)
        self.assertEqual(audit['train_label_months'], 12)
        self.assertEqual(audit['query_scope'], 'cohort')
        self.assertEqual(len(audit['train_query_keys']), 24)
        self.assertEqual(audit['train_query_keys'][0],
                         [dates[0].strftime('%Y-%m-%d'), .01, .02, .03, .7])
        self.assertEqual(audit['train_query_keys'][1],
                         [dates[0].strftime('%Y-%m-%d'), .02, .04, .06, .8])
        training = ranker.context_features(features[
            features.label_date.lt(dates[13]) & features.signal_date.lt(dates[13])])
        cursor = 0
        for key, count in zip(audit['train_query_keys'], audit['train_query_sizes']):
            selected = training.signal_date.eq(pd.Timestamp(key[0]))
            for column, value in zip(ranker.PEER_COLUMNS, key[1:]):
                selected &= training[column].eq(value)
            query = training[selected].sort_values('ts_code')
            self.assertEqual(len(query), count)
            np.testing.assert_allclose(x[cursor:cursor + count], query[ranker.FEATURE_COLUMNS])
            cursor += count
        self.assertEqual(cursor, len(x))

    @patch.object(ranker, 'LGBMRanker', DummyRanker)
    def test_cohort_future_mutations_and_input_order_do_not_change_prediction(self):
        features, dates = self.cohort_fixture()
        signal = dates[13]
        expected = ranker.walk_forward_context_scores(features, dates[13:15], query_scope='cohort')
        self.assertEqual(len(DummyRanker.fits), 1)
        changed = features.copy()
        changed.loc[changed.label_date.ge(signal), 'label_return'] = 9999.
        changed.loc[changed.signal_date.gt(dates[14]), ranker.SOURCE_FEATURE_COLUMNS] = -9999.
        changed = changed.sample(frac=1, random_state=88)
        actual = ranker.walk_forward_context_scores(changed, dates[13:15], query_scope='cohort')
        expected.attrs = actual.attrs = {}
        pd.testing.assert_frame_equal(expected, actual)

    @patch.object(ranker, 'LGBMRanker', DummyRanker)
    def test_cohort_all_singletons_cannot_supply_training_months(self):
        features, dates = self.cohort_fixture()
        features['peer_mom1'] = features.groupby('signal_date').cumcount().astype(float)
        actual = ranker.walk_forward_context_scores(features, [dates[13]], query_scope='cohort')
        self.assertTrue(actual.empty)
        self.assertEqual(actual.attrs['model_audit'][0]['train_rows'], 0)
        self.assertEqual(actual.attrs['model_audit'][0]['train_query_keys'], [])
        self.assertEqual(DummyRanker.fits, [])

    @patch.object(ranker, 'LGBMRanker', DummyRanker)
    def test_explicit_monthly_scope_preserves_original_predictions_and_metadata(self):
        features, dates = self.cohort_fixture()
        default = ranker.walk_forward_context_scores(features, dates[13:15])
        explicit = ranker.walk_forward_context_scores(features, dates[13:15], query_scope='monthly')
        self.assertEqual(default.attrs['model_audit'], explicit.attrs['model_audit'])
        self.assertNotIn('query_scope', default.attrs['model_audit'][0])
        self.assertNotIn('train_query_keys', default.attrs['model_audit'][0])
        default.attrs = explicit.attrs = {}
        pd.testing.assert_frame_equal(default, explicit)

    @patch.object(ranker, 'LGBMRanker', DummyRanker)
    def test_cohort_adapter_manifest_is_bound_to_query_scope_and_caches(self):
        with tempfile.TemporaryDirectory() as temporary:
            out, raw, monthly, daily, dates = self.source_fixture(temporary)
            with patch.object(ranker, 'ROOT', raw):
                ranker.prepare_context_scores(out, monthly, daily, [dates[13]])
                count = len(DummyRanker.fits)
                expected = ranker.prepare_context_scores(out, monthly, daily, [dates[13]], query_scope='cohort')
                self.assertGreater(len(DummyRanker.fits), count)
                count = len(DummyRanker.fits)
                actual = ranker.prepare_context_scores(out, monthly, daily, [dates[13]], query_scope='cohort')
            self.assertEqual(len(DummyRanker.fits), count)
            pd.testing.assert_frame_equal(expected, actual)
            manifest = json.loads((out / 'context_ranker_feature_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['specification']['protocol']['query_scope'], 'cohort')
            audit = json.loads((out / 'forecast_model_audit.json').read_text(encoding='utf-8'))
            self.assertEqual(audit['quarterly_models'][0]['query_scope'], 'cohort')

    def test_real_lightgbm_ranker_accepts_cohort_queries(self):
        features, dates = self.cohort_fixture()
        actual = ranker.walk_forward_context_scores(features, [dates[13]], query_scope='cohort')
        self.assertEqual(len(actual), 5)
        self.assertTrue(np.isfinite(actual.forecast_return).all())
        self.assertTrue(actual.train_label_end.lt(actual.model_fit_cutoff).all())
        model = actual.attrs['models'][str(dates[13].to_period('Q'))]['model']
        self.assertEqual(model.get_params()['objective'], 'lambdarank')
        self.assertEqual(model.n_features_in_, 23)


if __name__ == '__main__':
    unittest.main()
