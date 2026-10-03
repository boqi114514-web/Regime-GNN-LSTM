import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import research_dc_forecast as forecast


class DummyModel:
    fits = []

    def __init__(self, **parameters):
        self.parameters = parameters

    def fit(self, x, y):
        self.mean = float(np.mean(y))
        self.fits.append((np.asarray(x).copy(), np.asarray(y).copy(), self.parameters))
        return self

    def predict(self, x):
        return self.mean + np.asarray(x)[:, 0] * .01


class DCForecastTests(unittest.TestCase):
    def setUp(self):
        DummyModel.fits = []

    def feature_fixture(self, periods=32):
        dates = pd.date_range('2023-01-31', periods=periods, freq='ME')
        rows = []
        for index, day in enumerate(dates):
            for number, code in enumerate(('000001.SZ', '300001.SZ', '688001.SH')):
                row = dict(signal_date=day, ts_code=code,
                           label_date=dates[index + 1] if index + 1 < len(dates) else pd.NaT,
                           label_return=.01 * (index % 5 - 2) + .001 * number)
                row.update({name: .01 * (number + 1) + .0001 * index
                            for name in forecast.FEATURE_COLUMNS})
                rows.append(row)
        return pd.DataFrame(rows), dates

    def quote_fixture(self):
        days = pd.bdate_range('2022-01-03', '2024-05-31')
        months = pd.Series(days, index=days).groupby(days.to_period('M')).last().to_numpy()
        monthly, daily = [], []
        for number, code in enumerate(('000001.SZ', '300001.SZ', '688001.SH')):
            trend = .02 * np.sin(np.arange(len(months)) * .7) + .005 * (number + 1)
            price = 10 * np.cumprod(1 + trend)
            monthly.append(pd.DataFrame(dict(date=months, ts_code=code, close=price, adj_factor=1.)))
            d = days[days >= pd.Timestamp('2022-12-01')]
            pressure = .001 * (number + 1) + .0002 * np.sin(np.arange(len(d)) * .2)
            daily.append(pd.DataFrame(dict(date=d, ts_code=code, open=10.,
                close=10 * (1 + pressure), high=10.1, low=9.9, volume=100.,
                amount=1000. * (number + 1) + np.arange(len(d)))))
        return pd.concat(monthly, ignore_index=True), pd.concat(daily, ignore_index=True)

    @patch.object(forecast, 'HistGradientBoostingRegressor', DummyModel)
    def test_minimum_twelve_completed_label_months_and_strict_purge(self):
        features, dates = self.feature_fixture()
        early = forecast.walk_forward_predictions(features, [dates[12]])
        self.assertTrue(early.empty)  # February--December: only 11 completed labels.
        actual = forecast.walk_forward_predictions(features, [dates[13]])
        self.assertEqual(len(actual), 3)
        self.assertTrue(actual.train_label_end.lt(actual.model_fit_cutoff).all())
        self.assertEqual(actual.train_rows.iloc[0], 36)
        self.assertEqual(actual.train_label_end.iloc[0], dates[12])
        self.assertEqual(DummyModel.fits[-1][2], forecast.MODEL_PARAMS)

    @patch.object(forecast, 'HistGradientBoostingRegressor', DummyModel)
    def test_model_is_frozen_for_later_signals_in_same_quarter(self):
        features, dates = self.feature_fixture()
        actual = forecast.walk_forward_predictions(features, dates[13:15])
        self.assertEqual(len(DummyModel.fits), 1)
        self.assertEqual(actual.model_fit_cutoff.nunique(), 1)
        self.assertEqual(actual.train_rows.nunique(), 1)
        self.assertEqual(actual.signal_date.nunique(), 2)

    @patch.object(forecast, 'HistGradientBoostingRegressor', DummyModel)
    def test_future_labels_and_features_cannot_change_earlier_prediction(self):
        features, dates = self.feature_fixture()
        signal = dates[13]
        expected = forecast.walk_forward_predictions(features, [signal])
        modified = features.copy()
        modified.loc[modified.label_date.ge(signal), 'label_return'] = 10000.
        modified.loc[modified.signal_date.gt(signal), forecast.FEATURE_COLUMNS] = 99999.
        actual = forecast.walk_forward_predictions(modified, [signal])
        expected.attrs = actual.attrs = {}
        pd.testing.assert_frame_equal(expected, actual)

    @patch.object(forecast, 'HistGradientBoostingRegressor', DummyModel)
    def test_prediction_percentiles_include_chinext_and_star(self):
        features, dates = self.feature_fixture()
        actual = forecast.walk_forward_predictions(features, [dates[13]])
        self.assertEqual(set(actual.ts_code), {'000001.SZ', '300001.SZ', '688001.SH'})
        np.testing.assert_allclose(actual.forecast_percentile, actual.forecast_return.rank(pct=True))
        self.assertFalse(actual.duplicated(['signal_date', 'ts_code']).any())

    @patch.object(forecast, 'HistGradientBoostingRegressor', DummyModel)
    def test_label_clipping_is_only_for_fitting(self):
        features, dates = self.feature_fixture()
        features.loc[features.signal_date.eq(dates[0]), 'label_return'] = 9.
        before = features.copy(deep=True)
        forecast.walk_forward_predictions(features, [dates[13]])
        self.assertEqual(np.max(DummyModel.fits[0][1]), .5)
        pd.testing.assert_frame_equal(before, features)

    def test_adjusted_monthly_features_and_future_quote_causality(self):
        monthly, daily = self.quote_fixture()
        signal = pd.Timestamp('2024-02-29')
        expected = forecast.forecast_features(monthly, daily)
        modified_monthly, modified_daily = monthly.copy(), daily.copy()
        modified_monthly.loc[modified_monthly.date.gt(signal), 'close'] *= 500
        modified_daily.loc[modified_daily.date.gt(signal), ['open', 'high', 'low', 'close']] *= 100
        modified_daily.loc[modified_daily.date.gt(signal), 'amount'] *= 1000
        actual = forecast.forecast_features(modified_monthly, modified_daily)
        # The signal's next-month label is allowed to change, but its inputs are not.
        columns = ['signal_date', 'ts_code'] + forecast.FEATURE_COLUMNS
        pd.testing.assert_frame_equal(expected.loc[expected.signal_date.le(signal), columns],
                                      actual.loc[actual.signal_date.le(signal), columns])
        self.assertEqual(expected.signal_date.min(), pd.Timestamp('2023-01-31'))
        self.assertTrue(expected.label_date.dropna().gt(expected.loc[expected.label_date.notna(), 'signal_date']).all())
        self.assertTrue(expected[forecast.FEATURE_COLUMNS].notna().all(axis=None))
        self.assertTrue(expected.loc[expected.signal_date.eq(pd.Timestamp('2024-05-31')), 'label_return'].isna().all())

    def test_month_hole_does_not_shorten_momentum_or_volatility_window(self):
        monthly, daily = self.quote_fixture()
        monthly = monthly[~(monthly.ts_code.eq('000001.SZ') & monthly.date.eq(pd.Timestamp('2023-10-31')))]
        actual = forecast.forecast_features(monthly, daily)
        forbidden = actual.ts_code.eq('000001.SZ') & actual.signal_date.between('2023-11-01', '2024-04-30')
        self.assertFalse(forbidden.any())

    def test_duplicate_keys_and_nonfuture_labels_fail(self):
        features, dates = self.feature_fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate forecast stock'):
            forecast.walk_forward_predictions(pd.concat([features, features.iloc[:1]]), [dates[13]])
        features.loc[0, 'label_date'] = features.loc[0, 'signal_date']
        with self.assertRaisesRegex(ValueError, 'labels must follow'):
            forecast.walk_forward_predictions(features, [dates[13]])

    @patch.object(forecast, 'HistGradientBoostingRegressor', DummyModel)
    def test_cache_records_source_inputs_models_and_invalidates_changed_frame(self):
        monthly, daily = self.quote_fixture()
        signals = pd.DatetimeIndex(['2024-02-29', '2024-03-29'])
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            expected = forecast.prepare_forecasts(out, monthly, daily, signals)
            fits = len(DummyModel.fits)
            actual = forecast.prepare_forecasts(out, monthly, daily, signals)
            self.assertEqual(len(DummyModel.fits), fits)
            pd.testing.assert_frame_equal(expected, actual)
            manifest = json.loads((out / 'forecast_feature_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(set(manifest['artifacts']), {'forecast_features.pkl',
                'forecast_predictions.pkl', 'forecast_models.pkl', 'forecast_model_audit.json'})
            self.assertEqual(len(manifest['specification']['source_inputs']), 3)
            self.assertEqual(len(pd.read_pickle(out / 'forecast_models.pkl')), 1)
            modified = daily.copy()
            modified.loc[modified.date.eq(pd.Timestamp('2024-02-29')), 'amount'] *= 2
            forecast.prepare_forecasts(out, monthly, modified, signals)
            self.assertGreater(len(DummyModel.fits), fits)


if __name__ == '__main__':
    unittest.main()
