import sys
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from research_improvements import (make_panel, exposure_series, portfolio_returns,
                                    metrics, rolling_predictions)
import research_improvements as research
from s3_ensemble_backtest import run_topk_strategy


class ResearchTests(unittest.TestCase):
    def fixtures(self, n=110):
        rng = np.random.default_rng(1)
        rows, tech = [], []
        for i in range(6):
            dates = pd.date_range('2010-01-31', periods=n, freq='ME')
            rets = rng.normal(.004, .05, n)
            close = 100 * np.cumprod(1 + rets)
            rows.extend({'ts_code': f'I{i}', 'date': d, 'close': c,
                          'ret': r, 'pe': 10.+i, 'pb': 1.+i/10}
                        for d, c, r in zip(dates, close, rets))
            tech.extend({'ts_code': f'I{i}', 'date': d, 'factor': r}
                        for d, r in zip(dates, rets))
        return pd.DataFrame(rows), pd.DataFrame(tech)

    def test_feature_prefix_invariance(self):
        market, tech = self.fixtures()
        cutoff = pd.Timestamp('2017-12-31')
        full, cols, _, _ = make_panel(market, tech)
        prefix, _, _, _ = make_panel(market[market.date <= cutoff], tech[tech.date <= cutoff])
        features = ['ts_code', 'date'] + list(dict.fromkeys(sum(cols.values(), [])))
        pd.testing.assert_frame_equal(full.loc[full.date <= cutoff, features].reset_index(drop=True),
                                      prefix[features].reset_index(drop=True))

    def test_exposure_uses_no_future_and_has_no_leverage(self):
        market, _ = self.fixtures()
        bench = market.groupby('date').ret.mean()
        for mode in ('full', 'trend', 'vol_target'):
            exposure = exposure_series(bench, mode)
            pd.testing.assert_series_equal(exposure.iloc[:80], exposure_series(bench.iloc[:80], mode))
            self.assertTrue(exposure.between(0, 1).all())

    def test_rebuilt_returns_reproduce_existing_engine(self):
        market, tech = self.fixtures()
        pred = market[['ts_code', 'date', 'ret']].rename(columns={'ret': 'actual_ret'})
        pred = pred.sort_values(['date', 'ts_code'])
        pred['score'] = np.random.default_rng(2).normal(size=len(pred))
        expected, _ = run_topk_strategy(pred, 'score', inertia=.2)
        actual, _ = portfolio_returns(pred, 'score', pd.Series(1., index=sorted(pred.date.unique())))
        np.testing.assert_allclose(actual.ret, expected)

    def test_drawdown_includes_starting_cash_nav(self):
        result = metrics(pd.Series([-.1, 0.]))
        self.assertAlmostEqual(result['max_drawdown'], -.1)

    def test_legacy_ties_preserve_source_row_order(self):
        market, _ = self.fixtures(3)
        pred = market[['ts_code', 'date', 'ret']].rename(columns={'ret': 'actual_ret'})
        pred = pred.sort_values(['date', 'ts_code'])
        for col in research.RULES + research.LEARNED + research.BLENDS + research.BASELINES[:-1]:
            pred[col] = 0.
        source = pred[['date', 'ts_code', 'actual_ret']].iloc[::-1].copy()
        source['pred_ensemble'] = 0.
        expected, _ = run_topk_strategy(source, 'pred_ensemble', inertia=.2)
        benchmark = pred.groupby('date').actual_ret.mean()
        with TemporaryDirectory() as directory, patch.object(research.pd, 'read_pickle', return_value=source):
            monthly, _, _ = research.evaluate(pred, benchmark, Path(directory), save_holdings=False)
        np.testing.assert_allclose(monthly.original_repaired, expected)

    def test_training_cutoff_and_future_label_mutation(self):
        market, tech = self.fixtures(111)
        panel, sets, _, _ = make_panel(market, tech)
        end = pd.Timestamp('2019-03-31')
        before, audit, _ = rolling_predictions(panel, sets, end)
        panel.loc[panel.label_date >= pd.Timestamp('2019-01-31'), 'rank_target'] *= -1
        panel.loc[panel.label_date >= pd.Timestamp('2019-01-31'), 'relevance'] = 4
        after, _, _ = rolling_predictions(panel, sets, end)
        names = ['ridge_price', 'gbdt_price', 'ranker_price', 'gbdt_price_value', 'ridge_tech12', 'gbdt_tech12']
        np.testing.assert_allclose(before[names], after[names])
        self.assertTrue((pd.to_datetime(audit.latest_training_label) < pd.to_datetime(audit.first_signal)).all())


if __name__ == '__main__':
    unittest.main()
