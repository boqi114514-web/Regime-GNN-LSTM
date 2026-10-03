import copy
from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from research_daily_opportunity_leadership import EXTRA_NAMES, leadership_factors, masked_tree_rows
from research_daily_opportunity_masked_features import build_masked_feature_store, encode_missingness
from research_daily_opportunity_trend_residual import (
    causal_reference, native_rally_events, reference_for_rows, residual_targets,
    reconstruct_native_returns,
)


class TrendResidualTests(unittest.TestCase):
    def test_reference_availability_equal_weight_single_and_unknown(self):
        own = np.array([[.1,.2,np.nan],[.1,.2,.3]])
        peer = np.array([[.3,np.nan,np.nan],[.5,.6,.7]])
        expected = np.array([[.2,.2,0],[.1,.2,.7]])
        result = causal_reference(own, peer, [[1,1,1],[1,1,0]], [[1,1,1],[0,0,1]])
        np.testing.assert_allclose(result, expected)

    def test_true_future_residual_reconstructs_without_clipping_corrections(self):
        native = np.array([[.2,.3,.4],[-.3,-.2,-.1]])
        ref = np.array([[.9,.9,.9],[.5,.5,.5]])
        residual = residual_targets(native, ref)
        np.testing.assert_allclose(reconstruct_native_returns(residual, ref), native)
        self.assertTrue((residual < 0).all())
        self.assertLess(reconstruct_native_returns(residual, ref)[1,0], 0)
        missing = native.copy(); missing[0,1] = np.nan
        self.assertTrue(np.isnan(residual_targets(missing, ref)[0,1]))

    def test_rally_events_use_native_future_returns_not_negative_residual(self):
        native = np.array([[.2,.3,.4],[.05,.1,.2]])
        ref = np.ones((2,3))
        self.assertTrue((residual_targets(native,ref) < 0).all())
        np.testing.assert_array_equal(native_rally_events(native), [[1,1,1],[0,0,0]])

    def fixture(self, mutate=False):
        days = pd.bdate_range('2024-09-02', periods=130)
        rows = []
        for j in range(7):
            for t, day in enumerate(days):
                price = (10+j)*(1.001+j*.0002)**t
                if mutate and t>105:
                    price *= 3
                rows.append(dict(date=day,ts_code=f'60000{j}.SH',open=price*.99,
                                 high=price*1.02,low=price*.98,close=price,
                                 amount=(100000+t*1000)*(20 if mutate and t>100 else 1),adj_factor=1.))
        daily = pd.DataFrame(rows)
        store = build_masked_feature_store(daily)
        edges = pd.DataFrame([dict(snapshot_date=days[0],theme_code='BK0001.DC',ts_code=c)
                              for c in store.stock_codes[:6]])
        amounts = daily.pivot(index='date',columns='ts_code',values='amount').reindex(
            index=store.dates,columns=store.stock_codes).to_numpy()
        context = leadership_factors(store,edges,amounts)
        extended = replace(store,features=np.concatenate((store.features,context),axis=-1),
                           feature_names=tuple(store.feature_names)+EXTRA_NAMES)
        return encode_missingness(extended),edges

    def test_reference_future_invariance_and_labels_not_feature_selection(self):
        store, edges = self.fixture()
        changed, _ = self.fixture(mutate=True)
        for i in (70,90,100):
            np.testing.assert_array_equal(reference_for_rows(store,i), reference_for_rows(changed,i))
        altered = copy.deepcopy(store)
        altered.label_returns[:] = 999
        altered.label_downside[:] = np.nan
        np.testing.assert_array_equal(reference_for_rows(store,90), reference_for_rows(altered,90))
        self.assertFalse(np.array_equal(store.label_returns[100], changed.label_returns[100]))
        # Original missing-future graph-node retention is unchanged.
        self.assertEqual(len(masked_tree_rows(store,edges,129)[0]),7)
        self.assertTrue(np.isnan(masked_tree_rows(store,edges,129)[2]).all())

    def test_unknown_theme_zero_is_not_observed_peer_return(self):
        store, _ = self.fixture()
        j = store.stock_codes.index('600006.SH')
        columns = {name:i for i,name in enumerate(store.feature_names)}
        reference = reference_for_rows(store,90)
        own = [store.features[90,j,columns[f'return{h}']] for h in (5,10,20)]
        np.testing.assert_allclose(reference[j], own)
        changed = copy.deepcopy(store)
        for h in (5,10,20):
            changed.features[90,j,columns[f'theme_peer_return{h}']] = 1.
        np.testing.assert_array_equal(reference_for_rows(changed,90)[j], reference[j])

    def test_invalid_dimensions_and_nonfinite_inference_rejected(self):
        with self.assertRaises(ValueError):
            causal_reference(np.zeros((3,2)),np.zeros((3,2)))
        with self.assertRaises(ValueError):
            reconstruct_native_returns([[np.nan,0,0]],[[0,0,0]])
        with self.assertRaises(ValueError):
            residual_targets([[np.inf,0,0]],[[0,0,0]])


if __name__ == '__main__':
    unittest.main()
