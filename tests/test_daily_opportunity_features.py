import numpy as np
import pandas as pd
import unittest

from research_daily_opportunity_features import (
    FEATURE_NAMES, LazyDailyOpportunityBatch, build_feature_store, expected_utility,
)


def fixture_daily(n=120):
    days = pd.bdate_range('2024-09-02', periods=n)
    rows = []
    for k, code in enumerate(('000001.SZ', '300001.SZ', '688001.SH')):
        for index, day in enumerate(days):
            price = (10+k*3)*(1.001+ k*.0002)**index
            rows.append(dict(date=day, ts_code=code, open=price*.995,
                             high=price*1.02, low=price*.98, close=price,
                             amount=100000+index*1000, adj_factor=1.))
    return pd.DataFrame(rows)


def test_next_open_targets_use_actual_market_horizons():
    data = fixture_daily()
    store = build_feature_store(data)
    t, code = 80, store.stock_codes[0]
    stock = data[data.ts_code.eq(code)].set_index('date').reindex(store.dates)
    for j, h in enumerate((5, 10, 20)):
        assert np.isclose(store.label_returns[t, 0, j], stock.close.iloc[t+h]/stock.open.iloc[t+1]-1)
        assert np.isclose(store.label_downside[t, 0, j], max(0, 1-stock.low.iloc[t+1:t+h+1].min()/stock.open.iloc[t+1]))
        assert store.label_end_dates[t, j] == store.dates[t+h].to_datetime64().astype('datetime64[D]')


def test_future_changes_cannot_change_features_or_stage():
    data = fixture_daily()
    base = build_feature_store(data)
    cutoff = base.dates[85]
    changed = data.copy()
    changed.loc[changed.date.gt(base.dates[90]), ['open', 'high', 'low', 'close']] *= 2
    changed.loc[changed.date.gt(cutoff), 'amount'] *= 10
    after = build_feature_store(changed)
    np.testing.assert_allclose(base.features[:86], after.features[:86], equal_nan=True)
    np.testing.assert_array_equal(base.stages[:86], after.stages[:86])
    assert not np.array_equal(base.label_returns[85], after.label_returns[85])


def test_adjustment_split_does_not_make_fake_crash():
    data = fixture_daily()
    base = build_feature_store(data)
    changed = data.copy()
    mask = changed.date.ge(base.dates[75]) & changed.ts_code.eq('000001.SZ')
    changed.loc[mask, ['open', 'high', 'low', 'close']] /= 2
    changed.loc[mask, 'adj_factor'] = 2
    adjusted = build_feature_store(changed)
    np.testing.assert_allclose(base.features, adjusted.features, equal_nan=True)
    np.testing.assert_allclose(base.label_returns, adjusted.label_returns, equal_nan=True)
    np.testing.assert_allclose(base.label_downside, adjusted.label_downside, equal_nan=True)


def test_missing_stock_session_invalidates_window_not_compressed():
    data = fixture_daily()
    days = pd.DatetimeIndex(sorted(data.date.unique()))
    data = data[~(data.ts_code.eq('000001.SZ') & data.date.eq(days[90]))]
    store = build_feature_store(data, sessions=days)
    assert np.isnan(store.label_returns[85, 0, 0])
    assert not store.valid_features[95, 0]
    assert store.valid_features[95, 1]
    batch = LazyDailyOpportunityBatch(store, 110, pd.DataFrame(columns=['snapshot_date', 'theme_code', 'ts_code']))
    assert '000001.SZ' not in batch.stock_codes
    assert '300001.SZ' in batch.stock_codes and '688001.SH' in batch.stock_codes


def test_inference_keeps_stocks_without_future_labels():
    store = build_feature_store(fixture_daily())
    edges = pd.DataFrame([dict(snapshot_date=store.dates[85], theme_code='BK0001.DC', ts_code='000001.SZ'),
                          dict(snapshot_date=store.dates[85], theme_code='BK0002.DC', ts_code='000001.SZ'),
                          dict(snapshot_date=store.dates[85], theme_code='BK0001.DC', ts_code='300001.SZ')])
    batch = LazyDailyOpportunityBatch(store, 119, edges)
    batch.validate(supervised=True)
    assert len(batch.stock_codes) == 3
    assert np.isnan(batch.returns).all()
    assert tuple(batch.membership.shape) == (3, 2)
    assert batch.membership.to_dense()[0].sum() == 2
    assert batch.sequences.shape == (3, 20, len(FEATURE_NAMES))


def test_graph_cannot_leak_future_membership_and_replaces_snapshot():
    store = build_feature_store(fixture_daily())
    edges = pd.DataFrame([dict(snapshot_date=store.dates[85], theme_code='BK0001.DC', ts_code='000001.SZ'),
                          dict(snapshot_date=store.dates[105], theme_code='BK0002.DC', ts_code='300001.SZ')])
    earlier = LazyDailyOpportunityBatch(store, 100, edges)
    later = LazyDailyOpportunityBatch(store, 110, edges)
    assert earlier.membership.to_dense()[0, 0] == 1
    assert later.membership.to_dense()[0, 0] == 0
    assert later.membership.to_dense()[1, 0] == 1


def check_bad_source(mutation):
    data = fixture_daily()
    if mutation == 'duplicates':
        data = pd.concat([data, data.iloc[[0]]])
    elif mutation == 'no_factor':
        data = data.drop(columns='adj_factor')
    elif mutation == 'bad_ohlc':
        data.loc[0, 'high'] = 1
    elif mutation == 'negative_factor':
        data.loc[0, 'adj_factor'] = -1
    else:
        data.loc[0, 'date'] += pd.Timedelta(hours=1)
    with unittest.TestCase().assertRaises(ValueError):
        build_feature_store(data)


def test_reject_bad_sources():
    for mutation in ('duplicates', 'no_factor', 'bad_ohlc', 'negative_factor', 'intraday'):
        check_bad_source(mutation)


def test_expected_return_minus_downside_is_not_rank_alias():
    mu = np.array([[.1, .2, .3], [.2, .1, -.1]])
    risk = np.array([[.1, .1, .2], [.1, .2, .3]])
    score = expected_utility(mu, risk)
    np.testing.assert_allclose(score, (.2*mu[:, 0]+.3*mu[:, 1]+.5*mu[:, 2]) - .5*(.2*risk[:, 0]+.3*risk[:, 1]+.5*risk[:, 2]))
    with unittest.TestCase().assertRaises(ValueError):
        expected_utility(mu, -risk)


class DailyOpportunityFeatureTests(unittest.TestCase):
    pass


for _name, _test in list(globals().items()):
    if _name.startswith('test_') and callable(_test):
        setattr(DailyOpportunityFeatureTests, _name, lambda self, function=_test: function())
del _name, _test


if __name__ == '__main__':
    unittest.main()
