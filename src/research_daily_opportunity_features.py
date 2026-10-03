"""Causal all-board daily features and next-open multi-horizon targets.

Stock histories are reindexed to the market calendar, not compressed across
suspensions. Features use actual same-day adjustment factors and completed
sessions only. Future targets are stored separately and never enter sequences.
The lazy batches keep the full feature-valid graph cross-section; missing or
not-yet-completed labels are masked in the loss, not in graph construction.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


HORIZONS = (5, 10, 20)
FEATURE_NAMES = (
    'return1', 'return5', 'return10', 'return20', 'return60',
    'volatility20', 'downside_volatility20', 'ma5_vs_ma20', 'ma20_vs_ma60',
    'breakout20', 'breakout60', 'drawdown20', 'drawdown60',
    'intraday_range', 'close_location', 'intraday_pressure', 'overnight_gap',
    'amount5_vs_prior20', 'amount_percentile', 'return5_percentile',
    'return20_percentile', 'market_breadth5', 'market_breadth20',
    'market_return1', 'market_dispersion20', 'market_amount5_vs_prior20',
)
FEATURE_PROTOCOL = {
    'version': 'daily_opportunity_features_v1',
    'features': list(FEATURE_NAMES), 'horizons': list(HORIZONS),
    'calendar': 'Complete market sessions; never bridge missing stock quotes',
    'adjustment': 'Actual daily adj_factor; no future month-end interpolation',
    'label_return': 'adj_close[t+h] / adj_open[t+1] - 1',
    'label_downside': 'max(0, 1 - min(adj_low[t+1:t+h]) / adj_open[t+1])',
    'stage_labels': 'Past-only weak context: breakout, reversal, continuation, range',
    'universe': 'All supplied SH/SZ boards including ChiNext and STAR; buy restriction is execution-only',
    'scaling': 'Fixed clipping bounds here; train-only standardization in model',
}


def normalized_dates(values, name):
    days = pd.DatetimeIndex(pd.to_datetime(values, errors='raise'))
    if days.hasnans or days.tz is not None or not days.equals(days.normalize()):
        raise ValueError(f'{name} must contain timezone-free date-only values')
    return days


@dataclass
class DailyFeatureStore:
    dates: pd.DatetimeIndex
    stock_codes: tuple
    features: np.ndarray
    valid_features: np.ndarray
    label_returns: np.ndarray
    label_downside: np.ndarray
    label_end_dates: np.ndarray
    stages: np.ndarray
    feature_names: tuple = FEATURE_NAMES

    def signal_indices(self, start=None, end=None, sequence_length=20):
        good = []
        for index in range(sequence_length - 1, len(self.dates)):
            if start is not None and self.dates[index] < pd.Timestamp(start):
                continue
            if end is not None and self.dates[index] > pd.Timestamp(end):
                continue
            if self.valid_features[index-sequence_length+1:index+1].all(axis=0).any():
                good.append(index)
        return good


def build_feature_store(daily, *, sessions=None):
    required = {'date', 'ts_code', 'open', 'high', 'low', 'close', 'amount', 'adj_factor'}
    if not isinstance(daily, pd.DataFrame) or not daily.columns.is_unique or not required <= set(daily):
        raise ValueError('Daily raw OHLC, amount and actual daily adj_factor are required')
    frame = daily[list(required)].copy()
    frame['date'] = normalized_dates(frame.date, 'Daily dates')
    if frame.empty or frame.duplicated(['date', 'ts_code']).any():
        raise ValueError('Empty or duplicate daily observations')
    if not frame.ts_code.str.fullmatch(r'\d{6}\.(SH|SZ)').all():
        raise ValueError('This daily protocol requires SH/SZ stock codes')
    values = frame[['open', 'high', 'low', 'close', 'amount', 'adj_factor']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, [0, 1, 2, 3, 5]] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError('Invalid raw prices, amounts or actual adjustment factors')
    if (frame.high < frame[['open', 'close', 'low']].max(axis=1)-1e-6).any() or (frame.low > frame[['open', 'close', 'high']].min(axis=1)+1e-6).any():
        raise ValueError('Inconsistent daily OHLC')
    observed_days = pd.DatetimeIndex(sorted(frame.date.unique()))
    days = observed_days if sessions is None else normalized_dates(sessions, 'Market sessions')
    if not days.is_unique or not days.is_monotonic_increasing or not observed_days.isin(days).all():
        raise ValueError('Market sessions must be unique, ordered and include observed days')
    codes = tuple(sorted(frame.ts_code.unique()))

    def matrix(column):
        return frame.pivot(index='date', columns='ts_code', values=column).reindex(index=days, columns=codes)

    factor = matrix('adj_factor')
    close, opening, high, low = [matrix(name)*factor for name in ('close', 'open', 'high', 'low')]
    amount = matrix('amount')
    returns = {h: close.div(close.shift(h)).sub(1).where(
        close.notna().rolling(h+1, min_periods=h+1).sum().eq(h+1)) for h in (1, 5, 10, 20, 60)}
    # Explicit fill_method=None prevents pandas from silently bridging halts.
    r1 = close.pct_change(fill_method=None)
    ma = {h: close.rolling(h, min_periods=h).mean() for h in (5, 20, 60)}
    prior_high = {h: high.shift(1).rolling(h, min_periods=h).max() for h in (20, 60)}
    peak = {h: close.rolling(h, min_periods=h).max() for h in (20, 60)}
    raw_range = high-low
    location = close.sub(low).div(raw_range.where(raw_range.ne(0))).where(raw_range.ne(0), .5)
    amount_ratio = amount.rolling(5, min_periods=5).mean().div(amount.shift(1).rolling(20, min_periods=20).mean()).sub(1)
    series = {f'return{h}': returns[h] for h in returns}
    series.update({
        'volatility20': r1.rolling(20, min_periods=20).std(ddof=0),
        'downside_volatility20': r1.clip(upper=0).rolling(20, min_periods=20).std(ddof=0),
        'ma5_vs_ma20': ma[5].div(ma[20]).sub(1),
        'ma20_vs_ma60': ma[20].div(ma[60]).sub(1),
        'breakout20': close.div(prior_high[20]).sub(1),
        'breakout60': close.div(prior_high[60]).sub(1),
        'drawdown20': close.div(peak[20]).sub(1),
        'drawdown60': close.div(peak[60]).sub(1),
        'intraday_range': raw_range.div(close), 'close_location': location,
        'intraday_pressure': close.div(opening).sub(1),
        'overnight_gap': opening.div(close.shift(1)).sub(1),
        'amount5_vs_prior20': amount_ratio,
        'amount_percentile': amount.rank(axis=1, pct=True),
        'return5_percentile': returns[5].rank(axis=1, pct=True),
        'return20_percentile': returns[20].rank(axis=1, pct=True),
    })
    total_amount = amount.sum(axis=1, min_count=1)
    context = {
        'market_breadth5': returns[5].gt(0).where(returns[5].notna()).mean(axis=1),
        'market_breadth20': returns[20].gt(0).where(returns[20].notna()).mean(axis=1),
        'market_return1': r1.median(axis=1),
        'market_dispersion20': returns[20].std(axis=1, ddof=0),
        'market_amount5_vs_prior20': total_amount.rolling(5, min_periods=5).mean().div(total_amount.shift(1).rolling(20, min_periods=20).mean()).sub(1),
    }
    shape = (len(days), len(codes))
    features = np.empty((*shape, len(FEATURE_NAMES)), dtype=np.float32)
    for column, name in enumerate(FEATURE_NAMES):
        if name in context:
            values = np.broadcast_to(context[name].to_numpy(dtype=float)[:, None], shape)
        else:
            values = series[name].to_numpy(dtype=float)
        bound = 5. if 'amount5_vs_prior20' in name else 2.
        # Clipping infinity would falsely turn an undefined ratio into a valid
        # large signal (e.g. a zero-denominator amount window).
        features[..., column] = np.where(np.isfinite(values), np.clip(values, -bound, bound), np.nan)
    valid = np.isfinite(features).all(axis=-1)
    features[~valid] = np.nan
    stages = np.full(shape, 3, dtype=np.int8)
    continuation = returns[20].gt(0) & close.gt(ma[20])
    reversal = returns[20].lt(0) & returns[5].gt(0)
    breakout = close.gt(prior_high[20]) & returns[5].gt(0)
    stages[continuation.to_numpy()] = 2
    stages[reversal.to_numpy()] = 1
    stages[breakout.to_numpy()] = 0
    stages[~valid] = -1
    label_returns = np.full((*shape, 3), np.nan, dtype=np.float32)
    label_downside = np.full_like(label_returns, np.nan)
    label_ends = np.full((len(days), 3), np.datetime64('NaT', 'D'), dtype='datetime64[D]')
    c, o, lo, hi = [value.to_numpy(dtype=float) for value in (close, opening, low, high)]
    quote_valid = np.isfinite(c) & np.isfinite(o) & np.isfinite(lo) & np.isfinite(hi)
    for horizon_index, horizon in enumerate(HORIZONS):
        for index in range(len(days)-horizon):
            complete = quote_valid[index+1:index+horizon+1].all(axis=0)
            entry = o[index+1]
            ret = c[index+horizon]/entry-1
            risk = np.maximum(0., 1-np.min(lo[index+1:index+horizon+1], axis=0)/entry)
            label_returns[index, complete, horizon_index] = ret[complete]
            label_downside[index, complete, horizon_index] = risk[complete]
            label_ends[index, horizon_index] = days[index+horizon].to_datetime64().astype('datetime64[D]')
    return DailyFeatureStore(days, codes, features, valid, label_returns,
                             label_downside, label_ends, stages)


class LazyDailyOpportunityBatch:
    """Array-compatible daily batch, materializing a single sequence on demand."""
    def __init__(self, store, index, edges, *, sequence_length=20, use_graph=True):
        if index < sequence_length-1 or index >= len(store.dates):
            raise ValueError('Incomplete sequence index')
        self.store, self.index, self.edges = store, index, edges
        self.sequence_length, self.use_graph = sequence_length, use_graph
        self.indices = np.flatnonzero(store.valid_features[index-sequence_length+1:index+1].all(axis=0))
        if not len(self.indices):
            raise ValueError('No complete stock sequences')
        self.stock_codes = tuple(store.stock_codes[number] for number in self.indices)
        self.signal_date = store.dates[index]
        self.sequence_dates = store.dates[index-sequence_length+1:index+1].to_numpy(dtype='datetime64[D]')
        available = edges.snapshot_date[edges.snapshot_date.le(self.signal_date)] if len(edges) else []
        self.membership_asof = available.max() if len(available) and use_graph else None

    @property
    def sequences(self):
        values = self.store.features[self.index-self.sequence_length+1:self.index+1, self.indices]
        return np.ascontiguousarray(values.transpose(1, 0, 2))

    @property
    def membership(self):
        import torch
        from research_daily_graph_data import asof_graph
        if not self.use_graph:
            return torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), torch.empty(0), (len(self.indices), 0)).coalesce()
        graph = asof_graph(self.edges, self.signal_date, self.stock_codes, max_age_days=None, validated=True)
        coo = graph.incidence.tocoo()
        coords = torch.from_numpy(np.stack((coo.row, coo.col)).astype(np.int64))
        return torch.sparse_coo_tensor(coords, torch.from_numpy(coo.data.astype(np.float32)), coo.shape).coalesce()

    @property
    def returns(self):
        return self.store.label_returns[self.index, self.indices]

    @property
    def downside(self):
        return self.store.label_downside[self.index, self.indices]

    @property
    def label_end_dates(self):
        return np.broadcast_to(self.store.label_end_dates[self.index], (len(self.indices), 3))

    @property
    def stage_targets(self):
        return self.store.stages[self.index, self.indices]

    def validate(self, supervised=False):
        from research_daily_opportunity_model import DailyOpportunityBatch
        return DailyOpportunityBatch.validate(self, supervised=supervised)


def expected_utility(return_mu, downside, risk_aversion=.5):
    returns, risks = np.asarray(return_mu), np.asarray(downside)
    if returns.shape != risks.shape or returns.shape[-1] != 3 or not np.isfinite(returns).all() or not np.isfinite(risks).all() or (risks < 0).any():
        raise ValueError('Finite native-unit [stocks,3] forecasts and nonnegative downside are required')
    if not np.isfinite(risk_aversion) or risk_aversion < 0:
        raise ValueError('Nonnegative risk aversion required')
    weights = np.asarray((.2, .3, .5))
    return (returns-risk_aversion*risks) @ weights
