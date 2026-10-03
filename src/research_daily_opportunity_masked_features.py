"""Isolated calendar-preserving, missingness-aware daily feature variant.

Undefined indicators are not prices, returns, or labels. They remain NaN in
the partial store and are encoded as value zero + an explicit availability
mask only at the model-input boundary. A seasoned stock with a current actual
quote can resume forecasting immediately after a missing quote; a stock with
no current quote is never a current prediction / new-purchase candidate.
"""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from research_daily_opportunity_features import (
    DailyFeatureStore, FEATURE_NAMES, LazyDailyOpportunityBatch,
    build_feature_store,
)


MASK_EXTRA_NAMES = ('traded_today', 'sessions_since_previous_quote')
MASK_PROTOCOL = {
    'version': 'daily_missingness_aware_features_v1',
    'minimum_observations': 60,
    'seasoning': 'At least minimum_observations actual quotes by the signal date; future observations do not count',
    'undefined_inputs': 'Zero model-input value plus explicit per-feature availability mask; never invented prices/returns/labels',
    'current_eligibility': 'Seasoned stock with actual current OHLC, amount and adjustment factor',
    'sequence': 'Complete market calendar, with finite value/mask encodings on no-quote days; only current eligibility gates prediction',
    'quote_age': 'Calendar sessions since previous actual quote, capped at 252; first quote uses zero',
    'labels': 'Original complete-market-window labels remain unchanged, including NaN across missing quotes',
    'stage': 'Unknown (-1) unless all past inputs for weak stage classification are defined',
}


def encoded_feature_names(names):
    """Static protocol order; includes masks for subsequently appended factors."""
    names = tuple(names)
    result = names + tuple('available__'+name for name in names) + MASK_EXTRA_NAMES
    if len(set(result)) != len(result):
        raise ValueError('Encoded feature names must be unique')
    return result


@dataclass
class MaskedFeatureStore(DailyFeatureStore):
    quote_presence: object = None
    quote_age_sessions: object = None
    minimum_observations: int = 60
    missingness_encoded: bool = False

    def signal_indices(self, start=None, end=None, sequence_length=20):
        # Historical no-quote rows are explicitly encoded, not removed from
        # the calendar and not allowed to veto a resumed current quote.
        if sequence_length < 1:
            raise ValueError('Positive sequence length required')
        return [index for index in range(sequence_length-1, len(self.dates))
                if (start is None or self.dates[index] >= pd.Timestamp(start))
                and (end is None or self.dates[index] <= pd.Timestamp(end))
                and self.valid_features[index].any()]


def build_masked_feature_store(daily, *, sessions=None, minimum_observations=60):
    """Recompute the 26 base factors while retaining their individual NaNs.

    Source validation and forward labels use the unchanged original builder.
    Indicator calculations deliberately use full-calendar, complete-window
    rolling operations, never compressed observed-stock time or forward fill.
    """
    if not isinstance(minimum_observations, (int, np.integer)) or minimum_observations < 1:
        raise ValueError('minimum_observations must be a positive integer')
    original = build_feature_store(daily, sessions=sessions)
    days, codes = original.dates, original.stock_codes
    frame = daily.copy()
    frame['date'] = pd.to_datetime(frame.date, errors='raise')

    def matrix(name):
        return frame.pivot(index='date', columns='ts_code', values=name).reindex(index=days, columns=codes)

    factor = matrix('adj_factor')
    close, opening, high, low = [matrix(name)*factor for name in ('close', 'open', 'high', 'low')]
    amount = matrix('amount')
    presence = (close.notna() & opening.notna() & high.notna() & low.notna()
                & amount.notna() & factor.notna()).to_numpy()
    observed = np.cumsum(presence, axis=0)
    valid = presence & (observed >= minimum_observations)
    index = np.arange(len(days))[:, None]
    last = np.maximum.accumulate(np.where(presence, index, -1), axis=0)
    previous = np.concatenate((np.full((1, len(codes)), -1, dtype=int), last[:-1]), axis=0)
    age = np.where(previous >= 0, index-previous, 0).clip(0, 252).astype(np.float32)

    returns = {h: close.div(close.shift(h)).sub(1).where(
        close.notna().rolling(h+1, min_periods=h+1).sum().eq(h+1))
        for h in (1, 5, 10, 20, 60)}
    r1 = close.pct_change(fill_method=None)
    ma = {h: close.rolling(h, min_periods=h).mean() for h in (5, 20, 60)}
    prior_high = {h: high.shift(1).rolling(h, min_periods=h).max() for h in (20, 60)}
    peak = {h: close.rolling(h, min_periods=h).max() for h in (20, 60)}
    raw_range = high-low
    location = close.sub(low).div(raw_range.where(raw_range.ne(0))).where(raw_range.ne(0), .5)
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
        'amount5_vs_prior20': amount.rolling(5, min_periods=5).mean().div(
            amount.shift(1).rolling(20, min_periods=20).mean()).sub(1),
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
        'market_amount5_vs_prior20': total_amount.rolling(5, min_periods=5).mean().div(
            total_amount.shift(1).rolling(20, min_periods=20).mean()).sub(1),
    }
    values = np.empty((*valid.shape, len(FEATURE_NAMES)), dtype=np.float32)
    for column, name in enumerate(FEATURE_NAMES):
        array = (np.broadcast_to(context[name].to_numpy(dtype=float)[:, None], valid.shape)
                 if name in context else series[name].to_numpy(dtype=float))
        bound = 5. if 'amount5_vs_prior20' in name else 2.
        values[..., column] = np.where(np.isfinite(array), np.clip(array, -bound, bound), np.nan)

    # Stage supervision is not inferred from missing longer-term evidence.
    stage_known = (valid & np.isfinite(returns[5].to_numpy())
                   & np.isfinite(returns[20].to_numpy())
                   & np.isfinite(ma[20].to_numpy()) & np.isfinite(prior_high[20].to_numpy()))
    stages = np.full(valid.shape, -1, dtype=np.int8)
    stages[stage_known] = 3
    continuation = (returns[20].gt(0) & close.gt(ma[20])).to_numpy() & stage_known
    reversal = (returns[20].lt(0) & returns[5].gt(0)).to_numpy() & stage_known
    breakout = (close.gt(prior_high[20]) & returns[5].gt(0)).to_numpy() & stage_known
    stages[continuation], stages[reversal], stages[breakout] = 2, 1, 0
    return MaskedFeatureStore(
        dates=days, stock_codes=codes, features=values, valid_features=valid,
        label_returns=original.label_returns, label_downside=original.label_downside,
        label_end_dates=original.label_end_dates, stages=stages,
        feature_names=FEATURE_NAMES, quote_presence=presence,
        quote_age_sessions=age, minimum_observations=int(minimum_observations))


def encode_missingness(store):
    """Append masks to all supplied factors, preserving labels and eligibility."""
    if not isinstance(store, MaskedFeatureStore) or store.missingness_encoded:
        raise ValueError('An unencoded MaskedFeatureStore is required')
    values = np.asarray(store.features)
    if values.shape[:2] != store.valid_features.shape or values.shape[-1] != len(store.feature_names):
        raise ValueError('Feature values and names/eligibility shapes disagree')
    if store.quote_presence.shape != store.valid_features.shape or store.quote_age_sessions.shape != store.valid_features.shape:
        raise ValueError('Actual quote presence/age panel shape mismatch')
    if (store.valid_features & ~store.quote_presence).any():
        raise ValueError('A current no-quote stock cannot be eligible')
    if np.isinf(values).any() or not np.isfinite(store.quote_age_sessions).all():
        raise ValueError('Infinite factors or nonfinite quote ages are invalid')
    available = np.isfinite(values)
    encoded = np.concatenate((np.where(available, values, 0.).astype(np.float32),
                              available.astype(np.float32),
                              store.quote_presence[..., None].astype(np.float32),
                              store.quote_age_sessions[..., None].astype(np.float32)), axis=-1)
    return replace(store, features=encoded,
                   feature_names=encoded_feature_names(store.feature_names), missingness_encoded=True)


class MaskedDailyOpportunityBatch(LazyDailyOpportunityBatch):
    """The full-calendar sequence may contain explicitly masked no-quote rows."""
    def __init__(self, store, index, edges, *, sequence_length=20, use_graph=True):
        if not isinstance(store, MaskedFeatureStore) or not store.missingness_encoded:
            raise ValueError('Encode missingness before creating model batches')
        if sequence_length < 1 or index < sequence_length-1 or index >= len(store.dates):
            raise ValueError('Incomplete sequence index')
        self.store, self.index, self.edges = store, index, edges
        self.sequence_length, self.use_graph = sequence_length, use_graph
        self.indices = np.flatnonzero(store.valid_features[index])
        if not len(self.indices):
            raise ValueError('No seasoned stocks with actual current quotes')
        self.stock_codes = tuple(store.stock_codes[number] for number in self.indices)
        self.signal_date = store.dates[index]
        self.sequence_dates = store.dates[index-sequence_length+1:index+1].to_numpy(dtype='datetime64[D]')
        available = edges.snapshot_date[edges.snapshot_date.le(self.signal_date)] if len(edges) else []
        self.membership_asof = available.max() if len(available) and use_graph else None
