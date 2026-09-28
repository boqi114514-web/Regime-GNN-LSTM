# -*- coding: utf-8 -*-
"""全市场成交额/量价特征，作为 LSTM 的普通输入而非 HMM 状态。"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import LOCAL_DATA_PROCESSED, STOCK_DAILY_PATH


MARKET_FACTOR_COLS = [
    'market_turnover_rel_20_60',
    'market_turnover_rel_20_252',
    'market_turnover_change_20',
    'market_breadth_20',
    'market_up_amount_share_20',
]
CACHE = Path(LOCAL_DATA_PROCESSED) / 'market_liquidity_factors.pkl'


def build_market_factors(stock_daily: pd.DataFrame) -> pd.DataFrame:
    """只用每个信号月最后交易日及之前的日线计算。amount 单位会在比值中抵消。"""
    frame = stock_daily[['date', 'code', 'close', 'amount']].copy()
    frame['date'] = pd.to_datetime(frame['date'])
    frame['close'] = pd.to_numeric(frame['close'], errors='coerce')
    frame['amount'] = pd.to_numeric(frame['amount'], errors='coerce').clip(lower=0)
    frame = frame.sort_values(['code', 'date'])
    prior = frame.groupby('code')['close'].shift(1)
    valid_return = prior.gt(0) & frame['close'].gt(0)
    frame['up'] = frame['close'].gt(prior).where(valid_return)
    frame['up_amount'] = frame['amount'].where(frame['up'].fillna(False), 0)
    daily = frame.groupby('date').agg(
        turnover=('amount', 'sum'),
        up_count=('up', 'sum'),
        valid_count=('up', 'count'),
        up_amount=('up_amount', 'sum')).sort_index()
    daily['breadth'] = daily['up_count'] / daily['valid_count'].replace(0, np.nan)
    amt20 = daily['turnover'].rolling(20, min_periods=20).mean()
    amt60 = daily['turnover'].rolling(60, min_periods=60).mean()
    amt252 = daily['turnover'].rolling(252, min_periods=252).mean()
    daily['market_turnover_rel_20_60'] = np.log(amt20 / amt60)
    daily['market_turnover_rel_20_252'] = np.log(amt20 / amt252)
    daily['market_turnover_change_20'] = np.log(
        amt20 / daily['turnover'].shift(20).rolling(20, min_periods=20).mean())
    daily['market_breadth_20'] = daily['breadth'].rolling(20, min_periods=20).mean()
    daily['market_up_amount_share_20'] = (
        daily['up_amount'].rolling(20, min_periods=20).sum() /
        daily['turnover'].rolling(20, min_periods=20).sum())
    monthly = daily.groupby(daily.index.to_period('M')).tail(1).reset_index()
    monthly['ym'] = monthly['date'].dt.to_period('M')
    return monthly[['ym', 'date'] + MARKET_FACTOR_COLS]


def load_market_factors() -> pd.DataFrame:
    """原始日线更新时自动重算缓存，供训练及增量推理共用。"""
    raw = Path(STOCK_DAILY_PATH)
    if not raw.exists():
        raise FileNotFoundError(raw)
    if CACHE.exists() and CACHE.stat().st_mtime >= raw.stat().st_mtime:
        cached = pd.read_pickle(CACHE)
        if set(MARKET_FACTOR_COLS).issubset(cached.columns):
            return cached
    obj = pd.read_pickle(raw)
    stock = obj['df_stock'] if isinstance(obj, dict) else obj
    factors = build_market_factors(stock)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    factors.to_pickle(CACHE)
    return factors


def append_market_factors(tech: pd.DataFrame,
                          market: pd.DataFrame | None = None) -> pd.DataFrame:
    """按年月广播公共市场量价特征，保留行业自身技术因子。"""
    market = load_market_factors() if market is None else market
    left = tech.copy()
    left['date'] = pd.to_datetime(left['date'])
    left['ym'] = left['date'].dt.to_period('M')
    right = market[['ym'] + MARKET_FACTOR_COLS].drop_duplicates('ym', keep='last')
    joined = left.merge(right, on='ym', how='left', validate='many_to_one')
    return joined.drop(columns='ym')
