"""Rank-first whole-lot allocation, isolated from released optimizers.

The caller supplies already executable opening quotes. This policy buys at
most one new name: the highest causal score which has an affordable lot.
Residual cash is intentional and cannot improve a lower-ranked stock's rank.
"""
from decimal import Decimal, ROUND_FLOOR

import numpy as np
import pandas as pd

from s7_budget_portfolio import is_main_board


def leader_first_optimizer(candidates, budget, **kwargs):
    required = {'stock_code', 'ts_code', 'reference_price', 'leadership_score'}
    if not isinstance(candidates, pd.DataFrame) or not required.issubset(candidates):
        raise ValueError('Rank-first allocation missing required candidate columns')
    if not np.isfinite(budget) or budget <= 0 or kwargs.get('max_names', 1) < 1:
        raise ValueError('Rank-first allocation has no available budget/name slot')
    c = candidates.copy()
    if c.ts_code.isna().any() or c.ts_code.duplicated().any():
        raise ValueError('Missing or duplicate rank-first stock keys')
    c['stock_code'] = c.stock_code.astype(str).str.zfill(6)
    if not c.ts_code.astype(str).str[:6].eq(c.stock_code).all():
        raise ValueError('Rank-first stock keys do not match')
    for field in ('reference_price', 'leadership_score'):
        c[field] = pd.to_numeric(c[field], errors='coerce')
    c = c[c.stock_code.map(is_main_board)
          & np.isfinite(c[['reference_price', 'leadership_score']]).all(axis=1)
          & c.reference_price.gt(0) & c.leadership_score.gt(0)].copy()
    cash = Decimal(str(budget))
    c['lot_value'] = c.reference_price.map(lambda price: float(Decimal(str(price)) * 100))
    c['max_lots'] = c.reference_price.map(lambda price: int(
        (cash / (Decimal(str(price)) * 100)).to_integral_value(rounding=ROUND_FLOOR)))
    c = c[c.max_lots.ge(1)].sort_values(
        ['leadership_score', 'ts_code'], ascending=[False, True]).head(1).copy()
    if c.empty:
        # Reuse the executor's established no-affordable-lot signal, so a
        # future low-cash month stays in cash rather than aborting the account.
        raise ValueError('所有候选的一手价格超过可用预算')
    c['lots'] = c.max_lots.astype(int)
    c['shares'] = c.lots * 100
    c['planned_amount'] = c.lots * c.lot_value
    if c.planned_amount.sum() > budget + .000001:
        raise AssertionError('Rank-first whole-lot allocation exceeds available cash')
    return c, dict(budget=budget, invested=float(c.planned_amount.sum()),
                   residual_cash=float(budget-c.planned_amount.sum()),
                   objective='highest_score_first_then_maximum_affordable_lots')
