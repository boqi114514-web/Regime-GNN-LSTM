from fastapi import APIRouter
from dashboard.utils.data_loader import (
    get_nav_data, get_backtest_summary,
    get_etf_nav_data, get_etf_backtest_summary,
    get_history_reports,
    get_stock_nav_data, get_stock_summary, get_stock_annual,
)

router = APIRouter()

# ── s3 行业层 ──────────────────────────────────────────────────────────────
@router.get("/backtest/nav")
async def nav():          return get_nav_data()

@router.get("/backtest/summary")
async def summary():      return get_backtest_summary()

# ── ETF ───────────────────────────────────────────────────────────────────
@router.get("/backtest/etf/nav")
async def etf_nav():      return get_etf_nav_data()

@router.get("/backtest/etf/summary")
async def etf_summary():  return get_etf_backtest_summary()

# ── 历史信号 ───────────────────────────────────────────────────────────────
@router.get("/backtest/history")
async def history():      return get_history_reports()

# ── s4 股票选仓（主板）────────────────────────────────────────────────────
@router.get("/backtest/stock/nav")
async def stock_nav():    return get_stock_nav_data()

@router.get("/backtest/stock/summary")
async def stock_summary(): return get_stock_summary()

@router.get("/backtest/stock/annual")
async def stock_annual():  return get_stock_annual()
