from fastapi import APIRouter
from dashboard.utils.data_loader import (
    get_nav_data, get_backtest_summary,
    get_etf_nav_data, get_etf_backtest_summary,
    get_history_reports,
)

router = APIRouter()

@router.get("/backtest/nav")
async def nav():          return get_nav_data()

@router.get("/backtest/summary")
async def summary():      return get_backtest_summary()

@router.get("/backtest/etf/nav")
async def etf_nav():      return get_etf_nav_data()

@router.get("/backtest/etf/summary")
async def etf_summary():  return get_etf_backtest_summary()

@router.get("/backtest/history")
async def history():      return get_history_reports()
