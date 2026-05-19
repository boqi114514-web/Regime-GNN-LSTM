from typing import Optional

from fastapi import APIRouter
from dashboard.utils.data_loader import get_holdings_data

router = APIRouter()

@router.get("/holdings")
async def holdings(branch: Optional[str] = None):
    return get_holdings_data(branch)
