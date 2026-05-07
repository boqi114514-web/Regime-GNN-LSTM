from fastapi import APIRouter
from dashboard.utils.data_loader import get_report_data

router = APIRouter()

@router.get("/report")
async def report():
    return get_report_data()
