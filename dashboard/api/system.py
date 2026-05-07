from fastapi import APIRouter
from dashboard.utils.data_loader import get_system_status

router = APIRouter()

@router.get("/system")
async def system():
    return get_system_status()
