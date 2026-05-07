# -*- coding: utf-8 -*-
import os, sys, asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter()

_THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.normpath(os.path.join(_THIS, "..", ".."))
SRC_DIR = os.path.join(PROJECT_DIR, "src")

_running = False


async def _stream(cmd: list):
    global _running
    if _running:
        yield "data: [BUSY] 已有任务正在运行，请等待完成后再试\n\n"
        return

    _running = True
    try:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": SRC_DIR}
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=SRC_DIR,
            env=env,
        )
        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                yield f"data: {line}\n\n"
        await proc.wait()
        status = "完成" if proc.returncode == 0 else f"失败 (exit={proc.returncode})"
        yield f"data: [DONE] 任务{status}\n\n"
    except Exception as e:
        yield f"data: [ERROR] {e}\n\n"
    finally:
        _running = False


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _sse(cmd: list) -> StreamingResponse:
    return StreamingResponse(_stream(cmd), media_type="text/event-stream", headers=_SSE_HEADERS)


def _py(*args) -> list:
    return [sys.executable, *args]


@router.get("/run/weekly")
async def run_weekly():
    return _sse(_py("-m", "live.scheduler", "--once", "weekly"))


@router.get("/run/fetch_data")
async def run_fetch_data():
    """拉取最新市场数据（行情 + 宏观 + 因子增量更新）"""
    return _sse(_py("-c",
        f"import sys; sys.path.insert(0,r'{SRC_DIR}'); "
        "from data_pipeline import update; update.run()"))


@router.get("/run/update")
async def run_update():
    return _sse(_py("-c",
        f"import sys; sys.path.insert(0,r'{SRC_DIR}'); "
        "from data_pipeline import update; update.run()"))


@router.get("/run/pipeline")
async def run_pipeline(branch: str = "main"):
    """切换到指定分支，跑完整流水线，生成周报，然后恢复 main。"""
    allowed = {"main", "fix/macro-neutral-fill", "refactor/equal-weight-ensemble"}
    if branch not in allowed:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": f"unknown branch: {branch}"}, status_code=400)
    run_branch_py = os.path.join(SRC_DIR, "run_branch.py")
    return _sse(_py(run_branch_py, branch))


@router.get("/run/monitor")
async def run_monitor():
    return _sse(_py("-c",
        f"import sys; sys.path.insert(0,r'{SRC_DIR}'); "
        "from live import monitor; monitor.run()"))


@router.get("/run/train_quarterly")
async def run_train_quarterly():
    return _sse(_py("-c",
        f"import sys; sys.path.insert(0,r'{SRC_DIR}'); "
        "from live import train; train.train('quarterly')"))


@router.get("/run/update_processed")
async def run_update_processed():
    return _sse(_py("-c",
        f"import sys; sys.path.insert(0,r'{SRC_DIR}'); "
        "from data_pipeline import update; update.run(processed=True)"))


@router.get("/run/status")
async def run_status():
    return {"running": _running}
