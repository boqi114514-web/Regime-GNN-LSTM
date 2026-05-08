# -*- coding: utf-8 -*-
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

_STATIC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "static"))


def create_app() -> FastAPI:
    from dashboard.api import report, holdings, backtest, system, runner

    app = FastAPI(title="Regime Dashboard", docs_url=None, redoc_url=None)

    app.include_router(report.router,   prefix="/api")
    app.include_router(holdings.router, prefix="/api")
    app.include_router(backtest.router, prefix="/api")
    app.include_router(system.router,   prefix="/api")
    app.include_router(runner.router,   prefix="/api")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    app.mount("/static", StaticFiles(directory=_STATIC), name="static")

    @app.get("/")
    async def root():
        return FileResponse(os.path.join(_STATIC, "index.html"))

    return app
