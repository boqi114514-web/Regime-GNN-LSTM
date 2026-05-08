# -*- coding: utf-8 -*-
import os, sys, threading, time

_THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(_THIS)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, SRC_DIR)

import uvicorn
import webview
from dashboard.api import create_app

PORT = 8765


def _run_server():
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


def _wait_server(timeout: int = 15) -> bool:
    import urllib.request, urllib.error
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
            return True
        except Exception:
            time.sleep(0.25)
    return False


if __name__ == "__main__":
    t = threading.Thread(target=_run_server, daemon=True)
    t.start()

    if not _wait_server():
        print("FastAPI 启动超时，请检查环境", file=sys.stderr)
        sys.exit(1)

    window = webview.create_window(
        title="Regime 行业轮动 Dashboard",
        url=f"http://127.0.0.1:{PORT}",
        width=1440,
        height=900,
        min_size=(1024, 700),
        resizable=True,
    )
    webview.start()
