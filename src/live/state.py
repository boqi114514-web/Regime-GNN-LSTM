# -*- coding: utf-8 -*-
"""状态持久化：用 JSON 管理 last_run / last_holdings / model_version"""
import json
import os
from datetime import datetime
from typing import Any, Optional

from config import STATE_DIR

LAST_RUN_PATH = os.path.join(STATE_DIR, 'last_run.json')
LAST_HOLDINGS_PATH = os.path.join(STATE_DIR, 'last_holdings.json')
MODEL_VERSION_PATH = os.path.join(STATE_DIR, 'model_version.json')


def _read(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default if default is not None else {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[state] 读取 {path} 失败: {e}，返回默认值")
        return default if default is not None else {}


def _write(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


# ---------- last_run.json ----------

def get_last_run() -> dict:
    return _read(LAST_RUN_PATH, {})


def set_last_run(data: dict) -> None:
    _write(LAST_RUN_PATH, data)


def update_last_run(**kv) -> None:
    """合并式更新 last_run，自动加 updated_at 时间戳"""
    d = get_last_run()
    d.update(kv)
    d['updated_at'] = datetime.now().isoformat(timespec='seconds')
    set_last_run(d)


# ---------- last_holdings.json ----------

def get_last_holdings() -> Optional[dict]:
    d = _read(LAST_HOLDINGS_PATH, None)
    return d if d else None


def set_last_holdings(data: dict) -> None:
    _write(LAST_HOLDINGS_PATH, data)


# ---------- model_version.json ----------

def get_model_version() -> dict:
    return _read(MODEL_VERSION_PATH, {'current': {}, 'history': []})


def set_model_version(data: dict) -> None:
    _write(MODEL_VERSION_PATH, data)
