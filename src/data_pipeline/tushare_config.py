# -*- coding: utf-8 -*-
"""tushare 统一入口 —— 所有模块通过此文件初始化 pro 客户端

用法：
    from data_pipeline.tushare_config import get_pro, _call_with_retry
"""
import os
import time

import tushare as ts

TOKEN = 'tevkMsKnaXaWzSHuRBflgLTBniKPdMjsvITFKttACAhFCRKCHRsczkpRLmgTZjpj'
URL   = 'http://101.35.233.113:8020/'

API_SLEEP = 0.3   # 每次调用后的默认间隔（秒）

_pro = None


def get_pro():
    """获取 tushare pro_api 单例（支持环境变量覆盖 token/url）"""
    global _pro
    if _pro is not None:
        return _pro
    token = os.environ.get('TUSHARE_TOKEN', TOKEN)
    url   = os.environ.get('TUSHARE_URL',   URL)
    _pro = ts.pro_api(token)
    _pro._DataApi__http_url = url
    print(f'[tushare] token={token[:8]}... url={url}')
    return _pro


def _call_with_retry(fn, *args, retries: int = 3, sleep: float = 2.0, **kwargs):
    """带重试的 tushare 调用包装"""
    last_err = None
    for attempt in range(retries):
        try:
            df = fn(*args, **kwargs)
            time.sleep(API_SLEEP)
            return df
        except Exception as e:
            last_err = e
            print(f'    [retry {attempt+1}/{retries}] {type(e).__name__}: {e}')
            time.sleep(sleep * (attempt + 1))
    raise RuntimeError(f'tushare 调用失败: {last_err}')
