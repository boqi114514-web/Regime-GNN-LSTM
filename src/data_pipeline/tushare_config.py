# -*- coding: utf-8 -*-
"""tushare 统一入口 —— 所有模块通过此文件初始化 pro 客户端

用法：
    from data_pipeline.tushare_config import get_pro, _call_with_retry
"""
import os
import time
from functools import partial

import pandas as pd
import requests
import tushare as ts

TOKEN = ''
URL   = 'https://t.xiaodefa.top/'
GATEWAY_URL = 'https://tl.kaixin8.top/tushare/pro'

API_SLEEP = 0.3   # 每次调用后的默认间隔（秒）

_pro = None


class GatewayClient:
    """GET gateway with SDK-compatible DataFrame responses; TLS stays verified."""

    def __init__(self, key, url=GATEWAY_URL, session=None):
        self._key = key
        self._url = url.rstrip('/')
        self._session = session or requests.Session()

    def query(self, api_name, fields='', **params):
        if not api_name.replace('_', '').isalnum():
            raise ValueError('Invalid API name')
        if fields:
            params['fields'] = fields
        try:
            response = self._session.get(self._url+'/'+api_name, params=params,
                                         headers={'X-API-Key': self._key},
                                         verify=True, timeout=30)
        except requests.RequestException as exc:
            raise RuntimeError(f'{api_name}: transport {type(exc).__name__}') from None
        if response.status_code != 200:
            # Do not log request headers or arbitrary server bodies.
            raise RuntimeError(f'{api_name}: HTTP {response.status_code}')
        try:
            obj = response.json()
        except ValueError:
            raise RuntimeError(f'{api_name}: invalid JSON') from None
        if not isinstance(obj, dict) or obj.get('code') != 0 or obj.get('ok') is False:
            raise RuntimeError(f'{api_name}: API did not return success')
        data = obj.get('data')
        if not isinstance(data, dict) or not isinstance(data.get('fields'), list) or not isinstance(data.get('items'), list):
            raise RuntimeError(f'{api_name}: invalid response schema')
        return pd.DataFrame(data['items'], columns=data['fields'])

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return partial(self.query, name)


def get_pro():
    """获取 tushare pro_api 单例（支持环境变量覆盖 token/url）"""
    global _pro
    if _pro is not None:
        return _pro
    key = os.environ.get('TUSHARE_API_KEY', '').strip()
    if key:
        _pro = GatewayClient(key, os.environ.get('TUSHARE_GATEWAY_URL', GATEWAY_URL))
        return _pro
    token = os.environ.get('TUSHARE_TOKEN', TOKEN).strip()
    url   = os.environ.get('TUSHARE_URL',   URL)
    if not token:
        raise RuntimeError('缺少 TUSHARE_API_KEY（新 GET 接口）或 TUSHARE_TOKEN（旧 SDK 接口）环境变量')
    _pro = ts.pro_api(token)
    _pro._DataApi__token    = token   # 镜像要求：必须显式设置，否则取不到数据
    _pro._DataApi__http_url = url
    print(f'[tushare] connected url={url}')
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
