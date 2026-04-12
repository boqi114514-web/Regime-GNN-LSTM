# -*- coding: utf-8 -*-
"""交易日历：tushare trade_cal 的本地缓存封装

暴露接口：
    is_trading_day(d)                    → bool
    is_last_trading_day_of_month(d)      → bool
    is_last_trading_day_of_quarter(d)    → bool
    next_trading_day(d)                  → date  （d 之后的第一个交易日，不含 d）
    prev_trading_day(d)                  → date

缓存文件：data/cache/trade_cal.csv   列 = cal_date(yyyymmdd), is_open(0/1)
 - 首次调用自动拉取 [2015-01-01, today+60d]
 - 今天超过缓存覆盖范围时自动追拉到 today+60d
 - 拉取失败时回退到"周一~周五即交易日"的粗判，并打 warning

交易所固定 SSE（上交所），足够覆盖 A 股场景。
"""
import os
import sys
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import LOCAL_DATA_CACHE

CACHE_PATH = os.path.join(LOCAL_DATA_CACHE, 'trade_cal.csv')
EXCHANGE = 'SSE'
HISTORY_START = '20150101'
LOOKAHEAD_DAYS = 60  # 缓存要覆盖 today 之后多少天

_cache: Optional[pd.DataFrame] = None  # 进程内缓存


# ============================================================
#  加载 / 拉取
# ============================================================

def _to_date(d) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    return pd.Timestamp(d).date()


def _fetch_tushare(start_s: str, end_s: str) -> pd.DataFrame:
    """调 tushare trade_cal，返回 DataFrame(cal_date:str, is_open:int)"""
    from data_pipeline.update import get_pro, _call_with_retry
    pro = get_pro()
    df = _call_with_retry(pro.trade_cal, exchange=EXCHANGE,
                          start_date=start_s, end_date=end_s,
                          fields='cal_date,is_open')
    if df is None or len(df) == 0:
        raise RuntimeError('trade_cal 返回空')
    df['cal_date'] = df['cal_date'].astype(str)
    df['is_open'] = df['is_open'].astype(int)
    return df.sort_values('cal_date').reset_index(drop=True)


def _load_cache() -> pd.DataFrame:
    if not os.path.exists(CACHE_PATH):
        return pd.DataFrame(columns=['cal_date', 'is_open'])
    df = pd.read_csv(CACHE_PATH, dtype={'cal_date': str, 'is_open': int})
    return df.sort_values('cal_date').reset_index(drop=True)


def _save_cache(df: pd.DataFrame) -> None:
    os.makedirs(LOCAL_DATA_CACHE, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)


def _need_refresh(df: pd.DataFrame, target_end: str) -> bool:
    if len(df) == 0:
        return True
    return df['cal_date'].max() < target_end


def _ensure_loaded(force: bool = False) -> pd.DataFrame:
    """加载进程缓存；必要时追拉"""
    global _cache
    if _cache is not None and not force:
        return _cache

    df = _load_cache()
    today = date.today()
    target_end = (today + timedelta(days=LOOKAHEAD_DAYS)).strftime('%Y%m%d')

    if _need_refresh(df, target_end):
        try:
            start_s = df['cal_date'].max() if len(df) else HISTORY_START
            if len(df):
                # 从上次结尾后一天开始
                start_dt = pd.to_datetime(start_s, format='%Y%m%d') + pd.Timedelta(days=1)
                start_s = start_dt.strftime('%Y%m%d')
            print(f'[trade_cal] 追拉 {start_s} → {target_end}')
            new = _fetch_tushare(start_s, target_end)
            if len(df):
                df = pd.concat([df, new], ignore_index=True)
                df = df.drop_duplicates(subset='cal_date', keep='last')
                df = df.sort_values('cal_date').reset_index(drop=True)
            else:
                df = new
            _save_cache(df)
        except Exception as e:
            print(f'[trade_cal] WARN 拉取失败，回退到周一~周五粗判: {e}')

    _cache = df
    return df


# ============================================================
#  查询接口
# ============================================================

def _fallback_is_open(d: date) -> bool:
    """拉取失败时的兜底：工作日视为交易日"""
    return d.weekday() < 5


def _lookup(d: date) -> Optional[int]:
    """返回 0/1 或 None（超出缓存）"""
    df = _ensure_loaded()
    if len(df) == 0:
        return None
    key = d.strftime('%Y%m%d')
    row = df[df['cal_date'] == key]
    if len(row) == 0:
        return None
    return int(row.iloc[0]['is_open'])


def is_trading_day(d) -> bool:
    d = _to_date(d)
    v = _lookup(d)
    if v is None:
        return _fallback_is_open(d)
    return v == 1


def next_trading_day(d) -> date:
    d = _to_date(d)
    for _ in range(30):
        d = d + timedelta(days=1)
        if is_trading_day(d):
            return d
    raise RuntimeError(f'next_trading_day: 30 天内未找到交易日（起点 {d}）')


def prev_trading_day(d) -> date:
    d = _to_date(d)
    for _ in range(30):
        d = d - timedelta(days=1)
        if is_trading_day(d):
            return d
    raise RuntimeError(f'prev_trading_day: 30 天内未找到交易日（起点 {d}）')


def last_trading_day_of_month(d) -> date:
    """返回 d 所在自然月的最后一个交易日"""
    d = _to_date(d)
    first_next = (pd.Timestamp(d) + pd.offsets.MonthEnd(0)).date()
    # first_next 现在是 d 所在月最后一天（自然日）
    probe = first_next
    for _ in range(10):
        if is_trading_day(probe):
            return probe
        probe = probe - timedelta(days=1)
    raise RuntimeError(f'last_trading_day_of_month: 未找到交易日（月末 {first_next}）')


def is_last_trading_day_of_month(d) -> bool:
    d = _to_date(d)
    return is_trading_day(d) and d == last_trading_day_of_month(d)


def is_last_trading_day_of_quarter(d) -> bool:
    d = _to_date(d)
    if d.month not in (3, 6, 9, 12):
        return False
    return is_last_trading_day_of_month(d)


# ============================================================
#  CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='trade_cal 查询/刷新')
    parser.add_argument('--refresh', action='store_true', help='强制刷新缓存')
    parser.add_argument('--date', default=None, help='查询日期 yyyy-mm-dd（默认今天）')
    args = parser.parse_args()

    if args.refresh:
        global _cache
        _cache = None
        if os.path.exists(CACHE_PATH):
            os.remove(CACHE_PATH)
        _ensure_loaded(force=True)

    target = date.today() if not args.date else _to_date(args.date)
    print(f'日期          : {target}  ({["一","二","三","四","五","六","日"][target.weekday()]})')
    print(f'交易日        : {is_trading_day(target)}')
    print(f'月末最后交易日: {is_last_trading_day_of_month(target)}')
    print(f'季末最后交易日: {is_last_trading_day_of_quarter(target)}')
    try:
        print(f'下一交易日    : {next_trading_day(target)}')
        print(f'上一交易日    : {prev_trading_day(target)}')
    except RuntimeError as e:
        print(f'  {e}')


if __name__ == '__main__':
    main()
