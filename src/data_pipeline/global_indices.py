# -*- coding: utf-8 -*-
"""下载 HMM 使用的韩美指数日线；token 从环境变量读取。"""

from pathlib import Path
from datetime import date

import pandas as pd

from config import LOCAL_DATA_RAW
from data_pipeline.tushare_config import _call_with_retry, get_pro


GLOBAL_CODES = ('KS11', 'SPX', 'IXIC')
OUTPUT = Path(LOCAL_DATA_RAW) / 'ts_global_indices_daily.csv'
DATE_WINDOWS = (('20100101', '20181231'),
                ('20190101', date.today().strftime('%Y%m%d')))


def fetch_global_indices(pro=None) -> pd.DataFrame:
    pro = pro or get_pro()
    frames = []
    for code in GLOBAL_CODES:
        for start, end in DATE_WINDOWS:
            df = _call_with_retry(pro.index_global, ts_code=code,
                                  start_date=start, end_date=end,
                                  fields='ts_code,trade_date,close')
            if not df.empty:
                frames.append(df[['ts_code', 'trade_date', 'close']])
    if not frames:
        raise RuntimeError('韩美指数接口没有返回任何数据')
    result = pd.concat(frames, ignore_index=True)
    result['trade_date'] = pd.to_datetime(result['trade_date'].astype(str),
                                           format='%Y%m%d', errors='coerce')
    result['close'] = pd.to_numeric(result['close'], errors='coerce')
    result = result.dropna(subset=['trade_date', 'close'])
    result = result[result.close > 0]
    result = result.drop_duplicates(['ts_code', 'trade_date'], keep='last')
    return result.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)


def main() -> None:
    data = fetch_global_indices()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False, encoding='utf-8-sig',
                date_format='%Y%m%d')
    print(data.groupby('ts_code').agg(rows=('trade_date', 'size'),
                                      first=('trade_date', 'min'),
                                      last=('trade_date', 'max')).to_string())
    print(f'已保存 {OUTPUT}')


def update_global_indices(dry_run: bool = False, force_months: int = 0) -> dict:
    """每日/每月增量补齐，确保下次 HMM 不会沿用过期境外数据。"""
    if not OUTPUT.exists():
        if dry_run:
            return {'new_rows': 0, 'latest': None, 'would_fetch_full_history': True}
        main()
        return {'new_rows': len(pd.read_csv(OUTPUT)), 'latest': str(date.today())}
    old = pd.read_csv(OUTPUT)
    old['trade_date'] = pd.to_datetime(old['trade_date'].astype(str),
                                       format='%Y%m%d')
    pro = get_pro()
    fresh = []
    for code in GLOBAL_CODES:
        latest = old.loc[old.ts_code.eq(code), 'trade_date'].max()
        if pd.isna(latest):
            start = pd.Timestamp('2010-01-01')
        elif force_months > 0:
            start = latest - pd.DateOffset(months=force_months)
        else:
            start = latest + pd.Timedelta(days=1)
        if start.date() > date.today():
            continue
        fetched = _call_with_retry(pro.index_global, ts_code=code,
                                   start_date=start.strftime('%Y%m%d'),
                                   end_date=date.today().strftime('%Y%m%d'),
                                   fields='ts_code,trade_date,close')
        if fetched is not None and not fetched.empty:
            fresh.append(fetched[['ts_code', 'trade_date', 'close']])
    if not fresh:
        return {'new_rows': 0, 'latest': str(old.trade_date.max().date())}
    new = pd.concat(fresh, ignore_index=True)
    new['trade_date'] = pd.to_datetime(new['trade_date'].astype(str), format='%Y%m%d')
    combined = (pd.concat([old, new], ignore_index=True)
                .drop_duplicates(['ts_code', 'trade_date'], keep='last')
                .sort_values(['ts_code', 'trade_date']))
    if not dry_run:
        combined.to_csv(OUTPUT, index=False, encoding='utf-8-sig',
                        date_format='%Y%m%d')
    return {'new_rows': len(combined) - len(old),
            'latest': str(combined.trade_date.max().date())}


if __name__ == '__main__':
    main()
