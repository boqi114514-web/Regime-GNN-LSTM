# -*- coding: utf-8 -*-
"""拉个股月末 pe/pb（s4 选股加估值因子用）

做法同 gate3_l2_circ_mv.py：从 stock_daily 取每个月末交易日，
逐日调 daily_basic 拉全市场 ts_code/pe/pb，逐月缓存。

产物：data/raw/stock_pe_pb_monthly.csv  列：ts_code, trade_date, pe, pb
"""
import sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pandas as pd
from data_pipeline.tushare_config import get_pro
from config import LOCAL_DATA_RAW, load_stock_daily

OUT_CSV = os.path.join(LOCAL_DATA_RAW, "stock_pe_pb_monthly.csv")
CACHE = os.path.join(LOCAL_DATA_RAW, "_stock_pe_pb_cache")


def _safe_call(fn, **kw):
    for i in range(6):
        try:
            df = fn(**kw)
            time.sleep(0.5)
            return df
        except Exception as e:
            msg = str(e)
            if "速度过快" in msg or "rate" in msg.lower() or "频" in msg:
                wait = 20 * (i + 1)
                print(f"    限流，等 {wait}s ...")
                time.sleep(wait)
            else:
                time.sleep(3)
    return None


def main():
    os.makedirs(CACHE, exist_ok=True)
    pro = get_pro()

    sd = load_stock_daily()
    sd = sd[sd['date'] >= '2012-06-01']
    month_ends = sorted(sd.groupby(sd['date'].dt.to_period('M'))['date'].max())
    print(f"[pe_pb] {len(month_ends)} 个月末交易日，逐日拉 daily_basic pe/pb ...")

    rows, pulled, skipped, empty = [], 0, 0, []
    for i, me in enumerate(month_ends, 1):
        date_str = pd.Timestamp(me).strftime('%Y%m%d')
        fp = os.path.join(CACHE, f"{date_str}.pkl")
        if os.path.exists(fp):
            rows.append(pd.read_pickle(fp))
            skipped += 1
            continue
        df = _safe_call(pro.daily_basic, trade_date=date_str,
                        fields='ts_code,trade_date,pe,pb')
        if df is not None and len(df):
            df.to_pickle(fp)
            rows.append(df)
            pulled += 1
        else:
            empty.append(date_str)
        if i % 30 == 0 or i == len(month_ends):
            print(f"  [{i}/{len(month_ends)}] 新拉 {pulled} 跳过 {skipped} 空 {len(empty)}")

    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=['ts_code', 'trade_date'])
    out.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    print(f"\n[pe_pb] 写出 {OUT_CSV}")
    print(f"  {len(out)} 行, {out['ts_code'].nunique()} 只股票, "
          f"{out['trade_date'].min()}~{out['trade_date'].max()}")
    print(f"  pe 非空 {out['pe'].notna().mean():.1%}  pb 非空 {out['pb'].notna().mean():.1%}")
    if empty:
        print(f"  空的月末（{len(empty)}）：{empty}")


if __name__ == "__main__":
    main()
