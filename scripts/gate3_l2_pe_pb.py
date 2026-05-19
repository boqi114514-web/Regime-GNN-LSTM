# -*- coding: utf-8 -*-
"""方向 2 Gate 3 Step 5：给二级月度行情补 pe/pb

二级月度行情 ts_sw_l2_monthly.csv 初版（Gate 1）只拉了 index_monthly，无估值列。
GNN 把 pe/pb 的 60 月滚动分位（pe_pctile/pb_pctile）当节点特征，必须补上。

做法（对齐一级 update.py 的 update_sw_industry_monthly）：
  对 124 个在用二级行业拉 sw_daily 的 pe/pb，月末取最后一个交易日值，
  按 (ts_code, year-month) merge 进 ts_sw_l2_monthly.csv，新增 pe / pb 两列。
实测 sw_daily 对二级 2015-12 起即有 pe/pb（Gate 0 "2021+" 系误判），
完整覆盖月度行情区间（2019+），无需 ffill。
逐 code 缓存（_l2_pe_pb_cache），限流退避，支持断点续传。
"""
import sys, os, io, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
from data_pipeline.tushare_config import get_pro
from config import LOCAL_DATA_RAW

L2_CSV = os.path.join(LOCAL_DATA_RAW, "ts_sw_l2_monthly.csv")
CACHE = os.path.join(LOCAL_DATA_RAW, "_l2_pe_pb_cache")


def _safe_call(fn, **kw):
    """限流感知重试"""
    for i in range(6):
        try:
            df = fn(**kw)
            time.sleep(0.8)
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

    mkt = pd.read_csv(L2_CSV)
    codes = sorted(mkt['ts_code'].unique())
    print(f"[l2_pe_pb] 二级月度行情 {len(mkt)} 行 / {len(codes)} 行业，逐个拉 sw_daily pe/pb ...")

    rows, pulled, skipped, empty = [], 0, 0, []
    for i, code in enumerate(codes, 1):
        fp = os.path.join(CACHE, f"{code}.pkl")
        if os.path.exists(fp):
            rows.append(pd.read_pickle(fp))
            skipped += 1
            continue
        df = _safe_call(pro.sw_daily, ts_code=code,
                        start_date="20180101", end_date="20260531",
                        fields="ts_code,trade_date,pe,pb")
        if df is not None and len(df):
            df.to_pickle(fp)
            rows.append(df)
            pulled += 1
        else:
            empty.append(code)
        if i % 30 == 0 or i == len(codes):
            print(f"  [{i}/{len(codes)}] 新拉 {pulled} 跳过 {skipped} 空 {len(empty)}")

    daily = pd.concat(rows, ignore_index=True)
    daily['trade_date'] = pd.to_datetime(daily['trade_date'].astype(str), format='%Y%m%d')
    daily['ym'] = daily['trade_date'].dt.to_period('M')
    # 每月末最后一个交易日的 pe/pb
    month_pe_pb = (daily.sort_values(['ts_code', 'trade_date'])
                   .groupby(['ts_code', 'ym'], as_index=False).tail(1))
    month_pe_pb = month_pe_pb[['ts_code', 'ym', 'pe', 'pb']]

    # merge 进月度行情（按 ts_code + year-month，避免月末日不一致）
    mkt['ym'] = pd.to_datetime(mkt['trade_date'].astype(str), format='%Y%m%d').dt.to_period('M')
    mkt = mkt.drop(columns=['pe', 'pb'], errors='ignore')
    merged = mkt.merge(month_pe_pb, on=['ts_code', 'ym'], how='left')
    merged = merged.drop(columns=['ym'])
    merged.to_csv(L2_CSV, index=False, encoding='utf-8-sig')

    print(f"\n[l2_pe_pb] 写回 {L2_CSV}")
    print(f"  行数={len(merged)}  pe 覆盖={merged['pe'].notna().mean():.1%}  "
          f"pb 覆盖={merged['pb'].notna().mean():.1%}")
    # 按年覆盖率
    merged['_y'] = pd.to_datetime(merged['trade_date'].astype(str), format='%Y%m%d').dt.year
    cov = merged.groupby('_y')['pe'].apply(lambda s: s.notna().mean())
    print("  pe 按年覆盖率：" + "  ".join(f"{y}:{v:.0%}" for y, v in cov.items()))
    if empty:
        print(f"  sw_daily 空的行业（{len(empty)}）：{empty}")


if __name__ == "__main__":
    main()
