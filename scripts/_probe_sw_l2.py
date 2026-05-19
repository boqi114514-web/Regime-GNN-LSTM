# -*- coding: utf-8 -*-
"""方向 2 Gate 0：申万二级行业数据可得性测试"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from data_pipeline.tushare_config import get_pro, _call_with_retry

pro = get_pro()

print("=" * 60)
print("[1] 申万行业分类 index_classify")
print("=" * 60)
l1 = l2 = None
for lvl in ("L1", "L2"):
    try:
        df = _call_with_retry(pro.index_classify, level=lvl, src="SW2021")
        print(f"  {lvl}: {len(df) if df is not None else 0} 个  "
              f"列={list(df.columns) if df is not None else None}")
        if df is not None and len(df):
            print(f"      样本: {df.iloc[0].to_dict()}")
        if lvl == "L1": l1 = df
        else: l2 = df
    except Exception as e:
        print(f"  {lvl}: ERR {e}")

if l2 is None or len(l2) == 0:
    print("\n二级分类拿不到，Gate 0 失败")
    sys.exit()

# 找 index_code 列
code_col = "index_code" if "index_code" in l2.columns else l2.columns[0]
sample = l2.iloc[0][code_col]
print(f"\n二级指数数={len(l2)}，样本代码={sample}")

print("\n" + "=" * 60)
print(f"[2] 二级月度行情 index_monthly  ({sample})")
print("=" * 60)
try:
    df = _call_with_retry(pro.index_monthly, ts_code=sample,
                          start_date="20190101", end_date="20260430")
    if df is not None and len(df):
        df = df.sort_values("trade_date")
        print(f"  rows={len(df)}  范围={df['trade_date'].min()}~{df['trade_date'].max()}")
        print(f"  列={list(df.columns)}")
    else:
        print("  空 —— index_monthly 可能不覆盖申万二级")
except Exception as e:
    print(f"  ERR {e}")

print("\n" + "=" * 60)
print(f"[3] 二级行情 sw_daily  ({sample})")
print("=" * 60)
try:
    df = _call_with_retry(pro.sw_daily, ts_code=sample,
                          start_date="20190101", end_date="20260430")
    if df is not None and len(df):
        df = df.sort_values("trade_date")
        print(f"  rows={len(df)}  范围={df['trade_date'].min()}~{df['trade_date'].max()}")
        print(f"  列={list(df.columns)}")
    else:
        print("  空")
except Exception as e:
    print(f"  ERR {e}")

print("\n" + "=" * 60)
print("[4] 申万成分股 index_member_all")
print("=" * 60)
for api in ("index_member_all", "index_member"):
    fn = getattr(pro, api, None)
    if fn is None:
        print(f"  {api}: 接口不存在")
        continue
    try:
        if api == "index_member_all":
            df = _call_with_retry(fn, l2_code=sample)
        else:
            df = _call_with_retry(fn, index_code=sample)
        if df is not None and len(df):
            print(f"  {api}: rows={len(df)}  列={list(df.columns)}")
            print(f"      样本: {df.iloc[0].to_dict()}")
        else:
            print(f"  {api}: 空")
    except Exception as e:
        print(f"  {api}: ERR {str(e)[:80]}")
