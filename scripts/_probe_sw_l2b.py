# -*- coding: utf-8 -*-
"""方向 2 Gate 0（续）：区分在用/停用二级指数，测在用指数的行情历史深度"""
import sys, os, io
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from data_pipeline.tushare_config import get_pro, _call_with_retry

pro = get_pro()

l2 = _call_with_retry(pro.index_classify, level="L2", src="SW2021")
pub = l2[l2["is_pub"] == "1"]
unpub = l2[l2["is_pub"] != "1"]
print(f"申万二级 SW2021：共 {len(l2)} 个")
print(f"  在用 is_pub=1：{len(pub)} 个")
print(f"  停用 is_pub≠1：{len(unpub)} 个")
print(f"  停用样例：{unpub['industry_name'].head(8).tolist()}")

# 取 6 个在用二级指数测行情历史
samples = pub["index_code"].head(6).tolist()
print(f"\n测 6 个在用二级指数的行情历史（需覆盖 2019~2026）：")
print(f"{'代码':<14}{'名称':<12}{'index_monthly':<28}{'sw_daily'}")
for code in samples:
    name = pub[pub["index_code"] == code]["industry_name"].iloc[0]
    # index_monthly
    im = "-"
    try:
        df = _call_with_retry(pro.index_monthly, ts_code=code,
                              start_date="20190101", end_date="20260430")
        if df is not None and len(df):
            d = df["trade_date"]
            im = f"{len(df)}行 {d.min()}~{d.max()}"
        else:
            im = "空"
    except Exception as e:
        im = f"ERR {str(e)[:20]}"
    # sw_daily
    sw = "-"
    try:
        df = _call_with_retry(pro.sw_daily, ts_code=code,
                              start_date="20190101", end_date="20260430")
        if df is not None and len(df):
            d = df["trade_date"]
            sw = f"{len(df)}行 {d.min()}~{d.max()}"
        else:
            sw = "空"
    except Exception as e:
        sw = f"ERR {str(e)[:20]}"
    print(f"{code:<14}{name:<12}{im:<28}{sw}")
