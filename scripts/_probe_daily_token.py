# -*- coding: utf-8 -*-
"""烟雾测试：原 token 的 4 个日频接口权限。每接口调 1 次。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from phase_a import config as C

pro = C.get_pro_daily()
TD = "20260423"

tests = [
    ("index_weight", dict(index_code="000300.SH", start_date="20260401", end_date="20260430")),
    ("index_daily",  dict(ts_code="000300.SH", start_date="20260420", end_date="20260423")),
    ("daily",        dict(trade_date=TD)),
    ("daily_basic",  dict(trade_date=TD)),
    ("moneyflow",    dict(trade_date=TD)),
]

for name, params in tests:
    fn = getattr(pro, name, None)
    if fn is None:
        print(f"[NOAPI] {name}")
        continue
    try:
        df = fn(**params)
        if df is None or len(df) == 0:
            print(f"[EMPTY] {name:<14} 空表")
        else:
            print(f"[OK   ] {name:<14} rows={len(df)} cols={list(df.columns)[:8]}")
    except Exception as e:
        print(f"[ERR  ] {name:<14} {str(e).splitlines()[0][:90]}")
