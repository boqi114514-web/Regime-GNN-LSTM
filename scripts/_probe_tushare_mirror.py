"""探测代理可用接口（精简版）。仅测你购买的两个接口，间隔加长避免熔断。"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import tushare as ts
from tushare.pro import client as _ts_client

_ts_client.DataApi._DataApi__http_url = "http://47.109.59.144:8989/dataapi"
pro = ts.pro_api('iNW--EHGTqYsmBt4Zq78ohK9erwEMbFZaJUwf6SE344')

TS_CODE = "600519.SH"  # 贵州茅台
SLEEP = 8  # 间隔

def run_test(label, fn, *args, **kwargs):
    print(f"\n[测试] {label}")
    try:
        df = fn(*args, **kwargs)
        if df is None:
            print(f"  ✗ 返回 None")
            return
        if len(df) == 0:
            print(f"  ∅ 空表 cols={list(df.columns)}")
            return
        print(f"  ✓ rows={len(df)} cols={list(df.columns)}")
        print(f"  首2行:")
        for r in df.head(2).to_dict('records'):
            print(f"    {r}")
        if len(df) > 2:
            print(f"  末2行:")
            for r in df.tail(2).to_dict('records'):
                print(f"    {r}")
    except Exception as e:
        msg = str(e).split('\n')[0][:120]
        print(f"  ✗ ERR: {msg}")

# 测试 1: 历史 1 分钟 - 最近一周
run_test("stk_mins 1min @ 2026-04-23",
         pro.stk_mins, ts_code=TS_CODE, freq='1min',
         start_date='2026-04-23 09:00:00', end_date='2026-04-23 15:00:00')
time.sleep(SLEEP)

# 测试 2: 历史 1 分钟 - 1 年前
run_test("stk_mins 1min @ 2025-04-23 (1年前)",
         pro.stk_mins, ts_code=TS_CODE, freq='1min',
         start_date='2025-04-23 09:00:00', end_date='2025-04-23 15:00:00')
time.sleep(SLEEP)

# 测试 3: 历史 1 分钟 - 5 年前
run_test("stk_mins 1min @ 2021-04-23 (5年前)",
         pro.stk_mins, ts_code=TS_CODE, freq='1min',
         start_date='2021-04-23 09:00:00', end_date='2021-04-23 15:00:00')
time.sleep(SLEEP)

# 测试 4: 历史 5 分钟 - 跨月
run_test("stk_mins 5min @ 2026-04 整月",
         pro.stk_mins, ts_code=TS_CODE, freq='5min',
         start_date='2026-04-01 09:00:00', end_date='2026-04-30 15:00:00')
time.sleep(SLEEP)

# 测试 5: 实时日线 - 单股
run_test("rt_k 单股 600519.SH",
         pro.rt_k, ts_code=TS_CODE)
time.sleep(SLEEP)

# 测试 6: 实时日线 - 全市场（通配符）
run_test("rt_k 通配 6*.SH",
         pro.rt_k, ts_code='6*.SH')

print("\n" + "="*70)
print("探测结束")
