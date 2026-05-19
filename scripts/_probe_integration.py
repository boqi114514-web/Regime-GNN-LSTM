# -*- coding: utf-8 -*-
"""集成测试：universe 拉取 + 3 只票分钟 + 时段因子计算。约 2 分钟。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from phase_a import config as C
from phase_a import universe as U
from phase_a import data_fetch as F
from phase_a import features as FE

print("\n========== [1] universe 逐月拉取 ==========")
universe_all, union = U.fetch_monthly_universe()
print(f"universe_all shape={universe_all.shape}")
print(f"每月成分数：\n{universe_all.groupby('month').size().describe()}")

print("\n========== [2] 指数日线（交易日历）==========")
idx = F.fetch_index_daily()
tds = F.get_trade_dates()
print(f"交易日 {len(tds)} 天：{tds[0]} ~ {tds[-1]}")

print("\n========== [3] 3 只测试票分钟数据 ==========")
test_stocks = ["600519.SH", "000001.SZ", "300750.SZ"]
failed = F.fetch_minute_bars(test_stocks)
print(f"失败：{failed}")

print("\n========== [4] 时段因子计算 ==========")
seg = FE.compute_segment_returns(force=True)
print(f"segment_returns shape={seg.shape}")
print(f"列：{list(seg.columns)}")
print("\n样本（每票末 2 行）：")
for ts in test_stocks:
    sub = seg[seg["ts_code"] == ts].tail(2)
    print(sub.to_string(index=False))

print("\n========== 合理性检查 ==========")
for col in ["F1", "F2", "F3", "Y_target", "F8"]:
    s = seg[col].dropna()
    print(f"  {col}: n={len(s)} mean={s.mean():.5f} std={s.std():.5f} "
          f"min={s.min():.4f} max={s.max():.4f}")
print(f"  Y_sign 分布：{seg['Y_sign'].value_counts().to_dict()}")
print(f"  F1 缺失率：{seg['F1'].isna().mean():.1%}")
