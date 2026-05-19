# -*- coding: utf-8 -*-
"""烟雾测试：用合成因子面板跑通验证层。

合成数据植入已知动量信号 Y_target = 0.15*F1 + 0.05*F9 + 噪声，
验证 4 个验证模块 + 报告生成不崩，且能检出植入的信号。
测完删除合成 feature_panel.pkl，避免污染真实 run。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd
from phase_a import config as C
from phase_a.io_utils import save_df

np.random.seed(42)

panel_path = os.path.join(C.DIR_FEATURES, "feature_panel.pkl")
backup = panel_path + ".realbak"
if os.path.exists(panel_path):
    os.rename(panel_path, backup)   # 若已有真实面板，先备份

# ---- 合成面板 ----
n_stock, n_day = 300, 480
dates = pd.bdate_range("2024-05-06", periods=n_day).strftime("%Y%m%d")
stocks = [f"{600000+i}.SH" for i in range(n_stock)]

rows = []
for d in dates:
    f1 = np.random.normal(0, 0.0095, n_stock)
    f9 = np.random.normal(0, 1, n_stock)
    noise = np.random.normal(0, 0.0030, n_stock)
    y = 0.15 * f1 + 0.05 * 0.0095 * f9 + noise   # 植入动量 + OFI 信号
    rec = pd.DataFrame(dict(
        ts_code=stocks, date=d,
        F1=f1, F2=np.random.normal(0, 0.006, n_stock),
        F3=np.random.normal(0, 0.013, n_stock),
        F4=np.random.normal(0, 0.01, n_stock),
        F5=np.random.normal(0, 1, n_stock),
        F6=np.random.normal(0, 1, n_stock),
        F7=np.random.normal(0, 1, n_stock),
        F8=np.random.lognormal(0, 0.5, n_stock),
        F9=f9, F10=np.random.normal(0, 0.005, n_stock),
        Y_target=y, first_half_vol=np.random.lognormal(10, 1, n_stock),
        is_extreme=False))
    rec["Y_sign"] = np.sign(rec["Y_target"])
    rows.append(rec)

panel = pd.concat(rows, ignore_index=True)
save_df(panel, panel_path)
print(f"合成面板：{panel.shape}\n")

# ---- 跑验证层 ----
try:
    from phase_a import run_validate
    run_validate.main()
finally:
    # 清理：删除合成面板，恢复真实面板（如有）
    if os.path.exists(panel_path):
        os.remove(panel_path)
    if os.path.exists(backup):
        os.rename(backup, panel_path)
    print("\n[清理] 合成 feature_panel.pkl 已删除")
