# -*- coding: utf-8 -*-
"""受控实验：4 月 regime 衰退误判是否精确归因于 sf_yoy 单点

在【旧 sf_yoy 口径】（社融月增量同比，噪声大）下对比两组，其他一切不变：
  A 原样          —— 4 月 sf_yoy = -0.46
  B 仅替换 4 月    —— 4 月 sf_yoy = -0.03（= 12 月滚动口径的 4 月值），其余月份不动
若 A 判衰退、B 不判衰退 → 精确证明 4 月衰退就是这一个 sf_yoy 值造成的。
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pandas as pd
from config import load_macro_factors, load_csi300_monthly
from s0_regime import build_regime_features, rolling_hmm

NM = {-1: '失败', 0: '衰退', 1: '复苏', 2: '扩张', 3: '过热'}

macro = load_macro_factors()
csi = load_csi300_monthly()
feat, cols = build_regime_features(macro, csi)   # 当前 build 已是新口径

# 把 sf_yoy 列还原成【旧口径】（macro 原始 sf_yoy 列），其余 11 个特征不受影响
m = macro.sort_values('date').reset_index(drop=True)
old_map = dict(zip(m['date'].dt.to_period('M'), m['sf_yoy']))
feat_old = feat.copy()
feat_old['sf_yoy'] = feat_old['date'].dt.to_period('M').map(old_map).ffill()


def regime_apr(df):
    r = rolling_hmm(df, cols)
    r['date'] = pd.to_datetime(r['date'])
    row = r[r['date'].dt.to_period('M') == pd.Period('2026-04')]
    return int(row['regime'].iloc[0]) if len(row) else None


# A：旧口径原样
gA = regime_apr(feat_old)

# B：旧口径，仅把 4 月 sf_yoy 换成正常值
feat_B = feat_old.copy()
apr = feat_B['date'].dt.to_period('M') == pd.Period('2026-04')
print(f"4 月 sf_yoy：旧口径 {feat_B.loc[apr, 'sf_yoy'].iloc[0]:.3f}  →  受控替换为 -0.03")
feat_B.loc[apr, 'sf_yoy'] = -0.03
gB = regime_apr(feat_B)

print(f"\nA  旧口径原样（4月 sf_yoy=-0.46）   → 4月 regime = {gA} {NM[gA]}")
print(f"B  仅改 4月 sf_yoy=-0.03（其余不动）→ 4月 regime = {gB} {NM[gB]}")
print()
if gA == 0 and gB != 0:
    print("→ 精确证明：其他 11 个特征一字未动，仅 4 月 sf_yoy 这一个值")
    print(f"  从 -0.46 改到 -0.03，4 月 regime 就从「衰退」翻成「{NM[gB]}」。")
    print("  4 月衰退误判 = sf_yoy 单点噪声，归因成立。")
else:
    print(f"→ 未完全隔离：A={NM[gA]} B={NM[gB]}，sf_yoy 之外可能还有因素。")
