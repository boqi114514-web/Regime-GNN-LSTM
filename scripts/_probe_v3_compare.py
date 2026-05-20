# -*- coding: utf-8 -*-
"""对比新旧 v3 选股结果：选的行业差多少、同行业里挑的票差多少。

旧 = Regime_GNN_LSTM_旧/结果v2/stock_selections.pkl（年化 19.1%）
当前 = Regime-GNN-LSTM/results/stock_selections.pkl（年化 7.4%）
"""
import pandas as pd
import numpy as np

OLD = r"D:\desktop\有意思的事情\量化\项目\Regime_GNN_LSTM_旧\结果v2\stock_selections.pkl"
CUR = r"D:\desktop\有意思的事情\量化\项目\Regime-GNN-LSTM\results\stock_selections.pkl"

old = pd.read_pickle(OLD)
cur = pd.read_pickle(CUR)
print("旧列 :", list(old.columns))
print("新列 :", list(cur.columns))


def norm(df):
    df = df.copy()
    df['ym'] = pd.to_datetime(df['month']).dt.to_period('M')
    df['code'] = df['stock_code'].astype(str).str.extract(r'(\d{6})')[0]
    return df


old, cur = norm(old), norm(cur)
common = sorted(set(old['ym']) & set(cur['ym']))
print(f"\n旧 {old['ym'].min()}~{old['ym'].max()} ({old['ym'].nunique()}月)  "
      f"当前 {cur['ym'].min()}~{cur['ym'].max()} ({cur['ym'].nunique()}月)  "
      f"共同 {len(common)} 月")

ind_ov, stk_ov, stk_ov_in_common = [], [], []
rows = []
for ym in common:
    o, c = old[old['ym'] == ym], cur[cur['ym'] == ym]
    oi, ci = set(o['ind_code']), set(c['ind_code'])
    osk, csk = set(o['code']), set(c['code'])
    ind_ov.append(len(oi & ci) / max(len(oi | ci), 1))
    stk_ov.append(len(osk & csk) / max(len(osk | csk), 1))
    # 只看新旧都选中的行业，里面挑的票重合度
    for ind in (oi & ci):
        oc = set(o[o['ind_code'] == ind]['code'])
        cc = set(c[c['ind_code'] == ind]['code'])
        if oc or cc:
            stk_ov_in_common.append(len(oc & cc) / max(len(oc | cc), 1))
    rows.append((str(ym), len(oi & ci), len(oi), len(ci),
                 len(osk & csk), len(osk), len(csk)))

print(f"\n=== 重合度 ===")
print(f"  选的行业  Jaccard 重合均值 : {np.mean(ind_ov)*100:.0f}%")
print(f"  选的个股  Jaccard 重合均值 : {np.mean(stk_ov)*100:.0f}%")
print(f"  共同行业内 挑的票 重合均值 : {np.mean(stk_ov_in_common)*100:.0f}%")
print(f"\n抽样（每 12 月）：月份 | 共同行业/旧/新 | 共同股/旧/新")
for r in rows[::12]:
    print("  %s   行业 %d / %d / %d   个股 %d / %d / %d" % r)
