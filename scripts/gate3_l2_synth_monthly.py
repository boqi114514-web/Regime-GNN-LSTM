# -*- coding: utf-8 -*-
"""方向 2 Gate 3 Step 5.5：自建二级行业月度行情（回溯到 2013）

问题：申万二级 SW2021 是 2021 年发布的新分类，Tushare 的 index_monthly 对二级
      只回溯到 2019（81 月）。walk-forward 60+12 月窗下回测期仅 ~9 个月，
      无法与一级（2013+，Sharpe 1.38）公平对比。

解法：二级行业月收益率用成分股自己合成 —— 个股月收益按当月在册成分股的
      **流通市值加权**平均（权重取上月末 circ_mv，即期初权重，贴近申万指数编制法），
      即为该二级行业的月收益。个股日线 stock_daily 有 2009+ 长历史，
      故自建行情可一路回溯到 2013，与一级对齐。

依赖：data/raw/stock_circ_mv_monthly.csv（个股月末流通市值，先跑 gate3_l2_circ_mv.py）。
      某行业某月成分股全无 circ_mv 时退回等权兜底。

产物：data/processed/sw_l2_monthly_synth.csv
      列对齐一级 ts_sw_industry_monthly.csv：
      ts_code,date,open,high,low,close,pct_chg,vol,amount,pe,pb（+ n_cons 成分股数）
      pe/pb 从 raw/ts_sw_l2_monthly.csv（sw_daily，2021+）merge，pre-2021 留空。

验证：2019+ 区间自建 pct_chg 与官方 index_monthly 的 pct_chg 逐行业算相关性。
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd
from config import (load_stock_daily, load_industry_members_l2,
                    LOCAL_DATA_RAW, LOCAL_DATA_PROCESSED)

OFFICIAL_CSV = os.path.join(LOCAL_DATA_RAW, "ts_sw_l2_monthly.csv")
CIRC_MV_CSV = os.path.join(LOCAL_DATA_RAW, "stock_circ_mv_monthly.csv")
OUT_CSV = os.path.join(LOCAL_DATA_PROCESSED, "sw_l2_monthly_synth.csv")
BASE_INDEX = 1000.0   # 合成指数基期点位


def main():
    print("=" * 60)
    print("  Gate 3 Step 5.5：自建二级行业月度行情")
    print("=" * 60)

    print("\n[1/5] 加载个股日线...")
    sd = load_stock_daily()
    sd = sd[sd['date'] >= '2012-06-01'].copy()   # 留一个月余量给首月收益
    sd['ym'] = sd['date'].dt.to_period('M')
    print(f"  {len(sd):,} 行, {sd['code'].nunique()} 只股票")

    print("\n[2/5] 个股月末收盘 + 月收益 + 流通市值权重...")
    g = sd.sort_values('date').groupby(['code', 'ym'])
    m = g.agg(me_date=('date', 'last'), close=('close', 'last'),
              amount=('amount', 'sum'), vol=('volume', 'sum')).reset_index()
    m = m.sort_values(['code', 'ym'])
    m['ret'] = m.groupby('code')['close'].pct_change()

    # 流通市值：权重取上月末 circ_mv（期初权重）
    if not os.path.exists(CIRC_MV_CSV):
        raise FileNotFoundError(f"缺 {CIRC_MV_CSV}，请先运行 scripts/gate3_l2_circ_mv.py")
    cmv = pd.read_csv(CIRC_MV_CSV)
    cmv['code'] = cmv['ts_code'].str.replace(r'\.\w+$', '', regex=True)
    cmv['ym'] = pd.to_datetime(cmv['trade_date'].astype(str), format='%Y%m%d').dt.to_period('M')
    m = m.merge(cmv[['code', 'ym', 'circ_mv']], on=['code', 'ym'], how='left')
    m['w'] = m.groupby('code')['circ_mv'].shift(1)   # 上月末市值作本月权重
    print(f"  个股月度记录：{len(m):,} 行，circ_mv 命中 {m['circ_mv'].notna().mean():.1%}")

    print("\n[3/5] 按二级成分（逐期在册）流通市值加权聚合为行业月收益...")
    members = load_industry_members_l2()
    j = m.merge(members[['l2_code', 'code', 'in_date', 'out_date']], on='code', how='inner')
    mask = (j['in_date'] <= j['me_date']) & \
           (j['out_date'].isna() | (j['out_date'] > j['me_date']))
    j = j[mask].copy()

    def _agg(grp):
        r = grp['ret']
        w = grp['w']
        ok = r.notna()
        n = int(ok.sum())
        if n == 0:
            return pd.Series({'pct_chg': np.nan, 'n_cons': 0,
                              'amount': grp['amount'].sum(), 'vol': grp['vol'].sum()})
        wok = ok & w.notna() & (w > 0)
        if wok.sum() > 0:                              # 流通市值加权
            pct = np.average(r[wok], weights=w[wok]) * 100
        else:                                          # 兜底：等权
            pct = r[ok].mean() * 100
        return pd.Series({'pct_chg': pct, 'n_cons': n,
                          'amount': grp['amount'].sum(), 'vol': grp['vol'].sum()})

    ind = j.groupby(['l2_code', 'ym']).apply(_agg).reset_index()
    ind = ind.dropna(subset=['pct_chg'])
    ind['n_cons'] = ind['n_cons'].astype(int)
    ind = ind.sort_values(['l2_code', 'ym']).reset_index(drop=True)
    print(f"  行业月度：{len(ind)} 行, {ind['l2_code'].nunique()} 个二级行业")
    print(f"  区间：{ind['ym'].min()} ~ {ind['ym'].max()}")

    print("\n[4/5] 合成指数点位 + merge pe/pb...")
    # close 由 pct_chg 累乘（基期 1000）
    ind['close'] = BASE_INDEX * ind.groupby('l2_code')['pct_chg'].transform(
        lambda s: (1 + s / 100).cumprod())
    ind['date'] = ind['ym'].dt.to_timestamp('M').dt.normalize()
    # OHLC 下游不使用，用 close 占位保持列结构与一级一致
    for c in ('open', 'high', 'low'):
        ind[c] = ind['close']

    # pe/pb：从官方 ts_sw_l2_monthly.csv（sw_daily，2021+）按 (ts_code, ym) merge
    off = pd.read_csv(OFFICIAL_CSV)
    off['ym'] = pd.to_datetime(off['trade_date'].astype(str), format='%Y%m%d').dt.to_period('M')
    pe_pb = off[['ts_code', 'ym', 'pe', 'pb']].rename(columns={'ts_code': 'l2_code'})
    ind = ind.merge(pe_pb, on=['l2_code', 'ym'], how='left')

    out = ind.rename(columns={'l2_code': 'ts_code'})[
        ['ts_code', 'date', 'open', 'high', 'low', 'close',
         'pct_chg', 'vol', 'amount', 'pe', 'pb', 'n_cons']]
    os.makedirs(LOCAL_DATA_PROCESSED, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f"  写出 {OUT_CSV}")
    print(f"  {len(out)} 行, {out['ts_code'].nunique()} 行业, "
          f"{out['date'].min().date()}~{out['date'].max().date()}")
    print(f"  pe 覆盖 {out['pe'].notna().mean():.1%}  "
          f"成分股数 中位{out['n_cons'].median():.0f} 最小{out['n_cons'].min()}")

    print("\n[5/5] 与官方 index_monthly（2019+）相关性验证...")
    off2 = off.copy()
    off2['ym'] = off2['ym'].astype(str)
    syn = out.copy()
    syn['ym'] = syn['date'].dt.to_period('M').astype(str)
    cmp = syn[['ts_code', 'ym', 'pct_chg']].merge(
        off2[['ts_code', 'ym', 'pct_chg']], on=['ts_code', 'ym'],
        suffixes=('_syn', '_off'))
    corrs = []
    for code, grp in cmp.groupby('ts_code'):
        if len(grp) >= 12:
            c = grp['pct_chg_syn'].corr(grp['pct_chg_off'])
            if np.isfinite(c):
                corrs.append(c)
    corrs = np.array(corrs)
    print(f"  重叠区间 {cmp['ym'].min()}~{cmp['ym'].max()}，{len(corrs)} 个行业有效")
    print(f"  自建 vs 官方 月收益相关性：中位 {np.median(corrs):.3f}  "
          f"均值 {corrs.mean():.3f}  最小 {corrs.min():.3f}  "
          f">0.9 占比 {(corrs > 0.9).mean():.0%}")


if __name__ == "__main__":
    main()
