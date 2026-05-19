# -*- coding: utf-8 -*-
"""两层结构便宜验证：一级主轮动 + 二级在已选行业内择强，有没有信息空间？

方向 2「二级平替一级」已证伪。退一步问：若一级轮动不动（Sharpe 1.29），
只在「一级选中的行业」内部用二级子行业做择强，值不值得做？

两个判据：
  Q1 空间：一级行业内部，二级子行业的收益分化够大吗？
           —— 若子行业齐涨齐跌，择强无意义。
  Q2 可抓：这个分化能用现成因子（mom_1m）预测吗？
           —— 有空间但抓不住也白搭。

只用现成数据，不写两层代码。区间对齐 Gate 3 回测（2018-07+）。
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from config import LOCAL_DATA_RAW, LOCAL_DATA_PROCESSED, SW_EXCLUDE

START = '2018-07-01'   # 对齐 Gate 3 二级回测起点


def main():
    print("=" * 64)
    print("  两层结构便宜验证：一级行业内 二级子行业择强 有无空间")
    print("=" * 64)

    # ---- 一级 ↔ 二级 映射（排除金融，对齐一级 main 的 29 行业）----
    mem = pd.read_csv(os.path.join(LOCAL_DATA_RAW, 'ts_sw_l2_members.csv'))
    l1l2 = mem[['l1_code', 'l2_code']].drop_duplicates()
    l1l2 = l1l2[~l1l2['l1_code'].isin(SW_EXCLUDE)]
    l2_to_l1 = dict(zip(l1l2['l2_code'], l1l2['l1_code']))
    n_l2_per_l1 = l1l2.groupby('l1_code')['l2_code'].nunique()
    print(f"\n一级行业 {n_l2_per_l1.size} 个（排除金融），旗下二级子行业数："
          f"中位 {n_l2_per_l1.median():.0f}  最小 {n_l2_per_l1.min()}  最大 {n_l2_per_l1.max()}")
    print(f"  ≥2 个子行业（可择强）的一级行业：{(n_l2_per_l1 >= 2).sum()} 个")

    # ---- 二级月收益（自建行情）----
    syn = pd.read_csv(os.path.join(LOCAL_DATA_PROCESSED, 'sw_l2_monthly_synth.csv'))
    syn['date'] = pd.to_datetime(syn['date'])
    syn['ym'] = syn['date'].dt.to_period('M')
    syn['l1'] = syn['ts_code'].map(l2_to_l1)
    syn = syn.dropna(subset=['l1'])
    syn['ret'] = syn['pct_chg'] / 100.0
    syn = syn[syn['date'] >= START].copy()
    print(f"\n二级月收益区间：{syn['ym'].min()} ~ {syn['ym'].max()}")

    # ================= Q1：分化空间 =================
    print("\n" + "-" * 64)
    print("Q1  分化空间")
    print("-" * 64)
    # 一级行业内部：每 (l1, ym) 算其二级子行业收益的离散度
    grp = syn.groupby(['l1', 'ym'])['ret']
    within = grp.agg(std='std', n='count', mean='mean', mx='max', mn='min').reset_index()
    within = within[within['n'] >= 2]
    # 一级行业之间：每 ym 各一级（用旗下二级等权均值代理）收益的离散度
    l1_ret = syn.groupby(['l1', 'ym'])['ret'].mean().reset_index()
    between = l1_ret.groupby('ym')['ret'].std()

    print(f"  一级行业【之间】月收益 std   ：中位 {between.median()*100:.2f}%")
    print(f"  一级行业【内部】子行业 std   ：中位 {within['std'].median()*100:.2f}%")
    ratio = within['std'].median() / between.median()
    print(f"  内部/之间 比值：{ratio:.2f}  "
          f"（接近或超过 1 → 行业内分化与行业间轮动同量级，有空间）")

    # oracle 择强增厚（上限）：每月每个一级行业内若完美选中最强子行业
    within['oracle'] = within['mx'] - within['mean']
    within['drag'] = within['mean'] - within['mn']   # 选到最差的拖累（对称参照）
    o_m = within['oracle'].mean()
    print(f"\n  完美择强（oracle，上限）月均增厚 {o_m*100:.2f}%  → 年化 ~{o_m*12*100:.1f}%")
    print(f"  选到最差子行业 月均拖累 {within['drag'].mean()*100:.2f}%（对称参照）")

    # ================= Q2：可预测性（全因子扫描）=================
    print("\n" + "-" * 64)
    print("Q2  现成 12 因子（9 价量 + 3 走势复刻）能否抓住")
    print("-" * 64)
    pv = pd.read_pickle(os.path.join(LOCAL_DATA_PROCESSED, 'price_volume_factors_l2.pkl'))
    pt = pd.read_pickle(os.path.join(LOCAL_DATA_PROCESSED, 'pattern_factors_l2.pkl'))
    pv['ym'] = pd.to_datetime(pv['date']).dt.to_period('M')
    pt['ym'] = pd.to_datetime(pt['date']).dt.to_period('M')
    PV_COLS = ['mom_1m', 'mom_3m', 'mom_6m', 'vol_1m', 'vol_3m',
               'turnover_chg', 'vol_price_corr', 'max_drawdown_1m', 'high_low_pos']
    PT_COLS = ['pattern_median', 'bullish_ratio', 'signal_strength']
    src = {c: pv for c in PV_COLS}
    src.update({c: pt for c in PT_COLS})

    # t 月因子 → t+1 月收益
    syn_n = syn.sort_values(['ts_code', 'ym']).copy()
    syn_n['ret_next'] = syn_n.groupby('ts_code')['ret'].shift(-1)
    base = syn_n[['ts_code', 'ym', 'l1', 'ret_next']]

    print(f"  {'因子':<18}{'行业内IC':>10}{'ICIR':>9}{'IC>0':>8}{'择强年化':>10}")
    rows = []
    for fcol in PV_COLS + PT_COLS:
        ff = src[fcol][['ts_code', 'ym', fcol]].dropna()
        m = ff.merge(base, on=['ts_code', 'ym']).dropna(subset=[fcol, 'ret_next'])
        ics, ups = [], []
        for (_l1, _ym), g in m.groupby(['l1', 'ym']):
            if len(g) >= 3:
                ic, _ = spearmanr(g[fcol], g['ret_next'])
                if np.isfinite(ic):
                    ics.append(ic)
        ics = np.array(ics)
        if len(ics) == 0:
            continue
        icm = ics.mean()
        # 按 IC 符号决定择强方向，算 top-1 子行业相对行业均值的增厚
        for (_l1, _ym), g in m.groupby(['l1', 'ym']):
            if len(g) >= 2:
                idx = g[fcol].idxmax() if icm >= 0 else g[fcol].idxmin()
                ups.append(g.loc[idx, 'ret_next'] - g['ret_next'].mean())
        upm = np.mean(ups)
        rows.append((fcol, icm, icm / ics.std(), (ics > 0).mean(), upm))
        print(f"  {fcol:<16}{icm:>10.4f}{icm/ics.std():>9.3f}"
              f"{(ics>0).mean():>7.0%}{upm*12*100:>9.1f}%")

    best = max(rows, key=lambda r: abs(r[1]))

    # ================= 结论 =================
    print("\n" + "=" * 64)
    print("结论")
    print("=" * 64)
    space_ok = ratio >= 0.7
    best_ic = abs(best[1])
    catch_ok = best_ic > 0.03 and best[4] > 0
    print(f"  Q1 分化空间：{'有' if space_ok else '弱'}"
          f"（内部/之间 std 比 {ratio:.2f}，oracle 年化 {o_m*12*100:.1f}%）")
    print(f"  Q2 可预测性：最强因子 {best[0]} 行业内 |IC|={best_ic:.4f}、"
          f"择强年化 {best[4]*12*100:.1f}%")
    if space_ok and catch_ok:
        print(f"  → 两层结构值得做：用 {best[0]} 在一级选中行业内择强")
    elif not space_ok:
        print("  → 放弃：一级行业内子行业齐涨齐跌，择强无空间")
    else:
        print("  → 现成 12 因子都抓不住行业内分化。空间真实但无可用信号 ——")
        print("     两层结构不值得现在做；除非另找针对子行业的专门因子。")


if __name__ == "__main__":
    main()
