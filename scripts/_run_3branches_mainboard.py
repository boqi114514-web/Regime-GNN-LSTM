# -*- coding: utf-8 -*-
"""
三分支 × 主板过滤 对比实验

分支对应关系：
  main (Regime集成)         → results/predictions_ensemble.pkl
  fix/macro-neutral-fill    → 从 git 读取 CSV，rank 等权重建 pred_ensemble
  refactor/equal-weight     → results/predictions_ensemble_equal.pkl

主板过滤：在 s4_beta_selection 中 MAIN_BOARD_ONLY=True 已生效。
行业轮动训练（s0-s3）不变，双创股票仍参与行业日收益构建（ind_dict）。

输出：逐年净收益对比表
"""
import sys, os, time, subprocess, io
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

ROOT = r'D:/desktop/有意思的事情/量化/项目/Regime-GNN-LSTM'
sys.path.insert(0, os.path.join(ROOT, 'src'))

import s4_beta_selection as s4
from config import TOP_K, OUTPUT_DIR

assert s4.MAIN_BOARD_ONLY, "MAIN_BOARD_ONLY 未开启！"
assert s4.TOPN_PER_IND == 5, f"TOPN_PER_IND={s4.TOPN_PER_IND}，应为 5"
print(f"MAIN_BOARD_ONLY={s4.MAIN_BOARD_ONLY}  TOPN={s4.TOPN_PER_IND}")
print(f"权重: γ={s4.KF_GAMMA}  β={s4.W_BETA} 动={s4.W_MOM} 质={s4.W_QUAL} 估={s4.W_VAL}")

# ─── 加载数据（一次）────────────────────────────────────────────────────────────
print("\n加载数据...")
t0 = time.time()
stock_to_ind, ind_to_name = s4.load_stock_industry_map()
df_stock, stock_dict = s4.load_stock_daily()
ind_daily, ind_dict = s4.load_industry_daily(df_stock, stock_to_ind)
fund_dict = s4.load_fundamental_features()
pe_pb_dict = s4.load_stock_pe_pb()
print(f"加载完成 ({time.time()-t0:.0f}s)")

# ─── 重建 fix 分支的 pred_ensemble ───────────────────────────────────────────
def build_fix_pred():
    """从 git 读取 fix 分支的 GNN/LSTM CSV，rank 等权合并为 pred_ensemble"""
    def read_branch_csv(fname):
        raw = subprocess.check_output(
            ['git', 'show', f'fix/macro-neutral-fill:results/{fname}'],
            cwd=ROOT
        )
        df = pd.read_csv(io.StringIO(raw.decode('utf-8-sig', errors='replace')))
        df.columns = [c.lstrip('﻿') for c in df.columns]  # 去 BOM
        df['date'] = pd.to_datetime(df['date'])
        return df

    gnn  = read_branch_csv('predictions_gnn.csv')
    lstm = read_branch_csv('predictions_lstm_b.csv')

    # 跨月截面 rank（0~1），与 pred_ensemble 口径一致
    for df, col in [(gnn, 'pred_gnn'), (lstm, 'pred_lstm_b')]:
        df['_rank'] = df.groupby('date')[col].rank(pct=True)

    merged = pd.merge(
        gnn[['ts_code', 'date', '_rank']].rename(columns={'_rank': 'rank_gnn'}),
        lstm[['ts_code', 'date', '_rank']].rename(columns={'_rank': 'rank_lstm'}),
        on=['ts_code', 'date'], how='inner'
    )
    merged['pred_ensemble'] = (merged['rank_gnn'] + merged['rank_lstm']) / 2
    print(f"  fix 分支 pred: {merged['date'].nunique()} 个月, "
          f"{merged['date'].min().strftime('%Y-%m')} ~ {merged['date'].max().strftime('%Y-%m')}")
    return merged

# ─── 三路预测文件 ─────────────────────────────────────────────────────────────
print("\n准备三路预测...")
pred_main   = pd.read_pickle(os.path.join(OUTPUT_DIR, 'predictions_ensemble.pkl'))
pred_equal  = pd.read_pickle(os.path.join(OUTPUT_DIR, 'predictions_ensemble_equal.pkl'))
pred_fix    = build_fix_pred()

for df in [pred_main, pred_equal, pred_fix]:
    df['date'] = pd.to_datetime(df['date'])

BRANCHES = [
    ('main (Regime集成)',           pred_main),
    ('fix (等权重建)',              pred_fix),
    ('refactor (等权50/50)',        pred_equal),
]

# ─── 核心：单路选股 ────────────────────────────────────────────────────────────
def run_branch(label, pred_df):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    months = sorted(pred_df['date'].unique())
    t1 = time.time()
    all_sel = []
    prev = set()
    for i, month in enumerate(months):
        m_pred = pred_df[pred_df['date'] == month].sort_values('pred_ensemble', ascending=False)
        top_k = m_pred.head(TOP_K)
        sel = s4.select_stocks_for_month(
            pred_month=pd.Timestamp(month),
            top_industries=top_k['ts_code'].tolist(),
            stock_dict=stock_dict,
            ind_dict=ind_dict,
            stock_to_ind=stock_to_ind,
            fund_dict=fund_dict,
            ind_scores=dict(zip(top_k['ts_code'], top_k['pred_ensemble'])),
            prev_holdings=prev,
            pe_pb_dict=pe_pb_dict,
        )
        if not sel.empty:
            all_sel.append(sel)
            prev = set(sel['stock_code'])
        else:
            prev = set()
        if (i+1) % 25 == 0 or i == 0:
            n = len(sel) if not sel.empty else 0
            print(f"  [{i+1}/{len(months)}] {pd.Timestamp(month).strftime('%Y-%m')}: "
                  f"{n} 只  ({time.time()-t1:.0f}s)")

    monthly = pd.concat(all_sel, ignore_index=True)
    bt = s4.backtest_stock_portfolio(monthly, stock_dict)
    bt['date'] = pd.to_datetime(bt['date'])
    print(f"  完成，耗时 {(time.time()-t1)/60:.1f} 分钟")
    return bt

# ─── 执行三路 ─────────────────────────────────────────────────────────────────
results = {}
for label, pred in BRANCHES:
    results[label] = run_branch(label, pred)

# ─── 逐年对比 ────────────────────────────────────────────────────────────────
def annual(bt):
    rows = {}
    for yr, g in bt.groupby(bt['date'].dt.year):
        rows[yr] = (1 + g['ret_net']).prod() - 1
    return rows

print("\n\n" + "="*70)
print("  三分支 逐年净收益对比（主板过滤，TOPN=5）")
print("="*70)
labels = [l for l, _ in BRANCHES]
ann_all = {l: annual(results[l]) for l in labels}

years = sorted(set(y for a in ann_all.values() for y in a))
hdr = f"  {'年份':<6}" + "".join(f"{l[:12]:>20}" for l in labels)
print(hdr)
print("  " + "-"*66)
for yr in years:
    row = f"  {yr:<6}"
    for l in labels:
        v = ann_all[l].get(yr, float('nan'))
        row += f"{v*100:>19.2f}%"
    print(row)

print()
def fmt(m):
    return (f"年化={m['annual_return']*100:.2f}%  "
            f"Sharpe={m['sharpe_ratio']:.3f}  "
            f"回撤={m['max_drawdown']*100:.1f}%  "
            f"胜率={m['win_rate']*100:.1f}%")

for l in labels:
    bt = results[l]
    m = s4.calc_metrics(bt.set_index('date')['ret_net'])
    print(f"  {l}: {fmt(m)}")

# ─── 保存 ─────────────────────────────────────────────────────────────────────
for l, bt in results.items():
    safe = l.split('(')[0].strip().replace(' ', '_').replace('/', '_')
    bt.to_csv(os.path.join(OUTPUT_DIR, f'stock_backtest_{safe}_mainboard.csv'),
              index=False, encoding='utf-8-sig')

print(f"\n  总耗时 {(time.time()-t0)/60:.1f} 分钟")
