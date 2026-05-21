# -*- coding: utf-8 -*-
"""
三分支完全独立运行 + 主板过滤（隔离版）

每个分支用 git worktree + 独立 OUTPUT_DIR 隔离，保证不互相污染：
  main (Regime集成)        : results/predictions_ensemble.pkl（已有）
  fix/macro-neutral-fill   : D:/temp/fix_out/ → predictions_ensemble_fix.pkl
  refactor/equal-weight    : D:/temp/ref_out/ → predictions_ensemble_refactor.pkl

s4 主板过滤（MAIN_BOARD_ONLY=True，TOPN=5）统一在 main 分支代码里执行。
"""
import sys, os, time, subprocess, io, shutil
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

ROOT     = r'D:\desktop\有意思的事情\量化\项目\Regime-GNN-LSTM'
RESULTS  = os.path.join(ROOT, 'results')
WT_FIX   = r'D:\temp\regime_fix_wt'
WT_REF   = r'D:\temp\regime_refactor_wt'
OUT_FIX  = r'D:\temp\fix_out'
OUT_REF  = r'D:\temp\ref_out'

sys.path.insert(0, os.path.join(ROOT, 'src'))
import s4_beta_selection as s4
from config import TOP_K

assert s4.MAIN_BOARD_ONLY, "MAIN_BOARD_ONLY 未开启"
assert s4.TOPN_PER_IND == 5, "TOPN_PER_IND 应为 5"
print(f"MAIN_BOARD_ONLY={s4.MAIN_BOARD_ONLY}  TOPN={s4.TOPN_PER_IND}")
print(f"权重: γ={s4.KF_GAMMA} β={s4.W_BETA} 动={s4.W_MOM} 质={s4.W_QUAL} 估={s4.W_VAL}")


# ─── 工具函数 ─────────────────────────────────────────────────────────────────
def run_git(*args, cwd=ROOT):
    subprocess.run(['git'] + list(args), cwd=cwd,
                   capture_output=True, text=True, encoding='utf-8', errors='replace')

def setup_output_dir(out_dir):
    """建立独立输出目录，复制缓存文件"""
    os.makedirs(out_dir, exist_ok=True)
    for f in ['_cache_stock_daily.pkl', '_cache_ind_daily.pkl',
              '_cache_fund_monthly.pkl', '_cache_stock_pe_pb.pkl',
              'predictions_gnn.pkl', 'predictions_lstm_b.pkl']:
        src = os.path.join(RESULTS, f)
        dst = os.path.join(out_dir, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
    print(f"  输出目录: {out_dir}")

def inject_output_dir(wt_dir, out_dir):
    """在 worktree 的 config.py 末尾追加 OUTPUT_DIR 覆盖（末尾追加确保覆盖原定义）"""
    cfg_path = os.path.join(wt_dir, 'src', 'config.py')
    with open(cfg_path, encoding='utf-8', errors='replace') as f:
        content = f.read()
    marker = '# __INJECTED_OUTPUT_DIR__'
    if marker not in content:
        override = (f"\n{marker}\n"
                    f"import os as _os_inj\n"
                    f"OUTPUT_DIR = r'{out_dir}'\n"
                    f"_os_inj.makedirs(OUTPUT_DIR, exist_ok=True)\n")
        with open(cfg_path, 'a', encoding='utf-8') as f:
            f.write(override)
        print(f"  已注入 OUTPUT_DIR → {out_dir}")

def run_script(script_path, cwd, label):
    print(f"  运行: {os.path.basename(script_path)}")
    r = subprocess.run(
        [sys.executable, script_path], cwd=cwd,
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        timeout=600
    )
    lines = (r.stdout + r.stderr).splitlines()
    # 只打印关键行
    for l in lines:
        if any(k in l for k in ['年化', '夏普', 'Regime', '保存', 'ERROR', '完成',
                                  'annual', 'sharpe', 'error', 'Error', '失败']):
            print(f"    {l}")
    if r.returncode != 0 and 'ERROR' not in r.stdout + r.stderr:
        print(f"    [exit {r.returncode}] last: {lines[-3:]}")
    return r.returncode == 0

def create_worktree(wt_path, branch):
    if os.path.exists(wt_path):
        run_git('worktree', 'remove', '--force', wt_path)
        shutil.rmtree(wt_path, ignore_errors=True)
    run_git('worktree', 'add', wt_path, branch)
    print(f"  worktree: {wt_path}  ({branch})")

def remove_worktree(wt_path):
    run_git('worktree', 'remove', '--force', wt_path)
    shutil.rmtree(wt_path, ignore_errors=True)
    print(f"  worktree 已清理")


# ─── 步骤 1：fix 分支 (全量HMM) ──────────────────────────────────────────────
FIX_PRED = os.path.join(RESULTS, 'predictions_ensemble_fix.pkl')

print("="*60)
print("  1/3  fix/macro-neutral-fill 分支：s0(全量HMM) + s3")
print("="*60)

if os.path.exists(FIX_PRED):
    print(f"  已有缓存 → 跳过  ({FIX_PRED})")
else:
    create_worktree(WT_FIX, 'fix/macro-neutral-fill')
    setup_output_dir(OUT_FIX)
    inject_output_dir(WT_FIX, OUT_FIX)

    # s0 → regime_labels.pkl 写入 OUT_FIX
    ok0 = run_script(os.path.join(WT_FIX, 'src', 's0_regime.py'), WT_FIX, 'fix-s0')
    if not ok0:
        print("  [WARN] s0 失败，s3 将无 regime labels（降级为自适应集成）")

    # s3 → predictions_ensemble.pkl 写入 OUT_FIX
    run_script(os.path.join(WT_FIX, 'src', 's3_ensemble_backtest.py'), WT_FIX, 'fix-s3')

    # 拷贝到 results/
    src = os.path.join(OUT_FIX, 'predictions_ensemble.pkl')
    if os.path.exists(src):
        shutil.copy2(src, FIX_PRED)
        print(f"  ✓ 已保存: predictions_ensemble_fix.pkl")
    else:
        print("  [ERROR] 未生成预测文件")

    remove_worktree(WT_FIX)
    shutil.rmtree(OUT_FIX, ignore_errors=True)


# ─── 步骤 2：refactor 分支 (等权，无HMM) ────────────────────────────────────
REF_PRED = os.path.join(RESULTS, 'predictions_ensemble_refactor.pkl')

print("\n" + "="*60)
print("  2/3  refactor/equal-weight-ensemble 分支：s3(等权)")
print("="*60)

if os.path.exists(REF_PRED):
    print(f"  已有缓存 → 跳过  ({REF_PRED})")
else:
    create_worktree(WT_REF, 'refactor/equal-weight-ensemble')
    setup_output_dir(OUT_REF)
    inject_output_dir(WT_REF, OUT_REF)

    # 不跑 s0（refactor 分支无 HMM），直接 s3
    run_script(os.path.join(WT_REF, 'src', 's3_ensemble_backtest.py'), WT_REF, 'ref-s3')

    # refactor 保存 equal 版本
    for fname in ['predictions_ensemble.pkl', 'predictions_ensemble_equal.pkl']:
        src = os.path.join(OUT_REF, fname)
        if os.path.exists(src):
            shutil.copy2(src, REF_PRED)
            print(f"  ✓ 已保存: predictions_ensemble_refactor.pkl  (from {fname})")
            break
    else:
        print("  [ERROR] 未生成预测文件")

    remove_worktree(WT_REF)
    shutil.rmtree(OUT_REF, ignore_errors=True)


# ─── 步骤 3：加载数据（一次）────────────────────────────────────────────────
print("\n" + "="*60)
print("  3/3  s4 主板选股（三路预测对比）")
print("="*60)

t0 = time.time()
stock_to_ind, ind_to_name = s4.load_stock_industry_map()
df_stock, stock_dict       = s4.load_stock_daily()
ind_daily, ind_dict        = s4.load_industry_daily(df_stock, stock_to_ind)
fund_dict                  = s4.load_fundamental_features()
pe_pb_dict                 = s4.load_stock_pe_pb()
print(f"数据加载 ({time.time()-t0:.0f}s)")


# ─── 三路预测 ─────────────────────────────────────────────────────────────────
def load_pred(path):
    if not os.path.exists(path):
        return None
    df = pd.read_pickle(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

BRANCHES = [
    ('main (Regime集成)',     load_pred(os.path.join(RESULTS, 'predictions_ensemble.pkl'))),
    ('fix (全量HMM)',         load_pred(FIX_PRED)),
    ('refactor (等权50/50)', load_pred(REF_PRED)),
]
BRANCHES = [(l, p) for l, p in BRANCHES if p is not None]
print(f"可用分支: {[l for l,_ in BRANCHES]}")


# ─── 选股 ────────────────────────────────────────────────────────────────────
def run_branch(label, pred_df):
    print(f"\n  [{label}]")
    months = sorted(pred_df['date'].unique())
    t1 = time.time()
    all_sel, prev = [], set()
    for i, month in enumerate(months):
        m_pred = pred_df[pred_df['date'] == month].sort_values('pred_ensemble', ascending=False)
        top_k = m_pred.head(TOP_K)
        sel = s4.select_stocks_for_month(
            pred_month=pd.Timestamp(month),
            top_industries=top_k['ts_code'].tolist(),
            stock_dict=stock_dict, ind_dict=ind_dict,
            stock_to_ind=stock_to_ind, fund_dict=fund_dict,
            ind_scores=dict(zip(top_k['ts_code'], top_k['pred_ensemble'])),
            prev_holdings=prev, pe_pb_dict=pe_pb_dict,
        )
        if not sel.empty:
            all_sel.append(sel)
            prev = set(sel['stock_code'])
        else:
            prev = set()
        if (i+1) % 25 == 0 or i == 0:
            print(f"  [{i+1}/{len(months)}] {pd.Timestamp(month).strftime('%Y-%m')}: "
                  f"{len(sel) if not sel.empty else 0} 只  ({time.time()-t1:.0f}s)")

    bt = s4.backtest_stock_portfolio(
        pd.concat(all_sel, ignore_index=True), stock_dict)
    bt['date'] = pd.to_datetime(bt['date'])
    print(f"  完成 ({(time.time()-t1)/60:.1f} 分)")
    return bt


results = {l: run_branch(l, p) for l, p in BRANCHES}


# ─── 输出 ─────────────────────────────────────────────────────────────────────
def annual(bt):
    return {yr: (1 + g['ret_net']).prod() - 1
            for yr, g in bt.groupby(bt['date'].dt.year)}

labels = [l for l, _ in BRANCHES]
ann    = {l: annual(results[l]) for l in labels}
years  = sorted(set(y for a in ann.values() for y in a))

print("\n\n" + "="*72)
print("  三分支 逐年净收益（主板过滤，5行业×5只=25股）")
print("="*72)
col = 20
hdr = f"  {'年份':<6}" + "".join(f"{l[:col]:>{col}}" for l in labels)
print(hdr)
print("  " + "-"*68)
for yr in years:
    row = f"  {yr:<6}"
    for l in labels:
        v = ann[l].get(yr, float('nan'))
        row += f"{v*100:>{col-1}.2f}%"
    print(row)

print()
def fmt(m):
    return (f"年化={m['annual_return']*100:.2f}%  Sharpe={m['sharpe_ratio']:.3f}  "
            f"回撤={m['max_drawdown']*100:.1f}%  胜率={m['win_rate']*100:.1f}%")

for l in labels:
    m = s4.calc_metrics(results[l].set_index('date')['ret_net'])
    print(f"  {l}: {fmt(m)}")

for l, bt in results.items():
    safe = l.split('(')[0].strip().replace(' ', '_')
    bt.to_csv(os.path.join(RESULTS, f'bt_{safe}_mainboard.csv'),
              index=False, encoding='utf-8-sig')

print(f"\n  总耗时 {(time.time()-t0)/60:.1f} 分钟")
