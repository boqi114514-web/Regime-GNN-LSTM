# -*- coding: utf-8 -*-
"""
重生成三分支已归档周报的「个股持仓」+「主板持仓」两章节。

口径（用户确认）：
  个股持仓（选股层）= 全市场最优（含双创，带名字）
  主板持仓          = 强制主板、每行业满 TOPN 只（带名字）

全部用 main 的 s4 选股逻辑（名字映射正常、估值因子、KF_GAMMA=0.2），
保证三分支口径一致。不切 git、不重训，只读各分支预测 pkl。

分支 → 主输出标签 → 持仓列表 [(label, pred_pkl), ...]：
  main     : Regime 集成 + 等权集成
  fix      : Regime 集成 + 等权集成（等权由 fix pkl 的 rank 推导）
  refactor : 等权集成（本身即等权）
"""
import sys, os
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

ROOT = r'D:\desktop\有意思的事情\量化\项目\Regime-GNN-LSTM'
sys.path.insert(0, os.path.join(ROOT, 'src'))

import s4_beta_selection as s4
from live.monitor import _stock_table_lines
from config import OUTPUT_DIR

REPORTS_BR = os.path.join(ROOT, 'reports', 'branches')


def make_equal_pkl(src_pkl, dst_name):
    """等权预测：pred_ensemble = -(0.5*rank_gnn + 0.5*rank_lstm)，与 s3 一致"""
    df = pd.read_pickle(os.path.join(OUTPUT_DIR, src_pkl)).copy()
    df['pred_ensemble'] = -(0.5 * df['rank_gnn'] + 0.5 * df['rank_lstm'])
    df.to_pickle(os.path.join(OUTPUT_DIR, dst_name))
    return dst_name


def select(pred_pkl, ckpt, main_board_only):
    return s4.run_live(pred_pkl=pred_pkl, ckpt_suffix=ckpt,
                       force_refresh_latest=True, main_board_only=main_board_only)


def section(label, df):
    if df is None or df.empty:
        return []
    n = df['stock_code'].nunique()
    lines = [f'### {label}（{n} 只）', '']
    lines += _stock_table_lines(df)
    lines.append('')
    return lines


def build_holdings(branch_key, items):
    """items: [(label, pred_pkl), ...]  → (个股持仓行, 主板持仓行)"""
    all_lines = ['## 📋 个股持仓（选股层）', '']
    mb_lines  = ['## 📋 主板持仓（保证主板，每行业满额）', '']
    for i, (label, pkl) in enumerate(items):
        df_all = select(pkl, f'_{branch_key}_{i}_all', main_board_only=False)
        df_mb  = select(pkl, f'_{branch_key}_{i}_mb',  main_board_only=True)
        all_lines += section(label, df_all)
        mb_lines  += section(f'{label}·主板', df_mb)
        print(f"    {label}: 全市场 {df_all['stock_code'].nunique() if df_all is not None and not df_all.empty else 0} 只 / "
              f"主板 {df_mb['stock_code'].nunique() if df_mb is not None and not df_mb.empty else 0} 只")
    return all_lines + mb_lines


def patch_report(branch_file, new_lines):
    path = os.path.join(REPORTS_BR, branch_file)
    if not os.path.exists(path):
        print(f"  [跳过] 报告不存在: {branch_file}")
        return
    with open(path, encoding='utf-8') as f:
        text = f.read()

    # 头部：截到第一个「## 📋」之前
    idx = text.find('## 📋')
    head = (text[:idx] if idx != -1 else text).rstrip()

    # 页脚：最后一个 '\n---\n' 起
    fidx = text.rfind('\n---\n')
    footer = text[fidx:] if fidx != -1 else (
        '\n\n---\n*自动生成 by live/monitor.py · 周日晚 20:00 出报 · 人工复核后周一执行*\n')

    text = head + '\n\n' + '\n'.join(new_lines).rstrip() + '\n' + footer
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"  [OK] {branch_file} 已重写持仓章节")


def main():
    # ── main 分支 ────────────────────────────────────────────────────
    print("\n[main] 重生成持仓...")
    patch_report('main.md', build_holdings('main', [
        ('Regime 集成', 'predictions_ensemble.pkl'),
        ('等权集成',   'predictions_ensemble_equal.pkl'),
    ]))

    # ── fix 分支 ─────────────────────────────────────────────────────
    print("\n[fix] 重生成持仓...")
    fix_eq = make_equal_pkl('predictions_ensemble_fix.pkl', '_tmp_fix_equal.pkl')
    patch_report('fix_macro-neutral-fill.md', build_holdings('fix', [
        ('Regime 集成', 'predictions_ensemble_fix.pkl'),
        ('等权集成',   fix_eq),
    ]))

    # ── refactor 分支（仅等权）───────────────────────────────────────
    print("\n[refactor] 重生成持仓...")
    patch_report('refactor_equal-weight-ensemble.md', build_holdings('refactor', [
        ('等权集成', 'predictions_ensemble_refactor.pkl'),
    ]))

    # 清理临时
    tmp = os.path.join(OUTPUT_DIR, '_tmp_fix_equal.pkl')
    if os.path.exists(tmp):
        os.remove(tmp)
    print("\n完成。")


if __name__ == '__main__':
    main()
