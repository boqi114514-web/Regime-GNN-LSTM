# -*- coding: utf-8 -*-
"""
给 fix / refactor 分支的已归档周报追加「主板持仓」章节。

背景：fix/refactor 分支用旧 monitor 生成报告，没有主板持仓章节，导致 dashboard
持仓页「主板」tab 为空。本脚本用 main 的 s4（force_main_board=True）针对各分支的
预测重新选股，生成正确标注的主板章节，插入到对应报告的页脚之前。

不切 git、不重训，只读各分支预测 pkl + 当前 main 的 s4 选股逻辑。

分支 → 模式 → 预测来源：
  fix      : Regime（predictions_ensemble_fix.pkl）+ 等权（由同 pkl 的 rank 推导）
  refactor : 等权（predictions_ensemble_refactor.pkl，本身即等权）
"""
import sys, os
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

ROOT = r'D:\desktop\有意思的事情\量化\项目\Regime-GNN-LSTM'
sys.path.insert(0, os.path.join(ROOT, 'src'))

import s4_beta_selection as s4
from live.monitor import _stock_section_lines
from config import OUTPUT_DIR

REPORTS_BR = os.path.join(ROOT, 'reports', 'branches')

# 等权预测：pred_ensemble = -(0.5*rank_gnn + 0.5*rank_lstm)（与 s3 simple_rank_ensemble 一致）
def make_equal_pkl(src_pkl, dst_name):
    df = pd.read_pickle(os.path.join(OUTPUT_DIR, src_pkl)).copy()
    df['pred_ensemble'] = -(0.5 * df['rank_gnn'] + 0.5 * df['rank_lstm'])
    dst = os.path.join(OUTPUT_DIR, dst_name)
    df.to_pickle(dst)
    return dst_name


def select_mainboard(pred_pkl, ckpt_suffix):
    """用 main 的 s4 强制主板选股，返回最新月持仓 DataFrame"""
    return s4.run_live(
        pred_pkl=pred_pkl,
        ckpt_suffix=ckpt_suffix,
        force_refresh_latest=True,
        force_main_board=True,
    )


def build_section(sections):
    """sections: [(label, df), ...]  → 主板持仓 markdown 行"""
    lines = ['## 📋 主板持仓（保证主板，每行业满额）', '']
    for label, df in sections:
        if df is None or df.empty:
            continue
        lines += _stock_section_lines(df, label)
    return lines


def patch_report(branch_file, section_lines):
    path = os.path.join(REPORTS_BR, branch_file)
    if not os.path.exists(path):
        print(f"  [跳过] 报告不存在: {branch_file}")
        return
    with open(path, encoding='utf-8') as f:
        text = f.read()

    if '## 📋 主板持仓' in text:
        # 已有则替换（删除旧的主板章节到文末页脚之间）
        head = text.split('## 📋 主板持仓')[0].rstrip()
        footer_idx = text.rfind('\n---\n')
        footer = text[footer_idx:] if footer_idx != -1 else '\n'
        text = head + '\n\n' + '\n'.join(section_lines) + footer
    else:
        # 插到页脚 '---' 之前
        footer_idx = text.rfind('\n---\n')
        if footer_idx != -1:
            text = (text[:footer_idx].rstrip() + '\n\n'
                    + '\n'.join(section_lines) + '\n'
                    + text[footer_idx:])
        else:
            text = text.rstrip() + '\n\n' + '\n'.join(section_lines) + '\n'

    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    n = sum(1 for l in section_lines if l.startswith('| ') and '---' not in l and '行业 |' not in l)
    print(f"  [OK] {branch_file} 已追加主板章节（约 {n} 行持仓）")


def main():
    print("加载 s4 数据（一次性缓存）...")
    # 预热：第一次 run_live 会加载全部缓存

    # ── fix 分支：Regime + 等权 ──────────────────────────────────────
    print("\n[fix] 生成主板持仓...")
    fix_regime = select_mainboard('predictions_ensemble_fix.pkl', '_fix_re_mb')
    fix_eq_pkl = make_equal_pkl('predictions_ensemble_fix.pkl', '_tmp_fix_equal.pkl')
    fix_equal  = select_mainboard(fix_eq_pkl, '_fix_eq_mb')
    patch_report('fix_macro-neutral-fill.md', build_section([
        ('Regime 集成·主板', fix_regime),
        ('等权集成·主板',   fix_equal),
    ]))

    # ── refactor 分支：等权 ──────────────────────────────────────────
    print("\n[refactor] 生成主板持仓...")
    ref_equal = select_mainboard('predictions_ensemble_refactor.pkl', '_ref_eq_mb')
    patch_report('refactor_equal-weight-ensemble.md', build_section([
        ('等权集成·主板', ref_equal),
    ]))

    # 清理临时 pkl
    tmp = os.path.join(OUTPUT_DIR, '_tmp_fix_equal.pkl')
    if os.path.exists(tmp):
        os.remove(tmp)

    print("\n完成。")


if __name__ == '__main__':
    main()
