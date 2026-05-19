# -*- coding: utf-8 -*-
"""方向 2 Gate 1：二级行业月度收益拉取 + GLASSO 稳定性诊断

Part 1：拉 124 个在用二级月度行情 -> data/raw/ts_sw_l2_monthly.csv（带缓存）
Part 2：GLASSO 退化诊断
  对每个月的滚动窗口估图，统计：
    - GraphicalLassoCV 成功 / 退化到 corrcoef 兜底 / 失败 的占比
    - 协方差条件数（爆掉=欠定）
  对比：一级(31) vs 二级(124) × 窗口 {12,24,36} × GLASSO vs Ledoit-Wolf
"""
import sys, os, io, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV, LedoitWolf

from data_pipeline.tushare_config import get_pro
from config import LOCAL_DATA_RAW, load_industry_monthly

L2_CSV = os.path.join(LOCAL_DATA_RAW, "ts_sw_l2_monthly.csv")
L2_CACHE = os.path.join(LOCAL_DATA_RAW, "_l2_cache")


# ==================== Part 1：拉二级月度行情 ====================

def _safe_call(fn, **kw):
    """限流感知的重试：'速度过快' 时长退避，逐 code 调用"""
    for i in range(6):
        try:
            df = fn(**kw)
            time.sleep(0.8)                       # 常规节流
            return df
        except Exception as e:
            msg = str(e)
            if "速度过快" in msg or "rate" in msg.lower() or "频" in msg:
                wait = 20 * (i + 1)
                print(f"    限流，等 {wait}s ...")
                time.sleep(wait)
            else:
                time.sleep(3)
    return None


def pull_l2_monthly(force=False):
    if os.path.exists(L2_CSV) and not force:
        df = pd.read_csv(L2_CSV)
        print(f"[l2] 命中缓存 {L2_CSV}  rows={len(df)}")
        return df

    os.makedirs(L2_CACHE, exist_ok=True)
    pro = get_pro()
    l2 = _safe_call(pro.index_classify, level="L2", src="SW2021")
    codes = l2[l2["is_pub"] == "1"]["index_code"].tolist()
    print(f"[l2] 在用二级 {len(codes)} 个，逐 code 拉取（带缓存断点续传）...")

    rows, pulled, skipped = [], 0, 0
    for i, code in enumerate(codes, 1):
        cache_fp = os.path.join(L2_CACHE, f"{code}.pkl")
        if os.path.exists(cache_fp):
            rows.append(pd.read_pickle(cache_fp))
            skipped += 1
            continue
        df = _safe_call(pro.index_monthly, ts_code=code,
                        start_date="20190101", end_date="20260430")
        if df is not None and len(df):
            df.to_pickle(cache_fp)
            rows.append(df)
            pulled += 1
        else:
            print(f"  [{i}] {code} 拉取失败/空")
        if i % 30 == 0 or i == len(codes):
            print(f"  [{i}/{len(codes)}] 新拉 {pulled} 跳过 {skipped}")

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(L2_CSV, index=False)
    print(f"[l2] 写出 {L2_CSV}  rows={len(out)}  覆盖 {out['ts_code'].nunique()} 个二级")
    return out


# ==================== Part 2：GLASSO 诊断 ====================

def _ret_pivot(df, code_col, date_col, ret_col):
    p = df.pivot_table(index=date_col, columns=code_col, values=ret_col,
                       aggfunc="first").sort_index()
    return p


def diag_window(window, method):
    """对单个 (T,N) 收益窗口估协方差/精度，返回诊断 dict"""
    rc = np.nan_to_num(window, nan=0.0)
    T, N = rc.shape
    if T < 5 or np.std(rc) < 1e-8:
        return dict(status="too_few")
    if method == "glasso":
        try:
            m = GraphicalLassoCV(cv=3, max_iter=200)
            m.fit(rc)
            cov = m.covariance_
            cond = np.linalg.cond(cov)
            prec = np.abs(m.precision_)
            np.fill_diagonal(prec, 0)
            sparsity = (prec < 1e-6).mean()
            return dict(status="glasso_ok", cond=cond, sparsity=sparsity)
        except Exception as e:
            # 现有代码会在此退化到 corrcoef 兜底
            return dict(status="glasso_fail", err=type(e).__name__)
    else:  # ledoit-wolf
        try:
            lw = LedoitWolf().fit(rc)
            cond = np.linalg.cond(lw.covariance_)
            return dict(status="lw_ok", cond=cond, shrink=lw.shrinkage_)
        except Exception as e:
            return dict(status="lw_fail", err=type(e).__name__)


def run_diagnostic(pivot, label, windows=(12, 24, 36)):
    dates = list(pivot.index)
    arr = pivot.values
    N = arr.shape[1]
    print(f"\n{'='*64}\n[{label}]  {N} 个行业 × {len(dates)} 个月\n{'='*64}")
    for w in windows:
        for method in ("glasso", "lw"):
            stats = []
            for end in range(w, len(dates)):
                win = arr[end - w:end]
                stats.append(diag_window(win, method))
            ok = [s for s in stats if s["status"] in ("glasso_ok", "lw_ok")]
            fail = [s for s in stats if "fail" in s["status"]]
            conds = [s["cond"] for s in ok if np.isfinite(s.get("cond", np.nan))]
            tag = "GLASSO  " if method == "glasso" else "LedoitW "
            line = (f"  窗口{w:>2}月 {tag}  成功{len(ok):>3}/{len(stats):<3} "
                    f"失败{len(fail):>2}")
            if conds:
                line += (f"  条件数 中位={np.median(conds):.1e} "
                         f"最大={np.max(conds):.1e}")
            if method == "glasso" and ok:
                sp = np.mean([s["sparsity"] for s in ok])
                line += f"  稀疏度={sp:.0%}"
            print(line)


def main():
    # 二级
    l2 = pull_l2_monthly()
    l2 = l2.copy()
    l2["ret"] = l2["pct_chg"] / 100.0
    l2["date"] = pd.to_datetime(l2["trade_date"].astype(str), format="%Y%m%d")
    piv_l2 = _ret_pivot(l2, "ts_code", "date", "ret")

    # 一级（现有数据基准）
    mkt = load_industry_monthly()
    piv_l1 = _ret_pivot(mkt, "ts_code", "date", "ret")

    run_diagnostic(piv_l1, "一级基准 L1")
    run_diagnostic(piv_l2, "二级 L2")

    print(f"\n{'='*64}\n解读：")
    print("  - GLASSO 成功率低 / 条件数爆（>1e10）= 协方差严重欠定，图不可靠")
    print("  - LedoitWolf 条件数应显著更小（shrinkage 保证正定良态）")
    print("  - 若二级 12 月窗 GLASSO 退化严重 -> 改用 LedoitWolf 或拉长窗口")
    print("="*64)


if __name__ == "__main__":
    main()
