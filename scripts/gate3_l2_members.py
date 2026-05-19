# -*- coding: utf-8 -*-
"""方向 2 Gate 3 Step 1：申万二级行业成分股拉取

落地 data/raw/ts_sw_l2_members.csv
列：l1_code, l1_name, l2_code, l2_name, ts_code, name, in_date, out_date, is_new

⚠️ 接口选择（2026-05-18 修正）：
    旧版本用 index_member_all(l2_code=...)，实测只返回**当前成分**
    （is_new 全为 Y、out_date 全空），历史聚合会有幸存者偏差。
    改用 index_member(index_code=<l2_code>)，返回**逐期成分历史**，
    out_date 完整（含历史退出记录），支持点对点成分快照。
逐 code 缓存（_l2_members_cache_v2），限流长退避，支持断点续传。
"""
import sys, os, io, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
from data_pipeline.tushare_config import get_pro
from config import LOCAL_DATA_RAW

OUT_CSV = os.path.join(LOCAL_DATA_RAW, "ts_sw_l2_members.csv")
CACHE = os.path.join(LOCAL_DATA_RAW, "_l2_members_cache_v2")


def _safe_call(fn, **kw):
    """限流感知重试"""
    for i in range(6):
        try:
            df = fn(**kw)
            time.sleep(0.8)
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


def main():
    os.makedirs(CACHE, exist_ok=True)
    pro = get_pro()

    # 一级 / 二级分类表 —— 用于补 l1_code / 名称
    l1 = _safe_call(pro.index_classify, level="L1", src="SW2021")
    l2 = _safe_call(pro.index_classify, level="L2", src="SW2021")
    # parent_code 是 SW industry_code（如 110000），映射到 L1 的 index_code
    l1_by_indcode = l1.set_index("industry_code")
    l1_code_map = l1_by_indcode["index_code"].to_dict()
    l1_name_map = l1_by_indcode["industry_name"].to_dict()

    pub = l2[l2["is_pub"] == "1"].copy()
    codes = pub["index_code"].tolist()
    l2_name_map = dict(zip(pub["index_code"], pub["industry_name"]))
    l2_parent_map = dict(zip(pub["index_code"], pub["parent_code"]))
    print(f"[l2_members] 在用二级 {len(codes)} 个，逐个拉 index_member（逐期历史）...")

    rows, pulled, skipped, empty = [], 0, 0, []
    for i, code in enumerate(codes, 1):
        fp = os.path.join(CACHE, f"{code}.pkl")
        if os.path.exists(fp):
            rows.append(pd.read_pickle(fp))
            skipped += 1
            continue
        df = _safe_call(pro.index_member, index_code=code)
        if df is not None and len(df):
            df = df.drop_duplicates()
            df.to_pickle(fp)
            rows.append(df)
            pulled += 1
        else:
            empty.append(code)
        if i % 30 == 0 or i == len(codes):
            print(f"  [{i}/{len(codes)}] 新拉 {pulled} 跳过 {skipped} 空 {len(empty)}")

    out = pd.concat(rows, ignore_index=True)
    # index_member 返回：index_code, con_code, in_date, out_date, is_new
    out = out.rename(columns={"index_code": "l2_code", "con_code": "ts_code"})
    # index_member 的 out_date 对当前成分返回空串而非 None —— 统一成缺失值
    out["out_date"] = out["out_date"].replace("", pd.NA)
    out = out.drop_duplicates(subset=["l2_code", "ts_code", "in_date", "out_date"])

    # 补行业层级与名称
    out["l2_name"] = out["l2_code"].map(l2_name_map)
    out["l1_code"] = out["l2_code"].map(l2_parent_map).map(l1_code_map)
    out["l1_name"] = out["l2_code"].map(l2_parent_map).map(l1_name_map)

    # 用 stock_basic 补股票中文名
    print("  拉取 stock_basic 补充股票名称...")
    basic_frames = []
    for status in ("L", "D", "P"):
        b = _safe_call(pro.stock_basic, exchange="", list_status=status,
                       fields="ts_code,name")
        if b is not None and len(b):
            basic_frames.append(b)
    name_map = {}
    if basic_frames:
        name_map = dict(zip(pd.concat(basic_frames)["ts_code"],
                            pd.concat(basic_frames)["name"]))
    out["name"] = out["ts_code"].map(name_map)

    keep = ["l1_code", "l1_name", "l2_code", "l2_name",
            "ts_code", "name", "in_date", "out_date", "is_new"]
    out = out[keep].sort_values(["l2_code", "ts_code", "in_date"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n[l2_members] 写出 {OUT_CSV}")
    print(f"  总行数={len(out)}  二级数={out['l2_code'].nunique()}  "
          f"个股数={out['ts_code'].nunique()}")
    print(f"  out_date 非空={out['out_date'].notna().sum()}（历史退出记录），"
          f"is_new 分布={out['is_new'].value_counts().to_dict()}")
    # 每个二级当前成分股数（out_date 为空 = 当前在册）
    cur = out[out["out_date"].isna()]
    cnt = cur.groupby("l2_code")["ts_code"].nunique()
    print(f"  每二级当前成分股数：中位={cnt.median():.0f}  "
          f"最小={cnt.min()}  最大={cnt.max()}  "
          f"少于10只的二级={int((cnt < 10).sum())} 个")
    if empty:
        print(f"  空二级（{len(empty)}）：{empty}")


if __name__ == "__main__":
    main()
