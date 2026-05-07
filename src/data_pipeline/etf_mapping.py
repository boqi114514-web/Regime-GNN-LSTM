# -*- coding: utf-8 -*-
"""
data_pipeline/etf_mapping.py —— 申万一级行业 → 最优行业ETF 映射表构建

逻辑：
  1. etf_basic         拉取全部上市境内ETF，得到 ETF→跟踪指数 的映射
  2. index_weight      对每个唯一指数代码拉取最新月成份股（月度数据）
  3. 本地 ts_sw_members 拿申万一级行业成份股集合
  4. 计算"纯度"：ETF指数成份中属于该申万行业的股票占比
     purity = |ETF成份 ∩ SW行业成份| / |ETF成份|
  5. etf_share_size    拉取最新ETF规模，同指数下选规模最大的那只
  6. 输出 data/raw/etf_sw_mapping.csv：每个行业对应最优ETF及其纯度

用法：
    python -m data_pipeline.etf_mapping
    python -m data_pipeline.etf_mapping --top3   # 每行业输出前3只ETF
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import LOCAL_DATA_RAW, SW_MEMBERS_PATH
from data_pipeline.update import get_pro, _call_with_retry

OUTPUT_PATH      = os.path.join(LOCAL_DATA_RAW, 'etf_sw_mapping.csv')
CACHE_ETF_BASIC  = os.path.join(LOCAL_DATA_RAW, '_cache_etf_basic.pkl')
CACHE_IDX_WEIGHT = os.path.join(LOCAL_DATA_RAW, '_cache_idx_weight.pkl')
CACHE_ETF_SIZE   = os.path.join(LOCAL_DATA_RAW, '_cache_etf_size.pkl')

MIN_PURITY   = 0.30   # 低于此纯度的 ETF 不纳入候选
MIN_SIZE_WAN = 5000   # 最低规模门槛（万元），过滤迷你ETF


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def _recent_month_range(offset_months=0):
    """返回最近某个月的 (start_date, end_date)，格式 YYYYMMDD"""
    today = datetime.today()
    # 往前推 offset_months 个月
    month = today.month - offset_months
    year = today.year
    while month <= 0:
        month += 12
        year -= 1
    first = datetime(year, month, 1)
    if month == 12:
        last = datetime(year, 12, 31)
    else:
        last = datetime(year, month + 1, 1) - timedelta(days=1)
    return first.strftime('%Y%m%d'), last.strftime('%Y%m%d')


def _call(fn, **kwargs):
    return _call_with_retry(fn, retries=3, **kwargs)


# ──────────────────────────────────────────────
# Step 1: 拉取全部上市境内ETF
# ──────────────────────────────────────────────

def fetch_etf_basic(pro):
    import pickle
    if os.path.exists(CACHE_ETF_BASIC):
        print('[1/5] 读取 ETF 基本信息缓存 ...')
        with open(CACHE_ETF_BASIC, 'rb') as f:
            df = pickle.load(f)
    else:
        print('[1/5] 拉取 ETF 基本信息 ...')
        df = _call(pro.etf_basic, list_status='L',
                   fields='ts_code,extname,index_code,index_name,etf_type,mgt_fee,exchange')
        df = df[df['etf_type'] == '纯境内'].dropna(subset=['index_code']).copy()
        df = df[df['index_code'].str.strip() != ''].copy()
        with open(CACHE_ETF_BASIC, 'wb') as f:
            pickle.dump(df, f)
    print(f'    境内上市ETF: {len(df)} 只，唯一跟踪指数: {df["index_code"].nunique()} 个')
    return df


# ──────────────────────────────────────────────
# Step 2: 拉取每个指数的成份股
# ──────────────────────────────────────────────

def fetch_index_members(pro, index_codes):
    """对每个 index_code 拉取最新月成份股，返回 {index_code: set(ts_code)}"""
    import pickle
    if os.path.exists(CACHE_IDX_WEIGHT):
        print(f'[2/5] 读取指数成份股缓存 ...')
        with open(CACHE_IDX_WEIGHT, 'rb') as f:
            index_members = pickle.load(f)
        print(f'    缓存命中: {len(index_members)} 个指数')
        return index_members

    print(f'[2/5] 拉取 {len(index_codes)} 个指数成份股（index_weight）...')
    index_members = {}
    failed = []

    for i, idx_code in enumerate(index_codes):
        if i % 50 == 0:
            print(f'    进度: {i}/{len(index_codes)}')
        for offset in (0, 1, 2):
            start, end = _recent_month_range(offset)
            try:
                df = _call(pro.index_weight, index_code=idx_code,
                           start_date=start, end_date=end,
                           fields='index_code,con_code,weight')
                if df is not None and len(df) > 0:
                    index_members[idx_code] = set(df['con_code'].unique())
                    break
            except Exception:
                continue
        else:
            failed.append(idx_code)

    if failed:
        print(f'    无成份数据的指数（跳过）: {len(failed)} 个')
    print(f'    成功获取成份数据的指数: {len(index_members)} 个')
    with open(CACHE_IDX_WEIGHT, 'wb') as f:
        pickle.dump(index_members, f)
    return index_members


# ──────────────────────────────────────────────
# Step 3: 加载申万一级行业成份股（本地）
# ──────────────────────────────────────────────

def load_sw_members():
    print('[3/5] 加载申万一级行业成份股 ...')
    mem = pd.read_csv(SW_MEMBERS_PATH)
    # 只取当前在册的成份股（out_date 为空）
    current = mem[mem['out_date'].isna() | (mem['out_date'] == '')].copy()
    # 按 l1_code 分组
    sw = current.groupby('l1_code')['ts_code'].apply(set).to_dict()
    # 同时保留行业名称
    name_map = current.drop_duplicates('l1_code').set_index('l1_code')
    if 'l1_name' in name_map.columns:
        name_map = name_map['l1_name'].to_dict()
    else:
        name_map = {k: k for k in sw}
    print(f'    申万一级行业: {len(sw)} 个')
    return sw, name_map


# ──────────────────────────────────────────────
# Step 4: 计算纯度矩阵
# ──────────────────────────────────────────────

def compute_purity(etf_df, index_members, sw_members):
    """
    对每只ETF计算其对每个申万行业的纯度。
    返回 DataFrame: ts_code, index_code, l1_code, purity, coverage, n_overlap
    """
    print('[4/5] 计算纯度矩阵 ...')
    rows = []
    etf_has_members = etf_df[etf_df['index_code'].isin(index_members)]

    for _, etf_row in etf_has_members.iterrows():
        idx_code = etf_row['index_code']
        etf_stocks = index_members[idx_code]
        n_etf = len(etf_stocks)
        if n_etf == 0:
            continue

        for l1_code, sw_stocks in sw_members.items():
            overlap = etf_stocks & sw_stocks
            n_overlap = len(overlap)
            if n_overlap == 0:
                continue
            purity = n_overlap / n_etf
            coverage = n_overlap / len(sw_stocks) if sw_stocks else 0.0
            rows.append({
                'ts_code':    etf_row['ts_code'],
                'extname':    etf_row['extname'],
                'index_code': idx_code,
                'index_name': etf_row['index_name'],
                'mgt_fee':    etf_row['mgt_fee'],
                'l1_code':    l1_code,
                'purity':     round(purity, 4),
                'coverage':   round(coverage, 4),
                'n_overlap':  n_overlap,
                'n_etf':      n_etf,
            })

    result = pd.DataFrame(rows)
    print(f'    候选 (ETF, 行业) 对: {len(result)}')
    return result


# ──────────────────────────────────────────────
# Step 5: 拉取ETF规模（用于同指数内选最大）
# ──────────────────────────────────────────────

def fetch_etf_sizes(pro, ts_codes):
    """拉取最新交易日的ETF规模，返回 {ts_code: total_size(万元)}"""
    import pickle
    if os.path.exists(CACHE_ETF_SIZE):
        print('[5/5] 读取 ETF 规模缓存 ...')
        with open(CACHE_ETF_SIZE, 'rb') as f:
            size_map = pickle.load(f)
        found = sum(1 for c in ts_codes if c in size_map)
        print(f'    缓存命中: {found}/{len(ts_codes)}')
        return size_map

    print('[5/5] 拉取 ETF 规模 ...')
    size_map = {}
    for exch in ('SSE', 'SZSE'):
        for offset in (0, 1, 5):
            start, end = _recent_month_range(offset)
            try:
                df = _call(pro.etf_share_size, exchange=exch,
                           start_date=start, end_date=end,
                           fields='ts_code,trade_date,total_size')
                if df is not None and len(df) > 0:
                    latest = (df.sort_values('trade_date', ascending=False)
                                .drop_duplicates('ts_code')
                                .set_index('ts_code')['total_size']
                                .to_dict())
                    size_map.update(latest)
                    break
            except Exception:
                continue

    found = sum(1 for c in ts_codes if c in size_map)
    print(f'    获取到规模数据的ETF: {found}/{len(ts_codes)}')
    with open(CACHE_ETF_SIZE, 'wb') as f:
        pickle.dump(size_map, f)
    return size_map


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def build_mapping(top_n=1):
    pro = get_pro()

    # Step 1
    etf_df = fetch_etf_basic(pro)

    # Step 2
    unique_index_codes = list(etf_df['index_code'].unique())
    index_members = fetch_index_members(pro, unique_index_codes)

    # Step 3
    sw_members, sw_name_map = load_sw_members()

    # Step 4
    purity_df = compute_purity(etf_df, index_members, sw_members)
    if purity_df.empty:
        print('警告：纯度矩阵为空，请检查数据')
        return

    # Step 5
    size_map = fetch_etf_sizes(pro, etf_df['ts_code'].tolist())
    purity_df['total_size'] = purity_df['ts_code'].map(size_map).fillna(0.0)

    # 过滤低纯度和迷你ETF
    filtered = purity_df[
        (purity_df['purity'] >= MIN_PURITY) &
        (purity_df['total_size'] >= MIN_SIZE_WAN)
    ].copy()
    print(f'\n过滤后候选对（purity≥{MIN_PURITY}, size≥{MIN_SIZE_WAN}万）: {len(filtered)}')

    # 每个行业选 top_n，同指数内先按规模选最大ETF，再按纯度排名
    filtered = filtered.sort_values(['l1_code', 'purity', 'total_size'],
                                    ascending=[True, False, False])
    # 同一指数只保留规模最大那只
    filtered = filtered.drop_duplicates(subset=['l1_code', 'index_code'], keep='first')
    # 再按纯度取 top_n（数据已预排序，直接 head）
    mapping = (filtered.groupby('l1_code', group_keys=False)
                       .head(top_n)
                       .reset_index(drop=True))

    mapping['l1_name'] = mapping['l1_code'].map(sw_name_map)
    mapping['rank'] = mapping.groupby('l1_code').cumcount() + 1
    mapping['total_size_亿'] = (mapping['total_size'] / 10000).round(2)

    cols = ['l1_code', 'l1_name', 'rank', 'ts_code', 'extname',
            'index_code', 'index_name', 'purity', 'coverage',
            'n_overlap', 'n_etf', 'total_size_亿', 'mgt_fee']
    mapping = mapping[cols].sort_values(['l1_code', 'rank'])

    mapping.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f'\n映射表已保存: {OUTPUT_PATH}')
    print(f'覆盖行业数: {mapping["l1_code"].nunique()} / {len(sw_members)}')

    # 打印结果
    print('\n' + '=' * 90)
    print(f'{"行业":<10} {"排名":>4} {"ETF代码":<14} {"ETF名称":<18} {"纯度":>6} {"覆盖":>6} {"规模(亿)":>9}')
    print('-' * 90)
    for _, row in mapping.iterrows():
        ind_name = str(row.get('l1_name', row['l1_code']))[:8]
        print(f'{ind_name:<10} {int(row["rank"]):>4}  {row["ts_code"]:<14} '
              f'{str(row["extname"])[:16]:<18} {row["purity"]:>6.1%} '
              f'{row["coverage"]:>6.1%} {row["total_size_亿"]:>9.1f}亿')
    print('=' * 90)

    return mapping


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--top3', action='store_true', help='每行业输出前3只ETF')
    parser.add_argument('--refresh', action='store_true', help='忽略缓存，重新从API拉取')
    args = parser.parse_args()
    if args.refresh:
        for f in (CACHE_ETF_BASIC, CACHE_IDX_WEIGHT, CACHE_ETF_SIZE):
            if os.path.exists(f):
                os.remove(f)
        print('已清除缓存，将重新拉取数据')
    build_mapping(top_n=3 if args.top3 else 1)
