# -*- coding: utf-8 -*-
"""ETF→申万行业映射 v2（纯统计：全样本 OLS R²）

方法：
  1. etf_basic         拉取全部上市境内ETF列表
  2. fund_daily        按月末日期批量拉取收盘价（每个月末日期一次调用，覆盖所有ETF）
  3. 月末收盘价连续相除 → 月度收益率
  4. 对每只ETF，与31个申万一级行业跑全样本OLS
     R² = 皮尔逊相关系数²，beta = cov(ETF,SW) / var(SW)
  5. 每个行业按R²降序输出全部ETF（不设阈值、不预筛规模）

产物：data/raw/etf_sw_mapping_v2.csv
      每行是一个(申万行业, ETF)对，含 r2, beta, n_obs, etf_name, aum_亿

缓存：data/raw/_cache_v2_etf_prices.pkl（月末价格矩阵，重跑免重拉）

用法：
    python -m data_pipeline.etf_mapping_v2
    python -m data_pipeline.etf_mapping_v2 --refresh   # 清除价格缓存，重新拉取
    python -m data_pipeline.etf_mapping_v2 --top5      # 每行业显示前5
    python -m data_pipeline.etf_mapping_v2 --summary   # 只打印每行业最优ETF
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import LOCAL_DATA_RAW
from data_pipeline.tushare_config import get_pro, _call_with_retry

OUTPUT_PATH      = os.path.join(LOCAL_DATA_RAW, 'etf_sw_mapping_v2.csv')
CACHE_PRICES     = os.path.join(LOCAL_DATA_RAW, '_cache_v2_etf_prices.pkl')
SW_MONTHLY_PATH  = os.path.join(LOCAL_DATA_RAW, 'ts_sw_industry_monthly.csv')

MIN_OBS = 12   # 不足12个月历史的ETF跳过


# ─────────────────────────────────────────────
# Step 1: ETF 基本信息
# ─────────────────────────────────────────────

def fetch_etf_basic(pro):
    print('[1/4] 拉取境内上市ETF列表 ...')
    df = _call_with_retry(
        pro.etf_basic, list_status='L',
        fields='ts_code,extname,setup_date,list_date,etf_type',
    )
    df = df[df['etf_type'] == '纯境内'].copy()
    print(f'    境内上市ETF: {len(df)} 只')
    return df


# ─────────────────────────────────────────────
# Step 2: 月末收盘价矩阵
# ─────────────────────────────────────────────

def _get_month_end_dates():
    """从申万行业月度CSV中提取月末交易日列表（已是实际交易日）"""
    df = pd.read_csv(SW_MONTHLY_PATH)
    dates = pd.to_datetime(df['date']).dt.strftime('%Y%m%d').unique()
    return sorted(dates)


def fetch_etf_prices(pro, refresh=False):
    """
    对每个月末日期调用 fund_daily(trade_date=D)，
    一次返回所有ETF当天收盘价，最终汇成宽表。
    """
    if not refresh and os.path.exists(CACHE_PRICES):
        print('[2/4] 读取ETF月末价格缓存 ...')
        prices = pd.read_pickle(CACHE_PRICES)
        print(f'    {prices.shape[0]} 个月 × {prices.shape[1]} 只ETF')
        return prices

    dates = _get_month_end_dates()
    print(f'[2/4] 拉取ETF月末收盘价，共 {len(dates)} 个月末日期 ...')

    frames = []
    for i, d in enumerate(dates):
        df = _call_with_retry(
            pro.fund_daily, trade_date=d,
            fields='ts_code,trade_date,close',
        )
        if df is not None and len(df) > 0:
            frames.append(df)
        if (i + 1) % 20 == 0 or i == len(dates) - 1:
            print(f'    进度: {i+1}/{len(dates)}')

    if not frames:
        raise RuntimeError('未拉取到任何ETF日线数据，请检查接口和日期范围')

    all_data = pd.concat(frames, ignore_index=True)
    all_data['trade_date'] = pd.to_datetime(all_data['trade_date'], format='%Y%m%d')
    prices = all_data.pivot_table(
        index='trade_date', columns='ts_code', values='close', aggfunc='last'
    )
    prices = prices.sort_index()
    prices.to_pickle(CACHE_PRICES)
    print(f'    价格矩阵: {prices.shape[0]} 月 × {prices.shape[1]} ETF，已缓存')
    return prices


# ─────────────────────────────────────────────
# Step 3: 申万行业月收益 & ETF月收益
# ─────────────────────────────────────────────

def load_sw_names(pro):
    """从 index_basic 拉取申万一级行业名称，返回 {ts_code: name}"""
    df = _call_with_retry(pro.index_basic, market='SW', fields='ts_code,name')
    sw_codes = set(pd.read_csv(SW_MONTHLY_PATH)['ts_code'].unique())
    return df[df['ts_code'].isin(sw_codes)].set_index('ts_code')['name'].to_dict()


def load_sw_returns():
    """申万一级行业月度收益率（来自 ts_sw_industry_monthly.csv 的 pct_chg 列）"""
    df = pd.read_csv(SW_MONTHLY_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ts_code', 'date'])
    pivot = df.pivot_table(index='date', columns='ts_code', values='pct_chg', aggfunc='last')
    pivot = pivot / 100.0   # % → 小数
    # 排除综合指数和非一级行业（保留 801xxx.SI 格式）
    sw_cols = [c for c in pivot.columns if str(c).startswith('8') and str(c).endswith('.SI')]
    return pivot[sw_cols].sort_index()


def compute_etf_returns(prices):
    """月末收盘价连续相除得到月度收益率"""
    returns = prices.pct_change(fill_method=None)
    returns = returns.iloc[1:]   # 去掉第一行（全 NaN）
    return returns


# ─────────────────────────────────────────────
# Step 4: 全样本 OLS  R² = corr²
# ─────────────────────────────────────────────

def compute_r2_matrix(etf_returns, sw_returns):
    """
    对每个(行业, ETF)对计算全样本R²和beta。
    R² = corr(ETF_ret, SW_ret)²
    beta = cov / var(SW)
    返回 list of dicts。
    """
    print('[3/4] 计算 R² 矩阵 ...')

    # 按日期取交集（申万月末日和ETF月末日可能有1天偏差，用月份对齐）
    sw = sw_returns.copy()
    sw.index = sw.index.to_period('M')
    etf = etf_returns.copy()
    etf.index = pd.to_datetime(etf.index).to_period('M')

    common = sw.index.intersection(etf.index)
    sw = sw.loc[common]
    etf = etf.loc[common]

    sw_arr  = sw.values    # T × 31
    sw_cols = list(sw.columns)
    etf_arr = etf.values   # T × n_etf
    etf_cols = list(etf.columns)
    T = len(common)

    results = []
    n_etf = len(etf_cols)

    for j, sw_code in enumerate(sw_cols):
        s = sw_arr[:, j].astype(float)
        s_notna = ~np.isnan(s)
        if s_notna.sum() < MIN_OBS:
            continue
        s_var = np.nanvar(s)

        for i, etf_code in enumerate(etf_cols):
            e = etf_arr[:, i].astype(float)
            mask = s_notna & ~np.isnan(e)
            n = mask.sum()
            if n < MIN_OBS:
                continue
            e_v, s_v = e[mask], s[mask]
            corr = np.corrcoef(e_v, s_v)[0, 1]
            if np.isnan(corr):
                continue
            r2 = corr ** 2
            beta = np.cov(e_v, s_v)[0, 1] / np.var(s_v) if np.var(s_v) > 1e-12 else np.nan
            results.append({
                'sw_code':  sw_code,
                'etf_code': etf_code,
                'r2':       round(float(r2), 4),
                'beta':     round(float(beta), 4) if not np.isnan(beta) else np.nan,
                'n_obs':    int(n),
            })

        if (j + 1) % 5 == 0 or j == len(sw_cols) - 1:
            print(f'    行业进度: {j+1}/{len(sw_cols)}  ({sw_code})')

    print(f'    共 {len(results)} 个有效(行业,ETF)对')
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 辅助：拉最新AUM
# ─────────────────────────────────────────────

def fetch_latest_aum(pro):
    """拉最近7天内最新有效交易日所有ETF的规模，返回 {ts_code: aum_亿}"""
    print('    拉取最新ETF规模 ...')
    start = (pd.Timestamp.today() - pd.Timedelta(days=7)).strftime('%Y%m%d')
    end   = pd.Timestamp.today().strftime('%Y%m%d')
    frames = []
    for exch in ('SSE', 'SZSE'):
        df = _call_with_retry(
            pro.etf_share_size, exchange=exch,
            start_date=start, end_date=end,
            fields='ts_code,trade_date,total_size',
        )
        if df is not None and len(df) > 0:
            frames.append(df)

    if not frames:
        return {}
    all_aum = pd.concat(frames, ignore_index=True)
    all_aum = all_aum.dropna(subset=['total_size'])   # 剔除 NaN（数据未入库的交易日）
    latest = (all_aum.sort_values('trade_date', ascending=False)
                     .drop_duplicates('ts_code')
                     .set_index('ts_code')['total_size'])
    return (latest / 10000).round(2).to_dict()   # 万元 → 亿元


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def build_mapping(top_n=1, refresh=False):
    pro = get_pro()

    etf_basic = fetch_etf_basic(pro)
    etf_name_map = etf_basic.set_index('ts_code')['extname'].to_dict()

    prices  = fetch_etf_prices(pro, refresh=refresh)
    sw_ret  = load_sw_returns()
    etf_ret = compute_etf_returns(prices)

    # 只保留在 etf_basic 中的境内ETF列
    domestic = set(etf_basic['ts_code'])
    etf_ret  = etf_ret[[c for c in etf_ret.columns if c in domestic]]

    r2_df = compute_r2_matrix(etf_ret, sw_ret)

    # 附加ETF名称和AUM
    print('[4/4] 附加ETF名称和规模 ...')
    aum_map = fetch_latest_aum(pro)
    r2_df['etf_name'] = r2_df['etf_code'].map(etf_name_map)
    r2_df['aum_亿']   = r2_df['etf_code'].map(aum_map)

    # 申万行业名称（从 index_basic 拉取）
    sw_name_map = load_sw_names(pro)
    r2_df['sw_name'] = r2_df['sw_code'].map(sw_name_map)

    # 按行业、R²降序排列
    r2_df = r2_df.sort_values(['sw_code', 'r2'], ascending=[True, False]).reset_index(drop=True)
    r2_df['rank'] = r2_df.groupby('sw_code').cumcount() + 1

    cols = ['sw_code', 'sw_name', 'rank', 'etf_code', 'etf_name',
            'r2', 'beta', 'n_obs', 'aum_亿']
    r2_df = r2_df[cols]

    r2_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f'\n映射表已保存: {OUTPUT_PATH}')
    print(f'覆盖行业数: {r2_df["sw_code"].nunique()}  总行数: {len(r2_df)}')

    return r2_df


def print_summary(r2_df, top_n=1):
    """打印每个行业 Top-N 的 ETF"""
    top = r2_df[r2_df['rank'] <= top_n].copy()
    print('\n' + '=' * 100)
    header = f'{"行业":<10} {"排名":>4}  {"ETF代码":<12} {"ETF名称":<20} {"R²":>6} {"beta":>6} {"n_obs":>6} {"规模(亿)":>9}'
    print(header)
    print('-' * 100)
    for _, row in top.iterrows():
        ind = str(row.get('sw_name') or row['sw_code'])[:8]
        name = str(row['etf_name'] or '')[:18]
        aum  = f'{row["aum_亿"]:.1f}' if pd.notna(row.get('aum_亿')) else '  N/A'
        print(f'{ind:<10} {int(row["rank"]):>4}  {row["etf_code"]:<12} {name:<20} '
              f'{row["r2"]:>6.3f} {row["beta"]:>6.3f} {int(row["n_obs"]):>6} {aum:>9}')
    print('=' * 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true', help='清除价格缓存，重新拉取')
    parser.add_argument('--top5',    action='store_true', help='每行业显示前5只ETF')
    parser.add_argument('--summary', action='store_true', help='只打印每行业最优ETF（top1）')
    args = parser.parse_args()

    if args.refresh and os.path.exists(CACHE_PRICES):
        os.remove(CACHE_PRICES)
        print('已清除价格缓存')

    top_n = 5 if args.top5 else 1
    df = build_mapping(top_n=top_n, refresh=args.refresh)
    print_summary(df, top_n=top_n if not args.summary else 1)
