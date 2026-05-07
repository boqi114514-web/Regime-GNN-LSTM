# -*- coding: utf-8 -*-
"""data_pipeline/download.py —— 基础数据下载与迁移

覆盖的数据（保存到 data/raw/）：
  - stock_daily.pkl          全市场个股日K线（dict: df_stock, code_to_name）
  - ts_sw_members.csv        申万一级行业成分股映射
  - raw_income.pkl           利润表（tushare income_vip）
  - raw_balancesheet.pkl     资产负债表
  - raw_cashflow.pkl         现金流量表
  - raw_report_rc.pkl        券商一致预期

用法：
    # 首次迁移：从 ARIMAX 项目复制已有数据（最快，推荐首次使用）
    python -m data_pipeline.download --migrate

    # 增量更新个股日线（在已有 stock_daily.pkl 基础上追加最新数据）
    python -m data_pipeline.download --update-daily

    # 增量更新财务报表（追加最新季度）
    python -m data_pipeline.download --update-financial

    # 全量重新下载（极慢，仅在无法迁移时使用）
    python -m data_pipeline.download --full
"""
import argparse
import os
import pickle
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import LOCAL_DATA_RAW, SW_EXCLUDE

# ARIMAX 项目路径（仅 --migrate 使用）
_ARIMAX_DIR = r"D:\desktop\有意思的事情\量化\项目\ARIMAX_LSTM行业轮动\数据"
_ARIMAX_RAW = os.path.join(_ARIMAX_DIR, "原始数据")
_ARIMAX_EXISTING = os.path.join(_ARIMAX_DIR, "已有数据")
_ARIMAX_INDUSTRY = os.path.join(_ARIMAX_DIR, "行业数据")

# 输出路径
STOCK_DAILY_PKL = os.path.join(LOCAL_DATA_RAW, 'stock_daily.pkl')
SW_MEMBERS_CSV = os.path.join(LOCAL_DATA_RAW, 'ts_sw_members.csv')

# 财务报表文件列表
FINANCIAL_FILES = ['raw_income.pkl', 'raw_balancesheet.pkl',
                   'raw_cashflow.pkl', 'raw_report_rc.pkl']

# tushare 配置
API_SLEEP = 0.3
START_YEAR = 2012
END_YEAR = 2026


def _get_pro():
    from data_pipeline.update import get_pro
    return get_pro()


def _call(fn, *args, retries=3, **kwargs):
    from data_pipeline.update import _call_with_retry
    return _call_with_retry(fn, *args, retries=retries, **kwargs)


# ============================================================
#  --migrate: 从 ARIMAX 项目复制
# ============================================================

def migrate_from_arimax(dry_run=False):
    """一次性从 ARIMAX 项目复制所有基础数据"""
    print('=' * 60)
    print('  从 ARIMAX 项目迁移数据')
    print('=' * 60)

    copies = [
        # (source, dest, description)
        (os.path.join(_ARIMAX_INDUSTRY, 'full_market_data_v18.pkl'),
         STOCK_DAILY_PKL, '个股日K线'),
        (os.path.join(_ARIMAX_EXISTING, 'ts_sw_members.csv'),
         SW_MEMBERS_CSV, '行业成分股'),
    ]
    # 财务报表
    for fname in FINANCIAL_FILES:
        copies.append((
            os.path.join(_ARIMAX_RAW, fname),
            os.path.join(LOCAL_DATA_RAW, fname),
            fname,
        ))

    os.makedirs(LOCAL_DATA_RAW, exist_ok=True)
    results = {}
    for src, dst, desc in copies:
        if not os.path.exists(src):
            print(f'  [跳过] {desc}: 源文件不存在 {src}')
            results[desc] = 'missing'
            continue
        size_mb = os.path.getsize(src) / 1024 / 1024
        if os.path.exists(dst):
            print(f'  [已存在] {desc}: {dst} ({size_mb:.1f}MB)')
            results[desc] = 'exists'
            continue
        print(f'  复制 {desc}: {size_mb:.1f}MB ...', end=' ')
        if not dry_run:
            shutil.copy2(src, dst)
            print('完成')
        else:
            print('(dry-run)')
        results[desc] = 'copied'

    print('\n迁移结果:')
    for k, v in results.items():
        print(f'  {k}: {v}')
    return results


# ============================================================
#  --update-daily: 增量更新个股日线
# ============================================================

def _get_trade_dates(pro, start_s, end_s):
    """获取 [start_s, end_s] 之间的交易日列表"""
    cal = _call(pro.trade_cal, exchange='SSE',
                start_date=start_s, end_date=end_s,
                fields='cal_date,is_open')
    if cal is None or len(cal) == 0:
        return []
    return sorted(cal[cal['is_open'] == 1]['cal_date'].astype(str).tolist())


def update_stock_daily(dry_run=False):
    """在已有 stock_daily.pkl 基础上增量追加最新日线数据。

    策略：按交易日拉取（pro.daily(trade_date=xxx)），一次拿到全市场所有股票。
    ~60 个交易日 ≈ 60 次 API 调用，比逐股票拉快 30 倍以上。
    """
    print('=' * 60)
    print('  增量更新个股日K线')
    print('=' * 60)

    if not os.path.exists(STOCK_DAILY_PKL):
        print(f'[错误] {STOCK_DAILY_PKL} 不存在，请先 --migrate')
        return {'error': 'no_base_file'}

    print('[1/4] 加载现有数据...')
    with open(STOCK_DAILY_PKL, 'rb') as f:
        data = pickle.load(f)
    df = data['df_stock']
    df['date'] = pd.to_datetime(df['date'])
    latest = df['date'].max()
    print(f'  现有: {len(df):,} 行, {df["code"].nunique()} 只股票, 最新 {latest.date()}')

    start_dt = latest + pd.Timedelta(days=1)
    end_dt = pd.Timestamp.today()
    if start_dt >= end_dt:
        print('  已是最新，无需更新')
        return {'new_rows': 0}

    start_s = start_dt.strftime('%Y%m%d')
    end_s = end_dt.strftime('%Y%m%d')

    pro = _get_pro()

    print(f'\n[2/4] 获取交易日历 {start_s} → {end_s} ...')
    trade_dates = _get_trade_dates(pro, start_s, end_s)
    if not trade_dates:
        print('  该窗口内无交易日')
        return {'new_rows': 0}
    print(f'  共 {len(trade_dates)} 个交易日: {trade_dates[0]} → {trade_dates[-1]}')

    print(f'\n[3/4] 按交易日拉取全市场日线（{len(trade_dates)} 天）...')
    new_rows = []
    total_records = 0
    failed_dates = []

    for i, td in enumerate(trade_dates):
        print(f'  [{i+1:3d}/{len(trade_dates)}] {td}', end=' ', flush=True)
        try:
            bar = _call(pro.daily, trade_date=td)
            if bar is None or len(bar) == 0:
                print('(空)')
                continue

            # 转换格式：匹配现有 df 的列结构
            bar['code'] = bar['ts_code'].str.split('.').str[0]

            bar['date'] = pd.to_datetime(bar['trade_date'], format='%Y%m%d')
            bar = bar.rename(columns={'vol': 'volume'})

            keep_cols = ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']
            for col in keep_cols:
                if col not in bar.columns:
                    bar[col] = np.nan
            bar = bar[keep_cols].copy()

            new_rows.append(bar)
            total_records += len(bar)
            print(f'+{len(bar):,} 只  (累计 {total_records:,})')
        except Exception as e:
            print(f'失败: {e}')
            failed_dates.append(td)

    if failed_dates:
        print(f'\n  ⚠ {len(failed_dates)} 个交易日拉取失败: {failed_dates}')

    if not new_rows:
        print('\n  未拉到任何新数据')
        return {'new_rows': 0, 'failed_dates': failed_dates}

    print(f'\n[4/4] 合并写入...')
    new_df = pd.concat(new_rows, ignore_index=True)
    print(f'  新增原始: {len(new_df):,} 行, {new_df["code"].nunique()} 只股票')

    combined = pd.concat([df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['code', 'date'], keep='last')
    combined = combined.sort_values(['code', 'date']).reset_index(drop=True)

    added = len(combined) - len(df)
    new_latest = combined['date'].max()
    print(f'  合并后: {len(combined):,} 行, 净增 {added:,}, 最新 {new_latest.date()}')

    if not dry_run and added > 0:
        print(f'  写入 {STOCK_DAILY_PKL} ({os.path.getsize(STOCK_DAILY_PKL)/1024/1024:.0f}MB → ', end='')
        data['df_stock'] = combined
        with open(STOCK_DAILY_PKL, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'{os.path.getsize(STOCK_DAILY_PKL)/1024/1024:.0f}MB)')
    elif dry_run:
        print('  [dry-run] 未写入')

    return {'new_rows': added, 'latest': str(new_latest.date()),
            'trade_days': len(trade_dates), 'failed': failed_dates}


# ============================================================
#  --update-financial: 增量更新财务报表
# ============================================================

def _current_periods():
    """生成所有需要的报告期 yyyymmdd"""
    periods = []
    for year in range(START_YEAR, END_YEAR + 1):
        for q in ['0331', '0630', '0930', '1231']:
            periods.append(f"{year}{q}")
    return periods


def update_financial(dry_run=False, force_period: str = None):
    """增量更新财务报表（只拉缺失的报告期）。

    Args:
        force_period: 强制重拉某个报告期（如 '20260331'），会先删除该期已有行再重拉。
                      用于季报披露期末补全（如 4 月底一季报陆续出齐）。
    """
    print('=' * 60)
    print('  增量更新财务报表')
    print('=' * 60)
    if force_period:
        print(f'  [force-period] 将重拉 {force_period}')

    pro = _get_pro()
    all_periods = _current_periods()

    tables = {
        'raw_income.pkl': {
            'func': pro.income_vip,
            'fields': ('ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,'
                       'revenue,total_cogs,oper_cost,n_income,n_income_attr_p,'
                       'total_profit,ebit,int_exp'),
        },
        'raw_balancesheet.pkl': {
            'func': pro.balancesheet_vip,
            'fields': ('ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,'
                       'total_assets,total_hldr_eqy_exc_min_int,total_liab,'
                       'total_cur_assets,total_cur_liab,inventories'),
        },
        'raw_cashflow.pkl': {
            'func': pro.cashflow_vip,
            'fields': ('ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,'
                       'n_cashflow_act,n_cashflow_inv_act'),
        },
    }

    results = {}
    for fname, info in tables.items():
        path = os.path.join(LOCAL_DATA_RAW, fname)
        if os.path.exists(path):
            old = pd.read_pickle(path)
            # 强制重拉：先剔除该期已有行
            if force_period:
                before = len(old)
                old = old[old['end_date'].astype(str) != force_period].copy()
                print(f'\n[{fname}] 删除 {force_period} 旧行 {before - len(old)} 条')
            existing_periods = set(old['end_date'].astype(str).unique())
        else:
            old = pd.DataFrame()
            existing_periods = set()

        missing = [p for p in all_periods if p not in existing_periods]
        # 只拉最近 2 年的缺失期（更早的数据不太可能有新增）
        cutoff = str(datetime.now().year - 2)
        missing = [p for p in missing if p[:4] >= cutoff]

        print(f'\n[{fname}] 现有 {len(old)} 行, 缺失期数 {len(missing)}')
        if not missing:
            results[fname] = {'new_rows': 0}
            continue

        new_dfs = []
        for i, period in enumerate(missing):
            print(f'  [{i+1}/{len(missing)}] 拉取 {period}', end=' ')
            try:
                df = _call(info['func'], period=period, fields=info['fields'])
                if df is not None and len(df) > 0:
                    new_dfs.append(df)
                    print(f'+{len(df)}')
                else:
                    print('(empty)')
            except Exception as e:
                print(f'失败: {e}')

        if new_dfs:
            new = pd.concat(new_dfs, ignore_index=True)
            combined = pd.concat([old, new], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=['ts_code', 'end_date'], keep='last')
            added = len(combined) - len(old)
            print(f'  净增 {added} 行')
            if not dry_run:
                combined.to_pickle(path)
                print(f'  写入 {path}')
            results[fname] = {'new_rows': added}
        else:
            results[fname] = {'new_rows': 0}

    # 一致预期（report_rc）增量
    rc_path = os.path.join(LOCAL_DATA_RAW, 'raw_report_rc.pkl')
    if os.path.exists(rc_path):
        rc_old = pd.read_pickle(rc_path)
        rc_old['report_date'] = rc_old['report_date'].astype(str)
        latest_month = max(rc_old['report_date'].str[:6].unique())
        print(f'\n[raw_report_rc.pkl] 现有 {len(rc_old)} 行, 最新月 {latest_month}')

        # 拉最新月份之后的数据
        start_month = pd.to_datetime(latest_month, format='%Y%m') + pd.DateOffset(months=1)
        end_month = pd.Timestamp.today()
        months = pd.date_range(start_month, end_month, freq='MS')

        rc_fields = 'ts_code,report_date,quarter,op_rt,op_pr,tp,np,eps,roe,rd,org_name'
        new_rc = []
        for m in months:
            m_end = (m + pd.offsets.MonthEnd(0)).strftime('%Y%m%d')
            print(f'  report_rc {m_end}', end=' ')
            try:
                df = _call(pro.report_rc, report_date=m_end, fields=rc_fields)
                if df is not None and len(df) > 0:
                    new_rc.append(df)
                    print(f'+{len(df)}')
                else:
                    print('(empty)')
            except Exception:
                print('(failed)')

        if new_rc:
            new = pd.concat(new_rc, ignore_index=True)
            combined = pd.concat([rc_old, new], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=['ts_code', 'report_date', 'quarter', 'org_name'],
                keep='last')
            added = len(combined) - len(rc_old)
            print(f'  净增 {added} 行')
            if not dry_run:
                combined.to_pickle(rc_path)
            results['raw_report_rc.pkl'] = {'new_rows': added}
        else:
            results['raw_report_rc.pkl'] = {'new_rows': 0}
    else:
        print(f'\n[raw_report_rc.pkl] 不存在，请先 --migrate')
        results['raw_report_rc.pkl'] = {'error': 'missing'}

    return results


# ============================================================
#  --backfill: 补全缺失股票的完整历史（如创业板、北交所）
# ============================================================

def backfill_missing_stocks(dry_run=False):
    """一次性补下 stock_daily.pkl 中缺失的股票历史（创业板/北交所等）。

    策略：按交易日拉取（pro.daily(trade_date=xxx)），一次获得当天所有股票，
    只保留缺失的那些。与 update_stock_daily 使用相同接口，无频率限制问题。
    支持断点续传：进度存于 _backfill_ckpt.pkl，中断后重跑自动续接。
    """
    import time as _time
    print('=' * 60)
    print('  补全缺失股票历史数据（按交易日拉取）')
    print('=' * 60)

    if not os.path.exists(STOCK_DAILY_PKL):
        print(f'[错误] {STOCK_DAILY_PKL} 不存在，请先 --migrate')
        return {'error': 'no_base_file'}

    print('[1/4] 加载现有数据...')
    with open(STOCK_DAILY_PKL, 'rb') as f:
        data = pickle.load(f)
    df = data['df_stock']
    df['date'] = pd.to_datetime(df['date'])
    existing_codes = set(df['code'].unique())
    end_date_str = df['date'].max().strftime('%Y%m%d')
    print(f'  现有: {len(existing_codes)} 只股票, 最新 {df["date"].max().date()}')

    pro = _get_pro()

    print('\n[2/4] 获取全市场股票列表...')
    all_stocks = []
    for status in ('L', 'D', 'P'):
        batch = _call(pro.stock_basic, exchange='', list_status=status,
                      fields='ts_code,name,list_date,delist_date')
        if batch is not None and not batch.empty:
            all_stocks.append(batch)
    all_df = pd.concat(all_stocks, ignore_index=True)
    all_df['code'] = all_df['ts_code'].str[:6]

    missing = all_df[~all_df['code'].isin(existing_codes)].copy()
    print(f'  全市场 {len(all_df)} 只，缺失 {len(missing)} 只')
    for prefix, label in [('300', '创业板300'), ('301', '创业板301'),
                           ('688', '科创板688'), ('689', '科创板689'),
                           ('8', '北交所8xx'), ('43', '北交所43x')]:
        n = missing['code'].str.startswith(prefix).sum()
        if n:
            print(f'    {label}: {n}')

    if missing.empty:
        print('  无缺失，退出')
        return {'backfilled': 0}

    missing_codes = set(missing['code'])
    # 最早上市日（从这一天起才有创业板股票）
    valid_dates = missing['list_date'].dropna()
    valid_dates = valid_dates[valid_dates.astype(str).str.match(r'^\d{8}$')]
    start_date_str = valid_dates.min() if not valid_dates.empty else '20090101'
    print(f'  下载区间: {start_date_str} → {end_date_str}')

    # 断点续传
    ckpt_path = STOCK_DAILY_PKL.replace('.pkl', '_backfill_ckpt.pkl')
    accumulated = []
    done_dates  = set()
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, 'rb') as f:
                ckpt = pickle.load(f)
            done_dates   = ckpt.get('done_dates', set())
            accumulated  = ckpt.get('accumulated', [])
            print(f'  恢复检查点: 已处理 {len(done_dates)} 个交易日')
        except Exception:
            done_dates, accumulated = set(), []

    print('\n[3/4] 获取交易日历...')
    trade_dates = _get_trade_dates(pro, start_date_str, end_date_str)
    remaining   = [d for d in trade_dates if d not in done_dates]
    print(f'  共 {len(trade_dates)} 个交易日，待处理 {len(remaining)} 个')

    print(f'\n[4/4] 按交易日拉取（共 {len(remaining)} 天）...')
    for i, td in enumerate(remaining):
        if (i + 1) % 200 == 0 or i == 0:
            print(f'  [{i+1}/{len(remaining)}] {td}  '
                  f'(已收集 {sum(len(r) for r in accumulated):,} 行)', flush=True)
        try:
            bar = _call(pro.daily, trade_date=td)
            if bar is not None and not bar.empty:
                bar['code'] = bar['ts_code'].str.split('.').str[0]
                bar = bar[bar['code'].isin(missing_codes)]
                if not bar.empty:
                    bar['date'] = pd.to_datetime(td, format='%Y%m%d')
                    bar = bar.rename(columns={'vol': 'volume'})
                    keep = ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']
                    for col in keep:
                        if col not in bar.columns:
                            bar[col] = np.nan
                    accumulated.append(bar[keep])
        except Exception as e:
            print(f'  ⚠ {td} 失败: {e}', flush=True)

        done_dates.add(td)

        # 每 200 天存一次检查点
        if not dry_run and (i + 1) % 200 == 0:
            with open(ckpt_path, 'wb') as f:
                pickle.dump({'done_dates': done_dates, 'accumulated': accumulated}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)

    if not accumulated:
        print('  未拉到数据')
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        return {'backfilled': 0}

    print(f'\n  合并写入...')
    new_df = pd.concat(accumulated, ignore_index=True)
    print(f'  新增: {new_df["code"].nunique()} 只股票, {len(new_df):,} 条记录')

    combined = pd.concat([df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['code', 'date'], keep='last')
    combined = combined.sort_values(['code', 'date']).reset_index(drop=True)
    print(f'  合并后: {combined["code"].nunique()} 只股票, {len(combined):,} 条')

    if not dry_run:
        print(f'  写入 {STOCK_DAILY_PKL}...')
        data['df_stock'] = combined
        with open(STOCK_DAILY_PKL, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'  完成 ({os.path.getsize(STOCK_DAILY_PKL)/1024/1024:.0f}MB)')
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    return {'backfilled': new_df['code'].nunique()}


# ============================================================
#  --update-members: 更新行业成分股
# ============================================================

def update_sw_members(dry_run=False):
    """从 tushare 更新申万一级行业成分股映射"""
    print('=' * 60)
    print('  更新申万行业成分股')
    print('=' * 60)

    pro = _get_pro()

    # 获取所有申万一级行业
    idx = _call(pro.index_classify, level='L1', src='SW2021')
    if idx is None or len(idx) == 0:
        print('  无法获取行业列表')
        return {'error': 'no_index'}

    # 建立行业代码 → 名称映射（来自 index_classify 结果）
    name_col = next((c for c in ('industry_name', 'name', 'index_name') if c in idx.columns), None)
    ind_name_map = dict(zip(idx['index_code'], idx[name_col])) if name_col else {}

    all_members = []
    for _, row in idx.iterrows():
        code = row['index_code']
        if code in SW_EXCLUDE:
            continue
        l1_name = ind_name_map.get(code, '')
        print(f'  {code} {l1_name}', end=' ')
        df = _call(pro.index_member, index_code=code)
        if df is not None and len(df) > 0:
            # API 实际只返回 con_code / in_date / out_date，不带名称列
            df = df.rename(columns={'con_code': 'ts_code', 'con_name': 'name',
                                    'index_code': 'l1_code'})
            df['l1_code'] = code
            df['l1_name'] = l1_name
            # 标记现役：out_date 为空/None/空串 → is_new = 'Y'
            df['out_date'] = df['out_date'].replace('', pd.NA)
            df['is_new'] = df['out_date'].isna().map({True: 'Y', False: 'N'})
            all_members.append(df)
            print(f'+{len(df)}')
        else:
            print('(empty)')

    if not all_members:
        return {'error': 'no_data'}

    result = pd.concat(all_members, ignore_index=True)
    # 保留关键列，顺序与旧格式兼容
    keep = ['l1_code', 'l1_name', 'ts_code', 'in_date', 'out_date', 'is_new']
    if 'name' in result.columns:
        keep.insert(3, 'name')
    result = result[[c for c in keep if c in result.columns]]
    result = result.drop_duplicates(subset=['l1_code', 'ts_code', 'in_date'], keep='last')
    print(f'\n  总计 {len(result)} 条映射, {result["l1_code"].nunique()} 个行业')

    if not dry_run:
        result.to_csv(SW_MEMBERS_CSV, index=False, encoding='utf-8-sig')
        print(f'  写入 {SW_MEMBERS_CSV}')

    return {'rows': len(result)}


# ============================================================
#  入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='基础数据下载与迁移')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--migrate', action='store_true',
                       help='从 ARIMAX 项目复制数据（推荐首次使用）')
    group.add_argument('--update-daily', action='store_true',
                       help='增量更新个股日线')
    group.add_argument('--update-financial', action='store_true',
                       help='增量更新财务报表')
    group.add_argument('--update-members', action='store_true',
                       help='更新行业成分股')
    group.add_argument('--backfill', action='store_true',
                       help='补全缺失股票的完整历史（创业板/北交所，一次性操作）')
    group.add_argument('--full', action='store_true',
                       help='全量更新（日线+财务+成分股）')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force-period', type=str, default=None, metavar='YYYYMMDD',
                        help='强制重拉指定报告期（如 20260331），与 --update-financial 配合使用')
    args = parser.parse_args()

    if args.migrate:
        migrate_from_arimax(dry_run=args.dry_run)
    elif args.update_daily:
        update_stock_daily(dry_run=args.dry_run)
    elif args.update_financial:
        update_financial(dry_run=args.dry_run, force_period=args.force_period)
    elif args.update_members:
        update_sw_members(dry_run=args.dry_run)
    elif args.backfill:
        backfill_missing_stocks(dry_run=args.dry_run)
    elif args.full:
        update_stock_daily(dry_run=args.dry_run)
        update_financial(dry_run=args.dry_run)
        update_sw_members(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
