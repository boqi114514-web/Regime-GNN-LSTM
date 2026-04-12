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

def update_stock_daily(dry_run=False):
    """在已有 stock_daily.pkl 基础上增量追加最新日线数据"""
    print('=' * 60)
    print('  增量更新个股日K线')
    print('=' * 60)

    if not os.path.exists(STOCK_DAILY_PKL):
        print(f'[错误] {STOCK_DAILY_PKL} 不存在，请先 --migrate')
        return {'error': 'no_base_file'}

    print('加载现有数据...')
    with open(STOCK_DAILY_PKL, 'rb') as f:
        data = pickle.load(f)
    df = data['df_stock']
    df['date'] = pd.to_datetime(df['date'])
    latest = df['date'].max()
    print(f'  现有: {len(df):,} 行, {df["code"].nunique()} 只股票, 最新 {latest.date()}')

    # 需要更新的时间范围
    start_dt = latest + pd.Timedelta(days=1)
    end_dt = pd.Timestamp.today()
    if start_dt >= end_dt:
        print('  已是最新')
        return {'new_rows': 0}

    start_s = start_dt.strftime('%Y%m%d')
    end_s = end_dt.strftime('%Y%m%d')
    print(f'  拉取窗口: {start_s} → {end_s}')

    pro = _get_pro()

    # 获取所有A股列表
    stock_basic = _call(pro.stock_basic, exchange='', list_status='L',
                        fields='ts_code,name')
    all_ts_codes = stock_basic['ts_code'].tolist()
    # 也包含已退市的
    stock_d = _call(pro.stock_basic, exchange='', list_status='D',
                    fields='ts_code,name')
    if stock_d is not None and len(stock_d) > 0:
        all_ts_codes.extend(stock_d['ts_code'].tolist())

    # 过滤：只拉主板和中小板（排除创业板300/科创板688/北交所8）
    existing_codes = set(df['code'].unique())
    codes_to_pull = []
    for tc in all_ts_codes:
        code = tc.split('.')[0]
        if code.startswith(('300', '301', '688', '689', '8', '4', '9')):
            continue
        codes_to_pull.append(tc)

    print(f'  待拉取股票数: {len(codes_to_pull)}')

    new_rows = []
    for i, ts_code in enumerate(codes_to_pull):
        if (i + 1) % 200 == 0:
            print(f'  [{i+1}/{len(codes_to_pull)}] 拉取中...')
        try:
            bar = _call(pro.daily, ts_code=ts_code,
                        start_date=start_s, end_date=end_s)
            if bar is not None and len(bar) > 0:
                bar = bar.rename(columns={'trade_date': 'date', 'ts_code': 'ts_code_full'})
                bar['code'] = ts_code.split('.')[0]
                bar['date'] = pd.to_datetime(bar['date'], format='%Y%m%d')
                for col in ('open', 'high', 'low', 'close', 'vol', 'amount'):
                    if col in bar.columns:
                        bar[col] = pd.to_numeric(bar[col], errors='coerce')
                bar = bar.rename(columns={'vol': 'volume'})
                new_rows.append(bar[['date', 'code', 'open', 'high', 'low',
                                     'close', 'volume', 'amount']])
        except Exception:
            pass

    if not new_rows:
        print('  未拉到新数据')
        return {'new_rows': 0}

    new_df = pd.concat(new_rows, ignore_index=True)
    print(f'  新增 {len(new_df):,} 行')

    combined = pd.concat([df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['code', 'date'], keep='last')
    combined = combined.sort_values(['code', 'date']).reset_index(drop=True)

    added = len(combined) - len(df)
    print(f'  合并后 {len(combined):,} 行, 净增 {added:,}')

    if not dry_run and added > 0:
        data['df_stock'] = combined
        with open(STOCK_DAILY_PKL, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'  写入 {STOCK_DAILY_PKL}')

    return {'new_rows': added}


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


def update_financial(dry_run=False):
    """增量更新财务报表（只拉缺失的报告期）"""
    print('=' * 60)
    print('  增量更新财务报表')
    print('=' * 60)

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

    all_members = []
    for _, row in idx.iterrows():
        code = row['index_code']
        if code in SW_EXCLUDE:
            continue
        print(f'  {code} {row.get("industry_name", "")}', end=' ')
        df = _call(pro.index_member, index_code=code,
                   fields='index_code,index_name,con_code,con_name,in_date,out_date')
        if df is not None and len(df) > 0:
            df = df.rename(columns={
                'index_code': 'l1_code', 'index_name': 'l1_name',
                'con_code': 'ts_code', 'con_name': 'name',
            })
            all_members.append(df)
            print(f'+{len(df)}')
        else:
            print('(empty)')

    if not all_members:
        return {'error': 'no_data'}

    result = pd.concat(all_members, ignore_index=True)
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
    group.add_argument('--full', action='store_true',
                       help='全量更新（日线+财务+成分股）')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.migrate:
        migrate_from_arimax(dry_run=args.dry_run)
    elif args.update_daily:
        update_stock_daily(dry_run=args.dry_run)
    elif args.update_financial:
        update_financial(dry_run=args.dry_run)
    elif args.update_members:
        update_sw_members(dry_run=args.dry_run)
    elif args.full:
        update_stock_daily(dry_run=args.dry_run)
        update_financial(dry_run=args.dry_run)
        update_sw_members(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
