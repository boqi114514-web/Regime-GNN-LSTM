# -*- coding: utf-8 -*-
"""增量数据拉取：把 data/raw 的 csv 文件更新到最新月份

覆盖的数据：
  - ts_sw_industry_monthly.csv   (pro.index_monthly + pro.index_dailybasic)
  - ts_csi300_monthly.csv        (pro.index_monthly)
  - ts_macro_factors.csv         (pro.cn_pmi + pro.cn_m + pro.cn_sf + pro.shibor)

不覆盖（Phase 2 的活）：
  - prosperity_indicators_clean.pkl  (要重跑 ARIMAX 的 01→02→03→04 链条)
  - price_volume_factors.pkl         (要个股日线 + 重算)
  - pattern_factors.pkl              (同上)

用法：
    python -m data_pipeline.update              # 增量更新到今天
    python -m data_pipeline.update --force      # 强制重拉最近 3 个月
    python -m data_pipeline.update --dry-run    # 只打印不写文件

配置：
    TUSHARE_TOKEN 和 TUSHARE_URL 从环境变量读，若缺省用 data_pipeline/tushare_config.py 里的值。
"""
import argparse
import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import LOCAL_DATA_RAW, LOCAL_DATA_PROCESSED, SW_EXCLUDE
from live import state
from data_pipeline.tushare_config import get_pro, _call_with_retry


# ============================================================
#  日期辅助
# ============================================================

def _next_day_str(ts: pd.Timestamp) -> str:
    """返回 ts 的下一个自然日 yyyymmdd"""
    return (ts + pd.Timedelta(days=1)).strftime('%Y%m%d')


def _today_str() -> str:
    return datetime.today().strftime('%Y%m%d')


def _month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return ts + pd.offsets.MonthEnd(0)


# ============================================================
#  1. 申万一级行业月度
# ============================================================

def update_sw_industry_monthly(dry_run: bool = False, force_months: int = 0) -> dict:
    """拉最新月份的申万一级行业数据（含 pe/pb），追加到 ts_sw_industry_monthly.csv"""
    path = os.path.join(LOCAL_DATA_RAW, 'ts_sw_industry_monthly.csv')
    old = pd.read_csv(path)
    old['date'] = pd.to_datetime(old['date'])
    latest = old['date'].max()
    print(f'\n[sw_industry_monthly] 现有 {len(old)} 行, 最新 {latest.date()}')

    if force_months > 0:
        start = (latest - pd.DateOffset(months=force_months - 1)).replace(day=1)
    else:
        start = latest + pd.Timedelta(days=1)

    end = pd.Timestamp.today()
    if start > end:
        print('  已是最新，无需更新')
        return {'new_rows': 0, 'latest': str(latest.date())}

    start_s = start.strftime('%Y%m%d')
    end_s = end.strftime('%Y%m%d')
    print(f'  拉取窗口: {start_s} -> {end_s}')

    pro = get_pro()

    # 1.1 所有申万一级行业代码（从旧数据推断，避免依赖外部清单）
    industries = sorted(set(old['ts_code']) - set(SW_EXCLUDE))
    print(f'  行业数: {len(industries)}')

    # 1.2 拉月度行情
    monthly_rows = []
    for i, code in enumerate(industries):
        print(f'  [{i+1:2d}/{len(industries)}] index_monthly  {code}', end=' ')
        df = _call_with_retry(pro.index_monthly, ts_code=code,
                              start_date=start_s, end_date=end_s)
        if df is not None and len(df) > 0:
            monthly_rows.append(df)
            print(f'+{len(df)}')
        else:
            print('(empty)')

    if not monthly_rows:
        print('  [sw_industry_monthly] 新窗口内全为空')
        return {'new_rows': 0, 'latest': str(latest.date())}

    monthly = pd.concat(monthly_rows, ignore_index=True)
    monthly = monthly.rename(columns={'trade_date': 'date'})
    monthly['date'] = pd.to_datetime(monthly['date'].astype(str), format='%Y%m%d')

    # 1.3 拉日度估值（pe/pb），月末取最后一个交易日的值
    # 注意：index_dailybasic 不覆盖申万行业指数，必须走 sw_daily
    print(f'\n  拉取 sw_daily (pe/pb) ...')
    pb_rows = []
    for i, code in enumerate(industries):
        print(f'  [{i+1:2d}/{len(industries)}] sw_daily      {code}', end=' ')
        df = _call_with_retry(pro.sw_daily, ts_code=code,
                              start_date=start_s, end_date=end_s,
                              fields='ts_code,trade_date,pe,pb')
        if df is not None and len(df) > 0:
            pb_rows.append(df)
            print(f'+{len(df)}')
        else:
            print('(empty)')

    pe_pb_col = None
    if pb_rows:
        dailybasic = pd.concat(pb_rows, ignore_index=True)
        dailybasic['trade_date'] = pd.to_datetime(
            dailybasic['trade_date'].astype(str), format='%Y%m%d')
        dailybasic['ym'] = dailybasic['trade_date'].dt.to_period('M')
        # 每月末最后一个交易日的 pe/pb
        monthly_pe_pb = (dailybasic.sort_values(['ts_code', 'trade_date'])
                         .groupby(['ts_code', 'ym'], as_index=False)
                         .tail(1))
        monthly_pe_pb['date_month_end'] = monthly_pe_pb['ym'].dt.to_timestamp('M').dt.normalize()
        pe_pb_col = monthly_pe_pb[['ts_code', 'date_month_end', 'pe', 'pb']]

    # 1.4 合并 pe/pb 到月度行情
    monthly['date_month_end'] = monthly['date'] + pd.offsets.MonthEnd(0)
    if pe_pb_col is not None:
        monthly = monthly.merge(pe_pb_col, on=['ts_code', 'date_month_end'], how='left')
    else:
        monthly['pe'] = pd.NA
        monthly['pb'] = pd.NA

    # 对齐 old 的列
    for col in old.columns:
        if col not in monthly.columns:
            monthly[col] = pd.NA
    monthly = monthly[old.columns.tolist()]

    # 日期统一成月末（和 old 的格式一致）
    monthly['date'] = monthly['date'] + pd.offsets.MonthEnd(0)

    # 去重合并
    combined = pd.concat([old, monthly], ignore_index=True)
    combined = combined.drop_duplicates(subset=['ts_code', 'date'], keep='last')
    combined = combined.sort_values(['ts_code', 'date']).reset_index(drop=True)

    added = len(combined) - len(old)
    new_latest = combined['date'].max()
    print(f'\n  合并后 {len(combined)} 行, 净增 {added}, 最新 {new_latest.date()}')

    # 新数据的 pe/pb 覆盖率（用 monthly 而不是 combined，只看新增部分）
    if 'pe' in monthly.columns:
        pe_cov = monthly['pe'].notna().sum()
        pb_cov = monthly['pb'].notna().sum()
        print(f'  新增 {len(monthly)} 行中 pe 非空 {pe_cov}, pb 非空 {pb_cov}')
        print(monthly[['ts_code', 'date', 'close', 'pe', 'pb']].head(6).to_string(index=False))

    if not dry_run:
        combined.to_csv(path, index=False)
        print(f'  写入 {path}')

    return {'new_rows': added, 'latest': str(new_latest.date())}


# ============================================================
#  2. 沪深300 月度
# ============================================================

def update_csi300_monthly(dry_run: bool = False, force_months: int = 0) -> dict:
    path = os.path.join(LOCAL_DATA_RAW, 'ts_csi300_monthly.csv')
    old = pd.read_csv(path)
    # csi300 的 date 是 yyyymmdd 整数形式
    old['_d'] = pd.to_datetime(old['date'].astype(str), format='%Y%m%d')
    latest = old['_d'].max()
    print(f'\n[csi300_monthly] 现有 {len(old)} 行, 最新 {latest.date()}')

    if force_months > 0:
        start = (latest - pd.DateOffset(months=force_months - 1)).replace(day=1)
    else:
        start = latest + pd.Timedelta(days=1)
    end = pd.Timestamp.today()
    if start > end:
        print('  已是最新')
        return {'new_rows': 0, 'latest': str(latest.date())}

    start_s, end_s = start.strftime('%Y%m%d'), end.strftime('%Y%m%d')
    print(f'  拉取窗口: {start_s} -> {end_s}')

    pro = get_pro()
    df = _call_with_retry(pro.index_monthly, ts_code='000300.SH',
                          start_date=start_s, end_date=end_s)
    if df is None or len(df) == 0:
        print('  (empty)')
        return {'new_rows': 0, 'latest': str(latest.date())}

    df = df.rename(columns={'trade_date': 'date'})
    # 保持 old 的 yyyymmdd 字符串格式
    df['date'] = df['date'].astype(str)

    for col in old.columns:
        if col == '_d':
            continue
        if col not in df.columns:
            df[col] = pd.NA
    df = df[[c for c in old.columns if c != '_d']]

    combined = pd.concat([old.drop(columns='_d'), df], ignore_index=True)
    combined['date'] = combined['date'].astype(str)
    combined = combined.drop_duplicates(subset=['ts_code', 'date'], keep='last')
    combined['_sort'] = pd.to_datetime(combined['date'], format='%Y%m%d')
    combined = combined.sort_values('_sort').drop(columns='_sort').reset_index(drop=True)

    added = len(combined) - len(old)
    new_latest = combined['date'].max()
    print(f'  合并后 {len(combined)} 行, 净增 {added}, 最新 {new_latest}')

    if not dry_run:
        combined.to_csv(path, index=False)
        print(f'  写入 {path}')

    return {'new_rows': added, 'latest': str(new_latest)}


# ============================================================
#  3. 宏观因子
# ============================================================

def _month_start_end(year: int, month: int) -> tuple:
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(0)
    return start.strftime('%Y%m%d'), end.strftime('%Y%m%d')


def update_macro_factors(dry_run: bool = False, force_months: int = 0) -> dict:
    """只补 s0_regime 真正用到的 4 列：pmi_mfg, term_spread, m1_m2_spread, sf_yoy"""
    path = os.path.join(LOCAL_DATA_RAW, 'ts_macro_factors.csv')
    old = pd.read_csv(path)
    old['date'] = pd.to_datetime(old['date'])
    latest = old['date'].max()
    print(f'\n[macro_factors] 现有 {len(old)} 行, 最新 {latest.date()}')

    # --- 回填已有数据中的 sf_yoy 空洞 ---
    sf_holes = old[old['sf_yoy'].isna() & old['date'].dt.year >= 2015]
    if len(sf_holes) > 0:
        print(f'  发现 {len(sf_holes)} 个月 sf_yoy 为空，尝试回填...')
        hole_periods = sf_holes['date'].dt.to_period('M')
        # 为了算 yoy，需要往前拉一年
        all_needed = set()
        for p in hole_periods:
            all_needed.add(p)
            all_needed.add(p - 12)
        all_needed = sorted(all_needed)

        pro = get_pro()
        sf_all_backfill = {}
        for p in all_needed:
            m_str = f'{p.year}{p.month:02d}'
            try:
                df = _call_with_retry(pro.sf_month, start_m=m_str, end_m=m_str,
                                      fields='month,inc_month')
                if df is not None and len(df) > 0:
                    sf_all_backfill[m_str] = float(df.iloc[0]['inc_month'])
            except Exception as e:
                print(f'    sf_month {m_str} failed: {e}')

        filled_count = 0
        for idx_row, row in sf_holes.iterrows():
            p = row['date'].to_period('M')
            m_str = f'{p.year}{p.month:02d}'
            prev = f'{p.year - 1}{p.month:02d}'
            if m_str in sf_all_backfill and prev in sf_all_backfill and sf_all_backfill[prev] != 0:
                yoy = (sf_all_backfill[m_str] - sf_all_backfill[prev]) / abs(sf_all_backfill[prev])
                old.loc[idx_row, 'sf_yoy'] = yoy
                old.loc[idx_row, 'sf_inc_month'] = sf_all_backfill[m_str]
                filled_count += 1
                print(f'    回填 {m_str} sf_yoy={yoy:.4f}')
        print(f'  回填完成: {filled_count}/{len(sf_holes)} 个月')

        if filled_count > 0 and not dry_run:
            old.to_csv(path, index=False)
            print(f'  已写入回填结果')
        # 重新读取（回填后的 old 继续用于后续增量逻辑）
    # --- 回填结束 ---

    if force_months > 0:
        start_period = (latest - pd.DateOffset(months=force_months - 1)).to_period('M')
    else:
        start_period = (latest + pd.DateOffset(months=1)).to_period('M')

    today_period = pd.Timestamp.today().to_period('M')
    if start_period > today_period:
        print('  已是最新')
        return {'new_rows': 0, 'latest': str(latest.date())}

    # 生成需要补的月份列表
    periods = pd.period_range(start=start_period, end=today_period, freq='M')
    print(f'  需补月份: {[str(p) for p in periods]}')

    pro = get_pro()

    # 3.1 PMI（cn_pmi）
    pmi_data = {}  # {yyyy-mm: pmi_mfg}
    for p in periods:
        m_str = f'{p.year}{p.month:02d}'
        df = _call_with_retry(pro.cn_pmi, start_m=m_str, end_m=m_str,
                              fields='month,pmi010000')
        if df is not None and len(df) > 0:
            pmi_data[m_str] = float(df.iloc[0]['pmi010000'])
            print(f'  pmi {m_str} = {pmi_data[m_str]}')

    # 3.2 货币供应（cn_m） → m1_m2_spread
    m1m2_data = {}
    for p in periods:
        m_str = f'{p.year}{p.month:02d}'
        df = _call_with_retry(pro.cn_m, start_m=m_str, end_m=m_str,
                              fields='month,m1_yoy,m2_yoy')
        if df is not None and len(df) > 0:
            row = df.iloc[0]
            m1m2_data[m_str] = {
                'm1_yoy_pct': float(row['m1_yoy']),
                'm2_yoy_pct': float(row['m2_yoy']),
                'm1_m2_spread': float(row['m1_yoy']) - float(row['m2_yoy']),
            }
            print(f'  cn_m {m_str} m1={row["m1_yoy"]} m2={row["m2_yoy"]}')

    # 3.3 社会融资（sf_month）→ sf_yoy
    # tushare 的社融接口 sf_month 返回当月数据，需要计算同比
    sf_data = {}
    # 为了算 yoy，需要往前拉一年
    sf_periods = pd.period_range(start=start_period - 12, end=today_period, freq='M')
    sf_all = {}
    for p in sf_periods:
        m_str = f'{p.year}{p.month:02d}'
        try:
            df = _call_with_retry(pro.sf_month, start_m=m_str, end_m=m_str,
                                  fields='month,inc_month')
            if df is not None and len(df) > 0:
                sf_all[m_str] = float(df.iloc[0]['inc_month'])
        except Exception as e:
            print(f'  sf_month {m_str} failed: {e}')
    # 计算 yoy
    for p in periods:
        m_str = f'{p.year}{p.month:02d}'
        prev = f'{p.year - 1}{p.month:02d}'
        if m_str in sf_all and prev in sf_all and sf_all[prev] != 0:
            sf_data[m_str] = {
                'sf_inc_month': sf_all[m_str],
                'sf_yoy': (sf_all[m_str] - sf_all[prev]) / abs(sf_all[prev]),
            }
            print(f'  sf {m_str} inc={sf_all[m_str]} yoy={sf_data[m_str]["sf_yoy"]:.4f}')

    # 3.4 Shibor 利率 → term_spread = shibor_1y - shibor_on
    # 这个接口按天拉，取每月最后一个交易日
    shibor_data = {}
    for p in periods:
        start_s, end_s = _month_start_end(p.year, p.month)
        df = _call_with_retry(pro.shibor, start_date=start_s, end_date=end_s)
        if df is not None and len(df) > 0:
            df = df.sort_values('date')
            last = df.iloc[-1]
            shibor_data[f'{p.year}{p.month:02d}'] = {
                'shibor_on': float(last['on']),
                'shibor_1w': float(last.get('1w', 0) or 0),
                'shibor_1m': float(last.get('1m', 0) or 0),
                'shibor_3m': float(last.get('3m', 0) or 0),
                'shibor_6m': float(last.get('6m', 0) or 0),
                'shibor_1y': float(last.get('1y', 0) or 0),
                'term_spread': float(last.get('1y', 0) or 0) - float(last['on']),
            }
            print(f'  shibor {p.year}-{p.month:02d} on={last["on"]} 1y={last.get("1y")}')

    # 3.5 合成新行
    new_rows = []
    for p in periods:
        m_str = f'{p.year}{p.month:02d}'
        row = {'date': p.to_timestamp('M').normalize()}
        for c in old.columns:
            if c != 'date':
                row[c] = pd.NA

        if m_str in shibor_data:
            row.update(shibor_data[m_str])
        if m_str in pmi_data:
            row['pmi_mfg'] = pmi_data[m_str]
        if m_str in m1m2_data:
            row.update(m1m2_data[m_str])
        if m_str in sf_data:
            row.update(sf_data[m_str])

        new_rows.append(row)

    if not new_rows:
        print('  没有拉到任何月份的数据')
        return {'new_rows': 0, 'latest': str(latest.date())}

    new = pd.DataFrame(new_rows)
    new = new[old.columns.tolist()]

    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=['date'], keep='last')
    combined = combined.sort_values('date').reset_index(drop=True)

    added = len(combined) - len(old)
    new_latest = combined['date'].max()
    print(f'  合并后 {len(combined)} 行, 净增 {added}, 最新 {new_latest.date()}')

    # 关键列的覆盖情况
    for key in ('pmi_mfg', 'term_spread', 'm1_m2_spread', 'sf_yoy'):
        n_nan = combined[key].isna().sum()
        print(f'    {key}: {len(combined) - n_nan}/{len(combined)} 非空')

    if not dry_run:
        combined.to_csv(path, index=False)
        print(f'  写入 {path}')

    return {'new_rows': added, 'latest': str(new_latest.date())}


# ============================================================
#  主流程
# ============================================================

def run(dry_run: bool = False, force_months: int = 0,
        processed: bool = False) -> dict:
    """增量更新数据。

    Args:
        dry_run: 只打印不写文件
        force_months: 强制重拉最近 N 个月
        processed: 是否同时更新 processed 数据（prosperity/tech_factors/pattern_factors）
    """
    print('=' * 60)
    print(f'  data_pipeline.update  dry_run={dry_run}  force={force_months}  processed={processed}')
    print('=' * 60)

    results = {}
    try:
        results['sw_industry'] = update_sw_industry_monthly(dry_run, force_months)
    except Exception as e:
        results['sw_industry'] = {'error': str(e)}
        print(f'[sw_industry] 失败: {e}')

    try:
        results['csi300'] = update_csi300_monthly(dry_run, force_months)
    except Exception as e:
        results['csi300'] = {'error': str(e)}
        print(f'[csi300] 失败: {e}')

    try:
        results['macro'] = update_macro_factors(dry_run, force_months)
    except Exception as e:
        results['macro'] = {'error': str(e)}
        print(f'[macro] 失败: {e}')

    # processed 数据更新（较慢，仅在 --processed / 月末训练前使用）
    if processed:
        import traceback

        def _step(name, fn, *args, **kwargs):
            """带进度提示和完整错误输出的步骤包装"""
            print(f'\n{"─"*60}')
            print(f'  ▶ {name}')
            print(f'{"─"*60}')
            try:
                r = fn(*args, **kwargs)
                results[name] = r
                print(f'  ✓ {name} 完成')
                return r
            except Exception as e:
                results[name] = {'error': str(e)}
                print(f'  ✗ {name} 失败: {e}')
                traceback.print_exc()
                return None

        # Step A: 增量更新个股日线（按交易日拉取，~60 次 API）
        from data_pipeline.download import update_stock_daily
        _step('stock_daily', update_stock_daily, dry_run)

        # Step B: 增量更新财务报表（按季度拉取）
        from data_pipeline.download import update_financial
        _step('financial', update_financial, dry_run)

        # Step C: 重算景气度指标（纯本地计算，几分钟）
        from data_pipeline.prosperity import run as prosperity_run
        _step('prosperity', prosperity_run, dry_run)

        # Step D: 增量更新价量因子
        def _run_tech_factors():
            from data_pipeline.tech_factors import run as tf_run
            pv_path = os.path.join(LOCAL_DATA_PROCESSED, 'price_volume_factors.pkl')
            since = None
            if os.path.exists(pv_path):
                pv_old = pd.read_pickle(pv_path)
                since = pv_old['date'].max().strftime('%Y-%m')
                print(f'  增量起点: {since}')
            return tf_run(since=since, dry_run=dry_run)
        _step('tech_factors', _run_tech_factors)

        # Step E: 增量更新走势复刻因子
        def _run_pattern_factors():
            from data_pipeline.pattern_factors import run as pf_run
            pt_path = os.path.join(LOCAL_DATA_PROCESSED, 'pattern_factors.pkl')
            since = None
            if os.path.exists(pt_path):
                pt_old = pd.read_pickle(pt_path)
                since = pt_old['date'].max().strftime('%Y-%m')
                print(f'  增量起点: {since}')
            return pf_run(since=since, dry_run=dry_run)
        _step('pattern_factors', _run_pattern_factors)

    print('\n' + '=' * 60)
    print('  汇总:')
    for key, val in results.items():
        status = val.get('error', 'ok') if isinstance(val, dict) else val
        print(f'  {key}: {status}')

    if not dry_run:
        state.update_last_run(
            last_data_update=datetime.now().isoformat(timespec='seconds'),
            last_data_update_results=results,
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='只打印不写文件')
    parser.add_argument('--force', type=int, default=0, metavar='N',
                        help='强制重拉最近 N 个月（用于修复数据）')
    parser.add_argument('--processed', action='store_true',
                        help='同时更新 processed 数据（景气度/价量/走势复刻因子）')
    args = parser.parse_args()
    run(dry_run=args.dry_run, force_months=args.force, processed=args.processed)


if __name__ == '__main__':
    main()
