# -*- coding: utf-8 -*-
"""data_pipeline/prosperity.py —— 景气度指标全链路

合并自 ARIMAX 项目的 02_calc_industry_data + 03_ttm_and_indicators + 04_clean_outliers。

链路：
  raw_income/balancesheet/cashflow/report_rc.pkl
      → 行业财务汇总
      → TTM 平滑
      → 21 个基础景气度指标 + 16 个一致预期指标
      → 极端值清洗
      → prosperity_indicators_clean.pkl

产物：data/processed/prosperity_indicators_clean.pkl

用法：
    python -m data_pipeline.prosperity              # 全量重算
    python -m data_pipeline.prosperity --dry-run     # 只打印不写文件
    python -m data_pipeline.prosperity --level l2    # 方向2：二级行业景气度，产物 *_l2.pkl

level 说明（方向2 Gate 3 Step 4）：
    l1（默认）→ ts_sw_members.csv，行业=申万一级，产物 prosperity_indicators_clean.pkl
    l2         → ts_sw_l2_members.csv，行业=申万二级，产物 prosperity_indicators_clean_l2.pkl
    景气度链条（清洗/汇总/TTM/指标/一致预期/极端值清洗）完全不变。
    实现上下游沿用列名 `l1_code` 作为"行业列"占位，L2 模式下该列装 l2_code 值，
    ARIMAX / 指标算法零改动 —— 即文档所述"成分映射换 + 重跑"。
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import LOCAL_DATA_RAW, LOCAL_DATA_PROCESSED, SW_EXCLUDE

# ==================== 指标定义 ====================

# 21 个基础景气度指标
BASIC_INDICATORS = [
    ('nptocostexpense',      '成本费用利润率',           'qoq_diff',  1),
    ('netprofitmargin',      '销售净利率',               'qoq_diff',  1),
    ('roe',                  '净资产收益率',             'qoq_diff',  1),
    ('roa',                  '总资产收益率',             'qoq_diff',  1),
    ('grossprofitmargin',    '销售毛利率',               'qoq_diff',  1),
    ('netprofitincl',        '净利润同比增长率增速',     'yoy_accel', 1),
    ('totprofit',            '利润总额同比增长率增速',   'yoy_accel', 1),
    ('netprofitexcl',        '归母净利润同比增长率增速', 'yoy_accel', 1),
    ('operrev',              '营业收入同比增长率增速',   'yoy_accel', 1),
    ('operatecaptialturn',   '营运资本周转率',           'qoq_diff',  1),
    ('invturn',              '存���周转率',               'qoq_diff',  1),
    ('assetsturn',           '总资产周转率',             'qoq_diff',  1),
    ('caturn',               '流动资产周转率',           'qoq_diff',  1),
    ('current',              '流动比率',                 'yoy_diff', -1),
    ('quick',                '速动比率',                 'yoy_diff', -1),
    ('debttoequity',         '净资产负债率',             'yoy_diff',  1),
    ('ebittointerest',       '已获利息倍数',             'qoq_diff',  1),
    ('debttoassets',         '资产负债率',               'yoy_diff',  1),
    ('opercash',             '经营现金流同比增长率增速', 'yoy_accel', 1),
    ('invcash',              '投资现金流同比增长率增速', 'yoy_accel', 1),
    ('netprofitcashcover',   '净利润现金含量',           'qoq_diff',  1),
]

# 利润表和现金流量表的累计值字段（需要TTM处理）
CUMULATIVE_COLS = [
    'revenue', 'total_cogs', 'oper_cost', 'n_income',
    'n_income_attr_p', 'total_profit', 'ebit', 'int_exp',
    'n_cashflow_act', 'n_cashflow_inv_act',
]

# 资产负债表的时点值字段（不需要TTM处理）
POINT_COLS = [
    'total_assets', 'total_hldr_eqy_exc_min_int', 'total_liab',
    'total_cur_assets', 'total_cur_liab', 'inventories',
]

# yoy_accel 类指标用到的绝对值列映射
ABS_VALUE_MAP = {
    'netprofitincl': 'n_income',
    'totprofit': 'total_profit',
    'netprofitexcl': 'n_income_attr_p',
    'operrev': 'revenue',
    'opercash': 'n_cashflow_act',
    'invcash': 'n_cashflow_inv_act',
}


# ============================================================
#  Step 1: 加载原始数据 + 行业映射
# ============================================================

def _load_members(level='l1'):
    """加载行业成分股映射，返回 {ts_code: (行业code, 行业name)}

    level='l1' → ts_sw_members.csv，行业=申万一级（排除金融）
    level='l2' → ts_sw_l2_members.csv，行业=申万二级（124 个，含金融子行业）
                 个股按"当前成分(out_date 空)优先、否则 in_date 最新"取唯一行业归属。
    返回字典的"行业 code"在 l2 模式下是 l2_code 值，但下游一律写入列名 `l1_code`
    作为占位 —— 景气度链条逻辑因此无需改动。
    """
    if level == 'l1':
        path = os.path.join(LOCAL_DATA_RAW, 'ts_sw_members.csv')
        members = pd.read_csv(path)
        members = members[~members['l1_code'].isin(SW_EXCLUDE)]
        ind_col, name_col = 'l1_code', 'l1_name'
    else:
        path = os.path.join(LOCAL_DATA_RAW, 'ts_sw_l2_members.csv')
        members = pd.read_csv(path)
        # 一只股票横跨多个二级（再入/重分类）时取唯一归属：当前成分优先、再按 in_date 最新
        members['_cur'] = members['out_date'].isna()
        members = members.sort_values(['ts_code', '_cur', 'in_date'])
        members = members.drop_duplicates(subset='ts_code', keep='last')
        ind_col, name_col = 'l2_code', 'l2_name'
    mapping = {}
    for _, row in members.iterrows():
        mapping[row['ts_code']] = (row[ind_col], row.get(name_col, ''))
    return mapping


def _clean_financial(df, name):
    """清洗财务报表：合并报表 + 一般工商业 + 去重"""
    print(f'  清洗 {name}: {len(df)} 行', end='')
    df['report_type'] = df['report_type'].astype(str).str.strip()
    df = df[df['report_type'] == '1'].copy()
    df['comp_type'] = df['comp_type'].astype(str).str.strip()
    df = df[df['comp_type'] == '1'].copy()
    df['ann_date'] = df['ann_date'].fillna(df['f_ann_date'])
    df = df.sort_values(['ts_code', 'end_date', 'ann_date'],
                        ascending=[True, True, False])
    df = df.drop_duplicates(subset=['ts_code', 'end_date'], keep='first')
    print(f' → {len(df)} 行')
    return df


def _add_industry(df, mapping):
    """给 DataFrame 添加行业标签"""
    df['l1_code'] = df['ts_code'].map(lambda x: mapping.get(x, (None,))[0])
    df['l1_name'] = df['ts_code'].map(lambda x: mapping.get(x, (None, ''))[1]
                                       if x in mapping else None)
    return df.dropna(subset=['l1_code'])


# ============================================================
#  Step 2: 行业汇总
# ============================================================

def _aggregate_industry(income, balance, cashflow):
    """按行���汇总绝对值数据（三张财务报表）"""
    print('\n  行业汇总...')
    inc_cols = [c for c in ['revenue', 'total_cogs', 'oper_cost', 'n_income',
                            'n_income_attr_p', 'total_profit', 'ebit', 'int_exp']
                if c in income.columns]
    inc_agg = income.groupby(['l1_code', 'l1_name', 'end_date'])[inc_cols].sum().reset_index()

    bal_cols = [c for c in POINT_COLS if c in balance.columns]
    bal_agg = balance.groupby(['l1_code', 'l1_name', 'end_date'])[bal_cols].sum().reset_index()

    cf_cols = [c for c in ['n_cashflow_act', 'n_cashflow_inv_act'] if c in cashflow.columns]
    cf_agg = cashflow.groupby(['l1_code', 'l1_name', 'end_date'])[cf_cols].sum().reset_index()

    result = inc_agg.merge(bal_agg, on=['l1_code', 'l1_name', 'end_date'], how='outer')
    result = result.merge(cf_agg, on=['l1_code', 'l1_name', 'end_date'], how='outer')
    result = result.sort_values(['l1_code', 'end_date']).reset_index(drop=True)
    print(f'    汇总: {len(result)} 行, {result["l1_code"].nunique()} 个行业')
    return result


# ============================================================
#  Step 3: TTM 平滑
# ============================================================

def _ttm_transform(industry_data):
    """利润表/现金流量表做 TTM（滚动12个月），资产负债表保持时点值"""
    print('\n  TTM 平滑...')
    df = industry_data.copy()
    df['end_date'] = df['end_date'].astype(str)
    df['year'] = df['end_date'].str[:4].astype(int)
    df['quarter'] = df['end_date'].str[4:6].astype(int).map({3: 1, 6: 2, 9: 3, 12: 4})
    df = df.sort_values(['l1_code', 'year', 'quarter']).reset_index(drop=True)

    results = []
    for l1_code, group in df.groupby('l1_code'):
        group = group.sort_values(['year', 'quarter']).reset_index(drop=True)
        for idx, row in group.iterrows():
            r = {c: row[c] for c in ['l1_code', 'l1_name', 'end_date', 'year', 'quarter']}
            for col in POINT_COLS:
                if col in row.index:
                    r[col] = row[col]

            if row['quarter'] == 4:
                for col in CUMULATIVE_COLS:
                    if col in row.index:
                        r[col] = row[col]
            else:
                prev_q = group[(group['year'] == row['year'] - 1) &
                               (group['quarter'] == row['quarter'])]
                prev_y = group[(group['year'] == row['year'] - 1) &
                               (group['quarter'] == 4)]
                if not prev_q.empty and not prev_y.empty:
                    ps, py = prev_q.iloc[0], prev_y.iloc[0]
                    for col in CUMULATIVE_COLS:
                        if col in row.index:
                            cv = row[col] if pd.notna(row[col]) else 0
                            pv = ps[col] if pd.notna(ps[col]) else 0
                            av = py[col] if pd.notna(py[col]) else 0
                            r[col] = cv - pv + av
                else:
                    scale = {1: 4, 2: 2, 3: 4/3}[row['quarter']]
                    for col in CUMULATIVE_COLS:
                        if col in row.index:
                            r[col] = row[col] * scale if pd.notna(row[col]) else np.nan
            results.append(r)

    ttm = pd.DataFrame(results)
    print(f'    TTM: {len(ttm)} 行')
    return ttm


# ============================================================
#  Step 4: 计算比率型指标
# ============================================================

def _calc_ratios(df):
    """基于 TTM 后的绝对值数据计算比率型指标"""
    d = df.copy()
    d['nptocostexpense'] = d['n_income'] / d['total_cogs'].replace(0, np.nan)
    d['netprofitmargin'] = d['n_income'] / d['revenue'].replace(0, np.nan)
    d['roe'] = d['n_income'] / d['total_hldr_eqy_exc_min_int'].replace(0, np.nan)
    d['roa'] = d['n_income'] / d['total_assets'].replace(0, np.nan)
    d['grossprofitmargin'] = (d['revenue'] - d['oper_cost']) / d['revenue'].replace(0, np.nan)

    wc = d['total_cur_assets'] - d['total_cur_liab']
    d['operatecaptialturn'] = d['revenue'] / wc.replace(0, np.nan)
    d['invturn'] = d['oper_cost'] / d['inventories'].replace(0, np.nan)
    d['assetsturn'] = d['revenue'] / d['total_assets'].replace(0, np.nan)
    d['caturn'] = d['revenue'] / d['total_cur_assets'].replace(0, np.nan)

    d['current'] = d['total_cur_assets'] / d['total_cur_liab'].replace(0, np.nan)
    d['quick'] = (d['total_cur_assets'] - d['inventories']) / d['total_cur_liab'].replace(0, np.nan)
    d['debttoequity'] = d['total_liab'] / d['total_hldr_eqy_exc_min_int'].replace(0, np.nan)
    d['ebittointerest'] = d['ebit'] / d['int_exp'].replace(0, np.nan)
    d['debttoassets'] = d['total_liab'] / d['total_assets'].replace(0, np.nan)
    d['netprofitcashcover'] = d['n_cashflow_act'] / d['n_income'].replace(0, np.nan)

    # 1%~99% winsorize
    ratio_cols = [
        'nptocostexpense', 'netprofitmargin', 'roe', 'roa', 'grossprofitmargin',
        'operatecaptialturn', 'invturn', 'assetsturn', 'caturn',
        'current', 'quick', 'debttoequity', 'ebittointerest',
        'debttoassets', 'netprofitcashcover',
    ]
    for col in ratio_cols:
        lo, hi = d[col].quantile(0.01), d[col].quantile(0.99)
        d[col] = d[col].clip(lo, hi)
    return d


# ============================================================
#  Step 5: 构建 21 个基础景气度指标
# ============================================================

def _build_prosperity(ttm_df):
    """构建 21 个景气度指标（qoq_diff / yoy_diff / yoy_accel）"""
    print('\n  构建景气度指标...')
    df = ttm_df.sort_values(['l1_code', 'end_date']).reset_index(drop=True)
    base = ['l1_code', 'l1_name', 'end_date', 'year', 'quarter']
    result = df[base].drop_duplicates().copy()

    for code, name, method, direction in BASIC_INDICATORS:
        vals = []
        for l1, grp in df.groupby('l1_code'):
            grp = grp.sort_values('end_date').reset_index(drop=True)
            if method == 'qoq_diff':
                s = grp[code].diff(1)
            elif method == 'yoy_diff':
                s = grp[code].diff(4)
            elif method == 'yoy_accel':
                abs_col = ABS_VALUE_MAP.get(code)
                if abs_col and abs_col in grp.columns:
                    shifted = grp[abs_col].shift(4)
                    yoy = (grp[abs_col] - shifted) / shifted.abs().replace(0, np.nan)
                    s = yoy.diff(1)
                else:
                    s = pd.Series(np.nan, index=grp.index)
            else:
                s = pd.Series(np.nan, index=grp.index)

            for i, v in s.items():
                vals.append({**{c: grp.loc[i, c] for c in base}, code: v})

        ind_df = pd.DataFrame(vals).drop_duplicates(subset=base, keep='last')
        result = result.merge(ind_df[base + [code]], on=base, how='left')

    # 后向填充
    indicator_cols = [ind[0] for ind in BASIC_INDICATORS]
    for _, gidx in result.groupby('l1_code').groups.items():
        grp = result.loc[gidx].sort_values('end_date')
        for col in indicator_cols:
            result.loc[grp.index, col] = grp[col].bfill()

    result = result.sort_values(['l1_code', 'end_date']).reset_index(drop=True)
    print(f'    景气度: {result.shape}')
    return result


# ============================================================
#  Step 6: 一致预期指标
# ============================================================

def _integrate_consensus(prosperity_df, members_map):
    """加工 raw_report_rc → 同比增速/复合增长率 → 合并到景气度宽表"""
    rc_path = os.path.join(LOCAL_DATA_RAW, 'raw_report_rc.pkl')
    if not os.path.exists(rc_path):
        print('\n  [一致预期] raw_report_rc.pkl 不存在，跳过')
        return prosperity_df

    print('\n  整合一致预期...')
    rc = pd.read_pickle(rc_path)
    rc = rc[rc['quarter'].astype(str).str.match(r'^\d{4}Q\d$', na=False)].copy()
    rc['forecast_year'] = rc['quarter'].str[:4].astype(int)
    rc['report_month'] = rc['report_date'].astype(str).str[:6]

    value_cols = ['op_rt', 'op_pr', 'tp', 'np', 'eps', 'roe', 'rd']
    # 个股一致预期
    stock_con = rc.groupby(['ts_code', 'forecast_year', 'report_month'])[value_cols].median().reset_index()
    stock_con['l1_code'] = stock_con['ts_code'].map(lambda x: members_map.get(x, (None,))[0])
    stock_con['l1_name'] = stock_con['ts_code'].map(
        lambda x: members_map.get(x, (None, ''))[1] if x in members_map else None)
    stock_con = stock_con.dropna(subset=['l1_code'])

    # 行业汇总
    abs_cols = ['op_rt', 'op_pr', 'tp', 'np']
    ratio_cols = ['eps', 'roe', 'rd']
    ind_abs = stock_con.groupby(['l1_code', 'l1_name', 'forecast_year', 'report_month'])[abs_cols].sum().reset_index()
    ind_rat = stock_con.groupby(['l1_code', 'l1_name', 'forecast_year', 'report_month'])[ratio_cols].median().reset_index()
    ind_con = ind_abs.merge(ind_rat, on=['l1_code', 'l1_name', 'forecast_year', 'report_month'], how='outer')

    # BPS 推算（roe=0 时分母置 NaN —— np.where 不短路，裸除会触发 ZeroDivisionError）
    _roe_safe = ind_con['roe'].replace(0, np.nan)
    ind_con['bps'] = np.where(ind_con['roe'].abs() > 0.001,
                               ind_con['eps'] / (_roe_safe / 100), np.nan)

    # 同比增速 (yoy)
    calc_cols = ['np', 'tp', 'op_pr', 'op_rt', 'eps', 'roe', 'bps']
    ind_con = ind_con.sort_values(['l1_code', 'report_month', 'forecast_year'])

    yoy_recs = []
    for (l1, rm), g in ind_con.groupby(['l1_code', 'report_month']):
        g = g.sort_values('forecast_year')
        for i in range(1, len(g)):
            curr, prev = g.iloc[i], g.iloc[i-1]
            if curr['forecast_year'] - prev['forecast_year'] != 1:
                continue
            rec = {'l1_code': l1, 'l1_name': curr['l1_name'], 'report_month': rm,
                   'forecast_year': int(curr['forecast_year'])}
            for c in calc_cols:
                pv, cv = prev[c], curr[c]
                if pd.notna(pv) and pd.notna(cv) and abs(pv) > 1e-8:
                    rec[f'{c}_yoy'] = (cv - pv) / abs(pv)
                else:
                    rec[f'{c}_yoy'] = np.nan
            yoy_recs.append(rec)
    yoy_df = pd.DataFrame(yoy_recs) if yoy_recs else pd.DataFrame()

    # 复合增长率 (cagh)
    cagh_recs = []
    for (l1, rm), g in ind_con.groupby(['l1_code', 'report_month']):
        g = g.sort_values('forecast_year')
        if len(g) < 2:
            continue
        first, last = g.iloc[0], g.iloc[-1]
        ny = last['forecast_year'] - first['forecast_year']
        if ny < 1:
            continue
        rec = {'l1_code': l1, 'l1_name': first['l1_name'], 'report_month': rm}
        for c in calc_cols:
            fv, lv = first[c], last[c]
            if pd.notna(fv) and pd.notna(lv) and fv > 0 and lv > 0:
                rec[f'{c}_cagh'] = (lv / fv) ** (1.0 / ny) - 1
            else:
                rec[f'{c}_cagh'] = np.nan
        cagh_recs.append(rec)
    cagh_df = pd.DataFrame(cagh_recs) if cagh_recs else pd.DataFrame()

    # 合并
    if not yoy_df.empty and not cagh_df.empty:
        consensus = yoy_df.merge(cagh_df, on=['l1_code', 'l1_name', 'report_month'], how='outer')
    elif not yoy_df.empty:
        consensus = yoy_df
    elif not cagh_df.empty:
        consensus = cagh_df
    else:
        print('    无一致预期数据')
        return prosperity_df

    # 映射到季度
    def _to_quarter_end(rm):
        y, m = rm[:4], int(rm[4:6])
        if m <= 3: return f'{y}0331'
        elif m <= 6: return f'{y}0630'
        elif m <= 9: return f'{y}0930'
        else: return f'{y}1231'

    consensus['end_date'] = consensus['report_month'].apply(_to_quarter_end)
    consensus = consensus.sort_values(['l1_code', 'end_date', 'report_month'])
    consensus = consensus.drop_duplicates(subset=['l1_code', 'end_date'], keep='last')

    con_cols = [c for c in consensus.columns if c.endswith('_yoy') or c.endswith('_cagh')]
    rename = {c: f'con_{c}' for c in con_cols if not c.startswith('con_')}
    consensus = consensus.rename(columns=rename)
    con_cols = [rename.get(c, c) for c in con_cols]

    result = prosperity_df.merge(
        consensus[['l1_code', 'end_date'] + con_cols],
        on=['l1_code', 'end_date'], how='left')

    # 后向填充
    for _, gidx in result.groupby('l1_code').groups.items():
        grp = result.loc[gidx].sort_values('end_date')
        for col in con_cols:
            if col in result.columns:
                result.loc[grp.index, col] = grp[col].bfill()

    print(f'    一致预期列: {len(con_cols)} 个, 合并后 shape={result.shape}')
    return result


# ============================================================
#  Step 7: 极端值清洗
# ============================================================

def _clean_outliers(df):
    """极端值清洗（去后向填充假值 + winsorize）"""
    print('\n  极端值清洗...')
    basic_cols = [ind[0] for ind in BASIC_INDICATORS]
    con_cols = [c for c in df.columns if c.startswith('con_')]
    all_cols = basic_cols + con_cols

    # 修复后向填充的早期重复值
    fix_count = 0
    for _, gidx in df.groupby('l1_code').groups.items():
        grp = df.loc[gidx].sort_values('end_date')
        for col in all_cols:
            if col not in df.columns:
                continue
            vals = grp[col].values
            if len(vals) < 2 or pd.isna(vals[0]):
                continue
            first_val = vals[0]
            first_real = 0
            for i in range(1, len(vals)):
                if pd.isna(vals[i]) or vals[i] != first_val:
                    first_real = i
                    break
            else:
                continue
            if first_real > 1:
                idxs = grp.index[:first_real]
                df.loc[idxs, col] = np.nan
                fix_count += len(idxs)
    print(f'    清除 {fix_count} 个后向填充假值')

    # ebittointerest 截断
    if 'ebittointerest' in df.columns:
        df['ebittointerest'] = df['ebittointerest'].clip(-100, 100)

    # 按截面 2.5%~97.5% winsorize
    winsor_count = 0
    for _, gidx in df.groupby('end_date').groups.items():
        for col in all_cols:
            if col not in df.columns:
                continue
            vals = df.loc[gidx, col]
            valid = vals.dropna()
            if len(valid) < 5:
                continue
            lo, hi = valid.quantile(0.025), valid.quantile(0.975)
            clipped = vals.clip(lo, hi)
            winsor_count += (clipped != vals).sum()
            df.loc[gidx, col] = clipped
    print(f'    winsorize {winsor_count} 个值')

    return df


# ============================================================
#  主流程
# ============================================================

def run(dry_run=False, level='l1'):
    if level not in ('l1', 'l2'):
        raise ValueError(f"level 必须是 'l1' 或 'l2'，收到 {level!r}")
    out_suffix = '' if level == 'l1' else '_l2'

    print('=' * 60)
    print(f'  data_pipeline.prosperity  景气度指标全链路  level={level}')
    print('=' * 60)

    # 检查原始数据
    required = ['raw_income.pkl', 'raw_balancesheet.pkl', 'raw_cashflow.pkl']
    for f in required:
        p = os.path.join(LOCAL_DATA_RAW, f)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f'缺少 {p}\n请先运行: python -m data_pipeline.download --migrate')

    members_map = _load_members(level)
    print(f'  行业映射: {len(members_map)} 只股票')

    # 加载 + 清洗
    print('\n[1/7] 加载原始财务数据...')
    income = pd.read_pickle(os.path.join(LOCAL_DATA_RAW, 'raw_income.pkl'))
    balance = pd.read_pickle(os.path.join(LOCAL_DATA_RAW, 'raw_balancesheet.pkl'))
    cashflow = pd.read_pickle(os.path.join(LOCAL_DATA_RAW, 'raw_cashflow.pkl'))

    print('\n[2/7] 清洗...')
    income = _clean_financial(income, '利润表')
    balance = _clean_financial(balance, '资产负债表')
    cashflow = _clean_financial(cashflow, '现金流量表')

    income = _add_industry(income, members_map)
    balance = _add_industry(balance, members_map)
    cashflow = _add_industry(cashflow, members_map)

    print('\n[3/7] 行业汇总...')
    industry = _aggregate_industry(income, balance, cashflow)

    print('\n[4/7] TTM 平滑...')
    ttm = _ttm_transform(industry)
    ttm = _calc_ratios(ttm)

    print('\n[5/7] 景气度指标...')
    prosperity = _build_prosperity(ttm)

    print('\n[6/7] 一致预期...')
    prosperity = _integrate_consensus(prosperity, members_map)

    print('\n[7/7] 极端值清洗...')
    prosperity = _clean_outliers(prosperity)

    # 输出
    out_path = os.path.join(LOCAL_DATA_PROCESSED,
                            f'prosperity_indicators_clean{out_suffix}.pkl')
    if dry_run:
        print(f'\n[dry-run] 未写入 {out_path}')
    else:
        os.makedirs(LOCAL_DATA_PROCESSED, exist_ok=True)
        prosperity.to_pickle(out_path)
        print(f'\n写入 {out_path}')

    # 概览
    basic_cols = [ind[0] for ind in BASIC_INDICATORS]
    con_cols = [c for c in prosperity.columns if c.startswith('con_')]
    print(f'\nshape: {prosperity.shape}')
    print(f'行业数: {prosperity["l1_code"].nunique()}')
    print(f'时间范围: {prosperity["end_date"].min()} ~ {prosperity["end_date"].max()}')
    print(f'\n指标覆盖率:')
    for col in basic_cols + con_cols:
        if col in prosperity.columns:
            pct = prosperity[col].notna().mean()
            print(f'  {col:30s} {pct:5.1%}')

    return {
        'rows': len(prosperity),
        'industries': int(prosperity['l1_code'].nunique()),
        'date_min': str(prosperity['end_date'].min()),
        'date_max': str(prosperity['end_date'].max()),
    }


def main():
    parser = argparse.ArgumentParser(description='景气度指标全链路')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--level', default='l1', choices=['l1', 'l2'],
                        help='行业聚合粒度：l1=申万一级（默认），l2=申万二级（方向2）')
    args = parser.parse_args()
    run(dry_run=args.dry_run, level=args.level)


if __name__ == '__main__':
    main()
