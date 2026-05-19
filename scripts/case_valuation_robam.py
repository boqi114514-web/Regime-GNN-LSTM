# -*- coding: utf-8 -*-
"""老板电器（002508.SZ）企业价值评估案例

对应教材章节（McKinsey/Koller 框架）：
  Ch3  风险与资本成本   -> WACC
  Ch6  增长             -> g 分析（ROIC × IR）
  Ch8  重组财务报表     -> NOPAT / Invested Capital
  Ch9  绩效分析         -> ROIC 拆解 + 行业对比
  Ch10 业绩预测         -> 10 年显性 + 永续增长 + DCF

输出：results/case_robam_valuation.xlsx
"""
import os
import warnings
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

warnings.filterwarnings('ignore')

# ============================ 配置 ============================
PROJECT = r'D:\desktop\有意思的事情\量化\项目\Regime-GNN-LSTM'
RAW = os.path.join(PROJECT, 'data', 'raw')
RESULTS = os.path.join(PROJECT, 'results')
OUTPUT = os.path.join(RESULTS, 'case_robam_valuation.xlsx')

MAIN_TS = '002508.SZ'
MAIN_NAME = '老板电器'
PEERS = {
    '002508.SZ': '老板电器',
    '002032.SZ': '苏泊尔',
    '002035.SZ': '华帝股份',
    '002242.SZ': '九阳股份',
    '603868.SH': '飞科电器',
}

# CAPM / WACC 参数
RF = 0.025
MRP = 0.05
STATUTORY_TAX = 0.25

# 预测期参数
N_FORECAST = 10
TERMINAL_G = 0.03

# ============================ 数据加载 ============================
def load_financials():
    inc = pd.read_pickle(os.path.join(RAW, 'raw_income.pkl'))
    bs  = pd.read_pickle(os.path.join(RAW, 'raw_balancesheet.pkl'))
    cf  = pd.read_pickle(os.path.join(RAW, 'raw_cashflow.pkl'))

    def filter_annual(df):
        df = df[df['ts_code'].isin(PEERS)].copy()
        df['end_date'] = df['end_date'].astype(str)
        df = df[df['end_date'].str.endswith('1231')]
        df = df[df['report_type'].astype(str) == '1']
        df['year'] = df['end_date'].str[:4].astype(int)
        df = df.drop_duplicates(['ts_code', 'year'], keep='last')
        return df.sort_values(['ts_code', 'year']).reset_index(drop=True)

    return filter_annual(inc), filter_annual(bs), filter_annual(cf)

def load_prices():
    """加载日线行情（老板电器 + 沪深300）."""
    import pickle
    path = os.path.join(RAW, 'stock_daily.pkl')
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    df = raw['df_stock'] if isinstance(raw, dict) and 'df_stock' in raw else raw
    df = pd.DataFrame(df)
    df['date'] = pd.to_datetime(df['date'])

    # 主体行情
    code_col = 'code' if 'code' in df.columns else 'ts_code'
    # 个股的 code 通常是 '002508.SZ'
    robam = df[df[code_col].astype(str).str.startswith('002508')].copy()
    robam = robam[['date', 'close']].sort_values('date').reset_index(drop=True)

    # 沪深300 用 monthly csv
    idx = pd.read_csv(os.path.join(RAW, 'ts_csi300_monthly.csv'))
    idx['date'] = pd.to_datetime(idx['date'].astype(str), format='%Y%m%d')
    idx = idx[idx['ts_code'] == '000300.SH'][['date', 'close']].sort_values('date').reset_index(drop=True)

    return robam, idx

# ============================ 指标计算 ============================
def build_metrics(inc, bs, cf):
    df = inc.merge(
        bs.drop(columns=['ann_date', 'f_ann_date', 'end_date',
                         'report_type', 'comp_type'], errors='ignore'),
        on=['ts_code', 'year'])
    df = df.merge(
        cf[['ts_code', 'year', 'n_cashflow_act', 'n_cashflow_inv_act']],
        on=['ts_code', 'year'], how='left')

    df = df.rename(columns={'total_hldr_eqy_exc_min_int': 'equity'})

    # 利润链
    df['gross_profit'] = df['revenue'] - df['total_cogs']
    df['gross_margin'] = df['gross_profit'] / df['revenue']
    df['ebit_margin'] = df['ebit'] / df['revenue']
    df['net_margin']  = df['n_income'] / df['revenue']

    eff = (df['total_profit'] - df['n_income']) / df['total_profit']
    df['eff_tax'] = eff.clip(0, 0.45).fillna(STATUTORY_TAX)
    df['nopat'] = df['ebit'] * (1 - df['eff_tax'])

    # 投入资本（简化：总资产 - 流动负债）
    df['ic']  = df['total_assets'] - df['total_cur_liab']
    df['nwc'] = df['total_cur_assets'] - df['total_cur_liab']

    df = df.sort_values(['ts_code', 'year']).reset_index(drop=True)
    df['ic_prev']  = df.groupby('ts_code')['ic'].shift(1)
    df['avg_ic']   = (df['ic'] + df['ic_prev']) / 2
    df['roic']     = df['nopat'] / df['avg_ic']
    df['cap_turn'] = df['revenue'] / df['avg_ic']

    df['rev_g']    = df.groupby('ts_code')['revenue'].pct_change()
    df['nopat_g']  = df.groupby('ts_code')['nopat'].pct_change()
    df['ic_g']     = df.groupby('ts_code')['ic'].pct_change()
    df['delta_ic'] = df['ic'] - df['ic_prev']
    df['ir']       = df['delta_ic'] / df['nopat']
    df['g_impl']   = df['roic'] * df['ir']

    df['asset_turn'] = df['revenue'] / df['total_assets']
    df['d_to_e']     = df['total_liab'] / df['equity']
    df['ocf_to_rev'] = df['n_cashflow_act'] / df['revenue']
    df['capex']      = -df['n_cashflow_inv_act']   # 投资CF为流出（负）
    df['capex_to_rev'] = df['capex'] / df['revenue']
    df['fcf']        = df['n_cashflow_act'] - df['capex']
    df['fcf_to_rev'] = df['fcf'] / df['revenue']

    df['name'] = df['ts_code'].map(PEERS)
    return df

# ============================ WACC ============================
def compute_wacc(metrics, robam_prices, idx_prices):
    # β：用月度收益对 CSI300 月度收益回归（近 5 年）
    rp = robam_prices.copy()
    rp = rp.set_index('date').resample('ME').last().dropna()
    rp['ret'] = rp['close'].pct_change()

    ip = idx_prices.copy().set_index('date')
    ip['ret_mkt'] = ip['close'].pct_change()

    merged = rp[['ret']].join(ip[['ret_mkt']], how='inner').dropna()
    merged = merged.tail(60)  # 近 5 年（60个月）
    if len(merged) >= 24:
        cov = merged.cov().iloc[0, 1]
        var_m = merged['ret_mkt'].var()
        beta = cov / var_m
        beta_window = f'{merged.index.min():%Y-%m} ~ {merged.index.max():%Y-%m}（{len(merged)} 月）'
    else:
        beta = 1.0
        beta_window = '回归窗口不足，取 β=1.0'

    ke = RF + beta * MRP

    # 债务成本：近5年利息支出 / 平均总负债
    robam_fin = metrics[metrics['ts_code'] == MAIN_TS].sort_values('year')
    recent = robam_fin.tail(5)
    avg_int = recent['int_exp'].abs().mean()
    avg_debt = recent['total_liab'].mean()
    kd_pre = (avg_int / avg_debt) if (avg_debt and avg_debt > 0) else 0.04
    if pd.isna(kd_pre) or kd_pre <= 0:
        kd_pre = 0.04
    kd_after = kd_pre * (1 - STATUTORY_TAX)

    # 权重：账面
    last = robam_fin.iloc[-1]
    E = float(last['equity'])
    D = float(last['total_liab'])
    we, wd = E / (E + D), D / (E + D)
    wacc = we * ke + wd * kd_after

    return {
        'beta': beta, 'beta_window': beta_window,
        'rf': RF, 'mrp': MRP, 'ke': ke,
        'avg_int': avg_int, 'avg_debt': avg_debt,
        'kd_pre': kd_pre, 'kd_after': kd_after,
        'E': E, 'D': D, 'we': we, 'wd': wd, 'wacc': wacc,
    }

# ============================ 预测 + DCF ============================
def forecast_and_dcf(metrics, wacc_info):
    """10 年显性预测 + DCF."""
    df = metrics[metrics['ts_code'] == MAIN_TS].sort_values('year').copy()
    last = df.iloc[-1]
    base_year = int(last['year'])

    # 历史均值（用于设定假设）
    hist5 = df.tail(5)
    avg_ebit_margin = float(hist5['ebit_margin'].mean())
    avg_capex_ratio = float(hist5['capex_to_rev'].mean())
    if pd.isna(avg_capex_ratio):
        avg_capex_ratio = 0.03
    avg_ic_to_rev = float(hist5['ic'].iloc[-1] / hist5['revenue'].iloc[-1])
    hist5_revg = df.tail(6)['rev_g'].iloc[1:].mean()
    if pd.isna(hist5_revg):
        hist5_revg = 0.05

    # 起始增速：近5年均值 与 8% 取较小者（避免过分外推）
    g_start = min(max(hist5_revg, TERMINAL_G), 0.08)

    years = list(range(base_year + 1, base_year + 1 + N_FORECAST))
    # 增速线性衰减到 TERMINAL_G
    growths = np.linspace(g_start, TERMINAL_G, N_FORECAST)
    # EBIT margin 线性回到长期均值（向 hist5 均值靠拢，保留近年水平）
    margins = np.full(N_FORECAST, avg_ebit_margin)

    rev_prev = float(last['revenue'])
    ic_prev = float(last['ic'])
    rows = []
    wacc = wacc_info['wacc']

    pv_fcf = 0.0
    for i, (yr, g, m) in enumerate(zip(years, growths, margins)):
        rev = rev_prev * (1 + g)
        ebit = rev * m
        nopat = ebit * (1 - STATUTORY_TAX)
        ic = rev * avg_ic_to_rev
        delta_ic = ic - ic_prev
        fcf = nopat - delta_ic
        disc = (1 + wacc) ** (i + 1)
        pv = fcf / disc
        pv_fcf += pv
        rows.append({
            'year': yr, 'rev': rev, 'rev_g': g,
            'ebit_margin': m, 'ebit': ebit, 'tax_rate': STATUTORY_TAX,
            'nopat': nopat, 'ic': ic, 'delta_ic': delta_ic,
            'fcf': fcf, 'disc_factor': 1 / disc, 'pv_fcf': pv,
        })
        rev_prev, ic_prev = rev, ic

    # 永续价值
    last_fcf = rows[-1]['fcf']
    tv_fcf_next = last_fcf * (1 + TERMINAL_G)
    tv = tv_fcf_next / (wacc - TERMINAL_G)
    pv_tv = tv / ((1 + wacc) ** N_FORECAST)
    enterprise_value = pv_fcf + pv_tv

    # 股权价值 = EV - 净负债（用账面有息负债近似为总负债，无现金扣减信息）
    equity_value = enterprise_value - wacc_info['D']

    return {
        'assumptions': {
            'base_year': base_year,
            'g_start': g_start, 'g_terminal': TERMINAL_G,
            'avg_ebit_margin': avg_ebit_margin,
            'avg_capex_ratio': avg_capex_ratio,
            'avg_ic_to_rev': avg_ic_to_rev,
            'hist5_revg': hist5_revg,
            'tax_rate': STATUTORY_TAX,
            'wacc': wacc,
            'n_forecast': N_FORECAST,
        },
        'rows': rows,
        'pv_fcf': pv_fcf,
        'tv': tv, 'pv_tv': pv_tv,
        'enterprise_value': enterprise_value,
        'equity_value': equity_value,
    }

# ============================ Excel 写出 ============================
HEADER_FILL = PatternFill('solid', fgColor='1F4E78')
HEADER_FONT = Font(bold=True, color='FFFFFF', size=11)
SUBHEADER_FILL = PatternFill('solid', fgColor='D9E1F2')
SUBHEADER_FONT = Font(bold=True, color='1F4E78')
ROBAM_FILL = PatternFill('solid', fgColor='FFF2CC')
BORDER = Border(left=Side(style='thin', color='BFBFBF'),
                right=Side(style='thin', color='BFBFBF'),
                top=Side(style='thin', color='BFBFBF'),
                bottom=Side(style='thin', color='BFBFBF'))

def style_header(ws, row_idx, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row_idx, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = BORDER

def write_df(ws, df, start_row=1, pct_cols=None, num_cols=None,
             highlight_label=None, label_col=1):
    """将 DataFrame 写入工作表。pct_cols/num_cols 是 0-索引的列号集合."""
    pct_cols = pct_cols or set()
    num_cols = num_cols or set()
    ncols = df.shape[1]
    # 表头
    for j, col in enumerate(df.columns):
        ws.cell(row=start_row, column=j + 1, value=str(col))
    style_header(ws, start_row, ncols)
    # 数据
    for i, (_, row) in enumerate(df.iterrows()):
        r = start_row + 1 + i
        is_highlight = (highlight_label is not None and
                        str(row.iloc[label_col - 1]) == highlight_label)
        for j, v in enumerate(row.values):
            cell = ws.cell(row=r, column=j + 1)
            if pd.isna(v):
                cell.value = None
            elif isinstance(v, (int, np.integer)):
                cell.value = int(v)
            elif isinstance(v, (float, np.floating)):
                cell.value = float(v)
            else:
                cell.value = v
            if j in pct_cols:
                cell.number_format = '0.00%'
            elif j in num_cols:
                cell.number_format = '#,##0'
            if is_highlight:
                cell.fill = ROBAM_FILL
            cell.border = BORDER
    # 列宽
    for j, col in enumerate(df.columns):
        max_len = max(len(str(col)), 12)
        ws.column_dimensions[get_column_letter(j + 1)].width = max_len + 2
    return start_row + 1 + len(df)

def write_notes(ws, lines, start_row=1):
    for i, line in enumerate(lines):
        cell = ws.cell(row=start_row + i, column=1, value=line)
        if line.startswith('## '):
            cell.font = Font(bold=True, size=13, color='1F4E78')
        elif line.startswith('# '):
            cell.font = Font(bold=True, size=15, color='1F4E78')
        else:
            cell.font = Font(size=11)
        cell.alignment = Alignment(wrap_text=True, vertical='top')
    ws.column_dimensions['A'].width = 100

# ============================ 主流程 ============================
def main():
    os.makedirs(RESULTS, exist_ok=True)
    print('[1/5] 加载数据...')
    inc, bs, cf = load_financials()
    robam_prices, idx_prices = load_prices()
    print(f'    财报：{len(inc)} 行 income / {len(bs)} 行 BS / {len(cf)} 行 CF')
    print(f'    行情：老板电器 {len(robam_prices)} 行日线，CSI300 {len(idx_prices)} 月度')

    print('[2/5] 计算指标...')
    metrics = build_metrics(inc, bs, cf)
    print(f'    指标表：{len(metrics)} 行（{len(PEERS)} 家公司 × ~13 年）')

    print('[3/5] 计算 WACC...')
    wacc_info = compute_wacc(metrics, robam_prices, idx_prices)
    print(f"    β={wacc_info['beta']:.3f}  ke={wacc_info['ke']:.2%}  "
          f"kd_after={wacc_info['kd_after']:.2%}  WACC={wacc_info['wacc']:.2%}")

    print('[4/5] 预测 + DCF...')
    dcf = forecast_and_dcf(metrics, wacc_info)
    print(f"    EV={dcf['enterprise_value']/1e8:.2f} 亿元，"
          f"股权价值={dcf['equity_value']/1e8:.2f} 亿元")

    print('[5/5] 写出 Excel...')
    write_excel(metrics, wacc_info, dcf)
    print(f'\n完成 → {OUTPUT}')

# ============================ Excel 各 Sheet ============================
def write_excel(metrics, wacc_info, dcf):
    wb = Workbook()
    wb.remove(wb.active)

    # ---------- Sheet 00 说明 ----------
    ws = wb.create_sheet('00_说明')
    notes = [
        '# 老板电器（002508.SZ）企业价值评估案例',
        '',
        '## 案例对象',
        '主体：老板电器（厨房电器龙头，2010 年上市）',
        '行业：申万二级 - 家用电器（白色家电/厨房电器）',
        '同业对标：苏泊尔 002032、华帝股份 002035、九阳股份 002242、飞科电器 603868',
        '',
        '## 方法学（McKinsey/Koller 框架）',
        '· Ch8 重组财务报表：把会计报表拆为"经营/非经营"两部分，得到 NOPAT 与 Invested Capital。',
        '· Ch9 绩效分析（ROIC）：ROIC = NOPAT / 平均 IC = 营业利润率 × 资本周转率。',
        '· Ch6 增长（g）：内生 g = ROIC × Investment Rate（IR = ΔIC/NOPAT）。',
        '· Ch3 风险与资本成本：CAPM 算 ke，财报近似 kd，账面权重加权得 WACC。',
        '· Ch10 业绩预测：10 年显性期 + 永续增长，按 capital intensity / margin 假设展开三表，DCF 估值。',
        '',
        '## 数据来源（全部本地）',
        '· 财报三表：data/raw/raw_income.pkl、raw_balancesheet.pkl、raw_cashflow.pkl',
        '· 行情：data/raw/stock_daily.pkl（老板电器日线）+ ts_csi300_monthly.csv（沪深300 月度）',
        '· 行业映射：data/raw/ts_sw_members.csv',
        '· 时间窗口：2012 - 2024 年报（现金流到 2023）',
        '',
        '## 关键简化与数据局限',
        '· IC = 总资产 - 流动负债（无"货币资金/有息负债"细分，未剥离超额现金；这是教学案例标准简化）。',
        '· CapEx 用"投资活动现金流净额"近似（包含部分非经营性投资）。',
        '· 没有 D&A 单列，预测期 FCF = NOPAT - ΔIC（McKinsey 推荐做法）。',
        '· 有效税率 = (利润总额 - 净利润) / 利润总额，限制在 [0, 45%]；预测期统一用法定 25%。',
        '· β 用近 60 个月（5 年）月度收益对沪深300 月度收益回归，未做行业 unlever/relever。',
        '',
        '## 主要发现（详见各 sheet）',
        f"· 老板电器近 5 年平均 ROIC ≈ {wacc_info.get('placeholder_roic', 'N/A')}（见 02_ROIC）。",
        f"· WACC = {wacc_info['wacc']:.2%}，β = {wacc_info['beta']:.3f}。",
        f"· 10 年 DCF 企业价值 ≈ {dcf['enterprise_value']/1e8:.1f} 亿元；股权价值 ≈ {dcf['equity_value']/1e8:.1f} 亿元。",
        '',
        '## Sheet 索引',
        '00_说明        本表',
        '01_报表重组    老板电器重组利润表 + 重组资产负债表（NOPAT / IC 链路）',
        '02_ROIC        老板 ROIC 时序 + DuPont 拆解 + 5 家同业对比',
        '03_增长g       收入/NOPAT/IC 增速 + 内生 g = ROIC × IR + 行业对比',
        '04_经营绩效    毛利/净利/周转/杠杆/FCF + 行业对比',
        '05_WACC        β 回归参数 + CAPM ke + kd + 加权 WACC',
        '06_预测假设    10 年显性期增速/margin/资本强度假设面板',
        '07_预测三表DCF 预测利润表 + IC + FCF + 永续 + DCF 估值',
        '',
        '生成日期：2026-05-12  ·  脚本：scripts/case_valuation_robam.py',
    ]
    write_notes(ws, notes)

    # ---------- Sheet 01 报表重组 ----------
    ws = wb.create_sheet('01_报表重组')
    rb = metrics[metrics['ts_code'] == MAIN_TS].sort_values('year').copy()
    # 重组利润表
    inc_view = pd.DataFrame({
        '年度': rb['year'],
        '营业收入': rb['revenue'],
        '营业成本': rb['total_cogs'],
        '毛利': rb['gross_profit'],
        '毛利率': rb['gross_margin'],
        'EBIT': rb['ebit'],
        'EBIT利润率': rb['ebit_margin'],
        '有效税率': rb['eff_tax'],
        'NOPAT': rb['nopat'],
        'NOPAT/营收': rb['nopat'] / rb['revenue'],
        '归母净利润': rb['n_income_attr_p'],
    })
    ws.cell(row=1, column=1, value='【1】重组利润表（单位：元）').font = Font(bold=True, size=13, color='1F4E78')
    end_row = write_df(ws, inc_view, start_row=2,
                       pct_cols={4, 6, 7, 9},
                       num_cols={1, 2, 3, 5, 8, 10})
    # 重组资产负债表
    bs_view = pd.DataFrame({
        '年度': rb['year'],
        '总资产': rb['total_assets'],
        '流动资产': rb['total_cur_assets'],
        '存货': rb['inventories'],
        '流动负债': rb['total_cur_liab'],
        '总负债': rb['total_liab'],
        '股东权益(归母)': rb['equity'],
        '净营运资本': rb['nwc'],
        '投入资本IC': rb['ic'],
        '平均IC': rb['avg_ic'],
        '资产负债率': rb['total_liab'] / rb['total_assets'],
    })
    ws.cell(row=end_row + 2, column=1,
            value='【2】重组资产负债表（IC = 总资产 - 流动负债；单位：元）').font = \
        Font(bold=True, size=13, color='1F4E78')
    write_df(ws, bs_view, start_row=end_row + 3,
             pct_cols={10},
             num_cols={1, 2, 3, 4, 5, 6, 7, 8, 9})

    # ---------- Sheet 02 ROIC ----------
    ws = wb.create_sheet('02_ROIC')
    # 老板电器时序
    roic_view = pd.DataFrame({
        '年度': rb['year'],
        'NOPAT': rb['nopat'],
        '平均IC': rb['avg_ic'],
        'ROIC': rb['roic'],
        'EBIT利润率': rb['ebit_margin'],
        '税后利润率(NOPAT/营收)': rb['nopat'] / rb['revenue'],
        '资本周转率(营收/平均IC)': rb['cap_turn'],
    })
    ws.cell(row=1, column=1, value='【1】老板电器 ROIC 时序与 DuPont 拆解').font = Font(bold=True, size=13, color='1F4E78')
    end_row = write_df(ws, roic_view, start_row=2,
                       pct_cols={3, 4, 5},
                       num_cols={1, 2})

    # 5 家同业 ROIC 对比（透视）
    pivot_roic = metrics.pivot_table(index='year', columns='name', values='roic')
    # 重排列：老板电器在第一列
    cols = [MAIN_NAME] + [c for c in pivot_roic.columns if c != MAIN_NAME]
    pivot_roic = pivot_roic.reindex(columns=cols).reset_index()
    pivot_roic.columns = ['年度'] + cols
    pivot_roic['行业中位'] = pivot_roic[cols].median(axis=1)
    pivot_roic['行业均值'] = pivot_roic[cols].mean(axis=1)
    ws.cell(row=end_row + 2, column=1,
            value='【2】5 家公司 ROIC 行业对比').font = Font(bold=True, size=13, color='1F4E78')
    write_df(ws, pivot_roic, start_row=end_row + 3,
             pct_cols=set(range(1, pivot_roic.shape[1])),
             num_cols={0})

    # ---------- Sheet 03 增长 g ----------
    ws = wb.create_sheet('03_增长g')
    growth_view = pd.DataFrame({
        '年度': rb['year'],
        '营收增速': rb['rev_g'],
        'NOPAT增速': rb['nopat_g'],
        'IC增速': rb['ic_g'],
        '投资率IR(ΔIC/NOPAT)': rb['ir'].clip(-3, 3),
        '内生增长 g=ROIC×IR': (rb['roic'] * rb['ir']).clip(-1, 1),
        'ROIC': rb['roic'],
    })
    ws.cell(row=1, column=1, value='【1】老板电器 增长与内生 g').font = Font(bold=True, size=13, color='1F4E78')
    end_row = write_df(ws, growth_view, start_row=2,
                       pct_cols={1, 2, 3, 5, 6},
                       num_cols={0})

    # 滚动复合增长率
    rev_series = rb.set_index('year')['revenue']
    def cagr(s, n):
        if len(s) < n + 1:
            return np.nan
        return (s.iloc[-1] / s.iloc[-n - 1]) ** (1 / n) - 1
    cagr_df = pd.DataFrame({
        '指标': ['营业收入 CAGR'],
        '近3年': [cagr(rev_series, 3)],
        '近5年': [cagr(rev_series, 5)],
        '近10年': [cagr(rev_series, 10)],
        f'上市以来({rb["year"].min()}~{rb["year"].max()})':
            [cagr(rev_series, len(rev_series) - 1)],
    })
    ws.cell(row=end_row + 2, column=1, value='【2】老板电器 营收 CAGR').font = Font(bold=True, size=13, color='1F4E78')
    end_row2 = write_df(ws, cagr_df, start_row=end_row + 3, pct_cols={1, 2, 3, 4})

    # 5 家公司年化增速对比
    growth_pivot = metrics.pivot_table(index='year', columns='name', values='rev_g')
    cols = [MAIN_NAME] + [c for c in growth_pivot.columns if c != MAIN_NAME]
    growth_pivot = growth_pivot.reindex(columns=cols).reset_index()
    growth_pivot.columns = ['年度'] + cols
    ws.cell(row=end_row2 + 2, column=1, value='【3】5 家公司 营收增速对比').font = Font(bold=True, size=13, color='1F4E78')
    write_df(ws, growth_pivot, start_row=end_row2 + 3,
             pct_cols=set(range(1, growth_pivot.shape[1])),
             num_cols={0})

    # ---------- Sheet 04 经营绩效 ----------
    ws = wb.create_sheet('04_经营绩效')
    perf_view = pd.DataFrame({
        '年度': rb['year'],
        '毛利率': rb['gross_margin'],
        'EBIT利润率': rb['ebit_margin'],
        '净利率': rb['net_margin'],
        '总资产周转': rb['asset_turn'],
        '资产负债率': rb['total_liab'] / rb['total_assets'],
        'D/E': rb['d_to_e'],
        'OCF/营收': rb['ocf_to_rev'],
        'CapEx/营收': rb['capex_to_rev'],
        'FCF/营收': rb['fcf_to_rev'],
    })
    ws.cell(row=1, column=1, value='【1】老板电器 综合经营绩效').font = Font(bold=True, size=13, color='1F4E78')
    end_row = write_df(ws, perf_view, start_row=2,
                       pct_cols={1, 2, 3, 5, 7, 8, 9},
                       num_cols={0})

    # 行业对比（最近一年）
    latest_year = metrics['year'].max()
    peer_perf = metrics[metrics['year'] == latest_year].copy()
    peer_view = pd.DataFrame({
        '公司': peer_perf['name'],
        '毛利率': peer_perf['gross_margin'],
        'EBIT利润率': peer_perf['ebit_margin'],
        '净利率': peer_perf['net_margin'],
        'ROIC': peer_perf['roic'],
        '总资产周转': peer_perf['asset_turn'],
        '资产负债率': peer_perf['total_liab'] / peer_perf['total_assets'],
        '营收(亿元)': peer_perf['revenue'] / 1e8,
        'NOPAT(亿元)': peer_perf['nopat'] / 1e8,
    })
    # 排序：老板电器置顶
    peer_view = peer_view.sort_values(
        '公司', key=lambda s: s.map({MAIN_NAME: 0}).fillna(1))
    ws.cell(row=end_row + 2, column=1,
            value=f'【2】行业对比（{latest_year} 年）').font = \
        Font(bold=True, size=13, color='1F4E78')
    write_df(ws, peer_view, start_row=end_row + 3,
             pct_cols={1, 2, 3, 4, 6},
             num_cols={7, 8},
             highlight_label=MAIN_NAME, label_col=1)

    # ---------- Sheet 05 WACC ----------
    ws = wb.create_sheet('05_WACC')
    rows = [
        ('一、CAPM 股权成本', '', ''),
        ('无风险利率 Rf', wacc_info['rf'], 'pct'),
        ('市场风险溢价 MRP', wacc_info['mrp'], 'pct'),
        ('Beta', wacc_info['beta'], 'num4'),
        ('β 回归窗口', wacc_info['beta_window'], 'str'),
        ('股权成本 ke = Rf + β × MRP', wacc_info['ke'], 'pct'),
        ('', '', ''),
        ('二、债务成本', '', ''),
        ('近5年平均利息支出（元）', wacc_info['avg_int'], 'num'),
        ('近5年平均总负债（元）', wacc_info['avg_debt'], 'num'),
        ('税前 kd = 利息/平均负债', wacc_info['kd_pre'], 'pct'),
        ('法定税率', STATUTORY_TAX, 'pct'),
        ('税后 kd', wacc_info['kd_after'], 'pct'),
        ('', '', ''),
        ('三、加权（账面）', '', ''),
        ('股东权益 E（元）', wacc_info['E'], 'num'),
        ('总负债 D（元）', wacc_info['D'], 'num'),
        ('股权权重 we', wacc_info['we'], 'pct'),
        ('债务权重 wd', wacc_info['wd'], 'pct'),
        ('WACC = we·ke + wd·kd_after', wacc_info['wacc'], 'pct'),
    ]
    ws.cell(row=1, column=1, value='老板电器 WACC（简版 CAPM）').font = Font(bold=True, size=14, color='1F4E78')
    for i, (label, val, fmt) in enumerate(rows, start=3):
        c1 = ws.cell(row=i, column=1, value=label)
        c2 = ws.cell(row=i, column=2, value=val if val != '' else None)
        if label.startswith(('一、', '二、', '三、')):
            c1.font = Font(bold=True, color='1F4E78', size=12)
            c1.fill = SUBHEADER_FILL
            c2.fill = SUBHEADER_FILL
        else:
            c1.font = Font(size=11)
            c1.alignment = Alignment(indent=1)
        if fmt == 'pct' and val != '':
            c2.number_format = '0.00%'
        elif fmt == 'num' and val != '':
            c2.number_format = '#,##0'
        elif fmt == 'num4' and val != '':
            c2.number_format = '0.0000'
    ws.column_dimensions['A'].width = 38
    ws.column_dimensions['B'].width = 28

    # ---------- Sheet 06 预测假设 ----------
    ws = wb.create_sheet('06_预测假设')
    a = dcf['assumptions']
    rows = [
        ('基准年', a['base_year'], 'int'),
        ('显性期年数', a['n_forecast'], 'int'),
        ('永续 g', a['g_terminal'], 'pct'),
        ('', '', ''),
        ('—— 营收增速假设 ——', '', ''),
        ('近5年实际营收 CAGR', a['hist5_revg'], 'pct'),
        ('起始增速 g_1', a['g_start'], 'pct'),
        ('增速衰减路径', '线性从 g_1 衰减到 g_terminal', 'str'),
        ('', '', ''),
        ('—— 盈利能力假设 ——', '', ''),
        ('近5年平均 EBIT 利润率', a['avg_ebit_margin'], 'pct'),
        ('预测期 EBIT 利润率', a['avg_ebit_margin'], 'pct'),
        ('预测期税率', a['tax_rate'], 'pct'),
        ('', '', ''),
        ('—— 资本强度假设 ——', '', ''),
        ('近5年 IC/营收（资本强度）', a['avg_ic_to_rev'], 'num4'),
        ('近5年 CapEx/营收（参考）', a['avg_capex_ratio'], 'pct'),
        ('FCF = NOPAT - ΔIC', 'McKinsey 推荐做法', 'str'),
        ('', '', ''),
        ('—— 折现 ——', '', ''),
        ('WACC', a['wacc'], 'pct'),
    ]
    ws.cell(row=1, column=1, value='10 年财务预测 - 关键假设面板').font = Font(bold=True, size=14, color='1F4E78')
    for i, (label, val, fmt) in enumerate(rows, start=3):
        c1 = ws.cell(row=i, column=1, value=label)
        c2 = ws.cell(row=i, column=2, value=val if val != '' else None)
        if label.startswith('——'):
            c1.font = Font(bold=True, color='1F4E78', size=12)
            c1.fill = SUBHEADER_FILL
            c2.fill = SUBHEADER_FILL
        if fmt == 'pct' and val != '':
            c2.number_format = '0.00%'
        elif fmt == 'int' and val != '':
            c2.number_format = '0'
        elif fmt == 'num4' and val != '':
            c2.number_format = '0.0000'
    ws.column_dimensions['A'].width = 36
    ws.column_dimensions['B'].width = 35

    # ---------- Sheet 07 预测三表 + DCF ----------
    ws = wb.create_sheet('07_预测三表DCF')
    fc_df = pd.DataFrame(dcf['rows'])
    fc_view = pd.DataFrame({
        '年度': fc_df['year'].astype(int),
        '营业收入': fc_df['rev'],
        '营收增速 g': fc_df['rev_g'],
        'EBIT利润率': fc_df['ebit_margin'],
        'EBIT': fc_df['ebit'],
        '税率': fc_df['tax_rate'],
        'NOPAT': fc_df['nopat'],
        '投入资本IC': fc_df['ic'],
        'ΔIC（再投资）': fc_df['delta_ic'],
        'FCF = NOPAT - ΔIC': fc_df['fcf'],
        '折现因子': fc_df['disc_factor'],
        'PV(FCF)': fc_df['pv_fcf'],
    })
    ws.cell(row=1, column=1, value='【1】预测期三表精简（单位：元）').font = Font(bold=True, size=13, color='1F4E78')
    end_row = write_df(ws, fc_view, start_row=2,
                       pct_cols={2, 3, 5},
                       num_cols={0, 1, 4, 6, 7, 8, 9, 11})
    # 折现因子格式
    for r in range(3, 3 + len(fc_view)):
        ws.cell(row=r, column=11).number_format = '0.0000'

    # 估值汇总
    summary = [
        ('显性期 PV(FCF) 合计', dcf['pv_fcf'], 'num'),
        ('终值 TV（在 t=N 时点）', dcf['tv'], 'num'),
        ('终值现值 PV(TV)', dcf['pv_tv'], 'num'),
        ('企业价值 EV = ΣPV + PV(TV)', dcf['enterprise_value'], 'num'),
        ('减：账面总负债（净负债近似）', wacc_info['D'], 'num'),
        ('股权价值（估算）', dcf['equity_value'], 'num'),
        ('', '', ''),
        ('企业价值（亿元）', dcf['enterprise_value'] / 1e8, 'num4'),
        ('股权价值（亿元）', dcf['equity_value'] / 1e8, 'num4'),
    ]
    ws.cell(row=end_row + 2, column=1, value='【2】DCF 估值汇总').font = Font(bold=True, size=13, color='1F4E78')
    for i, (label, val, fmt) in enumerate(summary, start=end_row + 3):
        c1 = ws.cell(row=i, column=1, value=label)
        c2 = ws.cell(row=i, column=2, value=val if val != '' else None)
        c1.font = Font(size=11, bold=label.startswith(('企业价值', '股权价值')))
        if fmt == 'num' and val != '':
            c2.number_format = '#,##0'
        elif fmt == 'num4' and val != '':
            c2.number_format = '0.00'
        if '企业价值' in label or '股权价值' in label:
            c1.fill = ROBAM_FILL
            c2.fill = ROBAM_FILL

    wb.save(OUTPUT)


if __name__ == '__main__':
    main()
