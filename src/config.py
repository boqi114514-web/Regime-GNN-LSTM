# -*- coding: utf-8 -*-
"""
GNN + LSTM-B 行业轮动模型 —— 配置文件

架构：
  GNN分支：GAT 建模行业基本面联动（替代ARIMAX）
  LSTM-B分支：技术因子捕捉动量信号（沿用原方案）
  集成：自适应权重融合
"""

import pandas as pd
import numpy as np
import os
import sys
import io
import json
import pickle
import warnings

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# ==================== 路径 ====================
ARIMAX_PROJECT = r"D:\desktop\有意思的事情\量化\项目\ARIMAX_LSTM行业轮动"
PROJECT_DIR = r"D:\desktop\有意思的事情\量化\项目\Regime-GNN-LSTM"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据路径
INDUSTRY_DATA_DIR = os.path.join(ARIMAX_PROJECT, r"数据\行业数据")
EXISTING_DATA_DIR = os.path.join(ARIMAX_PROJECT, r"数据\已有数据")
FACTOR_DIR = os.path.join(ARIMAX_PROJECT, r"结果\改进模型\因子数据")

# ==================== 行业排除 ====================
SW_EXCLUDE = ['801780.SI', '801790.SI']

# ==================== Walk-Forward 参数 ====================
TRAIN_MONTHS = 60       # 训练窗口（月）
VAL_MONTHS = 12         # 验证窗口（月）
STEP_MONTHS = 3         # 滚动步长（月）
PREDICT_MONTHS = 3      # 预测步长（月）

# ==================== HMM Regime 参数 ====================
HMM_N_STATES = 4
HMM_COVARIANCE = 'full'
HMM_N_ITER = 200
HMM_TRAIN_WINDOW = 60  # 滚动窗口（月）

# ==================== GNN 参数 ====================
GAT_HIDDEN_DIM = 16
GAT_DROPOUT = 0.3
GAT_LR = 5e-3
GAT_WEIGHT_DECAY = 5e-3
GAT_EPOCHS = 300
GAT_PATIENCE = 40
GAT_N_SEEDS = 3

# GLASSO 图参数
GLASSO_ROLLING_MONTHS = 12
GRAPH_TOP_K = 8

# ==================== LSTM-B 参数 ====================
LSTM_LOOKBACK = 3
LSTM_HIDDEN = 64
LSTM_DROPOUT = 0.2
LSTM_LR = 1e-3
LSTM_EPOCHS = 80
LSTM_PATIENCE = 15
LSTM_BATCH_SIZE = 64
LSTM_N_SEEDS = 3

# ==================== 策略参数 ====================
TOP_K = 5
RF_ANNUAL = 0.03
RANDOM_SEED = 42

# ==================== 景气度指标（预筛选结果）====================
SELECTED_INDICATORS = [
    'totprofit', 'nptocostexpense', 'netprofitexcl', 'netprofitincl',
    'ebittointerest', 'opercash', 'grossprofitmargin',
    'con_op_pr_yoy', 'con_op_rt_yoy', 'con_np_cagh',
    'con_tp_cagh', 'con_op_pr_cagh', 'con_tp_yoy', 'operrev',
]

# ==================== 数据加载 ====================

def load_industry_monthly():
    """加载行业月度行情，返回 DataFrame"""
    path = os.path.join(EXISTING_DATA_DIR, 'ts_sw_industry_monthly.csv')
    mkt = pd.read_csv(path)
    mkt = mkt[~mkt['ts_code'].isin(SW_EXCLUDE)].copy()
    mkt['date'] = pd.to_datetime(mkt['date'])
    mkt['year'] = mkt['date'].dt.year
    mkt['month'] = mkt['date'].dt.month
    mkt['ret'] = mkt['pct_chg'] / 100.0
    mkt = mkt.sort_values(['ts_code', 'date']).reset_index(drop=True)
    return mkt


def load_prosperity_monthly():
    """
    加载景气度指标（季度→月度映射，滞后一个季度）
    返回 DataFrame：ts_code, year, month, indicator1, ...
    """
    path = os.path.join(INDUSTRY_DATA_DIR, 'prosperity_indicators_clean.pkl')
    indicators = pd.read_pickle(path)

    records = []
    for _, row in indicators.iterrows():
        q, y = int(row['quarter']), int(row['year'])
        if q == 1:
            months = [(y, 4), (y, 5), (y, 6)]
        elif q == 2:
            months = [(y, 7), (y, 8), (y, 9)]
        elif q == 3:
            months = [(y, 10), (y, 11), (y, 12)]
        elif q == 4:
            months = [(y + 1, 1), (y + 1, 2), (y + 1, 3)]
        else:
            continue
        for (my, mm) in months:
            rec = {'ts_code': row['l1_code'], 'year': my, 'month': mm}
            for col in SELECTED_INDICATORS:
                if col in row.index:
                    rec[col] = row[col]
            records.append(rec)

    df = pd.DataFrame(records)
    return df


def load_tech_factors():
    """加载技术因子（价量 + 走势复刻）"""
    pv_path = os.path.join(FACTOR_DIR, 'price_volume_factors.pkl')
    pt_path = os.path.join(FACTOR_DIR, 'pattern_factors.pkl')

    pv = pd.read_pickle(pv_path)
    pv['date'] = pd.to_datetime(pv['date'])

    pt = pd.read_pickle(pt_path)
    pt['date'] = pd.to_datetime(pt['date'])

    # 合并
    tech = pd.merge(pv, pt, on=['ts_code', 'date'], how='left')
    # pattern因子不可用时填0
    for col in ['pattern_median', 'bullish_ratio', 'signal_strength']:
        if col in tech.columns:
            tech[col] = tech[col].fillna(0.0)

    tech = tech.sort_values(['ts_code', 'date']).reset_index(drop=True)
    return tech


def get_industries(mkt=None):
    """获取行业代码列表"""
    if mkt is None:
        mkt = load_industry_monthly()
    return sorted(mkt['ts_code'].unique())


def get_available_months(mkt):
    """获取所有可用月份（排序）"""
    return sorted(mkt['date'].unique())


def load_macro_factors():
    """加载宏观因子"""
    path = os.path.join(EXISTING_DATA_DIR, 'ts_macro_factors.csv')
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_csi300_monthly():
    """加载沪深300月度行情"""
    path = os.path.join(EXISTING_DATA_DIR, 'ts_csi300_monthly.csv')
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    return df


def zscore_cross_section(df, cols, group_col='date'):
    """对指定列做截面 z-score 标准化"""
    df = df.copy()
    for col in cols:
        grp = df.groupby(group_col)[col]
        mean = grp.transform('mean')
        std = grp.transform('std')
        df[col] = (df[col] - mean) / (std + 1e-8)
        df[col] = df[col].fillna(0.0)
    return df
