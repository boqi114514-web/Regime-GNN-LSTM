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

# ==================== 路径（Phase 1：项目数据自包含） ====================
PROJECT_DIR = r"D:\desktop\有意思的事情\量化\项目\Regime-GNN-LSTM"

# ===== 方向2：行业粒度开关 =====
# 环境变量 REGIME_LEVEL=l2 启用申万二级行业流水线（124 行业）；
# 默认 l1（申万一级，29 行业），一级 main 分支行为完全不变。
# L2 模式下：数据读 *_l2 / 自建产物；产物输出隔离到 results/l2、models/*/l2。
LEVEL = os.environ.get('REGIME_LEVEL', 'l1').strip().lower()
if LEVEL not in ('l1', 'l2'):
    LEVEL = 'l1'

OUTPUT_DIR = os.path.join(PROJECT_DIR, "results", "l2") if LEVEL == 'l2' \
             else os.path.join(PROJECT_DIR, "results")

# 本项目自包含数据目录
LOCAL_DATA_DIR = os.path.join(PROJECT_DIR, "data")
LOCAL_DATA_RAW = os.path.join(LOCAL_DATA_DIR, "raw")              # tushare 原始下载
LOCAL_DATA_PROCESSED = os.path.join(LOCAL_DATA_DIR, "processed")  # 景气度 / 技术因子
LOCAL_DATA_CACHE = os.path.join(LOCAL_DATA_DIR, "cache")          # 推理结果缓存

# 模型 / 状态 / 报告（实盘化使用）
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
_MODELS_BASE = os.path.join(MODELS_DIR, "l2") if LEVEL == 'l2' else MODELS_DIR
MODELS_CURRENT_DIR = os.path.join(_MODELS_BASE, "current")
MODELS_QUARTERLY_DIR = os.path.join(_MODELS_BASE, "quarterly")
MODELS_MONTHLY_DIR = os.path.join(_MODELS_BASE, "monthly")
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")
STATE_DIR = os.path.join(PROJECT_DIR, "state")

for _d in (OUTPUT_DIR, LOCAL_DATA_RAW, LOCAL_DATA_PROCESSED, LOCAL_DATA_CACHE,
           MODELS_CURRENT_DIR, MODELS_QUARTERLY_DIR, MODELS_MONTHLY_DIR,
           REPORTS_DIR, STATE_DIR):
    os.makedirs(_d, exist_ok=True)

# ==================== 个股日线与行业成分股（本地自包含） ====================
STOCK_DAILY_PATH = os.path.join(LOCAL_DATA_RAW, "stock_daily.pkl")
SW_MEMBERS_PATH = os.path.join(LOCAL_DATA_RAW, "ts_sw_members.csv")
SW_L2_MEMBERS_PATH = os.path.join(LOCAL_DATA_RAW, "ts_sw_l2_members.csv")  # 方向2：二级成分股

# ==================== 行业排除 ====================
SW_EXCLUDE = ['801780.SI', '801790.SI']

# ==================== 走势复刻因子参数 ====================
PATTERN_WINDOW = 60        # 走势指纹窗口（交易日）
PATTERN_TOP_K = 50         # Top-K 匹配
PATTERN_MIN_CORR = 0.85    # 最低相似度
PATTERN_GAP_MONTHS = 6     # 历史匹配的最近间隔（避免数据穿越）

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

# 集成模式：'regime'（Regime条件集成，扩张期GNN独占）| 'equal'（等权集成，分散化更强）
ENSEMBLE_MODE = 'regime'

# ==================== 景气度指标（预筛选结果）====================
SELECTED_INDICATORS = [
    'totprofit', 'nptocostexpense', 'netprofitexcl', 'netprofitincl',
    'ebittointerest', 'opercash', 'grossprofitmargin',
    'con_op_pr_yoy', 'con_op_rt_yoy', 'con_np_cagh',
    'con_tp_cagh', 'con_op_pr_cagh', 'con_tp_yoy', 'operrev',
]

# ==================== 数据加载 ====================

def load_industry_monthly():
    """加载行业月度行情，返回 DataFrame

    LEVEL='l1' → ts_sw_industry_monthly.csv（申万一级官方指数）
    LEVEL='l2' → sw_l2_monthly_synth.csv（自建二级行业月行情，流通市值加权）
    """
    if LEVEL == 'l2':
        path = os.path.join(LOCAL_DATA_PROCESSED, 'sw_l2_monthly_synth.csv')
        mkt = pd.read_csv(path)
        # SW_EXCLUDE 是一级代码，对二级 universe 无意义，不过滤
    else:
        path = os.path.join(LOCAL_DATA_RAW, 'ts_sw_industry_monthly.csv')
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
    fname = 'prosperity_indicators_clean_l2.pkl' if LEVEL == 'l2' \
            else 'prosperity_indicators_clean.pkl'
    path = os.path.join(LOCAL_DATA_PROCESSED, fname)
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
    _sfx = '_l2' if LEVEL == 'l2' else ''
    pv_path = os.path.join(LOCAL_DATA_PROCESSED, f'price_volume_factors{_sfx}.pkl')
    pt_path = os.path.join(LOCAL_DATA_PROCESSED, f'pattern_factors{_sfx}.pkl')

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
    path = os.path.join(LOCAL_DATA_RAW, 'ts_macro_factors.csv')
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_csi300_monthly():
    """加载沪深300月度行情"""
    path = os.path.join(LOCAL_DATA_RAW, 'ts_csi300_monthly.csv')
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    return df


def load_stock_daily():
    """全市场个股日K线（Phase 2 前跨项目读 ARIMAX）
    返回 DataFrame: date, code, open, high, low, close, volume, amount
    """
    with open(STOCK_DAILY_PATH, 'rb') as f:
        data = pickle.load(f)
    df = data['df_stock'].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    return df


def load_industry_members():
    """个股-行业一级映射（Phase 2 前跨项目读 ARIMAX）
    返回 DataFrame: l1_code, ts_code(带后缀), code(纯数字), in_date, out_date
    """
    mem = pd.read_csv(SW_MEMBERS_PATH)
    mem = mem[~mem['l1_code'].isin(SW_EXCLUDE)].copy()
    mem['code'] = mem['ts_code'].str.replace(r'\.\w+$', '', regex=True)
    mem['in_date'] = pd.to_datetime(mem['in_date'], format='%Y%m%d', errors='coerce')
    mem['out_date'] = pd.to_datetime(mem['out_date'], format='%Y%m%d', errors='coerce')
    return mem[['l1_code', 'ts_code', 'code', 'in_date', 'out_date']].copy()


def load_industry_members_l2():
    """个股-行业二级映射（方向2：申万二级行业轮动）

    返回 DataFrame: l2_code, ts_code(带后缀), code(纯数字), in_date, out_date

    与 load_industry_members 的差异：
      - 行业列是 l2_code（124 个在用二级），不过滤 SW_EXCLUDE（方案 doc 定义
        二级 universe = 124，含金融子行业）
      - 来源文件 ts_sw_l2_members.csv 为当前成分快照，out_date 多为空
    """
    mem = pd.read_csv(SW_L2_MEMBERS_PATH)
    mem['code'] = mem['ts_code'].str.replace(r'\.\w+$', '', regex=True)
    mem['in_date'] = pd.to_datetime(mem['in_date'], format='%Y%m%d', errors='coerce')
    mem['out_date'] = pd.to_datetime(mem['out_date'], format='%Y%m%d', errors='coerce')
    return mem[['l2_code', 'ts_code', 'code', 'in_date', 'out_date']].copy()


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
