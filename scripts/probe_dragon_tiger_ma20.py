# -*- coding: utf-8 -*-
"""
龙虎榜量化/机构净买入 + 20日线回踩策略验证

策略逻辑：
  1. 股票出现在龙虎榜，流通市值 > 1000亿
  2. 机构专用席位净买入 > INST_THRESH（通过 top_inst 识别；1000亿+大票的机构席位
     实质上覆盖量化基金、公募等），且信号日收盘价 > 20日均线
  3. 信号日后 MAX_HOLD_DAYS 个交易日内，股价第一次回踩20日均线
     （close ≤ ma20 × (1+TOUCH_THRESH)，从上方触及）
  4. 回踩日收盘买入，分别持有 FWDS 个交易日后卖出
  5. 统计胜率、平均收益、t检验、与同期大盘对比

运行：
  cd D:\\desktop\\有意思的事情\\量化\\项目\\Regime-GNN-LSTM
  python scripts/probe_dragon_tiger_ma20.py
"""

import sys, io, time, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta

# ── 参数 ──────────────────────────────────────────────────────────────────────
START_DATE     = '20190101'   # 回测起始
END_DATE       = '20260601'   # 回测截止（不含）
FLOAT_MV_THRESH = 1e11        # 流通市值阈值：1000亿（元）
NET_AMOUNT_THRESH = 3e7       # top_list 净买入阈值：3000万（元，初步过滤）
INST_NET_THRESH   = 2e8       # 机构专用席位净买入阈值：2亿（元）
TOUCH_THRESH   = 0.01         # 回踩20日线容忍带：收盘在20MA上方0-1%内触发
MIN_ABOVE_MA   = 0.02         # 信号日收盘至少高于20MA 2%（避免已在均线附近）
MAX_WAIT_DAYS  = 60           # 最多等60个交易日等回踩
FWDS           = [5, 10, 20, 60]   # 持有天数
MA_WINDOW      = 20
MA_TREND       = 60           # 趋势判断用60日均线：要求信号日 MA20 > MA60
SLEEP_SHORT    = 0.4          # API调用间隔（秒）

# 只保留四个顶级外资/头部量化席位
QUANT_ORGS = {
    '瑞银证券有限责任公司上海花园石桥路证券营业部',
    '高盛（中国）证券有限责任公司上海浦东新区世纪大道证券营业部',
    '高盛(中国)证券有限责任公司上海浦东新区世纪大道证券营业部',  # 兼容半角括号
    '国泰海通证券股份有限公司总部',
    '中国国际金融股份有限公司上海分公司',
}
SLEEP_LONG     = 1.5          # 速率更慢时的间隔
CACHE_DIR      = 'results/_dragon_tiger_cache'

# ── tushare 初始化 ────────────────────────────────────────────────────────────
import os
os.environ['NO_PROXY'] = 'lianghua.nanyangqiankun.top,47.109.59.144,101.35.233.113,127.0.0.1'
os.environ['no_proxy'] = os.environ['NO_PROXY']

import tushare as ts
TOKEN = os.environ.get('TUSHARE_TOKEN', '').strip()
if not TOKEN:
    raise RuntimeError('Set TUSHARE_TOKEN in the process environment before running this script')
URL   = 'http://lianghua.nanyangqiankun.top'
pro = ts.pro_api(TOKEN)
pro._DataApi__token    = TOKEN
pro._DataApi__http_url = URL
print(f'[init] tushare mirror: {URL}')

os.makedirs(CACHE_DIR, exist_ok=True)

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def safe_call(fn, *args, retries=3, **kwargs):
    for i in range(retries):
        try:
            df = fn(*args, **kwargs)
            time.sleep(SLEEP_SHORT)
            return df
        except Exception as e:
            print(f'  [retry {i+1}] {e}')
            time.sleep(SLEEP_LONG * (i + 1))
    return None


def get_trade_dates(start, end):
    """获取交易日历"""
    df = safe_call(pro.trade_cal, exchange='SSE', start_date=start, end_date=end,
                   fields='cal_date,is_open')
    if df is None or df.empty:
        return []
    return df[df.is_open == 1]['cal_date'].tolist()


def fetch_top_list_date(date_str):
    """拉取单日龙虎榜，过滤大市值"""
    cache_f = os.path.join(CACHE_DIR, f'toplist_{date_str}.pkl')
    if os.path.exists(cache_f):
        return pd.read_pickle(cache_f)
    df = safe_call(pro.top_list, trade_date=date_str)
    if df is None or df.empty:
        return pd.DataFrame()
    big = df[df['float_values'] > FLOAT_MV_THRESH].copy()
    big.to_pickle(cache_f)
    return big


def fetch_top_inst_date(date_str):
    """拉取单日机构交易明细"""
    cache_f = os.path.join(CACHE_DIR, f'topinst_{date_str}.pkl')
    if os.path.exists(cache_f):
        return pd.read_pickle(cache_f)
    df = safe_call(pro.top_inst, trade_date=date_str)
    if df is None:
        df = pd.DataFrame()
    df.to_pickle(cache_f)
    return df


def compute_inst_net_buy(top_inst_df, ts_code):
    """计算该股票量化/机构席位净买入总额。
    覆盖两类席位：
      1. 机构专用（含量化基金、公募、险资等所有机构通道）
      2. QUANT_ORGS 中的已知量化营业部
    同一席位在 side=0/1 各出现一次，先去重再汇总。
    """
    if top_inst_df.empty:
        return 0.0
    sub = top_inst_df[top_inst_df['ts_code'] == ts_code].copy()
    if sub.empty:
        return 0.0
    mask = (sub['exalter'].str.contains('机构专用', na=False) |
            sub['exalter'].isin(QUANT_ORGS))
    inst = sub[mask]
    if inst.empty:
        return 0.0
    # 去重：同一席位只取 side=0，fallback 到 drop_duplicates
    inst_dedup = inst[inst['side'] == 0]
    if inst_dedup.empty:
        inst_dedup = inst.drop_duplicates(subset=['exalter'])
    return float(inst_dedup['net_buy'].sum())


def fetch_daily_price(ts_code, start_date, end_date):
    """拉取复权日线数据，返回含 close/ma20 的 DataFrame，index=trade_date"""
    cache_f = os.path.join(CACHE_DIR, f'daily_{ts_code}_{start_date}_{end_date}.pkl')
    if os.path.exists(cache_f):
        return pd.read_pickle(cache_f)
    df = safe_call(pro.daily, ts_code=ts_code, start_date=start_date, end_date=end_date,
                   fields='trade_date,close,open,high,low,vol')
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_values('trade_date').set_index('trade_date')
    df['ma20'] = df['close'].rolling(MA_WINDOW, min_periods=MA_WINDOW).mean()
    df['ma60'] = df['close'].rolling(MA_TREND,  min_periods=MA_TREND).mean()
    df.to_pickle(cache_f)
    return df


# ── Phase 1：收集信号 ──────────────────────────────────────────────────────────

print('\n' + '='*70)
print('Phase 1: 扫描龙虎榜，收集大市值净买入信号')
print('='*70)

cache_signals_f = os.path.join(CACHE_DIR, 'signals_raw.pkl')
if os.path.exists(cache_signals_f):
    signals_raw = pd.read_pickle(cache_signals_f)
    print(f'  [cache] 读取原始信号 {len(signals_raw)} 条')
else:
    trade_dates = get_trade_dates(START_DATE, END_DATE)
    print(f'  交易日 {len(trade_dates)} 个，从 {trade_dates[0]} 到 {trade_dates[-1]}')

    rows = []
    for i, d in enumerate(trade_dates):
        if i % 100 == 0:
            print(f'  [{i}/{len(trade_dates)}] {d}')
        big_df = fetch_top_list_date(d)
        if big_df.empty:
            continue
        # 初步过滤：龙虎榜净买入 > 阈值
        cands = big_df[big_df['net_amount'] > NET_AMOUNT_THRESH]
        for _, row in cands.iterrows():
            rows.append({
                'signal_date': d,
                'ts_code'    : row['ts_code'],
                'name'       : row['name'],
                'close'      : row['close'],
                'net_amount' : row['net_amount'],
                'float_values': row['float_values'],
                'reason'     : row.get('reason', ''),
            })

    signals_raw = pd.DataFrame(rows)
    signals_raw.to_pickle(cache_signals_f)
    print(f'  原始信号 {len(signals_raw)} 条（流通市值>1000亿 且 净买入>{NET_AMOUNT_THRESH/1e6:.0f}百万）')

print(f'  原始信号：{len(signals_raw)} 条')
if len(signals_raw) == 0:
    print('  没有信号，退出')
    sys.exit(0)

print(f'  涉及股票 {signals_raw.ts_code.nunique()} 只')
print(f'  日期分布（按年）:')
signals_raw['year'] = signals_raw['signal_date'].str[:4]
print(signals_raw.groupby('year').size().to_string())

# ── Phase 2：机构专用席位净买入过滤 ──────────────────────────────────────────

print('\n' + '='*70)
print('Phase 2: 机构专用席位净买入确认')
print('='*70)

cache_inst_f = os.path.join(CACHE_DIR, 'signals_with_inst.pkl')
if os.path.exists(cache_inst_f):
    signals_inst = pd.read_pickle(cache_inst_f)
    print(f'  [cache] 机构过滤后信号 {len(signals_inst)} 条')
else:
    unique_dates = signals_raw['signal_date'].unique()
    print(f'  需要拉取 top_inst 的日期：{len(unique_dates)} 个')

    inst_net_buy_map = {}  # (date, ts_code) -> inst_net_buy
    for i, d in enumerate(unique_dates):
        if i % 50 == 0:
            print(f'  [{i}/{len(unique_dates)}] {d}')
        top_inst = fetch_top_inst_date(d)
        for ts_code in signals_raw[signals_raw['signal_date'] == d]['ts_code']:
            inst_net_buy_map[(d, ts_code)] = compute_inst_net_buy(top_inst, ts_code)

    signals_raw['inst_net_buy'] = signals_raw.apply(
        lambda r: inst_net_buy_map.get((r.signal_date, r.ts_code), 0.0), axis=1)

    # 若 top_inst 无权限（全为0），直接用 top_list net_amount 作为代理
    has_inst_data = (signals_raw['inst_net_buy'] > 0).sum()
    print(f'  机构专用席位有数据的信号：{has_inst_data} 条')

    if has_inst_data < 5:
        print('  ⚠ top_inst 机构数据不足（可能积分不够），改用 top_list net_amount 作代理')
        print('  → 保留 top_list 净买入 > NET_AMOUNT_THRESH 的全部信号')
        signals_inst = signals_raw.copy()
        signals_inst['inst_net_buy'] = signals_inst['net_amount']
    else:
        signals_inst = signals_raw[signals_raw['inst_net_buy'] >= INST_NET_THRESH].copy()

    signals_inst.to_pickle(cache_inst_f)
    print(f'  机构过滤后：{len(signals_inst)} 条信号')

print(f'  机构确认信号：{len(signals_inst)} 条，涉及 {signals_inst.ts_code.nunique()} 只股票')

# ── Phase 3：获取日线数据 + 确认信号日在MA上方 ───────────────────────────────

print('\n' + '='*70)
print('Phase 3: 拉取日线数据，验证信号日在20MA上方')
print('='*70)

# 为每个信号股票获取覆盖区间的日线数据
# 覆盖范围：信号日前25日（算MA）+ 信号日后 MAX_WAIT_DAYS + max(FWDS)
LOOKBACK = 30   # MA计算需要的历史
FORWARD  = MAX_WAIT_DAYS + max(FWDS) + 5

unique_stocks = signals_inst['ts_code'].unique()
print(f'  需要拉取 {len(unique_stocks)} 只股票的日线')

price_cache = {}
for i, code in enumerate(unique_stocks):
    if i % 50 == 0:
        print(f'  [{i}/{len(unique_stocks)}] {code}')
    df = fetch_daily_price(code, START_DATE, END_DATE)
    if not df.empty:
        price_cache[code] = df

print(f'  成功获取 {len(price_cache)} 只股票日线')

# ── Phase 4：寻找回踩20MA的入场点 ────────────────────────────────────────────

print('\n' + '='*70)
print('Phase 4: 寻找回踩20日线入场点')
print('='*70)

entries = []
skipped_no_price = 0
skipped_not_above_ma = 0
skipped_no_uptrend = 0
skipped_no_touch = 0

for _, sig in signals_inst.iterrows():
    code      = sig['ts_code']
    sig_date  = sig['signal_date']

    if code not in price_cache:
        skipped_no_price += 1
        continue

    price_df = price_cache[code]
    all_dates = sorted(price_df.index.tolist())

    # 信号日当天收盘和MA20
    if sig_date not in price_df.index:
        skipped_no_price += 1
        continue

    sig_close = price_df.loc[sig_date, 'close']
    sig_ma20  = price_df.loc[sig_date, 'ma20']
    sig_ma60  = price_df.loc[sig_date, 'ma60']

    if pd.isna(sig_ma20):
        skipped_not_above_ma += 1
        continue

    # 信号日收盘必须高于20MA至少 MIN_ABOVE_MA
    if sig_close < sig_ma20 * (1 + MIN_ABOVE_MA):
        skipped_not_above_ma += 1
        continue

    # 上涨趋势：MA20 > MA60（中期均线多头排列）
    if pd.isna(sig_ma60) or sig_ma20 <= sig_ma60:
        skipped_no_uptrend += 1
        continue

    # 寻找信号日后 MAX_WAIT_DAYS 个交易日内，第一次回踩20MA
    try:
        sig_idx = all_dates.index(sig_date)
    except ValueError:
        skipped_no_price += 1
        continue

    future_dates = all_dates[sig_idx + 1: sig_idx + 1 + MAX_WAIT_DAYS]
    entry_date   = None

    for fd in future_dates:
        if fd not in price_df.index:
            continue
        fd_close = price_df.loc[fd, 'close']
        fd_ma20  = price_df.loc[fd, 'ma20']
        if pd.isna(fd_ma20):
            continue
        # 回踩条件：收盘在 [ma20, ma20*(1+TOUCH_THRESH)] 之间（从上方触及）
        if fd_close <= fd_ma20 * (1 + TOUCH_THRESH):
            entry_date = fd
            entry_close = fd_close
            entry_ma20  = fd_ma20
            break

    if entry_date is None:
        skipped_no_touch += 1
        continue

    # 计算各持有期收益
    try:
        entry_idx = all_dates.index(entry_date)
    except ValueError:
        continue

    fwd_returns = {}
    valid = True
    for fwd in FWDS:
        exit_idx = entry_idx + fwd
        if exit_idx >= len(all_dates):
            valid = False
            break
        exit_date  = all_dates[exit_idx]
        if exit_date not in price_df.index:
            valid = False
            break
        exit_close = price_df.loc[exit_date, 'close']
        fwd_returns[f'ret_{fwd}d'] = (exit_close - entry_close) / entry_close

    if not valid:
        continue

    entry_rec = {
        'ts_code'     : code,
        'name'        : sig.get('name', ''),
        'signal_date' : sig_date,
        'entry_date'  : entry_date,
        'days_to_entry': all_dates.index(entry_date) - sig_idx,
        'sig_close'   : sig_close,
        'sig_ma20'    : sig_ma20,
        'sig_above_ma': (sig_close / sig_ma20 - 1),
        'entry_close' : entry_close,
        'entry_ma20'  : entry_ma20,
        'net_amount'  : sig['net_amount'],
        'inst_net_buy': sig['inst_net_buy'],
        'float_values': sig['float_values'],
        'reason'      : sig.get('reason', ''),
    }
    entry_rec.update(fwd_returns)
    entries.append(entry_rec)

print(f'  跳过（无价格数据）：{skipped_no_price}')
print(f'  跳过（信号日未在20MA上方）：{skipped_not_above_ma}')
print(f'  跳过（信号日非上涨趋势 MA20≤MA60）：{skipped_no_uptrend}')
print(f'  跳过（等待期内未回踩）：{skipped_no_touch}')
print(f'  有效入场记录：{len(entries)}')

if len(entries) == 0:
    print('  没有有效入场，退出')
    sys.exit(0)

results = pd.DataFrame(entries)
results.to_csv(os.path.join(CACHE_DIR, 'entry_results.csv'), index=False, encoding='utf-8-sig')

# ── Phase 5：统计分析 ─────────────────────────────────────────────────────────

print('\n' + '='*70)
print('Phase 5: 统计分析')
print('='*70)

def analyze(df, label='全部样本'):
    n = len(df)
    print(f'\n  [{label}]  样本数 = {n}')
    if n < 5:
        print('    样本太少，跳过')
        return
    for fwd in FWDS:
        col = f'ret_{fwd}d'
        rets = df[col].dropna().values
        win_rate = (rets > 0).mean()
        mean_ret = rets.mean()
        med_ret  = np.median(rets)
        tstat, pval = stats.ttest_1samp(rets, 0)
        print(f'    持有{fwd:>2}日:  胜率={win_rate:.1%}  均值={mean_ret:+.2%}  '
              f'中位数={med_ret:+.2%}  t={tstat:.2f}  p={pval:.3f}')

analyze(results, '全部（流通>1000亿 + 机构净买入 + 回踩20MA）')

# 按流通市值分组
results['mv_tier'] = pd.cut(results['float_values'],
    bins=[1e11, 3e11, 5e11, 1e12, np.inf],
    labels=['1-3千亿', '3-5千亿', '5000亿-1万亿', '>1万亿'])
for tier, g in results.groupby('mv_tier', observed=True):
    analyze(g, f'流通市值 {tier}')

# 按入场等待天数分组
results['wait_tier'] = pd.cut(results['days_to_entry'],
    bins=[0, 5, 10, 20, 60],
    labels=['1-5天', '6-10天', '11-20天', '21-60天'])
for tier, g in results.groupby('wait_tier', observed=True):
    analyze(g, f'等待入场 {tier}')

# 按信号日涨幅偏离MA程度分组
results['ma_gap_tier'] = pd.cut(results['sig_above_ma'],
    bins=[MIN_ABOVE_MA, 0.05, 0.10, 0.20, np.inf],
    labels=['2-5%', '5-10%', '10-20%', '>20%'])
for tier, g in results.groupby('ma_gap_tier', observed=True):
    analyze(g, f'信号日高于20MA {tier}')

# ── 汇总表格 ──────────────────────────────────────────────────────────────────

print('\n' + '='*70)
print('汇总表（全部样本各持有期）')
print('='*70)
summary_rows = []
for fwd in FWDS:
    col = f'ret_{fwd}d'
    rets = results[col].dropna().values
    n    = len(rets)
    tstat, pval = stats.ttest_1samp(rets, 0)
    summary_rows.append({
        '持有天数'  : fwd,
        '样本量'    : n,
        '胜率'      : f"{(rets>0).mean():.1%}",
        '均值收益'  : f"{rets.mean():+.2%}",
        '中位数'    : f"{np.median(rets):+.2%}",
        '标准差'    : f"{rets.std():.2%}",
        't统计量'   : f"{tstat:.2f}",
        'p值'       : f"{pval:.4f}",
        '显著性'    : '***' if pval<0.01 else ('**' if pval<0.05 else ('*' if pval<0.1 else '')),
    })
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# ── 输出部分样本 ──────────────────────────────────────────────────────────────

print('\n前20条入场记录：')
show_cols = ['ts_code','name','signal_date','entry_date','days_to_entry',
             'sig_above_ma','ret_5d','ret_10d','ret_20d','ret_60d']
print(results[show_cols].head(20).to_string(index=False))

print(f'\n结果已保存至 {CACHE_DIR}/entry_results.csv')
print('\n完成。')
