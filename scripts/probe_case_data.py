# -*- coding: utf-8 -*-
"""探查案例所需数据：字段名 + 老板电器覆盖 + 家电同业."""
import pandas as pd
import os

RAW = r'D:\desktop\有意思的事情\量化\项目\Regime-GNN-LSTM\data\raw'

income = pd.read_pickle(os.path.join(RAW, 'raw_income.pkl'))
bs     = pd.read_pickle(os.path.join(RAW, 'raw_balancesheet.pkl'))
cf     = pd.read_pickle(os.path.join(RAW, 'raw_cashflow.pkl'))

print('=== columns ===')
print('income      :', list(income.columns))
print('balancesheet:', list(bs.columns))
print('cashflow    :', list(cf.columns))

# 老板电器 002508.SZ
TS = '002508.SZ'
print(f'\n=== {TS} (老板电器) 数据覆盖 ===')
for name, df in [('income', income), ('bs', bs), ('cf', cf)]:
    sub = df[df['ts_code'] == TS].copy()
    sub['end_date'] = sub['end_date'].astype(str)
    # 只看年报：12-31
    annual = sub[sub['end_date'].str.endswith('1231')]
    print(f'  {name}: 全部 {len(sub)} 行, 年报 {len(annual)} 行, '
          f'年份范围 {annual["end_date"].min()} ~ {annual["end_date"].max()}')

# report_type 分布（确认哪些是合并/母公司）
print('\n=== income report_type 分布 (002508) ===')
sub = income[income['ts_code'] == TS]
print(sub['report_type'].value_counts())
print('\n=== income comp_type 分布 (002508) ===')
print(sub['comp_type'].value_counts())

# 家电同业 - 申万行业
sw_path = os.path.join(RAW, 'ts_sw_members.csv')
sw = pd.read_csv(sw_path, dtype=str)
print('\n=== ts_sw_members 列 ===', list(sw.columns))
print(sw.head(3).to_string())

# 找家电
appliance = sw[sw.apply(lambda r: r.astype(str).str.contains('家用电器').any(), axis=1)]
print(f'\n=== 家用电器 行业成员 (前 20) ===')
print(appliance.head(20).to_string())
print(f'总数: {len(appliance)}')
