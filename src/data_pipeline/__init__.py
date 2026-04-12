# -*- coding: utf-8 -*-
"""data_pipeline/：数据拉取、处理与更新

- download.py        基础数据下载/迁移（个股日线、行业成分股、财务报表）
- prosperity.py      景气度指标全链路（财务报表→TTM→指标→清洗）
- tech_factors.py    价量因子���个股日K线→行业月频中位数）
- pattern_factors.py 走势复刻因子（指纹匹配→行业信号）
- update.py          增量更新统一入口（raw CSV + 可选 processed）
"""
