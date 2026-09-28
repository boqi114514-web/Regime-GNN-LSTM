"""Canonical monthly industry prices and returns.

The historical CSV stores ``pct_chg`` in percent while newer rows store it
as a decimal.  Close-to-close returns are the only consistent local source.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def canonicalize_industry_monthly(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one row per industry/month with decimal ``ret``.

    ``pct_chg`` is rewritten in *percent* units for legacy consumers.  A
    missing month never becomes a multi-month return: the next ``ret`` stays
    NaN until two consecutive monthly closes are available.
    """
    required = {'ts_code', 'date', 'close'}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f'行业月行情缺少字段: {sorted(missing)}')

    monthly = raw.copy()
    monthly['date'] = pd.to_datetime(monthly['date'].astype(str), errors='raise')
    monthly['close'] = pd.to_numeric(monthly['close'], errors='coerce')
    if monthly['ts_code'].isna().any() or monthly['close'].isna().any() or (monthly['close'] <= 0).any():
        raise ValueError('行业月行情有缺失代码、无效收盘价或非正收盘价')

    monthly['_period'] = monthly['date'].dt.to_period('M')
    monthly = (monthly.sort_values(['ts_code', 'date'], kind='stable')
               .drop_duplicates(['ts_code', '_period'], keep='last')
               .reset_index(drop=True))
    period_number = monthly['_period'].astype('int64')
    previous_period = period_number.groupby(monthly['ts_code']).shift(1)
    consecutive = period_number.sub(previous_period).eq(1)
    monthly['ret'] = monthly.groupby('ts_code')['close'].pct_change(fill_method=None)
    monthly.loc[~consecutive, 'ret'] = np.nan
    monthly['pct_chg'] = monthly['ret'] * 100.0
    monthly['date'] = monthly['_period'].dt.to_timestamp('M')
    return monthly.drop(columns='_period')


def repair_csv(path: Path, backup: Path) -> tuple[int, int]:
    """Normalize an existing CSV in place, preserving a byte-for-byte backup."""
    path = path.resolve()
    backup = backup.resolve()
    if path == backup or backup.exists():
        raise ValueError('备份文件已存在或与原文件相同，拒绝覆盖')
    raw = pd.read_csv(path)
    fixed = canonicalize_industry_monthly(raw)
    original_columns = list(raw.columns)
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup)
    descriptor, temp_name = tempfile.mkstemp(prefix='industry_monthly_', suffix='.csv', dir=path.parent)
    os.close(descriptor)
    try:
        fixed[original_columns].to_csv(temp_name, index=False, encoding='utf-8-sig')
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return len(raw), len(fixed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='备份并修复申万行业月度行情')
    parser.add_argument('path', type=Path)
    parser.add_argument('--backup', required=True, type=Path)
    args = parser.parse_args()
    before, after = repair_csv(args.path, args.backup)
    print(f'行业月行情已规范化: {before} → {after} 行；备份: {args.backup}')
