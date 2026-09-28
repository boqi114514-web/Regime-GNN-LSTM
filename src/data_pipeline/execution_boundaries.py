"""Fetch monthly execution-session opens/limits and final valuation quotes."""
import json
from pathlib import Path
import pandas as pd
from data_pipeline.execution_data import ROOT, fetch, fetch_pages, fetch_variants, validate_rows
from data_pipeline.tushare_config import get_pro
from config import STOCK_DAILY_PATH


def boundaries(pro, start='2023-01', end='2026-09-24'):
    root = ROOT/'boundaries'
    root.mkdir(parents=True, exist_ok=True)
    local_dates = pd.to_datetime(pd.read_pickle(STOCK_DAILY_PATH)['df_stock'].date).drop_duplicates()
    output = []
    for month in pd.period_range(start, pd.Timestamp(end).to_period('M'), freq='M'):
        begin = month.to_timestamp()
        finish = min(month.to_timestamp('M'), pd.Timestamp(end))
        calendar_path = root/f'calendar_{month}.pkl'
        cal = pd.read_pickle(calendar_path) if calendar_path.exists() else fetch(
            pro, 'trade_cal', exchange='SSE', start_date=begin.strftime('%Y%m%d'),
            end_date=finish.strftime('%Y%m%d'), limit=6000)
        if not {'cal_date', 'is_open'} <= set(cal):
            raise ValueError('Invalid calendar schema')
        values = cal[['cal_date', 'is_open']].drop_duplicates()
        if values.cal_date.duplicated().any():
            raise ValueError('Conflicting trading calendar')
        dates = pd.to_datetime(values.cal_date)
        # Some gateway responses omit closed days. Verify actual sessions
        # against the independent, all-stock local quote-date union instead.
        actual_sessions = set(dates[pd.to_numeric(values.is_open).eq(1)])
        local_sessions = set(local_dates[(local_dates >= begin) & (local_dates <= finish)])
        closed = set(dates[pd.to_numeric(values.is_open).eq(0)])
        if actual_sessions-local_sessions or closed & local_sessions or not dates.between(begin, finish).all():
            raise ValueError(f'Trading-session coverage disagrees with local market union: {month}')
        cal.to_pickle(calendar_path)
        # Omitted sessions are not holidays. Use observed all-market quote dates
        # and record the gateway omission, then fetch those actual boundary days.
        sessions = sorted(local_sessions)
        if not sessions:
            raise ValueError('No sessions for month')
        first, last = sessions[0], sessions[-1]
        was_cached = all((root/f'{api}_{d:%Y%m%d}.pkl').exists()
                         for d, api in [(first, 'daily'), (first, 'stk_limit'), (last, 'daily')])
        for date, apis in [(first, ['daily', 'stk_limit']), (last, ['daily'])]:
            for api in apis:
                path = root/f'{api}_{date:%Y%m%d}.pkl'
                if path.exists():
                    d = pd.read_pickle(path)
                elif api == 'daily' and (ROOT/'daily_month_end'/f'{date:%Y%m%d}.pkl').exists():
                    d = pd.read_pickle(ROOT/'daily_month_end'/f'{date:%Y%m%d}.pkl')
                elif api == 'stk_limit':
                    # Gateway ignores pagination for this endpoint and returns
                    # >6000 securities including funds. Validate dates and keys;
                    # missing stock limits never authorize a trade.
                    day = date.strftime('%Y%m%d')
                    d = fetch_variants(pro, api, [
                        dict(trade_date=day),
                        dict(trade_date=day, fields='trade_date,ts_code,up_limit,down_limit'),
                        dict(trade_date=day, fields='ts_code,trade_date,up_limit,down_limit'),
                        dict(trade_date=day, limit=10000, offset=0),
                        dict(start_date=day, end_date=day, limit=10000, offset=0)],
                        ['ts_code', 'trade_date', 'up_limit', 'down_limit'], day)
                else:
                    d = fetch_pages(pro, api, trade_date=date.strftime('%Y%m%d'))
                positive = ['open', 'high', 'low', 'close'] if api == 'daily' else ['up_limit', 'down_limit']
                # New listings may not have positive daily limits; keep only
                # finite positive bounds, never interpret absent limits as free trading.
                if api == 'stk_limit':
                    d = d[(pd.to_numeric(d.up_limit, errors='coerce') > 0) &
                          (pd.to_numeric(d.down_limit, errors='coerce') > 0)].copy()
                d = validate_rows(d, ['ts_code', 'trade_date'], positive, date.strftime('%Y%m%d'))
                d.to_pickle(path)
        output.append(dict(month=str(month), first=str(first.date()), last=str(last.date()), n_sessions=len(sessions),
                           calendar_omitted_sessions=len(local_sessions-actual_sessions)))
        (root/'sessions.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
        if not was_cached:
            print('boundary', output[-1], flush=True)
    print(f'Execution boundaries ready: {len(output)} months', flush=True)


if __name__ == '__main__':
    boundaries(get_pro())
