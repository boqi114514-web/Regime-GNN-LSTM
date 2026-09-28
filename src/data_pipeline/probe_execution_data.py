"""Small, isolated endpoint/coverage probe; does not update production datasets."""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from config import LOCAL_DATA_RAW
from data_pipeline.tushare_config import get_pro


def main(args):
    pro = get_pro()
    output = Path(LOCAL_DATA_RAW)/'execution_probe'/f'{args.code}_{args.start}_{args.end}'
    output.mkdir(parents=True, exist_ok=True)
    checks = [
        ('trade_cal', dict(exchange='SSE', start_date=args.start, end_date=args.end), 'cal_date'),
        ('daily', dict(ts_code=args.code, start_date=args.start, end_date=args.end), 'trade_date'),
        ('adj_factor', dict(ts_code=args.code, start_date=args.start, end_date=args.end), 'trade_date'),
        ('dividend', dict(ts_code=args.code), 'ex_date'),
        ('stk_limit', dict(ts_code=args.code, start_date=args.start, end_date=args.end), 'trade_date'),
        ('suspend_d', dict(ts_code=args.code, start_date=args.start, end_date=args.end), 'trade_date'),
    ]
    audit = {'purpose': 'endpoint_samples_not_complete_backtest_inputs', 'requests': []}
    for api, params, date_col in checks:
        row = {'api': api, 'params': params, 'status': 'failed'}
        for attempt in range(2):
            try:
                frame = pro.query(api, **params)
                row.update(status='ok', rows=len(frame), columns=list(frame))
                row['identical_duplicate_rows'] = int(frame.duplicated().sum())
                if 'ts_code' in frame and 'ts_code' in params:
                    row['wrong_code_rows'] = int(frame.ts_code.ne(args.code).sum())
                if date_col in frame:
                    keys = [c for c in ('exchange', 'ts_code', date_col) if c in frame]
                    if api != 'dividend':
                        row['duplicate_key_rows'] = int(frame.duplicated(keys).sum())
                    dates = pd.to_datetime(frame[date_col], errors='coerce')
                    row['first_date'] = str(dates.min())
                    row['last_date'] = str(dates.max())
                    if 'start_date' in params:
                        row['out_of_range_rows'] = int(((dates < pd.Timestamp(args.start)) | (dates > pd.Timestamp(args.end))).sum())
                frame.to_pickle(output/f'{api}.pkl')
                row.pop('error', None)
                break
            except Exception as exc:
                row['error'] = str(exc)
                if attempt == 0:
                    time.sleep(1.)
        audit['requests'].append(row)
        (output/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(row, ensure_ascii=True), flush=True)
    print(f'Sample output: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', default='000001.SZ')
    parser.add_argument('--start', default='20260924')
    parser.add_argument('--end', default='20260925')
    main(parser.parse_args())
