"""Corporate actions scoped to an explicitly empty-at-start research account.

Preserve complete raw histories; do not invent a listing date for an older
restructuring which occurred before this account existed.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

import research_leadership as leadership
import small_account_backtest as engine
from data_pipeline.execution_data import ROOT


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_history(raw, code, start, end):
    required = {'ts_code','end_date','div_proc','record_date','ex_date','pay_date',
                'div_listdate','cash_div_tax','stk_div'}
    if not isinstance(raw,pd.DataFrame) or raw.empty or not required.issubset(raw.columns):
        raise ValueError('Incomplete scoped corporate-action history')
    if not raw.ts_code.eq(code).all() or raw.end_date.astype(str).nunique()<2:
        raise ValueError('Wrong stock or single-period corporate-action history')
    implemented = raw[raw.div_proc.eq('实施')]
    dates = pd.to_datetime(implemented.ex_date,errors='coerce')
    record=pd.to_datetime(implemented.record_date,errors='coerce')
    paid=pd.to_datetime(implemented.pay_date,errors='coerce')
    listed=pd.to_datetime(implemented.div_listdate,errors='coerce')
    stock=pd.to_numeric(implemented.stk_div,errors='coerce').fillna(0)
    cash=pd.to_numeric(implemented.cash_div_tax,errors='coerce').fillna(0)
    settled_before=(record.lt(pd.Timestamp(start)) & (stock.le(0) | listed.lt(pd.Timestamp(start)))
                    & (cash.le(0) | paid.lt(pd.Timestamp(start))))
    if (dates.isna() & ~settled_before).any():
        raise ValueError('Implemented action has no valid ex-date')
    # All in-window validation remains in the unchanged ledger normalizer.
    return engine.normalize_actions(raw,code,start=start,end=end)


def load_scoped_actions(pro, code, out, start, end):
    start,end = f'{pd.Timestamp(start):%Y-%m-%d}',f'{pd.Timestamp(end):%Y-%m-%d}'
    folder = Path(out)/'scoped_actions'; folder.mkdir(parents=True,exist_ok=True)
    cache, manifest_path = folder/f'{code}.pkl',folder/f'{code}.json'
    if cache.exists() or manifest_path.exists():
        if not cache.exists() or not manifest_path.exists():
            raise ValueError('Incomplete scoped corporate-action cache')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (manifest.get('start'),manifest.get('end'),manifest.get('raw_sha256')) != (start,end,_sha(cache)):
            raise ValueError('Scoped corporate-action cache fingerprint/period changed')
        return validate_history(pd.read_pickle(cache),code,start,end)
    sources = [ROOT/'dividends'/f'{code}.canonical.pkl',ROOT/'dividends'/f'{code}.pkl',
               leadership.OUT/'dividends'/f'{code}.pkl']
    errors=[]; raw=None; source=None
    for path in sources:
        if not path.exists(): continue
        try:
            frame=pd.read_pickle(path)
            actions=validate_history(frame,code,start,end)
            raw,source=frame,str(path.resolve()); break
        except ValueError as exc:
            errors.append(type(exc).__name__)
    if raw is None:
        variants=[dict(ts_code=code),dict(ts_code=code,
                  fields='ts_code,end_date,div_proc,stk_div,cash_div_tax,record_date,ex_date,pay_date,div_listdate'),dict(ts_code=code,
                  fields='ts_code,end_date,ann_date,div_proc,stk_div,stk_bo_rate,stk_co_rate,cash_div_tax,record_date,ex_date,pay_date,div_listdate,imp_ann_date'),
                  dict(ts_code=code,div_proc='实施'),
                  dict(ts_code=code,limit=6000,offset=0)]
        variants += [dict(ts_code=code,end_date=period,offset=0)
                     for period in ('20251231','20241231','20231231','20221231')]
        for params in variants:
            try:
                frame=pro.query('dividend',**params)
                actions=validate_history(frame,code,start,end)
                raw,source=frame,'gateway/dividend'; break
            except Exception as exc:
                errors.append(type(exc).__name__)
    if raw is None:
        raise RuntimeError(f'{code}: no complete scoped history; error types={errors}')
    raw.to_pickle(cache)
    dates=pd.to_datetime(raw.loc[raw.div_proc.eq('实施'),'ex_date'],errors='coerce')
    manifest=dict(code=code,start=start,end=end,source=source,raw_sha256=_sha(cache),
                  raw_periods=int(raw.end_date.astype(str).nunique()),
                  out_of_account_implemented_records=int((~dates.between(start,end)).sum()),
                  in_account_actions=len(actions),
                  treatment='Only out-of-account events ignored; no missing in-window payment/listing date inferred')
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return actions
