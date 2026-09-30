"""As-reported positive-profit / positive-equity floor; never a ticker blacklist."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

TTM_VARIANTS=('state_quality_retry','daily_quality_retry')
REPORT_VARIANTS=('state_report_quality_retry','daily_report_quality_retry')
GUARDED_VARIANTS=('daily_guarded_retry',)
VARIANTS=TTM_VARIANTS+REPORT_VARIANTS+GUARDED_VARIANTS
PROTOCOL=dict(
    variants=list(TTM_VARIANTS),
    stage='Exploratory follow-up after a loss-making-company limit-down event in daily-pressure replay; not untouched validation',
    gate='Latest available trailing-12-month attributable net profit>0 AND latest available attributable net assets>0; missing or stale (>450-day report period) means ineligible',
    timing='Snapshot at the previous monthly signal trading date; each statement becomes usable one calendar day after max(ann_date,f_ann_date); later statements/revisions cannot enter earlier snapshots',
    ttm='Annual report: annual attributable net profit; otherwise current YTD + preceding full year - previous-year same YTD, all components available by snapshot',
    scope='Same generic gate for every stock and every year; no ticker, theme, ST-event-date or 2026-month whitelist',
    ranking='No change to technical ranking, risk exposure, lot allocation or stop rules; fundamental data is only an eligibility floor',
    state_quality='Gate the existing state_onset monthly candidates',
    daily_quality='Same monthly floor, plus unchanged daily_pressure supplementary entries with the same previous-month floor',
    inputs='Existing local original consolidated statements (report_type=1), dated availability; not a certified historical vendor-release archive',
    costs='25000 initial/new-investment cap, no topups, zero costs, mainboard whole-lot buys')
REPORT_PROTOCOL={**PROTOCOL,
    'variants':list(REPORT_VARIANTS),
    'gate':'Latest available reported YTD attributable net profit>0 AND latest available attributable net assets>0; missing/stale report (>450 days) means ineligible',
    'ttm':'TTM is diagnostic only, not used by this separate variant',
    'motivation':'Before any quality-account run, TTM coverage audit found only 378 local 2025 annual reports; gateway income_vip unavailable and income bulk query failed. Do not silently fill missing TTM or loosen its guard. This variant tests the separate latest-reported-profit rule, not TTM.',
    'limitation':'YTD profit has seasonality and one-off items; this is a minimal loss-making-company screen, not a financial-distress predictor'}
GUARDED_PROTOCOL={**REPORT_PROTOCOL,
    'variants':list(GUARDED_VARIANTS),
    'stage':'Follow-up after monthly-only report-floor replay; isolates a guarded daily add-on without changing the original monthly policy; all history already examined',
    'state_quality':'Original state_onset monthly selection remains exactly unchanged; no new monthly financial gate',
    'daily_quality':'Apply latest-reported-profit/net-assets floor only to supplementary daily-pressure entries',
    'motivation':'The broad monthly filter also changed ordinary technical selections and integer allocations. Test the explicitly requested supplemental upgrade separately from redesigning the monthly core.'}


def normalize_reports(frame, value):
    f=frame[frame.report_type.astype(str).eq('1')].copy()
    for col in ('ann_date','f_ann_date','end_date'):
        f[col]=pd.to_datetime(f[col].astype('string').str.replace(r'\.0$','',regex=True),format='%Y%m%d',errors='coerce')
    if f[['ann_date','f_ann_date','end_date']].isna().any().any():
        raise ValueError('Financial statements lack valid publication dates')
    f['available']=f[['ann_date','f_ann_date']].max(axis=1)+pd.Timedelta(days=1)
    if f.available.le(f.end_date).any(): raise ValueError('Statement available before report period ends')
    f[value]=pd.to_numeric(f[value],errors='coerce')
    f=f[['ts_code','end_date','available',value]].drop_duplicates()
    if f.duplicated(['ts_code','end_date','available']).any():
        raise ValueError('Ambiguous same-date statement values')
    return f.sort_values(['available','end_date','ts_code'])


def snapshot(income, balance, day):
    profit='n_income_attr_p'; equity='total_hldr_eqy_exc_min_int'
    inc=income[income.available.le(day)].drop_duplicates(['ts_code','end_date'],keep='last')
    bal=balance[balance.available.le(day)].drop_duplicates(['ts_code','end_date'],keep='last')
    latest=inc.sort_values('end_date').drop_duplicates('ts_code',keep='last').set_index('ts_code')
    b=bal.sort_values('end_date').drop_duplicates('ts_code',keep='last').set_index('ts_code')
    values=inc.set_index(['ts_code','end_date'])
    annual=pd.to_datetime((latest.end_date.dt.year-1).astype(str)+'1231',format='%Y%m%d')
    prior=latest.end_date-pd.DateOffset(years=1)
    def component(dates):
        return values.reindex(pd.MultiIndex.from_arrays([latest.index,dates]))
    yearly=component(annual); same=component(prior)
    tt=latest[profit].to_numpy()+yearly[profit].to_numpy()-same[profit].to_numpy()
    is_annual=latest.end_date.dt.month.eq(12)
    tt=np.where(is_annual,latest[profit],tt)
    f=pd.DataFrame(dict(ts_code=latest.index,ttm_profit=tt,
        reported_profit=latest[profit].to_numpy(),
        income_period=latest.end_date.to_numpy(),income_available=latest.available.to_numpy(),
        net_assets=b[equity].reindex(latest.index).to_numpy(),
        balance_period=b.end_date.reindex(latest.index).to_numpy(),
        balance_available=b.available.reindex(latest.index).to_numpy()))
    f['quality_ok']=(f.ttm_profit.gt(0)&f.net_assets.gt(0)
        &f.income_period.ge(day-pd.Timedelta(days=450))&f.balance_period.ge(day-pd.Timedelta(days=450)))
    f['report_quality_ok']=(f.reported_profit.gt(0)&f.net_assets.gt(0)
        &f.income_period.ge(day-pd.Timedelta(days=450))&f.balance_period.ge(day-pd.Timedelta(days=450)))
    return f


def quality_features(candidates, income, balance):
    inc=normalize_reports(income,'n_income_attr_p')
    bal=normalize_reports(balance,'total_hldr_eqy_exc_min_int')
    blocks=[]
    for month,c in candidates.groupby('month',sort=True):
        day=c.date.max()
        f=snapshot(inc,bal,day)
        f=c[['month','ts_code']].merge(f,on='ts_code',how='left',validate='one_to_one')
        f['quality_ok']=f.quality_ok.eq(True)
        f['quality_signal_day']=day
        blocks.append(f)
    return pd.concat(blocks,ignore_index=True)


def prepare_candidates(out, raw_dir, name='state_quality_retry'):
    if name not in VARIANTS: raise ValueError(name)
    guarded=name in GUARDED_VARIANTS
    report_only=name in REPORT_VARIANTS+GUARDED_VARIANTS
    protocol=GUARDED_PROTOCOL if guarded else (REPORT_PROTOCOL if report_only else PROTOCOL)
    path=out/('daily_guarded_protocol.json' if guarded else ('quality_report_floor_protocol.json' if report_only else 'quality_floor_protocol.json'))
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=protocol:
        raise ValueError('Quality floor protocol changed')
    path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    c=pd.read_pickle(out/'candidates.pkl')
    f=quality_features(c,pd.read_pickle(raw_dir/'raw_income.pkl'),pd.read_pickle(raw_dir/'raw_balancesheet.pkl'))
    f.to_pickle(out/'quality_floor_features.pkl')
    audit=f.groupby('month').agg(candidates=('ts_code','size'),known_ttm=('ttm_profit','count'),
                                 known_report=('reported_profit','count'),known_assets=('net_assets','count'),
                                 passed=('quality_ok','sum'),report_passed=('report_quality_ok','sum'))
    audit.to_csv(out/'quality_floor_coverage.csv')
    known=audit.known_report if report_only else audit.known_ttm
    if (known/audit.candidates<.9).any() or (audit.known_assets/audit.candidates<.9).any():
        raise ValueError('Financial data coverage below 90%; do not run a mostly-missing filter')
    c=c.merge(f,on=['month','ts_code'],how='left',validate='one_to_one')
    c['eligible']=c['report_quality_ok' if report_only else 'quality_ok'].eq(True)&c.size_bucket.eq(c.chosen_style)&~c.phase.isin(['retreat','overheat'])
    if guarded: c['eligible']=c.size_bucket.eq(c.chosen_style)&~c.phase.isin(['retreat','overheat'])
    suffix='_monthly' if name.startswith('daily_') else ''
    c.to_pickle(out/f'candidate_audit_{name}{suffix}.pkl')
    return c[c.eligible].copy()


def filter_daily(out, lookup, name):
    flag='report_quality_ok' if name in REPORT_VARIANTS+GUARDED_VARIANTS else 'quality_ok'
    f=pd.read_pickle(out/'quality_floor_features.pkl')[['month','ts_code',flag]].rename(columns={flag:'quality_ok'})
    blocks=[]; result={}
    for day,c in lookup.items():
        x=c.merge(f,on=['month','ts_code'],how='left',validate='many_to_one')
        x=x[x.quality_ok.eq(True)].copy()
        if len(x): result[day]=x; blocks.append(x)
    pd.concat(blocks,ignore_index=True).to_pickle(out/f'candidate_audit_{name}.pkl')
    return result
