"""Independent, read-only reconstruction of a daily opportunity account.

No import of its execution engine or Ledger. Raw quotes, predictions, official
limits and dividend histories are the authority; exported NAV/P&L are checks.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))
from data_pipeline.execution_data import ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def equal(a, b, label, tolerance=1e-6):
    check(np.isfinite(a) and np.isfinite(b) and abs(float(a)-float(b)) <= tolerance, label)


def mainboard(code):
    return bool(re.fullmatch(r'(?:600|601|603|605|000|001|002|003)\d{3}\.(?:SH|SZ)', str(code)))


def distribution_quantity(value):
    # The vendor represents an absent cash/stock component as null or an
    # empty field. Do not treat a nonempty malformed number as zero.
    return 0. if pd.isna(value) or (isinstance(value,str) and not value.strip()) else float(value)


def raw_actions(out, codes, start, end):
    """Independently normalize complete recorded raw histories; no zero guessing."""
    records, fingerprints = [], {}
    for code in sorted(codes):
        path = Path(out)/'scoped_actions'/f'{code}.pkl'
        manifest_path = path.with_suffix('.json')
        check(path.exists() and manifest_path.exists(), 'Missing raw corporate action history: '+code)
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        check(manifest.get('raw_sha256') == sha(path), 'Raw corporate action fingerprint mismatch: '+code)
        check((manifest.get('start'),manifest.get('end')) == (f'{start:%Y-%m-%d}',f'{end:%Y-%m-%d}'), 'Corporate action period mismatch')
        frame = pd.read_pickle(path)
        required = {'ts_code','end_date','div_proc','record_date','ex_date','pay_date','div_listdate','cash_div_tax','stk_div'}
        check(required.issubset(frame) and len(frame) > 0 and frame.ts_code.eq(code).all(), 'Incomplete raw action history')
        check(frame.end_date.astype(str).nunique() >= 2, 'Single-period history is insufficient')
        for row in frame[frame.div_proc.eq('实施')].itertuples():
            ex_date = pd.to_datetime(row.ex_date, errors='coerce')
            record_date = pd.to_datetime(row.record_date, errors='coerce')
            pay_date = pd.to_datetime(row.pay_date, errors='coerce')
            listing = pd.to_datetime(row.div_listdate, errors='coerce')
            cash = distribution_quantity(row.cash_div_tax)
            stock = distribution_quantity(row.stk_div)
            check(np.isfinite(cash) and np.isfinite(stock) and min(cash,stock) >= 0, 'Invalid action quantities')
            if pd.isna(ex_date):
                settled_before = (pd.notna(record_date) and record_date<start
                                  and (not stock or (pd.notna(listing) and listing<start))
                                  and (not cash or (pd.notna(pay_date) and pay_date<start)))
                check(settled_before,'Implemented action has no valid ex-date or proven old settlement')
                continue
            if not start <= ex_date <= end:
                continue
            if not cash and not stock:
                continue
            check(pd.notna(record_date) and record_date < ex_date, 'Invalid record/ex-date')
            check(not cash or (pd.notna(pay_date) and pay_date >= ex_date), 'Missing cash payment date')
            check(not stock or (pd.notna(listing) and listing >= ex_date), 'Missing bonus-share listing date')
            period = str(row.end_date)
            check(bool(re.fullmatch(r'\d{8}',period)), 'Noncanonical action report period')
            if hasattr(row,'stk_bo_rate') and hasattr(row,'stk_co_rate'):
                parts = sum(distribution_quantity(x) for x in (row.stk_bo_rate,row.stk_co_rate))
                check(not parts or abs(parts-stock) <= 1e-6, 'Inconsistent bonus-share units')
            records.append(dict(ts_code=code,record_date=record_date,ex_date=ex_date,
                                pay_date=pay_date if cash else pd.NaT,div_listdate=listing if stock else pd.NaT,
                                cash=cash,stock=stock,event=f'{code}:{ex_date:%Y%m%d}:{period}'))
        fingerprints[str(path.resolve())], fingerprints[str(manifest_path.resolve())] = sha(path),sha(manifest_path)
    columns = ['ts_code','record_date','ex_date','pay_date','div_listdate','cash','stock','event']
    result = pd.DataFrame(records,columns=columns).drop_duplicates()
    check(not result.event.duplicated().any(), 'Conflicting action events')
    return result, fingerprints


def load_exact_limits(out, status, dates_codes, injected=None):
    """Resolve original vendor rows, including blocked-order days."""
    if injected is not None:
        return dict(injected), {}
    out = Path(out)
    candidates = list((out/'daily_limits').glob('*.pkl'))
    # Verified legacy replacement/exit caches can be named code_date.pkl,
    # without "stk_limit" in the filename. Resolve by vendor schema, never
    # by filename spelling. All recorded source hashes are checked by caller.
    candidates += [Path(x) for x in status.get('source_sha256',{}) if Path(x).suffix == '.pkl']
    candidates += [ROOT/'boundaries'/f'stk_limit_{day:%Y%m%d}.pkl' for day in sorted(set(d for d,c in dates_codes))]
    values, fingerprints = {}, {}
    needed = set(dates_codes)
    for path in dict.fromkeys(candidates):
        if not path.exists():
            continue
        raw = pd.read_pickle(path)
        if not {'ts_code','trade_date','up_limit','down_limit'}.issubset(raw):
            continue
        raw = raw.assign(date=pd.to_datetime(raw.trade_date.astype(str),format='%Y%m%d',errors='coerce'))
        selected = raw[[((day,code) in needed) for day,code in zip(raw.date,raw.ts_code)]]
        if selected.empty:
            continue
        check(not selected.duplicated(['date','ts_code']).any(), 'Conflicting exact official limit rows')
        for row in selected.itertuples():
            check(np.isfinite(row.up_limit) and np.isfinite(row.down_limit) and 0 < row.down_limit < row.up_limit, 'Invalid official limit')
            key, item = (row.date,row.ts_code),(float(row.down_limit),float(row.up_limit))
            check(key not in values or values[key] == item, 'Contradictory exact-limit sources')
            values[key] = item
        fingerprints[str(path.resolve())] = sha(path)
    check(needed.issubset(values), 'Missing independent exact official-limit observation')
    return values, fingerprints


def compare_frames(actual, expected, keys, values, label):
    check(set(keys+values).issubset(actual), 'Missing '+label+' schema')
    check(not actual.duplicated(keys).any(), 'Duplicate '+label+' keys')
    a = actual[keys+values].sort_values(keys).reset_index(drop=True)
    e = expected[keys+values].sort_values(keys).reset_index(drop=True)
    check(len(a) == len(e), label+' row count mismatch')
    for key in keys:
        check(a[key].astype(str).equals(e[key].astype(str)),label+' key mismatch: '+key)
    for value in values:
        check(np.allclose(pd.to_numeric(a[value]),pd.to_numeric(e[value]),atol=1e-6,rtol=0),label+' mismatch: '+value)


def frame_sha(frame):
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes())
    digest.update(json.dumps([(str(k),str(v)) for k,v in frame.dtypes.items()]).encode('utf-8'))
    return digest.hexdigest()


def verify_input_binding(status, daily, predictions, sessions, policy, start, end):
    # Rebuild only immutable input transforms, without importing the engine.
    dates = pd.DatetimeIndex(pd.to_datetime(sessions))
    p = predictions.copy()
    p['signal_date'] = pd.to_datetime(p.signal_date)
    p['execution_date'] = p.signal_date.map(dict(zip(dates[:-1],dates[1:])))
    for column in ('mu5','mu10','mu20','risk5','risk10','risk20','prob5','prob10','prob20'):
        if column in p:
            p[column] = pd.to_numeric(p[column],errors='coerce')
    if 'utility' not in p:
        p['utility'] = p.mu20-policy['risk_aversion']*p.risk20
    p['utility'] = pd.to_numeric(p.utility,errors='coerce')
    d = daily.copy()
    if 'date' not in d:
        d['date'] = pd.to_datetime(d.trade_date.astype(str))
    d['date'] = pd.to_datetime(d.date)
    selected_dates = dates[(dates>=start)&(dates<=end)]
    d = d[d.date.isin(selected_dates)]
    check(status.get('prediction_frame_sha256')==frame_sha(p),'Prediction frame fingerprint does not match account input')
    check(status.get('active_daily_frame_sha256')==frame_sha(d),'Raw quote frame fingerprint does not match account input')
    check(status.get('sessions_sha256')==hashlib.sha256(dates.asi8.tobytes()).hexdigest(),'Session calendar fingerprint does not match account input')


def reconstruct(outputs, daily, predictions, sessions, actions, limits, policy, start, end, suspension_marks=None):
    """Pure independent cash/share book and causal signal reconstruction."""
    dates = pd.DatetimeIndex(pd.to_datetime(sessions))
    check(not dates.has_duplicates and dates.is_monotonic_increasing and not dates.hasnans, 'Invalid sessions')
    indexes = {day:i for i,day in enumerate(dates)}
    account_dates = dates[(dates >= start) & (dates <= end)]
    check(len(account_dates)>0 and indexes[account_dates[0]]>0,'Missing preceding session')
    check(policy['initial_cash'] == 25000 and policy['capital_cap'] == 25000,'Wrong capital account')
    check(1 <= policy['max_names'] <= 3 and policy['cooldown_sessions'] >= 2,'Invalid concentration/cooldown policy')
    d = daily.copy()
    if 'date' not in d:
        d['date'] = pd.to_datetime(d.trade_date.astype(str))
    d['date'] = pd.to_datetime(d.date)
    check(not d.duplicated(['date','ts_code']).any(),'Duplicate daily quote keys')
    q_by_day = {day:g.set_index('ts_code') for day,g in d[d.date.isin(account_dates)].groupby('date')}
    check(set(account_dates).issubset(q_by_day),'Missing daily account session')
    p = predictions.copy()
    p['signal_date'] = pd.to_datetime(p.signal_date)
    if 'utility' not in p:
        p['utility'] = p.mu20-policy['risk_aversion']*p.risk20
    check(not p.duplicated(['signal_date','ts_code']).any() and p.signal_date.isin(dates).all(),'Invalid prediction keys')
    check(np.isfinite(p.utility).all(),'Invalid prediction utilities')
    predictions_by_date = {day:g.set_index('ts_code') for day,g in p.groupby('signal_date')}
    trades = outputs['trades'].copy()
    for key in ('date','signal_date'):
        trades[key] = pd.to_datetime(trades[key])
    check(trades.date.isin(account_dates).all(),'Non-session trade')
    check(trades.side.isin(['buy','sell']).all(),'Unknown trade side')
    check(not trades.duplicated(['date','code','side']).any(),'Duplicate order fill')
    trade_days = {day:g for day,g in trades.groupby('date')}
    cash, shares, receivable, locked, recorded = 25000., {}, {}, {}, {}
    anchors, pending, sold_at, previous_values = {}, {}, {}, {}
    navs, contributions, holdings, events, allocations = [], [], [], [], []
    last_nav = 25000.
    marks = suspension_marks or {}
    for day in pd.date_range(start,end):
        dividends = {}
        for action in actions[actions.ex_date.eq(day)].itertuples():
            entitled = recorded.get(action.event,0)
            cash_entitlement, bonus = entitled*action.cash, math.floor(entitled*action.stock+1e-8)
            if cash_entitlement:
                receivable[action.event] = (cash_entitlement,action.pay_date)
                dividends[action.ts_code] = dividends.get(action.ts_code,0.)+cash_entitlement
            if bonus:
                shares[action.ts_code] = shares.get(action.ts_code,0)+bonus
                locked[action.event] = (action.ts_code,bonus,action.div_listdate)
            if cash_entitlement or bonus:
                events.append(dict(date=day,event=action.event,code=action.ts_code,cash_entitlement=cash_entitlement,
                                   bonus_shares=bonus,fractional_bonus_discarded=entitled*action.stock-bonus))
        for event,(amount,payday) in list(receivable.items()):
            if payday <= day:
                cash += amount
                del receivable[event]
        for event,(_,_,listing) in list(locked.items()):
            if listing <= day:
                del locked[event]
        for code,anchor in anchors.items():
            changes = actions[actions.ts_code.eq(code) & actions.ex_date.eq(day)]
            if len(changes):
                anchor['entry'] = (anchor['entry']-changes.cash.sum())/(1+changes.stock.sum())
                anchor['peak'] = (anchor['peak']-changes.cash.sum())/(1+changes.stock.sum())
        if day not in q_by_day:
            check(not dividends,'Ex-date outside account calendar')
            for a in actions[actions.record_date.eq(day)].itertuples():
                recorded[a.event] = shares.get(a.ts_code,0)
            continue
        index, q = indexes[day],q_by_day[day]
        price = lambda code,field: float(q.loc[code,field]) if code in q.index else marks[(day,code,field)]
        day_trades = trade_days.get(day,trades.iloc[:0])
        buys, sells = {}, {}
        # Pending risk/model exits remain until the first executable opening.
        for code,(reason,signal_day) in list(pending.items()):
            unlocked = shares[code]-sum(n for c,n,_ in locked.values() if c==code)
            fills = day_trades[day_trades.code.eq(code) & day_trades.side.eq('sell')]
            if unlocked<=0 or code not in q.index or index<=anchors[code]['entry_index']:
                check(fills.empty,'Blocked/T+1 sale executed')
                continue
            check((day,code) in limits,'Missing pending-order exact limit')
            down,up = limits[(day,code)]
            opening = price(code,'open')
            check(down-.005 <= opening <= up+.005,'Opening outside official limits')
            if opening <= down+.005:
                check(fills.empty,'Lower-limit blocked sale executed')
                continue
            check(len(fills)==1,'Executable pending sale was omitted')
            fill = fills.iloc[0]
            check(fill.reason == reason and fill.signal_date == signal_day,'Pending signal changed/cancelled')
            equal(fill.shares,unlocked,'Incorrect available sale quantity')
            equal(fill.price,opening,'Sale not at actual next executable opening',.005)
            cash += unlocked*opening
            shares[code] -= unlocked
            sells[code] = unlocked*opening
            sold_at[code] = index
            if shares[code]==0:
                del pending[code]
                del anchors[code]
        actual_sells = day_trades[day_trades.side.eq('sell')]
        check(set(actual_sells.code)==set(sells),'Sale has no previously known exit signal')
        retained = {c for c,n in shares.items() if n>0}
        held_open = sum(shares[c]*price(c,'open') for c in retained)
        equity_open = cash+held_open+sum(x[0] for x in receivable.values())
        available = max(0.,min(cash,min(equity_open,25000.)-held_open))
        actual_buys = day_trades[day_trades.side.eq('buy')]
        previous_date = dates[index-1]
        prior = predictions_by_date.get(previous_date,p.iloc[:0].set_index('ts_code'))
        for fill in actual_buys.itertuples():
            check(mainboard(fill.code),'Unauthorized buy board')
            check(fill.code not in retained and fill.code in prior.index,'Buy has no prior-session prediction or adds existing holding')
            check(fill.signal_date==previous_date,'Buy did not use immediately previous completed session')
            check(index-sold_at.get(fill.code,-1000000)>=policy['cooldown_sessions'],'Stock-specific cooldown violated')
            row = prior.loc[fill.code]
            check(row.utility>0 and ('eligible' not in row or row.eligible==True),'Nonpositive/ineligible opportunity bought')
            check(fill.shares>0 and fill.shares%100==0,'Buy not positive whole lots')
            check(fill.code in q.index and (day,fill.code) in limits,'Missing executable buy quote/official limits')
            opening = price(fill.code,'open')
            down,up = limits[(day,fill.code)]
            check(down+.005<opening<up-.005,'Limit-locked buy executed')
            equal(fill.price,opening,'Buy not actual next opening',.005)
            amount = fill.shares*opening
            cash -= amount
            shares[fill.code] = int(fill.shares)
            anchors[fill.code] = dict(entry=opening,peak=opening,entry_index=index)
            buys[fill.code] = amount
            allocations.append(dict(date=day,signal_date=previous_date,code=fill.code,shares=int(fill.shares),
                                    price=opening,amount=amount,utility=float(row.utility),
                                    target_budget=min(equity_open,25000.),available_budget=available))
        check(sum(buys.values())<=available+1e-6,'New investment exceeded residual 25000 cap/cash')
        check(cash>=-1e-6,'Cash became negative/hidden financing')
        check(sum(n>0 for n in shares.values())<=policy['max_names'],'Too many holdings')
        closing_predictions = predictions_by_date.get(day,p.iloc[:0].set_index('ts_code'))
        competitor = []
        equity_close = cash+sum(n*price(c,'close') for c,n in shares.items() if n)+sum(x[0] for x in receivable.values())
        for code,row in closing_predictions.iterrows():
            if (mainboard(code) and not shares.get(code,0) and code in q.index
                and price(code,'close')*100 <= min(equity_close,25000.)
                and index-sold_at.get(code,-1000000)>=policy['cooldown_sessions']
                and ('eligible' not in row or row.eligible==True)):
                competitor.append(float(row.utility))
        best = max(competitor) if competitor else None
        for code,n in shares.items():
            if not n:
                continue
            closing, anchor = price(code,'close'),anchors[code]
            anchor['peak'] = max(anchor['peak'],closing)
            utility = float(closing_predictions.loc[code,'utility']) if code in closing_predictions.index else None
            reason = None
            if closing<=anchor['entry']*(1-policy['hard_stop']):
                reason = 'hard_stop'
            elif (anchor['peak']>=anchor['entry']*(1+policy['trailing_activation'])
                  and closing<=anchor['peak']*(1-policy['trailing_distance'])):
                reason = 'trailing_stop'
            elif index-anchor['entry_index']>=policy['minimum_holding_sessions'] and utility is not None:
                if utility<=0:
                    reason = 'nonpositive_expected_utility'
                elif best is not None and best>utility+policy['switch_margin']:
                    reason = 'better_predicted_opportunity'
            if code not in pending and reason:
                pending[code] = (reason,day)
            holdings.append(dict(date=day,code=code,shares=n,price=closing,value=n*closing,
                                 locked_shares=sum(qty for c,qty,_ in locked.values() if c==code),
                                 entry=anchor['entry'],peak=anchor['peak'],pending_exit=code in pending))
        for action in actions[actions.record_date.eq(day)].itertuples():
            recorded[action.event] = shares.get(action.ts_code,0)
        current_values = {c:n*price(c,'close') for c,n in shares.items() if n}
        total_profit = 0.
        for code in set(previous_values)|set(current_values)|set(buys)|set(sells)|set(dividends):
            profit = current_values.get(code,0)-previous_values.get(code,0)-buys.get(code,0)+sells.get(code,0)+dividends.get(code,0)
            contributions.append(dict(date=day,code=code,profit=profit))
            total_profit += profit
        nav = cash+sum(current_values.values())+sum(x[0] for x in receivable.values())
        equal(nav-last_nav,total_profit,'Independent P&L cash conservation failed')
        navs.append(dict(date=day,cash=cash,stock_value=sum(current_values.values()),
                         receivable=sum(x[0] for x in receivable.values()),equity=nav,profit=nav-last_nav,return_=nav/last_nav-1))
        last_nav,previous_values = nav,current_values
    reconstructed = pd.DataFrame(navs)
    actual_nav = outputs['daily_nav'].copy()
    actual_nav['date'] = pd.to_datetime(actual_nav.date)
    compare_frames(actual_nav,reconstructed,['date'],['cash','stock_value','receivable','equity','profit'],'Daily account')
    actual_holds = outputs['holdings'].copy()
    actual_holds['date'] = pd.to_datetime(actual_holds.date)
    expected_holds = pd.DataFrame(holdings,columns=['date','code','shares','price','value','locked_shares','entry','peak','pending_exit'])
    compare_frames(actual_holds,expected_holds,['date','code'],['shares','price','value','locked_shares','entry','peak','pending_exit'],'Holdings')
    actual_contrib = outputs['stock_daily_contributions'].copy()
    actual_contrib['date'] = pd.to_datetime(actual_contrib.date)
    expected_contrib = pd.DataFrame(contributions,columns=['date','code','profit'])
    compare_frames(actual_contrib,expected_contrib,['date','code'],['profit'],'Stock daily attribution')
    expected_alloc = pd.DataFrame(allocations,columns=['date','signal_date','code','shares','price','amount','utility','target_budget','available_budget'])
    actual_alloc = outputs['allocations'].copy()
    for key in ('date','signal_date'):
        actual_alloc[key] = pd.to_datetime(actual_alloc[key])
    compare_frames(actual_alloc,expected_alloc,['date','code'],['shares','price','amount','utility','target_budget','available_budget'],'Allocation audit')
    actual_events = outputs['corporate_events'].copy()
    actual_events['date'] = pd.to_datetime(actual_events.date)
    expected_events = pd.DataFrame(events,columns=['date','event','code','cash_entitlement','bonus_shares','fractional_bonus_discarded'])
    compare_frames(actual_events,expected_events,['date','event','code'],['cash_entitlement','bonus_shares','fractional_bonus_discarded'],'Corporate events')
    monthly = reconstructed.groupby(reconstructed.date.dt.to_period('M')).tail(1).copy()
    previous = np.r_[25000.,monthly.equity.to_numpy()[:-1]]
    monthly['profit'] = monthly.equity-previous
    monthly['return'] = monthly.equity/previous-1
    monthly['budget_return'] = monthly.profit/25000.
    monthly['target_hit'] = monthly.profit>=7500.-1e-7
    monthly['month'] = monthly.date.dt.strftime('%Y-%m')
    actual_monthly = outputs['account'].copy()
    actual_monthly['date'] = pd.to_datetime(actual_monthly.date)
    compare_frames(actual_monthly,monthly,['date'],['equity','profit','return','budget_return','target_hit'],'Monthly account')
    nav = np.r_[25000.,reconstructed.equity.to_numpy()]
    checks = dict(passed=True,independent_book=True,initial_cash=25000.,capital_cap=25000.,
                  months=len(monthly),sessions=len(reconstructed),trades=len(trades),
                  ending_equity=float(nav[-1]),profit=float(nav[-1]-25000.),
                  max_daily_drawdown_pct=float(100*(1-(nav/np.maximum.accumulate(nav)).min())),
                  target_hit_months=int(monthly.target_hit.sum()),
                  mainboard_wholelot_nextopen_T1=True,exact_vendor_limits=True,
                  cash_share_NAV_corporate_actions_reconciled=True,pending_exits_independently_verified=True)
    return checks,monthly,expected_contrib,reconstructed


def audit_account(out, daily, predictions, sessions, output_dir=None, limit_frames=None, actions=None):
    out = Path(out)
    status = json.loads((out/'daily_run_status.json').read_text(encoding='utf-8'))
    check(status.get('account_complete') is True,'Incomplete account cannot be audited as final')
    protocol = json.loads((out/'execution_protocol.json').read_text(encoding='utf-8'))
    start,end = pd.Timestamp(status['start']),pd.Timestamp(status['end'])
    check(protocol.get('fees')==0 and protocol.get('external_topups') is False,'Unexpected fees/financing protocol')
    verify_input_binding(status,daily,predictions,sessions,protocol['policy'],start,end)
    sources = {}
    for path,hash_value in status.get('source_sha256',{}).items():
        check(Path(path).exists() and sha(path)==hash_value,'Source fingerprint changed after account run: '+Path(path).name)
        sources[str(Path(path).resolve())] = hash_value
    labels = ['account','daily_nav','trades','holdings','corporate_events','allocations','stock_daily_contributions','execution_checks']
    outputs = {name:pd.read_csv(out/f'{name}.csv') for name in labels}
    needed = set((pd.Timestamp(d),c) for d,c in zip(outputs['execution_checks'].date,outputs['execution_checks'].code))
    needed |= set((pd.Timestamp(d),c) for d,c in zip(outputs['trades'].date,outputs['trades'].code))
    limits,limit_sha = load_exact_limits(out,status,needed,limit_frames)
    sources.update(limit_sha)
    quote_data = daily.copy()
    if 'date' not in quote_data:
        quote_data['date'] = pd.to_datetime(quote_data.trade_date.astype(str))
    quote_data['date'] = pd.to_datetime(quote_data.date)
    quote_lookup = quote_data.set_index(['date','ts_code'],verify_integrity=True)['open']
    for row in outputs['execution_checks'].itertuples():
        key = (pd.Timestamp(row.date),row.code)
        check(key in quote_lookup.index and key in limits,'Execution check has no original quote/official limit')
        opening = float(quote_lookup.loc[key])
        down,up = limits[key]
        equal(float(row.price),opening,'Reported execution check opening mismatch',.005)
        equal(float(row.down_limit),down,'Reported official lower limit mismatch')
        equal(float(row.up_limit),up,'Reported official upper limit mismatch')
        expected = opening>down+.005 if row.side=='sell' else down+.005<opening<up-.005
        check(row.side in ('buy','sell') and bool(row.executable)==expected,'Reported execution eligibility mismatch')
    if actions is None:
        actions,action_sha = raw_actions(out,set(outputs['trades'].loc[outputs['trades'].side.eq('buy'),'code']),start,end)
        sources.update(action_sha)
    verified_marks = {}
    stale = pd.read_csv(out/'suspension_marks.csv')
    for row in stale.itertuples():
        day,code = pd.Timestamp(row.date),row.code
        history_path = ROOT/'suspension_checks'/f'{code}_{day:%Y%m%d}_daily.pkl'
        susp_path = ROOT/'suspension_checks'/f'{code}_{day:%Y%m%d}_suspend.pkl'
        check(history_path.exists() and susp_path.exists(),'Missing suspension source proof')
        quotes = pd.read_pickle(history_path)
        check(quotes.ts_code.eq(code).all(),'Wrong suspension quote stock')
        quotes = quotes.assign(date=pd.to_datetime(quotes.trade_date.astype(str))).sort_values('date')
        check(len(quotes)>0 and quotes.date.le(day).all() and quotes.date.iloc[-1]<day,'Suspension mark is stale/future traded quote')
        susp = pd.read_pickle(susp_path)
        check(susp.ts_code.eq(code).all(),'Wrong suspension history stock')
        susp = susp.assign(date=pd.to_datetime(susp.trade_date.astype(str)))
        valid = susp[susp.date.gt(quotes.date.iloc[-1]) & susp.date.le(day)]
        stops,resumes = valid.loc[valid.suspend_type.eq('S'),'date'].max(),valid.loc[valid.suspend_type.eq('R'),'date'].max()
        check(pd.notna(stops) and (pd.isna(resumes) or resumes<stops),'Suspension not independently confirmed')
        value = float(quotes.close.iloc[-1])
        changes = actions[actions.ts_code.eq(code)&actions.ex_date.gt(quotes.date.iloc[-1])&actions.ex_date.le(day)]
        for _,group in changes.groupby('ex_date'):
            value = (value-group.cash.sum())/(1+group.stock.sum())
        equal(float(row.mark),value,'Suspension valuation mismatch')
        verified_marks[(day,code,row.field)] = value
        sources[str(history_path.resolve())],sources[str(susp_path.resolve())] = sha(history_path),sha(susp_path)
    results = reconstruct(outputs,daily,predictions,sessions,actions,limits,protocol['policy'],start,end,verified_marks)
    checks,monthly,contributions,reconstructed = results
    metrics = json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    for key,value in [('ending_equity',checks['ending_equity']),('total_profit',checks['profit']),
                      ('max_daily_drawdown_pct',checks['max_daily_drawdown_pct']),('target_hit_months',checks['target_hit_months'])]:
        equal(metrics[key],value,'Published metric disagrees with independent book: '+key)
    sources[str(Path(__file__).resolve())] = sha(__file__)
    for label in labels+['suspension_marks']:
        sources[str((out/f'{label}.csv').resolve())] = sha(out/f'{label}.csv')
    checks['source_sha256'] = sources
    checks['input_binding_verified'] = True
    destination = Path(output_dir) if output_dir else out/'independent_audit'
    check(destination.resolve()!=out.resolve(),'Audit must not overwrite source account')
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'independent_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    monthly.to_csv(destination/'independent_monthly.csv',index=False)
    contributions.to_csv(destination/'independent_stock_daily_contributions.csv',index=False)
    reconstructed.to_csv(destination/'independent_daily_nav.csv',index=False)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account-dir',required=True)
    parser.add_argument('--daily',required=True)
    parser.add_argument('--predictions',required=True)
    parser.add_argument('--sessions',required=True)
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    read = lambda p: pd.read_pickle(p) if Path(p).suffix=='.pkl' else pd.read_csv(p)
    checks = audit_account(args.account_dir,read(args.daily),read(args.predictions),
                           json.loads(Path(args.sessions).read_text(encoding='utf-8')),args.output_dir)
    print(json.dumps({k:v for k,v in checks.items() if k!='source_sha256'},indent=2))


if __name__ == '__main__':
    main()
