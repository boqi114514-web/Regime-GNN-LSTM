"""Predeclared leadership ablations; never overwrite the released account outputs."""
import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

import config
import small_account_backtest as engine
from data_pipeline.execution_data import ROOT
from data_pipeline.tushare_config import get_pro
from s7_budget_portfolio import is_main_board, optimize_portfolio as old_optimizer

BASE = Path(config.OUTPUT_DIR)/'stock_execution_research'
OUT = Path(config.OUTPUT_DIR)/'leadership_research'
VARIANTS = ('fast_sector', 'global_fast', 'global_peer', 'global_peer_flexible')
PROTOCOL = dict(
    hypothesis='Hard L1 gate, slow defensive stock ranks, and 30% lot ceiling miss persistent leaders',
    scope='2023-01 to 2026-09-24, all previously examined; not untouched out-of-sample',
    fast_score={'mom3_rank': .5, 'mom6_rank': .3, 'mom1_rank': .2},
    peer_score='75% fast score + 25% contemporaneous historical L2 median 3-month momentum rank',
    standard=dict(stock_cap=.30, l1_cap=.60, names=[4, 5], max_names_per_l1=2),
    flexible=dict(stock_cap=.50, l1_cap=.75, names=[2, 4], max_names_per_l1=2),
    universe='Same original candidate rows; all boards in ranking; main board only in execution',
    optimizer='Top 120 feasible scores; maximize sum(score**4 * invested_amount); no theme whitelist',
    variants=list(VARIANTS), fees=0, capital_cap=25000,
    selection='Publish every variant and every year, do not select on 2026Q2 alone',
    attribution='fast_sector isolates stock score; global variants also change allocation objective and concentration')

BASE_ACTION_LOADER = engine.load_actions


def research_actions(pro, code):
    """Extra equivalent gateway query forms; never infer missing actions as zero."""
    if (ROOT/'dividends'/f'{code}.pkl').exists():
        return BASE_ACTION_LOADER(pro, code)
    local = OUT/'dividends'/f'{code}.pkl'
    if local.exists():
        return engine.normalize_actions(pd.read_pickle(local), code)
    errors = []
    for period in ('20251231','20241231','20231231','20221231'):
        try:
            raw = pro.query('dividend', ts_code=code, end_date=period, offset=0)
            if raw.empty: raise ValueError('Empty dividend response')
            actions = engine.normalize_actions(raw, code)
            # This gateway often ignores end_date and returns full history.
            # A single report-period slice is insufficient for 45-month replay.
            periods = pd.to_numeric(raw.end_date, errors='coerce')
            if periods.nunique() < 2 or periods.eq(int(period)).all():
                raise ValueError('Report-period query is not a complete multi-year response')
            local.parent.mkdir(parents=True, exist_ok=True)
            raw.to_pickle(local)
            print('validated research dividend history', code, len(raw), flush=True)
            return actions
        except Exception as exc:
            errors.append(str(exc))
    try:
        return BASE_ACTION_LOADER(pro, code)
    except Exception as exc:
        raise RuntimeError(f'{code}: no complete dividend history; {errors}; default: {exc}') from None


def enrich(candidates, monthly, l2_members):
    c = candidates.copy()
    m = monthly.copy()
    m['period'] = m.date.dt.to_period('M')
    price = m.pivot(index='period', columns='ts_code', values='close')*m.pivot(index='period', columns='ts_code', values='adj_factor')
    price = price.reindex(pd.period_range(price.index.min(), price.index.max(), freq='M'))
    price.index.name = 'period'
    for horizon in (1, 3):
        values = (price/price.shift(horizon)-1).stack().rename(f'mom{horizon}').reset_index()
        values['month'] = values.period.dt.to_timestamp('M')
        c = c.merge(values.drop(columns='period'), on=['month', 'ts_code'], how='left', validate='one_to_one')
    if c[['mom1', 'mom3']].isna().any().any():
        raise ValueError('Missing short-horizon history in existing universe')
    mem = l2_members.copy()
    for col in ('in_date', 'out_date'):
        mem[col] = pd.to_datetime(mem[col].astype('string').str.replace(r'\.0$', '', regex=True), format='%Y%m%d', errors='coerce')
    blocks = []
    for date, block in c.groupby('month', sort=True):
        block = block.copy()
        available = mem[mem.in_date.notna() & (mem.in_date <= block.date.max()) & (mem.out_date.isna() | (mem.out_date > block.date.max()))]
        available = available[['ts_code', 'l2_code']].drop_duplicates()
        if available.ts_code.duplicated().any():
            raise ValueError('Conflicting historical L2 memberships')
        block = block.merge(available, on='ts_code', how='left', validate='one_to_one')
        rank = block[['mom1', 'mom3', 'mom6']].rank(pct=True)
        block['fast_score'] = .5*rank.mom3+.3*rank.mom6+.2*rank.mom1
        peers = block.groupby('l2_code').mom3.agg(['median', 'count'])
        peer_rank = peers.loc[peers['count'] >= 5, 'median'].rank(pct=True)
        block['peer_rank'] = block.l2_code.map(peer_rank).fillna(.5)
        block['peer_score'] = .75*block.fast_score+.25*block.peer_rank
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def leader_optimizer(candidates, budget, flexible=False, stock_cap=None, sector_cap=None, name_limit=None,
                     minimum_names=None, sector_name_limit=2, score_power=4, **kwargs):
    cap, group_cap = (.5, .75) if flexible else (.3, .6)
    cap = cap if stock_cap is None else stock_cap
    group_cap = group_cap if sector_cap is None else sector_cap
    maximum = min(kwargs.get('max_names', 5), 4 if flexible else 5)
    if name_limit is not None: maximum = min(maximum,name_limit)
    minimum = min(kwargs.get('min_names', 4), 2 if flexible else 4, maximum)
    if minimum_names is not None: minimum = min(minimum_names, maximum)
    c = candidates.copy()
    c['lot_value'] = (100*c.reference_price).round(2)
    c['max_lots'] = np.floor(budget*cap/c.lot_value).astype(int)
    c = c[c.max_lots >= 1].sort_values(['leadership_score', 'ts_code'], ascending=[False, True]).head(120).reset_index(drop=True)
    if c.empty:
        raise ValueError('所有候选的一手价格超过仓位上限')
    n = len(c)
    if score_power not in (1,4):
        raise ValueError('Unsupported optimizer score power')
    objective = np.r_[-c.leadership_score.to_numpy()**score_power*c.lot_value.to_numpy()/budget, np.zeros(n)]
    rows, upper, lower = [], [], []
    def add(row, hi, lo=-np.inf):
        rows.append(row); upper.append(hi); lower.append(lo)
    add(np.r_[c.lot_value, np.zeros(n)], budget)
    for i in range(n):
        row = np.zeros(2*n); row[i] = 1; row[n+i] = -c.max_lots.iloc[i]; add(row, 0)
        row = np.zeros(2*n); row[i] = -1; row[n+i] = 1; add(row, 0)
    for inds in c.groupby('ind_code').indices.values():
        if sector_name_limit is not None:
            row = np.zeros(2*n); row[n+inds] = 1; add(row, sector_name_limit)
        row = np.zeros(2*n); row[inds] = c.lot_value.iloc[inds]; add(row, budget*group_cap)
    row = np.r_[np.zeros(n), np.ones(n)]
    solved = None
    for requested in range(minimum, 0, -1):
        result = milp(objective, integrality=np.ones(2*n),
            bounds=Bounds(np.zeros(2*n), np.r_[c.max_lots, np.ones(n)]),
            constraints=LinearConstraint(np.vstack(rows+[row]), np.r_[lower, requested], np.r_[upper, maximum]),
            options={'time_limit': 20})
        if result.success:
            solved = result; break
    if solved is None:
        raise ValueError('找不到可买组合')
    c['lots'] = np.rint(solved.x[:n]).astype(int)
    c = c[c.lots > 0].copy()
    c['shares'] = c.lots*100
    c['planned_amount'] = c.lots*c.lot_value
    assert c.planned_amount.sum() <= budget+.01
    assert (c.planned_amount <= budget*cap+.01).all()
    assert (c.groupby('ind_code').planned_amount.sum() <= budget*group_cap+.01).all()
    assert c.stock_code.map(is_main_board).all()
    return c, {'budget': budget}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_file = OUT/'protocol.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text(encoding='utf-8')) != PROTOCOL:
        raise ValueError('Protocol changed; use a new experiment directory')
    protocol_file.write_text(json.dumps(PROTOCOL, ensure_ascii=False, indent=2), encoding='utf-8')
    c = enrich(pd.read_pickle(BASE/'candidates.pkl'), pd.read_pickle(ROOT/'stock_month_end_verified.pkl'),
               pd.read_csv(Path(config.LOCAL_DATA_RAW)/'ts_sw_l2_members.csv', dtype=str))
    c.to_pickle(OUT/'candidates.pkl')
    print('Prepared', len(c), 'candidate rows; variants fixed before account evaluation', flush=True)


def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    names = pd.read_csv(Path(config.LOCAL_DATA_RAW)/'ts_stock_basic.csv')
    members = pd.read_csv(Path(config.LOCAL_DATA_RAW)/'ts_sw_members.csv', dtype=str)
    industries = members.drop_duplicates('l1_code').set_index('l1_code').l1_name
    plans = pd.read_pickle(BASE/'industry_plans.pkl')
    p = plans[plans.strategy.isin(['new_full', 'original']) & plans.date.between('2026-03-01', '2026-05-31')].copy()
    p['industry'] = p.ts_code.map(industries)
    p[['date', 'strategy', 'industry_rank', 'industry', 'actual_ret']].to_csv(OUT/'q2_sector_audit.csv', index=False)
    for name in ('new_full', 'original'):
        h = pd.read_csv(BASE/f'holdings_{name}.csv')
        h = h[h.date.between('2026-04-01', '2026-06-30')].merge(names, left_on='code', right_on='ts_code')
        h.to_csv(OUT/f'q2_holdings_{name}.csv', index=False)
        print(name, h[['date', 'name', 'value']].to_string(index=False))
    # Named examples diagnose user-mentioned themes only; never a trading whitelist.
    sample_names = ['东山精密','沪电股份','深南电路','生益科技','兴森科技','鹏鼎控股','德明利','兆易创新','光迅科技','华工科技','亨通光电','通鼎互联','中际旭创','新易盛','胜宏科技','佰维存储']
    c = pd.read_pickle(BASE/'candidates.pkl').merge(names, on='ts_code')
    c = c[c.month.between('2026-03-01','2026-05-31') & c.name.isin(sample_names)].copy()
    sessions = json.loads((ROOT/'boundaries/sessions.json').read_text(encoding='utf-8'))
    rows = []
    for s in sessions:
        first, last = pd.Timestamp(s['first']), pd.Timestamp(s['last'])
        if first.year != 2026 or first.month not in (4,5,6): continue
        month = (first.to_period('M')-1).to_timestamp('M')
        q = engine.boundary_frame(first)
        end = engine.boundary_frame(last)
        factor_end = pd.read_pickle(ROOT/'adj_month_end'/f'{last:%Y%m%d}.pkl').set_index('ts_code').adj_factor
        factor_start = pd.read_pickle(ROOT/'adj_month_end'/f'{c.loc[c.month.eq(month),"date"].max():%Y%m%d}.pkl').set_index('ts_code').adj_factor
        for r in c[c.month.eq(month)].itertuples():
            price = q.loc[r.ts_code, 'open'] if r.ts_code in q.index else np.nan
            final = end.loc[r.ts_code, 'close'] if r.ts_code in end.index else np.nan
            adj_return = final*factor_end.get(r.ts_code, np.nan)/(r.close*factor_start.get(r.ts_code, np.nan))-1
            rows.append(dict(month=str(first.to_period('M')),name=r.name,ts_code=r.ts_code,
                sector_selected=r.ind_code in set(p.loc[p.strategy.eq('new_full') & p.date.eq(month),'ts_code']),
                mainboard=is_main_board(r.stock_code),original_rank=r.rank_in_ind,
                execution_open=price,one_lot=100*price,eligible_30pct=100*price<=7500,
                within_total_25000=100*price<=25000,month_adjusted_return=adj_return))
    pd.DataFrame(rows).to_csv(OUT/'q2_stock_exclusions.csv',index=False)
    # Original pre-intervention artifacts, never relabel repaired output as original.
    raw = subprocess.check_output(['git','show','818423c:results/stock_backtest.csv'], text=True, encoding='utf-8')
    old = pd.read_csv(io.StringIO(raw))
    old.to_csv(OUT/'original_818423c_stock_backtest.csv', index=False)
    print('Original artifact tail:',old.tail(4).to_string(index=False))
    print('Sector audit:',p[['date','strategy','industry','actual_ret']].to_string(index=False))


def run_variant(name, offline):
    c = pd.read_pickle(OUT/'candidates.pkl')
    plans = pd.read_pickle(BASE/'industry_plans.pkl')
    if name.startswith('conviction'):
        plans=pd.read_csv(OUT/'industry_holdings.csv',parse_dates=['date'])
        plans=plans[plans.trend_weight.eq(.5)].copy()
        plans['sector_rank']=plans.groupby('date').selection_score.rank(pct=True)
        lookup=plans[['date','ts_code','sector_rank']].rename(columns={'date':'month','ts_code':'ind_code'})
        c=c.merge(lookup,on=['month','ind_code'],how='inner',validate='many_to_one')
        c['leadership_score']=.25*c.sector_rank+.375*c.score+.375*c.peer_score
        if name == 'conviction100':
            engine.optimize_portfolio=lambda candidates,budget,**kw: leader_optimizer(candidates,budget,
                stock_cap=1.,sector_cap=1.,minimum_names=1,sector_name_limit=None,**kw)
        else:
            cap=.8 if name.endswith('80') else .5
            engine.optimize_portfolio=lambda candidates,budget,**kw: leader_optimizer(candidates,budget,True,
                stock_cap=cap,sector_cap=max(.75,cap),name_limit=3,**kw)
    elif name in ('fast_sector','hybrid_sector'):
        if name == 'fast_sector':
            c['score'] = c.fast_score
            c['rank_in_ind'] = c.groupby(['month','ind_code']).fast_score.rank(ascending=False, method='first')
            plans = plans[plans.strategy.eq('new_full')].copy()
        else:
            plans = pd.read_csv(OUT/'industry_holdings.csv',parse_dates=['date'])
            plans = plans[plans.trend_weight.eq(.5)].copy()
            (OUT/'hybrid_account_protocol.json').write_text(json.dumps(dict(
                chosen_after='sector-layer 0/.25/.5/.75/1 exploration; not an independent holdout',
                trend_weight=.5,stock_layer='unchanged released stock candidates and integer allocator',
                period='full 2023-01 to 2026-09-24, no selective month substitution'),indent=2),encoding='utf-8')
        engine.optimize_portfolio = old_optimizer
    else:
        c['leadership_score'] = c.peer_score if name.startswith('global_peer') else c.fast_score
        plans = c[['month','ind_code']].drop_duplicates().rename(columns={'month':'date','ind_code':'ts_code'})
        plans['selection_score'] = 0.
        plans['risk_exposure'] = 1.
        engine.optimize_portfolio = lambda candidates,budget,**kw: leader_optimizer(candidates,budget,name.endswith('flexible'),**kw)
    plans['strategy'] = name
    engine.OUT = OUT
    engine.load_actions = research_actions
    sessions = json.loads((ROOT/'boundaries/sessions.json').read_text(encoding='utf-8'))
    return engine.run(name,c,plans,sessions,engine.OfflineClient() if offline else get_pro())


def summarize():
    rows=[]
    monthly=[]
    for name, root in [('new_full',BASE),('original',BASE)]+[(v,OUT) for v in VARIANTS+('hybrid_sector','conviction50','conviction80','conviction100')]:
        path=root/f'account_{name}.csv'
        if not path.exists(): continue
        df=pd.read_csv(path,parse_dates=['date'])
        df['profit']=df.equity-df.equity.shift(1,fill_value=25000.)
        df['budget_return_pct']=100*df.profit/25000.
        df['target_hit']=df.profit>=7500.-1e-7
        monthly.append(df.assign(strategy=name))
        for year, g in df.groupby(df.date.dt.year):
            rows.append(dict(strategy=name,year=year,months=len(g),return_pct=100*((1+g['return']).prod()-1),
                profit=g.profit.sum(),budget_return_pct=g.budget_return_pct.sum(),target_hit_months=int(g.target_hit.sum()),
                worst_month_profit=g.profit.min(),ending_equity=g.equity.iloc[-1]))
        q=df[df.date.between('2026-04-01','2026-06-30')]
        nav=np.r_[25000.,df.equity.to_numpy()]
        print(name,'Q2',100*((1+q['return']).prod()-1),'DD',100*np.min(nav/np.maximum.accumulate(nav)-1),'equity',nav[-1])
    pd.DataFrame(rows).to_csv(OUT/'annual.csv',index=False)
    pd.concat(monthly,ignore_index=True).to_csv(OUT/'monthly_account_summary.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def verify_completed():
    """Independent reconciliation and reproducibility fingerprints, completed runs only."""
    import verify_small_account as verifier
    verifier.OUT = OUT
    results=[]
    for path in sorted(OUT.glob('account_*.csv')):
        name=path.stem.removeprefix('account_')
        results.append(verifier.verify(name))
    files=list(ROOT.rglob('*.pkl'))+list((OUT/'dividends').glob('*.pkl'))+[
        ROOT/'boundaries/sessions.json', OUT/'candidates.pkl', BASE/'industry_plans.pkl',
        OUT/'industry_holdings.csv', Path(__file__), Path(engine.__file__),
        Path(verifier.__file__), Path(config.PROJECT_DIR)/'src/s7_budget_portfolio.py']
    files+=list(OUT.glob('*protocol.json'))
    fingerprints={str(p.relative_to(config.PROJECT_DIR)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(set(files))}
    result=dict(reconciliations=results, target_profit=7500,return_denominator=25000,
        end='2026-09-24', input_sha256=fingerprints)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))


def industry_ablation():
    """Independent sector-layer test, explicitly not a 25k stock account."""
    from price_risk_pipeline import industry_plans
    definition = dict(layer='industry only, no individual stocks or integer lots',
        trend='.5 rank(mom3) + .3 rank(mom6) + .2 rank(mom1)',
        trend_weights=[0.,.25,.5,.75,1.], top_k=5, inertia=.2,
        stage='additional exploration after inspecting the unblended trend result; no holdout claim',
        source='verified monthly index closes; signal t predicts holding month t+1')
    (OUT/'industry_protocol.json').write_text(json.dumps(definition,indent=2),encoding='utf-8')
    m=pd.read_csv(Path(config.LOCAL_DATA_RAW)/'ts_sw_industry_monthly.csv')
    m['month']=pd.to_datetime(m.date).dt.to_period('M')
    q=m.pivot(index='month',columns='ts_code',values='close').drop(columns=config.SW_EXCLUDE)
    q=q.reindex(pd.period_range(q.index.min(),q.index.max(),freq='M'))
    q.index.name='month'
    trend=.5*(q/q.shift(3)-1).rank(axis=1,pct=True)+.3*(q/q.shift(6)-1).rank(axis=1,pct=True)+.2*(q/q.shift(1)-1).rank(axis=1,pct=True)
    trend=trend.stack().rename('trend_score').reset_index()
    trend['date']=trend.month.dt.to_timestamp('M')
    pred=pd.read_pickle(Path(config.OUTPUT_DIR)/'price_risk_pipeline/predictions_ensemble.pkl')
    pred=pred.merge(trend[['date','ts_code','trend_score']],on=['date','ts_code'],validate='one_to_one')
    if pred.trend_score.isna().any(): raise ValueError('Missing sector momentum')
    returns, holdings=[],[]
    for weight in definition['trend_weights']:
        p=pred.copy()
        p['pred_ensemble']=(1-weight)*p.pred_ensemble+weight*p.trend_score
        p['risk_exposure']=1.
        plan=industry_plans(p)
        plan['trend_weight']=weight
        holdings.append(plan)
        for date,g in plan.groupby('date'):
            if g.actual_ret.isna().any(): continue
            returns.append(dict(trend_weight=weight,date=(date.to_period('M')+1).to_timestamp('M'),return_=g.actual_ret.mean()))
    result=pd.DataFrame(returns)
    rows=[]
    for (weight,year),g in result.groupby(['trend_weight',result.date.dt.year]):
        rows.append(dict(trend_weight=weight,year=year,months=len(g),return_pct=100*((1+g.return_).prod()-1)))
    pd.DataFrame(rows).to_csv(OUT/'industry_annual.csv',index=False)
    result.to_csv(OUT/'industry_monthly.csv',index=False)
    pd.concat(holdings).to_csv(OUT/'industry_holdings.csv',index=False)
    print(pd.DataFrame(rows).query('year>=2023').to_string(index=False))
    print(result[result.date.between('2026-04-01','2026-06-30')].to_string(index=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','audit','run','summary','industry','verify'])
    parser.add_argument('--variant',choices=VARIANTS+('hybrid_sector','conviction50','conviction80','conviction100'))
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='audit': audit()
    elif args.action=='summary': summarize()
    elif args.action=='industry': industry_ablation()
    elif args.action=='verify': verify_completed()
    else:
        if not args.variant: parser.error('--variant required')
        run_variant(args.variant,args.offline)
