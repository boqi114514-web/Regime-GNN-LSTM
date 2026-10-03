"""Publish complete, independently audited daily experiment accounts only."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from research_daily_graph_data import sha256,publish_manifest
from research_daily_opportunity import validate_predictions

VARIANTS={
    'neural_v1':'daily_opportunity_20261002',
    'no_graph':'daily_opportunity_no_graph_20261002',
    'trees_v1':'daily_opportunity_trees_20261002',
    'attention_rank_v2':'daily_opportunity_ranked_20261002',
    'v2_prior_band':'daily_opportunity_ranked_band_20261002',
    'v2_online':'daily_opportunity_online_20261002',
    'v2_online_prior_band':'daily_opportunity_online_band_20261002',
    'v2_prior_band_trail8':'daily_opportunity_ranked_band_trail8_20261002',
    'leadership_masked_v3':'daily_opportunity_leadership_band_account_20261002',
    'leadership_masked_trees':'daily_opportunity_leadership_trees_band_account_20261002',
    'trend_residual':'daily_opportunity_trend_residual_band_account_20261002',
    'trend_residual_liquid':'daily_opportunity_trend_residual_liquid_account_20261003',
    'trend_residual_liquid_cool':'daily_opportunity_trend_residual_liquid_cool_account_20261003',
    'path_payoff_prior_band':'daily_opportunity_path_band_account_20261003',
    'path_payoff_liquid':'daily_opportunity_path_liquid_account_20261003',
}


def summarize(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    months,metrics,records=[],[],{}
    for name,folder in VARIANTS.items():
        source=REPO/'results'/folder;account=source/'account'
        audit=account/'independent_audit/independent_checks.json'
        if not audit.exists():
            records[name]=dict(status='not_independently_completed',path=str(source))
            continue
        metadata=json.loads((source/'feature_manifest.json').read_text(encoding='utf-8'))
        if metadata['feature_protocol']['version']=='daily_opportunity_leadership_features_v3':
            from research_daily_opportunity_leadership import leadership_variant,validate_leadership
            context=leadership_variant()
        else:
            context=nullcontext()
        with context:
            if metadata['feature_protocol']['version']=='daily_opportunity_leadership_features_v3':
                validate_leadership(source)
            validate_predictions(source)
        if (source/'extension_manifest.json').exists():
            from research_daily_opportunity_extension_filter import validate_extension_inputs
            validate_extension_inputs(source)
        elif (source/'liquidity_manifest.json').exists():
            from research_daily_opportunity_liquid_filter import validate_liquidity_inputs
            validate_liquidity_inputs(source)
        if (source/'path_filter_manifest.json').exists():
            from research_daily_opportunity_path_runner import validate_path_filter
            validate_path_filter(source)
        checks=json.loads(audit.read_text(encoding='utf-8'))
        if not checks.get('passed') or not checks.get('input_binding_verified'):
            raise ValueError('Incomplete independent audit: '+name)
        for file,expected in checks['source_sha256'].items():
            if sha256(file)!=expected:
                raise ValueError('Changed independently audited input: '+file)
        status=json.loads((account/'daily_run_status.json').read_text(encoding='utf-8'))
        if not status.get('account_complete') or status['end']!='2026-09-24':
            raise ValueError('Incomplete / different account period')
        f=pd.read_csv(account/'account.csv')
        if len(f)!=9 or f.month.duplicated().any():
            raise ValueError('Invalid account month coverage')
        m=json.loads((account/'metrics.json').read_text(encoding='utf-8'))
        if not np.isclose(f.profit.sum(),m['total_profit']) or not np.isclose(m['total_profit'],checks['profit']):
            raise ValueError('Monthly profit disagreement')
        months.append(f.assign(variant=name))
        metrics.append(dict(variant=name,**m))
        records[name]=dict(status='complete_independently_audited',path=str(source),
            prediction_manifest_sha256=sha256(source/'prediction_manifest.json'),
            independent_audit_sha256=sha256(audit))
    if not months:
        raise ValueError('No independently completed accounts')
    monthly=pd.concat(months,ignore_index=True)
    monthly.to_csv(destination/'monthly_results.csv',index=False,encoding='utf-8-sig')
    monthly.pivot(index='month',columns='variant',values='profit').to_csv(destination/'monthly_profit_matrix.csv',encoding='utf-8-sig')
    pd.DataFrame(metrics).to_csv(destination/'metrics.csv',index=False,encoding='utf-8-sig')
    publish_manifest(records,destination/'accounts.json')
    publish_manifest(dict(status='complete_for_listed_audited_accounts',
        period=['2026-01-01','2026-09-24'],initial_cash=25000,budget_denominator=25000,
        target_monthly_profit=7500,month_splice=False,source_sha256=sha256(__file__),
        artifacts={file:sha256(destination/file) for file in ('monthly_results.csv','monthly_profit_matrix.csv','metrics.csv','accounts.json')}),
        destination/'comparison_manifest.json')
    print(pd.DataFrame(metrics).to_string(index=False))
    return monthly


def examples():
    for name in ('daily_opportunity_ranked_20261002','daily_opportunity_leadership_20261002','daily_opportunity_leadership_trees_20261002'):
        source=REPO/'results'/name
        p=pd.read_pickle(source/'predictions.pkl')
        f=p[p.ts_code.isin(['600183.SH','002384.SZ'])&p.signal_date.between('2026-04-01','2026-06-30')].copy()
        f['month']=f.signal_date.dt.strftime('%Y-%m')
        print(name)
        print(f.groupby(['month','ts_code']).agg(utility=('utility','mean'),mu20=('mu20','mean'),risk20=('risk20','mean'),
            positive_sessions=('utility',lambda x:int(x.gt(0).sum())),sessions=('utility','size')).to_string())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=REPO/'results/daily_opportunity_comparison')
    parser.add_argument('--examples',action='store_true')
    args=parser.parse_args()
    if args.examples:examples()
    else:summarize(args.out)
