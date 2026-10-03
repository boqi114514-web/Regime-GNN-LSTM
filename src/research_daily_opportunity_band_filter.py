"""A separately recorded, past-only official risk-band buy filter.

Uses the completed signal day's official limits, not a present-day ST list.
It does not change training, graph nodes, expected return or risk predictions.
Unknown limit data aborts collection; it is never classified as ordinary stock.
Existing positions are not forcibly sold by this filter.
"""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from research_daily_graph_data import publish_frame, publish_manifest, sha256
from research_daily_opportunity import validate_predictions
from research_daily_opportunity_execution import OfficialLimits, _mainboard


def filter_buy_risk_band(source,out,pro,minimum_fraction=.075,candidate_limit=40,cache_roots=()):
    source,out=Path(source),Path(out)
    if not 0<minimum_fraction<.1:
        raise ValueError('This filter requires a disclosed fraction between 0 and 10 percent')
    predictions=validate_predictions(source)
    out.mkdir(parents=True,exist_ok=True)
    for name in ('feature_store.pkl','feature_manifest.json','frozen_edges.pkl','fit_audits.json'):
        shutil.copy2(source/name,out/name)
    shutil.copytree(source/'checkpoints',out/'checkpoints',dirs_exist_ok=True)
    limits=OfficialLimits(out,pro=pro,cache_roots=cache_roots)
    predictions=predictions.copy();predictions['eligible']=False
    checks=[]
    for number,(day,frame) in enumerate(predictions.groupby('signal_date',sort=True)):
        pool=frame[frame.utility.gt(0)&frame.ts_code.map(_mainboard)].sort_values(['utility','ts_code'],ascending=[False,True]).head(candidate_limit)
        for index,row in pool.iterrows():
            official=limits(day,row.ts_code)
            fraction=(float(official.up_limit)-float(official.down_limit))/(float(official.up_limit)+float(official.down_limit))
            eligible=fraction>=minimum_fraction
            predictions.loc[index,'eligible']=eligible
            checks.append(dict(signal_date=day,ts_code=row.ts_code,up_limit=official.up_limit,
                               down_limit=official.down_limit,official_band_fraction=fraction,eligible=eligible))
        if number%20==0:
            print('official risk-band filter',number+1,'sessions','checks',len(checks),flush=True)
    pd.DataFrame(checks).to_csv(out/'prior_band_checks.csv',index=False)
    publish_frame(predictions,out/'predictions.pkl')
    predictions.to_csv(out/'predictions.csv',index=False,encoding='utf-8-sig')
    protocol=json.loads((source/'experiment_protocol.json').read_text(encoding='utf-8'))
    protocol['buy_filter']=dict(minimum_prior_signal_limit_fraction=minimum_fraction,
                                initial_model_pool=candidate_limit,
                                observed='Exact official signal-day limits, known before next open',
                                treatment='Mainboard 5%-band candidates blocked; not a current-name ST backfill',
                                retained_positions='Unchanged; existing stop/exit rules remain')
    protocol['source_sha256'][Path(__file__).name]=sha256(__file__)
    publish_manifest(protocol,out/'experiment_protocol.json')
    publish_manifest(dict(status='complete',checks=len(checks),accepted=int(predictions.eligible.sum()),minimum_fraction=minimum_fraction,
                          source_sha256=limits.sources,checks_sha256=sha256(out/'prior_band_checks.csv')),
                     out/'prior_band_manifest.json')
    artifacts={name:sha256(out/name) for name in ('predictions.pkl','fit_audits.json','experiment_protocol.json','feature_manifest.json',
                                               'prior_band_checks.csv','prior_band_manifest.json')}
    publish_manifest(dict(status='complete',rows=len(predictions),sessions=int(predictions.signal_date.nunique()),artifacts=artifacts),
                     out/'prediction_manifest.json')
    return predictions


def validate_band_inputs(out):
    """Recheck eligibility against frozen original vendor limits, not flags."""
    out=Path(out)
    manifest=json.loads((out/'prior_band_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status')!='complete' or sha256(out/'prior_band_checks.csv')!=manifest['checks_sha256']:
        raise ValueError('Incomplete/tampered band checks')
    checks=pd.read_csv(out/'prior_band_checks.csv')
    if len(checks)!=manifest['checks'] or checks.duplicated(['signal_date','ts_code']).any():
        raise ValueError('Band check coverage mismatch')
    originals={}
    for path,expected in manifest['source_sha256'].items():
        if sha256(path)!=expected:
            raise ValueError('Official prior band source fingerprint mismatch')
        raw=pd.read_pickle(path)
        if not {'ts_code','trade_date','up_limit','down_limit'}<=set(raw):
            raise ValueError('Missing official prior band schema')
        for r in raw.itertuples():
            key=(pd.Timestamp(str(r.trade_date)),r.ts_code)
            item=(float(r.down_limit),float(r.up_limit))
            if key in originals and originals[key]!=item:
                raise ValueError('Conflicting official prior bands')
            originals[key]=item
    p=validate_predictions(out).set_index(['signal_date','ts_code'])
    for row in checks.itertuples():
        key=(pd.Timestamp(row.signal_date),row.ts_code)
        if key not in originals or key not in p.index:
            raise ValueError('Band filter has no original signal-day observation')
        down,up=originals[key]
        if not 0<down<up or not np.isclose(row.down_limit,down) or not np.isclose(row.up_limit,up):
            raise ValueError('Band limits mismatch')
        fraction=(up-down)/(up+down)
        if not np.isclose(row.official_band_fraction,fraction) or bool(row.eligible)!=(fraction>=manifest['minimum_fraction']):
            raise ValueError('Band filter eligibility mismatch')
        if bool(p.loc[key,'eligible'])!=bool(row.eligible):
            raise ValueError('Prediction and original band eligibility disagree')
    return manifest


def run_filtered_account(out,inputs,pro,offline=False):
    import data_pipeline.tushare_config as cfg
    from research_daily_opportunity import run_daily_account
    out=Path(out)
    manifest=validate_band_inputs(out)
    cfg._pro=pro
    result=run_daily_account(validate_predictions(out),inputs,out,offline=offline)
    path=out/'account/daily_run_status.json'
    status=json.loads(path.read_text(encoding='utf-8'))
    status['source_sha256'].update(manifest['source_sha256'])
    for artifact in ('prior_band_manifest.json','prior_band_checks.csv'):
        status['source_sha256'][str((out/artifact).resolve())]=sha256(out/artifact)
    status['source_sha256'][str(Path(__file__).resolve())]=sha256(__file__)
    publish_manifest(status,path)
    return result
