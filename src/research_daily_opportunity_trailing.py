"""One declared trailing-risk variant of an unchanged daily forecast account."""
import argparse
import json
import os
from pathlib import Path
import shutil

from research_daily_graph_data import load_adjusted_daily,publish_manifest,sha256
from research_daily_opportunity import validate_predictions
from research_daily_opportunity_band_filter import validate_band_inputs
from research_daily_opportunity_execution import ExecutionPolicy,OfficialLimits,run_account
from research_daily_opportunity_runner import BackoffGateway,limits_caches,seed_actions


def run_trailing(source,out,inputs,trailing_distance=.08,offline=False):
    source,out,inputs=Path(source),Path(out),Path(inputs)
    p=validate_predictions(source)
    band=validate_band_inputs(source) if (source/'prior_band_manifest.json').exists() else None
    out.mkdir(parents=True,exist_ok=True)
    for file in source.iterdir():
        if file.is_file() and file.name not in ('predictions.csv',):
            shutil.copy2(file,out/file.name)
    shutil.copytree(source/'checkpoints',out/'checkpoints',dirs_exist_ok=True)
    daily,manifest=load_adjusted_daily(inputs)
    feature=json.loads((source/'feature_manifest.json').read_text(encoding='utf-8'))
    if manifest['adjusted_daily']['sha256']!=feature['raw_adjusted_daily']['sha256']:
        raise ValueError('Different raw daily source')
    account=out/'account';account.mkdir(exist_ok=True);seed_actions(account)
    key=os.environ.get('TUSHARE_API_KEY','').strip()
    if not offline and not key:raise ValueError('Missing API key environment variable')
    pro=None if offline else BackoffGateway(key)
    roots=limits_caches()+[source/'account/daily_limits',source/'daily_limits']
    limits=OfficialLimits(account,pro=pro,offline=offline,cache_roots=roots)
    policy=ExecutionPolicy(trailing_distance=trailing_distance)
    result=run_account(p,daily,sorted(daily.date.unique()),account,pro=pro,offline=offline,
                       policy=policy,limit_provider=limits)
    path=account/'daily_run_status.json'
    status=json.loads(path.read_text(encoding='utf-8'))
    status['source_sha256'][str(Path(__file__).resolve())]=sha256(__file__)
    if band:
        status['source_sha256'].update(band['source_sha256'])
        for file in ('prior_band_manifest.json','prior_band_checks.csv'):
            status['source_sha256'][str((source/file).resolve())]=sha256(source/file)
    publish_manifest(status,path)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,default=Path('data/raw/daily_opportunity_20261002'))
    parser.add_argument('--trailing-distance',type=float,default=.08)
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args();run_trailing(args.source,args.out,args.inputs,args.trailing_distance,args.offline)
