"""Isolated account runner with verified cache reuse and transport backoff."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import pandas as pd

from data_pipeline.tushare_config import GatewayClient
from research_daily_graph_data import load_adjusted_daily,publish_manifest,sha256
from research_daily_opportunity import validate_predictions
from research_daily_opportunity_execution import OfficialLimits,run_account
from research_scoped_actions import validate_history


REPO=Path(__file__).resolve().parents[1]


class BackoffGateway:
    def __init__(self,key,attempts=4):
        self.base=GatewayClient(key);self.attempts=attempts
    def query(self,api,**params):
        for attempt in range(self.attempts):
            try:
                return self.base.query(api,**params)
            except RuntimeError as exc:
                print('gateway_retry',api,attempt+1,str(exc),flush=True)
                if attempt+1==self.attempts:
                    raise
                time.sleep(min(8,2**attempt))
    def __getattr__(self,name):
        if name.startswith('_'):
            raise AttributeError(name)
        return lambda **params:self.query(name,**params)


def cache_directories():
    return [REPO/'results'/name for name in (
        'daily_opportunity_20261002','daily_opportunity_no_graph_20261002',
        'daily_opportunity_trees_20261002','daily_opportunity_ranked_20261002',
        'dc_member_relative_leader_2026_research','dc_member_relative_moderate_2026_research',
        'dc_structure_replay_20261002')]


def limits_caches():
    return [folder/name for folder in cache_directories() for name in
            ('account/daily_limits','daily_limits','exit_quotes','replacement_limits')]


def seed_actions(account_dir):
    target=Path(account_dir)/'scoped_actions';target.mkdir(parents=True,exist_ok=True)
    for root in cache_directories():
        for folder in (root/'scoped_actions',root/'account/scoped_actions'):
            for source in folder.glob('*.pkl'):
                destination=target/source.name
                if destination.exists() or source.resolve()==destination.resolve():
                    continue
                metadata=source.with_suffix('.json')
                if not metadata.exists():
                    continue
                manifest=json.loads(metadata.read_text(encoding='utf-8'))
                if (manifest.get('start'),manifest.get('end'),manifest.get('raw_sha256'))!=('2026-01-01','2026-09-24',sha256(source)):
                    continue
                validate_history(pd.read_pickle(source),source.stem,'2026-01-01','2026-09-24')
                shutil.copy2(source,destination);shutil.copy2(metadata,destination.with_suffix('.json'))


def run_verified(out,inputs,offline=False):
    out,inputs=Path(out),Path(inputs)
    predictions=validate_predictions(out)
    daily,meta=load_adjusted_daily(inputs)
    expected=json.loads((out/'feature_manifest.json').read_text(encoding='utf-8'))
    if meta['adjusted_daily']['sha256']!=expected['raw_adjusted_daily']['sha256']:
        raise ValueError('Account source differs from fitted daily data')
    band=None
    if (out/'prior_band_manifest.json').exists():
        from research_daily_opportunity_band_filter import validate_band_inputs
        band=validate_band_inputs(out)
    account_dir=out/'account';account_dir.mkdir(exist_ok=True)
    seed_actions(account_dir)
    key=os.environ.get('TUSHARE_API_KEY','').strip()
    if not offline and not key:
        raise ValueError('Missing API key environment variable')
    pro=None if offline else BackoffGateway(key)
    limits=OfficialLimits(account_dir,pro=pro,offline=offline,cache_roots=limits_caches())
    result=run_account(predictions,daily,sorted(daily.date.unique()),account_dir,
                       pro=pro,offline=offline,limit_provider=limits)
    path=account_dir/'daily_run_status.json'
    status=json.loads(path.read_text(encoding='utf-8'))
    status['source_sha256'][str(Path(__file__).resolve())]=sha256(__file__)
    if band:
        status['source_sha256'].update(band['source_sha256'])
        for file in ('prior_band_manifest.json','prior_band_checks.csv'):
            status['source_sha256'][str((out/file).resolve())]=sha256(out/file)
    publish_manifest(status,path)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,default=REPO/'data/raw/daily_opportunity_20261002')
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args()
    run_verified(args.out,args.inputs,args.offline)
