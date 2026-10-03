"""Isolated V3: explicit causal theme leadership and specialised experts.

The base 26 features, V1/V2 checkpoints and their provenance remain unchanged.
These extra factors describe a stock's position in its strongest observable
theme; they never prescribe a stock, theme, month or buy. Unknown graphs remain
valid learning/prediction nodes with a separate missing-relationship indicator.
"""
import argparse
from contextlib import contextmanager
import copy
from dataclasses import replace
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

import research_daily_opportunity as pipeline
import research_daily_opportunity_features as features
import research_daily_opportunity_model as model
import research_daily_opportunity_ranked as ranked
from research_daily_graph_data import asof_graph, publish_manifest, sha256


EXTRA_NAMES = (
    'theme_peer_return5', 'theme_peer_return10', 'theme_peer_return20',
    'excess_theme_return5', 'excess_theme_return20', 'theme_breadth20',
    'within_theme_return20_percentile', 'within_theme_relative_amount',
    'theme_peer_amount_growth5', 'theme_peer_drawdown20',
    'theme_peer_count', 'known_theme_context',
)
BASE_PROTOCOL = copy.deepcopy(features.FEATURE_PROTOCOL)
V3_PROTOCOL = dict(BASE_PROTOCOL, version='daily_opportunity_leadership_features_v3',
    features=list(features.FEATURE_NAMES)+list(EXTRA_NAMES),
    theme_selection='Strongest self-free peer return20 - .02*log(observed theme size); at least five feature-valid stocks; no named theme gate',
    theme_unknown='Zero context plus known_theme_context=0; stocks are not excluded',
    theme_causality='Latest whole historically available snapshot; current completed features only; no target in selection',
    amount_role='Raw same-session traded amount relative to selected theme average, not a market-cap proxy')


def leadership_factors(store, edges, amounts):
    """Build factors one day at a time, from visible edges and stock features."""
    if tuple(store.feature_names)!=features.FEATURE_NAMES:
        raise ValueError('Leadership extension requires the unchanged base feature order')
    amounts=np.asarray(amounts,dtype=float)
    if amounts.shape!=store.valid_features.shape:
        raise ValueError('Amount panel must follow complete market calendar and stock order')
    if (amounts[np.isfinite(amounts)]<0).any():
        raise ValueError('Negative raw amount')
    result=np.zeros((*store.valid_features.shape,len(EXTRA_NAMES)),dtype=np.float32)
    columns={name:i for i,name in enumerate(store.feature_names)}
    cached={}
    snapshot_dates=pd.DatetimeIndex(sorted(edges.snapshot_date.unique()))
    for i,day in enumerate(store.dates):
        prior=snapshot_dates[snapshot_dates<=day]
        if not len(prior):
            continue
        snapshot=prior[-1]
        if snapshot not in cached:
            cached[snapshot]=asof_graph(edges,day,store.stock_codes,max_age_days=None,validated=True).incidence
        valid=store.valid_features[i]&np.isfinite(amounts[i])
        b=cached[snapshot].multiply(valid[:,None]).tocsr()
        b.eliminate_zeros()
        count=np.asarray(b.sum(axis=0)).ravel()
        coo=b.tocoo();row,col=coo.row,coo.col
        if not len(row):
            continue
        x=store.features[i].copy()
        r5,r10,r20=[x[:,columns[f'return{h}']] for h in (5,10,20)]
        value=np.column_stack((r5,r10,r20,np.where(np.isfinite(r20),(r20>0).astype(float),np.nan),
            x[:,columns['amount5_vs_prior20']],x[:,columns['drawdown20']]))
        observed=np.isfinite(value)&valid[:,None]
        sums=np.asarray(b.T@np.where(observed,value,0.))
        supports=np.asarray(b.T@observed.astype(float))
        own=np.where(observed,value,0.)
        denominator=supports[col]-observed[row]
        peer=np.divide(sums[col]-own[row],denominator,
            out=np.full_like(sums[col],np.nan),where=denominator>0)
        scores=peer[:,2]-.02*np.log(np.maximum(count[col],1))
        scores[(count[col]<5)|(denominator[:,2]<4)|~np.isfinite(scores)]=-np.inf
        maximum=np.full(len(store.stock_codes),-np.inf)
        np.maximum.at(maximum,row,scores)
        eligible=np.isfinite(scores)&(scores==maximum[row])
        chosen=np.full(len(store.stock_codes),b.shape[1],dtype=int)
        np.minimum.at(chosen,row[eligible],col[eligible])
        selected=eligible&(col==chosen[row])
        rr,cc=row[selected],col[selected]
        if not len(rr):
            continue
        # Ranking uses the current quote cross-section, not realised targets.
        ranks=pd.DataFrame(dict(theme=col,ret=r20[row])).groupby('theme').ret.rank(pct=True).to_numpy()
        amount=np.where(valid,amounts[i],0.)
        total_amount=np.asarray(b.T@amount).ravel()
        relative_amount=amount[rr]*count[cc]/np.maximum(total_amount[cc],1e-12)
        pp=peer[selected]
        context=np.column_stack((pp[:,:3],r5[rr]-pp[:,0],r20[rr]-pp[:,2],pp[:,3],
            ranks[selected],relative_amount,pp[:,4],pp[:,5],np.log1p(count[cc]-1)/5.,np.ones(len(rr))))
        context=np.clip(context,-5.,5.)
        # Partial factors stay undefined, then receive explicit missingness
        # masks during encoding. No quote, return or peer mean is invented.
        result[i,rr]=context
    return result


def prepare_leadership(source,out):
    source,out=Path(source),Path(out)
    _,edges,metadata=pipeline.validate_prepared(source,use_graph=True)
    daily=pd.read_pickle(metadata['raw_adjusted_daily']['path'])
    from research_daily_opportunity_masked_features import build_masked_feature_store,encode_missingness,encoded_feature_names
    V3_PROTOCOL['features']=list(encoded_feature_names(tuple(features.FEATURE_NAMES)+EXTRA_NAMES))
    V3_PROTOCOL['calendar']='Complete market calendar; missing factors explicit 0+mask, no synthetic quotes/returns; seasoned resumed stocks remain predictable'
    store=build_masked_feature_store(daily)
    amount=daily.pivot(index='date',columns='ts_code',values='amount').reindex(
        index=store.dates,columns=store.stock_codes).to_numpy()
    context=leadership_factors(store,edges,amount)
    extended=replace(store,features=np.concatenate((store.features,context),axis=-1),
                     feature_names=tuple(store.feature_names)+EXTRA_NAMES)
    extended=encode_missingness(extended)
    out.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source/'frozen_edges.pkl',out/'frozen_edges.pkl')
    target=out/'feature_store.pkl'
    with target.open('wb') as handle:
        pickle.dump(extended,handle,protocol=pickle.HIGHEST_PROTOCOL)
    manifest=copy.deepcopy(metadata)
    manifest.update(feature_protocol=V3_PROTOCOL,feature_store_sha256=sha256(target),
        leadership_builder_sha256=sha256(__file__),
        missingness_builder_sha256=sha256(Path(__file__).with_name('research_daily_opportunity_masked_features.py')),
        base_feature_store=dict(path=str((source/'feature_store.pkl').resolve()),sha256=sha256(source/'feature_store.pkl')))
    manifest['edges']=dict(path=str((out/'frozen_edges.pkl').resolve()),sha256=sha256(out/'frozen_edges.pkl'))
    publish_manifest(manifest,out/'feature_manifest.json')
    print(json.dumps(dict(event='leadership_features_prepared',features=len(extended.feature_names),
        sessions=len(store.dates),stocks=len(store.stock_codes),
        known_theme_observations=int((context[...,-1]>0).sum()))),flush=True)
    return extended,edges


def specialised_loss(output,returns,downside,**kwargs):
    losses=ranked.ranked_loss(output,returns,downside,**kwargs)
    stages=kwargs.get('stage_targets')
    loss=output['return_mu'].new_zeros(())
    if stages is not None:
        device=loss.device
        y=torch.as_tensor(returns,device=device,dtype=torch.float32)
        risk=torch.as_tensor(downside,device=device,dtype=torch.float32)
        stages=torch.as_tensor(stages,device=device,dtype=torch.long)
        mask=kwargs.get('mask')
        good=(torch.isfinite(y).all(-1)&torch.isfinite(risk).all(-1)) if mask is None else torch.as_tensor(mask,device=device,dtype=torch.bool)
        good=good&(stages>=0)&(stages<4)
        if good.any():
            rows=torch.where(good)[0];expert=stages[rows]
            r=F.smooth_l1_loss(output['expert_return'][rows,expert]/.1,y[rows]/.1)
            d=F.smooth_l1_loss(output['expert_downside'][rows,expert]/.1,risk[rows]/.1)
            event=(y[rows]>=y.new_tensor((.1,.15,.3))).float()
            p=F.binary_cross_entropy_with_logits(output['expert_rally_logits'][rows,expert],event)
            loss=r+.5*d+.25*p
    losses['specialisation']=loss
    losses['total']=losses['total']+.25*loss
    return losses


@contextmanager
def leadership_variant():
    """Scoped variant: restore old pipeline/model protocols on every exit."""
    from research_daily_opportunity_masked_features import MaskedDailyOpportunityBatch,encoded_feature_names
    protocol=copy.deepcopy(features.FEATURE_PROTOCOL)
    old_batch=pipeline.LazyDailyOpportunityBatch
    V3_PROTOCOL['features']=list(encoded_feature_names(tuple(features.FEATURE_NAMES)+EXTRA_NAMES))
    V3_PROTOCOL['calendar']='Complete market calendar; missing factors explicit 0+mask, no synthetic quotes/returns; seasoned resumed stocks remain predictable'
    try:
        features.FEATURE_PROTOCOL.clear();features.FEATURE_PROTOCOL.update(V3_PROTOCOL)
        pipeline.LazyDailyOpportunityBatch=MaskedDailyOpportunityBatch
        with ranked.ranked_variant():
            model.opportunity_loss=specialised_loss
            model.PROTOCOL.update(version='daily_theme_leadership_specialised_opportunity_v3',
                specialisation='Past-stage-conditioned auxiliary native-unit expert losses, weight .25; soft routing retained')
            yield
    finally:
        pipeline.LazyDailyOpportunityBatch=old_batch
        features.FEATURE_PROTOCOL.clear();features.FEATURE_PROTOCOL.update(protocol)


def validate_leadership(out):
    out=Path(out)
    meta=json.loads((out/'feature_manifest.json').read_text(encoding='utf-8'))
    if meta.get('leadership_builder_sha256')!=sha256(__file__):
        raise ValueError('Leadership builder changed after preparation')
    if meta.get('missingness_builder_sha256')!=sha256(Path(__file__).with_name('research_daily_opportunity_masked_features.py')):
        raise ValueError('Missingness builder changed after preparation')
    artifact=meta['base_feature_store']
    if sha256(artifact['path'])!=artifact['sha256']:
        raise ValueError('Base feature-store provenance changed')
    store,edges,_=pipeline.validate_prepared(out,use_graph=True)
    if tuple(store.feature_names)!=tuple(V3_PROTOCOL['features']):
        raise ValueError('Leadership feature shape/order mismatch')
    return store,edges


def train_leadership(out,device='cuda',epochs=8,tree=False):
    out=Path(out)
    with leadership_variant():
        store,edges=validate_leadership(out)
        if tree:
            import research_daily_opportunity_trees as trees
            # Same-directory artifacts are already prepared; use a separate
            # train directory to preserve the original feature store.
            temporary=out/'prepared_inputs';temporary.mkdir(exist_ok=True)
            for name in ('feature_store.pkl','feature_manifest.json','frozen_edges.pkl'):
                shutil.copy2(out/name,temporary/name)
            previous=trees.daily_rows
            try:
                trees.daily_rows=masked_tree_rows
                predictions=trees.run_trees(temporary,out)
            finally:
                trees.daily_rows=previous
        else:
            predictions=pipeline.train_and_predict(store,edges,out,
                config=model.FitConfig(epochs=epochs),device=device)
    path=out/'experiment_protocol.json'
    protocol=json.loads(path.read_text(encoding='utf-8'))
    protocol['source_sha256'][Path(__file__).name]=sha256(__file__)
    protocol['source_sha256']['research_daily_opportunity_ranked.py']=sha256(ranked.__file__)
    protocol['source_sha256']['research_daily_opportunity_masked_features.py']=sha256(Path(__file__).with_name('research_daily_opportunity_masked_features.py'))
    protocol['factor_inputs']='Base 26 + causal theme leadership 12 + explicit masks/timing; current quote eligibility, not 80-session gap exclusion'
    protocol['hyperparameters']='V3 fixed after V2 diagnosis; no named stock/theme/month rules or monthwise strategy splice'
    publish_manifest(protocol,path)
    manifest=json.loads((out/'prediction_manifest.json').read_text(encoding='utf-8'))
    manifest['artifacts']['experiment_protocol.json']=sha256(path)
    publish_manifest(manifest,out/'prediction_manifest.json')
    return predictions


def masked_tree_rows(store,edges,index,sequence_length=20):
    indices=np.flatnonzero(store.valid_features[index])
    codes=tuple(store.stock_codes[i] for i in indices)
    sequence=store.features[index-sequence_length+1:index+1,indices]
    last=sequence[-1]
    graph=asof_graph(edges,store.dates[index],codes,max_age_days=None,validated=True)
    peer=graph.messages(last,exclude_self=True).astype(np.float32)
    degree=np.asarray(graph.incidence.sum(axis=1)).ravel()
    stage=store.stages[index,indices]
    onehot=np.zeros((len(indices),4));known=stage>=0
    onehot[known,stage[known]]=1.
    x=np.column_stack((last,sequence.mean(axis=0),peer,np.log1p(degree),onehot)).astype(np.float32)
    return codes,x,store.label_returns[index,indices],store.label_downside[index,indices],graph.snapshot_date


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','train','all','filter','account','quality'))
    parser.add_argument('--source',type=Path,default=Path('results/daily_opportunity_20261002'))
    parser.add_argument('--out',type=Path,default=Path('results/daily_opportunity_leadership_20261002'))
    parser.add_argument('--inputs',type=Path,default=Path('data/raw/daily_opportunity_20261002'))
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--tree',action='store_true')
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args()
    if args.action in ('prepare','all'):
        prepare_leadership(args.source,args.out)
    if args.action in ('train','all'):
        train_leadership(args.out,args.device,args.epochs,args.tree)
    if args.action=='filter':
        import os
        from research_daily_opportunity_band_filter import filter_buy_risk_band
        from research_daily_opportunity_runner import BackoffGateway,limits_caches
        with leadership_variant():
            validate_leadership(args.source)
            filter_buy_risk_band(args.source,args.out,
                pro=None if args.offline else BackoffGateway(os.environ['TUSHARE_API_KEY']),
                cache_roots=limits_caches()+[args.source/'account/daily_limits'])
    if args.action=='account':
        from research_daily_opportunity_trailing import run_trailing
        with leadership_variant():
            validate_leadership(args.source)
            run_trailing(args.source,args.out,args.inputs,.08,args.offline)
    if args.action=='quality':
        from research_daily_opportunity_diagnostics import quality_report
        with leadership_variant():
            validate_leadership(args.out)
            print(quality_report(args.out).to_string())
