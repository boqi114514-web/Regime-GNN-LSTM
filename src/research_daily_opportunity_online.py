"""Daily, past-only market/theme residual calibration for frozen forecasts.

Short-horizon evidence can update after its own label matures; 20-day evidence
is never read early. Peers use the graph observable at the current signal, not
a future or undated membership list. This is an explicitly disclosed adaptive
estimator in native return/risk/probability units, not a momentum-rank alias.
"""
import argparse
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd

from research_daily_graph_data import asof_graph,publish_frame,publish_manifest,sha256
from research_daily_opportunity import validate_predictions,validate_prepared
from research_daily_opportunity_features import expected_utility


def specific_peer_message(graph,values):
    """Self-free concept means, stock averaging favors narrower known themes."""
    b=graph.incidence
    counts=np.asarray(b.sum(axis=0)).ravel()
    useful=counts>1
    if not useful.any():
        return np.zeros_like(values),np.zeros(len(values),dtype=bool)
    c=b[:,useful]
    strength=1/np.sqrt(counts[useful])
    sums=c.T@values
    inv=1/(counts[useful]-1)
    denominator=np.asarray(c@strength).ravel()
    out=np.asarray(c@(sums*(inv*strength)[:,None]))-np.asarray(c@(inv*strength)).ravel()[:,None]*values
    return out/np.maximum(denominator[:,None],1e-12),denominator>0


def mature_indices(store,signal_index,horizon_index,earlier_indices,window=20):
    day=np.datetime64(store.dates[signal_index],'D')
    good=[i for i in earlier_indices if i<signal_index and not np.isnat(store.label_end_dates[i,horizon_index])
          and store.label_end_dates[i,horizon_index]<day]
    return good[-window:]


def run_online(source,out,window=20,minimum_sessions=5,market_weight=.5,theme_weight=.5):
    source,out=Path(source),Path(out)
    original=validate_predictions(source)
    store,edges,_=validate_prepared(source,use_graph=True)
    out.mkdir(parents=True,exist_ok=True)
    for name in ('feature_store.pkl','feature_manifest.json','frozen_edges.pkl','fit_audits.json'):
        shutil.copy2(source/name,out/name)
    shutil.copytree(source/'checkpoints',out/'checkpoints',dirs_exist_ok=True)
    dindex={date:i for i,date in enumerate(store.dates)}
    cindex={code:i for i,code in enumerate(store.stock_codes)}
    shape=(len(store.dates),len(store.stock_codes),3)
    mu=np.full(shape,np.nan,dtype=np.float32)
    risk=np.full_like(mu,np.nan);prob=np.full_like(mu,np.nan)
    for day,f in original.groupby('signal_date'):
        i=dindex[day];stocks=[cindex[code] for code in f.ts_code]
        mu[i,stocks]=f[['mu5','mu10','mu20']].to_numpy()
        risk[i,stocks]=f[['risk5','risk10','risk20']].to_numpy()
        prob[i,stocks]=f[['rally_probability5','rally_probability10','rally_probability20']].to_numpy()
    frames,audits=[],[]
    by_model={}
    for day,frame in original.groupby('signal_date',sort=True):
        index=dindex[day]
        cutoff=frame.model_fit_cutoff.iloc[0]
        if frame.model_fit_cutoff.nunique()!=1:
            raise ValueError('One frozen checkpoint per signal required')
        earlier=by_model.setdefault(cutoff,[])
        codes=store.stock_codes
        graph=asof_graph(edges,day,codes,max_age_days=None,validated=True)
        f=frame.copy()
        stocks=np.array([cindex[code] for code in f.ts_code])
        for j,horizon in enumerate((5,10,20)):
            dates=mature_indices(store,index,j,earlier,window)
            if len(dates)<minimum_sessions:
                audits.append(dict(signal_date=day,horizon=horizon,model_fit_cutoff=cutoff,
                                   mature_signal_count=len(dates),active=False,maximum_label_end=pd.NaT))
                continue
            labels=store.label_returns[dates,:,j]
            observed_risk=store.label_downside[dates,:,j]
            observed_prob=(labels>=(.10,.15,.30)[j]).astype(np.float32)
            observed_prob[~np.isfinite(labels)]=np.nan
            residuals=np.stack((labels-mu[dates,:,j],observed_risk-risk[dates,:,j],
                                observed_prob-prob[dates,:,j]),axis=-1)
            good=np.isfinite(residuals)
            count=good.sum(axis=0)
            sums=np.where(good,residuals,0.).sum(axis=0)
            average=sums/np.maximum(count,1)
            global_error=np.where(good,residuals,0.).sum(axis=(0,1))/np.maximum(good.sum(axis=(0,1)),1)
            average=np.where(count>=minimum_sessions,average,global_error)
            # Explicit fixed bounds on the adaptive correction, not label
            # leakage or replacing invalid observations with profitable ones.
            average=np.clip(average,[-.4,-.4,-1.],[.4,.4,1.])
            peers,known=specific_peer_message(graph,average)
            peers[~known]=global_error
            correction=market_weight*global_error+theme_weight*peers
            f[f'mu{horizon}']=mu[index,stocks,j]+correction[stocks,0]
            f[f'risk{horizon}']=np.maximum(0.,risk[index,stocks,j]+correction[stocks,1])
            f[f'rally_probability{horizon}']=np.clip(prob[index,stocks,j]+correction[stocks,2],1e-6,1-1e-6)
            audits.append(dict(signal_date=day,horizon=horizon,model_fit_cutoff=cutoff,
                               mature_signal_count=len(dates),active=True,
                               maximum_label_end=pd.Timestamp(max(store.label_end_dates[i,j] for i in dates)),
                               global_return_error=float(global_error[0]),global_risk_error=float(global_error[1])))
        f['utility']=expected_utility(f[['mu5','mu10','mu20']].to_numpy(),f[['risk5','risk10','risk20']].to_numpy())
        frames.append(f);earlier.append(index)
    predictions=pd.concat(frames,ignore_index=True)
    checks=pd.DataFrame(audits)
    active=checks[checks.active]
    if active.maximum_label_end.ge(active.signal_date).any():
        raise AssertionError('Future online correction label')
    checks.to_csv(out/'online_calibration_audit.csv',index=False)
    publish_frame(predictions,out/'predictions.pkl')
    predictions.to_csv(out/'predictions.csv',index=False,encoding='utf-8-sig')
    protocol=json.loads((source/'experiment_protocol.json').read_text(encoding='utf-8'))
    protocol['online_calibration']=dict(window=window,minimum_sessions=minimum_sessions,
        market_weight=market_weight,theme_weight=theme_weight,
        inputs='Only previously deployed same-checkpoint errors with this horizon label_end strictly before current signal',
        graph='Current known graph, self-free concept messages with inverse-square-root size weighting',
        checkpoint_change='Residual history resets; no incompatible older-model errors used',
        parameters='Fixed after diagnosis, before this account; no stock/theme/month whitelist')
    protocol['source_sha256'][Path(__file__).name]=sha256(__file__)
    publish_manifest(protocol,out/'experiment_protocol.json')
    artifacts={name:sha256(out/name) for name in ('predictions.pkl','fit_audits.json','experiment_protocol.json','feature_manifest.json',
                                               'online_calibration_audit.csv')}
    publish_manifest(dict(status='complete',rows=len(predictions),sessions=int(predictions.signal_date.nunique()),artifacts=artifacts),
                     out/'prediction_manifest.json')
    return predictions


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('results/daily_opportunity_ranked_20261002'))
    parser.add_argument('--out',type=Path,default=Path('results/daily_opportunity_online_20261002'))
    args=parser.parse_args();run_online(args.source,args.out)
