"""V2 experiment: adaptive theme attention and investment-utility ranking.

V1 source/checkpoints/results remain unchanged. A scoped runtime adapter reuses
its verified trainer, with a different network/loss/calibration whose source is
explicitly recorded. Pair labels are strictly past completed horizons. Native
return regression remains in the loss; no rank is called a predicted return.
Primary inspirations: HIST (2110.13716), RSR (1809.09441), TRA (2106.12950).
"""
import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import research_daily_opportunity_model as base
import research_daily_opportunity as pipeline
from research_daily_graph_data import publish_manifest, sha256


OriginalNet = base.DailyGraphOpportunityNet
OriginalLoss = base.opportunity_loss
OriginalCalibration = base._calibration
RANK_WEIGHT = .5


class AdaptiveThemeOpportunityNet(OriginalNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stock_query = nn.Linear(self.hidden,self.hidden,bias=False)
        self.theme_key = nn.Linear(self.hidden,self.hidden,bias=False)

    def forward(self, sequences, membership):
        if sequences.ndim!=3 or sequences.shape[-1]!=self.input_dim or not len(sequences) or sequences.shape[1]<self.sequence_length:
            raise ValueError('Bad stock sequence shape')
        if not torch.isfinite(sequences).all():
            raise ValueError('Nonfinite sequences')
        sequences=sequences[:,-self.sequence_length:]
        encoded = self.temporal(sequences)[0][:,-1]
        graph = membership.to_sparse_coo().coalesce()
        if graph.shape[0]!=len(encoded) or not torch.isfinite(graph.values()).all() or (graph.values()<0).any():
            raise ValueError('Invalid dated membership')
        messages = torch.zeros_like(encoded)
        if graph._nnz():
            stock, theme = graph.indices()
            count = torch.zeros(graph.shape[1],device=encoded.device).index_add(0,theme,graph.values())
            pooled = torch.zeros((graph.shape[1],self.hidden),device=encoded.device).index_add(
                0,theme,encoded[stock]*graph.values()[:,None])/count.clamp_min(1)[:,None]
            relevance = (self.stock_query(encoded)[stock]*self.theme_key(pooled)[theme]).sum(-1)/self.hidden**.5
            # A soft specificity prior, not a named-theme whitelist. Broad
            # generic concepts remain, but do not swamp small economic themes.
            logits = relevance-.5*torch.log(count[theme].clamp_min(1))
            maximum = torch.full((len(encoded),),-torch.inf,device=encoded.device).scatter_reduce(
                0,stock,logits,reduce='amax',include_self=True)
            weights = torch.exp(logits-maximum[stock])
            denominator = torch.zeros(len(encoded),device=encoded.device).index_add(0,stock,weights)
            weights = weights/denominator[stock].clamp_min(1e-12)
            messages = messages.index_add(0,stock,pooled[theme]*weights[:,None])
        market = encoded.mean(0,keepdim=True).expand_as(encoded)
        fused = self.fusion(torch.cat((encoded,messages,market),dim=-1))
        logits = self.router(fused)
        gate = torch.softmax(logits,dim=-1)
        heads = torch.stack([head(fused) for head in self.expert_heads],dim=1)
        expert_returns = heads[...,:3]
        expert_downside = F.softplus(heads[...,3:6])*.1
        expert_rally = heads[...,6:9]
        probability=(gate[:,:,None]*torch.sigmoid(expert_rally)).sum(1)
        return dict(return_mu=(gate[:,:,None]*expert_returns).sum(1),
                    downside=(gate[:,:,None]*expert_downside).sum(1),
                    rally_logits=torch.logit(probability.clamp(1e-6,1-1e-6)),
                    gate=gate,gate_logits=logits,expert_return=expert_returns,
                    expert_downside=expert_downside,expert_rally_logits=expert_rally)


def ranked_loss(output,returns,downside,**kwargs):
    losses = OriginalLoss(output,returns,downside,**kwargs)
    device = output['return_mu'].device
    y = torch.as_tensor(returns,device=device,dtype=torch.float32)
    r = torch.as_tensor(downside,device=device,dtype=torch.float32)
    mask = kwargs.get('mask')
    mask = (torch.isfinite(y).all(-1)&torch.isfinite(r).all(-1)) if mask is None else torch.as_tensor(mask,device=device,dtype=torch.bool)
    weights = y.new_tensor((.2,.3,.5))
    realized = ((y[mask]-.5*r[mask])*weights).sum(-1)
    predicted = ((output['return_mu'][mask]-.5*output['downside'][mask])*weights).sum(-1)
    rank_loss = predicted.new_zeros(())
    if len(realized)>1:
        terms=[]
        for shift in (1,17,101):
            shift %= len(realized)
            if not shift:
                continue
            difference = realized-realized.roll(shift)
            active = difference.abs()>1e-5
            logits = (predicted-predicted.roll(shift))/.025
            loss = F.binary_cross_entropy_with_logits(logits[active],(difference[active]>0).float(),reduction='none')
            magnitude = (difference[active].abs()/.1).clamp(max=2)
            if len(loss):
                terms.append((loss*magnitude).mean())
        if terms:
            rank_loss = torch.stack(terms).mean()
    losses['rank'] = rank_loss
    losses['total'] = losses['total']+RANK_WEIGHT*rank_loss
    return losses


def intercept_calibration(network,validation,mean,std,device):
    calibration = OriginalCalibration(network,validation,mean,std,device)
    if not validation:
        return calibration
    forecast,observed=[],[]
    network.eval()
    with torch.no_grad():
        for batch,mask in validation:
            forecast.append(base._batch_forward(network,batch,mean,std,device)['return_mu'].cpu().numpy()[mask])
            observed.append(np.asarray(batch.returns)[mask])
    x,y=np.concatenate(forecast),np.concatenate(observed)
    # Center native-unit predictions on historical validation, while retaining
    # the newly trained cross-sectional signal. V1's nonpositive slope could
    # collapse an entire horizon to a constant, eliminating opportunity order.
    calibration['return_slope']=np.ones(3)
    calibration['return_intercept']=np.clip(y.mean(0)-x.mean(0),-.25,.25)
    return calibration


@contextmanager
def ranked_variant():
    network,loss,calibration=base.DailyGraphOpportunityNet,base.opportunity_loss,base._calibration
    protocol=copy.deepcopy(base.PROTOCOL)
    try:
        base.DailyGraphOpportunityNet=AdaptiveThemeOpportunityNet
        base.opportunity_loss=ranked_loss
        base._calibration=intercept_calibration
        base.PROTOCOL.update(version='daily_adaptive_graph_ranked_opportunity_v2',
            graph='Learned stock/concept attention with soft inverse-size specificity prior',
            ranking='Pairwise completed future utility + native return/risk/rally losses',
            return_calibration='Past validation intercept only; no cross-sectional slope collapse')
        yield
    finally:
        base.DailyGraphOpportunityNet,base.opportunity_loss,base._calibration=network,loss,calibration
        base.PROTOCOL.clear();base.PROTOCOL.update(protocol)


def train_ranked(source,out,device='cuda',epochs=8):
    import shutil
    source,out=Path(source),Path(out)
    store,edges,_=pipeline.validate_prepared(source,use_graph=True)
    out.mkdir(parents=True,exist_ok=True)
    for file in ('feature_store.pkl','frozen_edges.pkl','feature_manifest.json'):
        shutil.copy2(source/file,out/file)
    with ranked_variant():
        predictions=pipeline.train_and_predict(store,edges,out,config=base.FitConfig(epochs=epochs),device=device)
    protocol_path=out/'experiment_protocol.json'
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    protocol['source_sha256'][Path(__file__).name]=sha256(__file__)
    protocol['ranking_weight']=RANK_WEIGHT
    protocol['checkpoint_reload']='Use ranked_variant() context then base.FittedOpportunityModel.load; custom attention keys must match'
    protocol['hyperparameters']='Fixed for V2 after V1 failure diagnosis; no named stock/theme/month whitelist'
    publish_manifest(protocol,protocol_path)
    manifest=json.loads((out/'prediction_manifest.json').read_text(encoding='utf-8'))
    manifest['artifacts']['experiment_protocol.json']=sha256(protocol_path)
    publish_manifest(manifest,out/'prediction_manifest.json')
    return predictions


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('results/daily_opportunity_20261002'))
    parser.add_argument('--out',type=Path,default=Path('results/daily_opportunity_ranked_20261002'))
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--epochs',type=int,default=8)
    args=parser.parse_args()
    train_ranked(args.source,args.out,args.device,args.epochs)
