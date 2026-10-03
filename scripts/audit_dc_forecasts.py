"""Independent forecast-label timing and post-account ranking diagnostics."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from s7_budget_portfolio import is_main_board


def audit(out,name):
    predictions=pd.read_pickle(out/'forecast_predictions.pkl')
    metadata=json.loads((out/'forecast_model_audit.json').read_text(encoding='utf-8'))
    context_ranker=metadata['protocol'].get('target_kind')=='context_ranker'
    features=pd.read_pickle(out/('context_ranker_features.pkl' if context_ranker else 'forecast_features.pkl'))
    columns=metadata['protocol']['features']
    if not predictions.train_label_end.lt(predictions.model_fit_cutoff).all():
        raise AssertionError('Forecast training labels not purged')
    if not predictions.model_fit_cutoff.le(predictions.signal_date).all():
        raise AssertionError('Model fitted after prediction signal')
    finite=np.isfinite(features[columns].to_numpy(dtype=float)).all(axis=1)
    models=pd.read_pickle(out/'forecast_models.pkl')
    max_error=0.
    for fit in metadata['quarterly_models']:
        if fit['status']!='fitted': continue
        cutoff=pd.Timestamp(fit['model_fit_cutoff'])
        train=features[finite & features.signal_date.lt(cutoff) & features.label_date.lt(cutoff)
                       & np.isfinite(features.label_return)]
        if context_ranker:
            query_columns=['signal_date']
            if metadata['protocol'].get('query_scope')=='cohort':
                query_columns+=['peer_mom1','peer_mom3','peer_mom6','peer_positive3']
            train=train[train.groupby(query_columns).ts_code.transform('size').ge(2)].sort_values(query_columns+['ts_code'])
            sizes=train.groupby(query_columns,sort=False).size()
            rank=train.groupby(query_columns).label_return.rank(method='average')
            counts=train.groupby(query_columns).label_return.transform('size')
            relevance=np.floor(10*(rank-1)/counts).clip(0,9).astype(int)
            dates=(sizes.index.get_level_values('signal_date') if len(query_columns)>1 else sizes.index)
            if sizes.tolist()!=fit['train_query_sizes'] or dates.strftime('%Y-%m-%d').tolist()!=fit['train_query_dates']:
                raise AssertionError('Context query grouping does not match separately stored labels')
            if len(query_columns)>1:
                keys=[[f'{key[0]:%Y-%m-%d}']+[float(v) for v in key[1:]] for key in sizes.index]
                if keys!=fit['train_query_keys']:
                    raise AssertionError('Context cohort keys do not match stored feature statistics')
            if {str(i):int(relevance.eq(i).sum()) for i in range(10)}!=fit['train_relevance_counts']:
                raise AssertionError('Context relevance deciles do not match independently reconstructed labels')
        if len(train)!=fit['train_rows'] or train.label_date.max()!=pd.Timestamp(fit['train_label_end']):
            raise AssertionError('Training audit does not match separately stored labels')
        current=predictions[predictions.model_fit_cutoff.eq(cutoff)]
        sample=current.merge(features[['signal_date','ts_code']+columns],on=['signal_date','ts_code'],validate='one_to_one')
        model=models[fit['quarter']]['model']
        if metadata['protocol'].get('target_kind')=='rally_classifier':
            rerun=model.predict_proba(sample[columns].to_numpy(dtype=float))[:,list(model.classes_).index(1)]
        else:
            rerun=model.predict(sample[columns].to_numpy(dtype=float))
        np.testing.assert_allclose(rerun,sample.forecast_return,atol=1e-12,rtol=0)
        max_error=max(max_error,float(np.max(np.abs(rerun-sample.forecast_return))))
    labelled=predictions.merge(features[['signal_date','ts_code','label_return','label_date']],
        on=['signal_date','ts_code'],validate='one_to_one')
    labelled=labelled[np.isfinite(labelled.label_return) & labelled.label_date.notna()]
    rows=[]
    for day,full in labelled.groupby('signal_date'):
        for scope,group in [('all_boards',full),('mainboard',full[full.ts_code.str[:6].map(is_main_board)])]:
            top=group[group.forecast_percentile.ge(.8) & (True if context_ranker else group.forecast_return.gt(0))]
            rows.append(dict(strategy=name,signal_date=day,label_date=group.label_date.max(),scope=scope,
                stocks=len(group),rank_ic=group.forecast_return.corr(group.label_return,method='spearman'),
                top_positive_forecast_stocks=len(top),top_mean_adjusted_close_return=top.label_return.mean(),
                all_mean_adjusted_close_return=group.label_return.mean(),
                top_excess_adjusted_close_return=top.label_return.mean()-group.label_return.mean()))
            rows[-1]['top_rally_label_rate']=top.label_return.ge(.30).mean()
            rows[-1]['all_rally_label_rate']=group.label_return.ge(.30).mean()
    return pd.DataFrame(rows),dict(strategy=name,quarterly_fits_checked=len(models),
        prediction_rows=len(predictions),max_saved_model_prediction_error=max_error,
        all_training_labels_strictly_before_fit=True,
        caveat='Adjusted close-label ranking diagnostic, not whole-lot account returns; September label is unavailable')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accounts',nargs='+',required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('results/dc_entry_comparison'))
    args=parser.parse_args();rows=[];checks=[]
    for item in args.accounts:
        directory,name=item.rsplit('=',1)
        result,check=audit(Path(directory),name);rows.append(result);checks.append(check)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    combined=pd.concat(rows,ignore_index=True)
    combined.to_csv(args.output_dir/'forecast_ranking_diagnostics.csv',index=False)
    (args.output_dir/'forecast_independent_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(combined[combined.scope.eq('mainboard')].to_string(index=False))


if __name__=='__main__': main()
