"""Quarterly forecast with contemporaneous, unsupervised price-cohort context."""
import json
from pathlib import Path

import pandas as pd
import sklearn

from data_pipeline.execution_data import ROOT
from research_dc_forecast import FEATURE_COLUMNS, FORECAST_PROTOCOL, _sha256, _frame_sha256, forecast_features, walk_forward_predictions


PEER_COLUMNS = ['peer_mom1','peer_mom3','peer_mom6','peer_positive3']
PROTOCOL = {**FORECAST_PROTOCOL,'version':'peer_context_v1',
    'features':FEATURE_COLUMNS+PEER_COLUMNS,
    'context':'Past 60-session gap-neutral intraday-pressure cohorts; fixed MiniBatchKMeans20, seed42, three initializations, maxiter60, batch1024; >=10 complete-history stocks; cohort adjusted-momentum medians and positive3 breadth; not industry membership',
    'ranking':'Forecast magnitude, linear expected-profit whole-lot allocation; dated DC price association remains an eligibility gate only'}


def prepare_peer_forecasts(out, monthly, daily, signal_dates):
    from research_dc_peer_context import peer_context
    out=Path(out)
    source_names=['research_dc_peer_forecast.py','research_dc_peer_context.py','research_dc_forecast.py',
                  'research_dc_themes.py','research_dc_entry.py']
    files=[Path(__file__).with_name(n) for n in source_names]+[out/'daily.pkl',ROOT/'stock_month_end_verified.pkl']
    spec=dict(protocol=PROTOCOL,sklearn_version=sklearn.__version__,signals=[f'{pd.Timestamp(d):%Y-%m-%d}' for d in sorted(signal_dates)],
              inputs={str(p.resolve()):_sha256(p) for p in files},
              monthly_frame_sha256=_frame_sha256(monthly,['date','ts_code','close','adj_factor']),
              daily_frame_sha256=_frame_sha256(daily,['date','ts_code','open','high','low','close','volume','amount']))
    names=['forecast_features.pkl','forecast_predictions.pkl','forecast_models.pkl','forecast_model_audit.json','peer_context.pkl']
    manifest_path=out/'forecast_feature_manifest.json'
    if manifest_path.exists() and all((out/n).exists() for n in names):
        old=json.loads(manifest_path.read_text(encoding='utf-8'))
        if old.get('specification')==spec and all(old.get('artifacts',{}).get(n)==_sha256(out/n) for n in names):
            return pd.read_pickle(out/'forecast_predictions.pkl')
    base=forecast_features(monthly,daily)
    context=peer_context(monthly,daily,sorted(base.signal_date.unique()))
    context.to_pickle(out/'peer_context.pkl')
    features=base.merge(context,on=['signal_date','ts_code'],how='inner',validate='one_to_one')
    features=features[features.peer_count.ge(10)].copy()
    predictions=walk_forward_predictions(features,signal_dates,feature_columns=FEATURE_COLUMNS+PEER_COLUMNS)
    models=predictions.attrs.pop('models');audit=predictions.attrs.pop('model_audit')
    features.to_pickle(out/'forecast_features.pkl')
    predictions.to_pickle(out/'forecast_predictions.pkl')
    pd.to_pickle(models,out/'forecast_models.pkl')
    payload=dict(protocol=PROTOCOL,quarterly_models=audit,feature_rows=len(features),prediction_rows=len(predictions),
                 train_label_purge_verified=bool(predictions.empty or predictions.train_label_end.lt(predictions.model_fit_cutoff).all()))
    (out/'forecast_model_audit.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    manifest_path.write_text(json.dumps(dict(status='complete',specification=spec,
        artifacts={n:_sha256(out/n) for n in names}),indent=2),encoding='utf-8')
    return predictions
