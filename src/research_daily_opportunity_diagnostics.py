"""Read-only model-quality diagnostics; not a trading return or account.

Future labels are used only to evaluate already-frozen predictions. Top-k is
selected BEFORE looking at label availability. Overlapping 20-session returns
are never added up and presented as calendar-month account profits.
"""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from research_daily_graph_data import publish_manifest, sha256
from research_daily_opportunity import validate_predictions


def quality_report(out):
    out = Path(out)
    predictions = validate_predictions(out)
    with (out/'feature_store.pkl').open('rb') as handle:
        store = pickle.load(handle)
    date_index = {date: i for i, date in enumerate(store.dates)}
    stock_index = {code: i for i, code in enumerate(store.stock_codes)}
    rows = []
    for day, frame in predictions.groupby('signal_date', sort=True):
        t = date_index[day]
        indices = np.array([stock_index[code] for code in frame.ts_code])
        labels = store.label_returns[t, indices]
        risks = store.label_downside[t, indices]
        frame = frame.copy()
        frame['mainboard'] = frame.ts_code.str.match(r'^(00[0123]|60[0135])\d{3}\.(SZ|SH)$')
        for scope, mask in [('all_boards', np.ones(len(frame), dtype=bool)),
                            ('buyable_mainboard', frame.mainboard.to_numpy())]:
            f, y, downside = frame.loc[mask], labels[mask], risks[mask]
            for j, h in enumerate((5, 10, 20)):
                observed = np.isfinite(y[:, j]) & np.isfinite(downside[:, j])
                mu = f[f'mu{h}'].to_numpy()
                risk = f[f'risk{h}'].to_numpy()
                valid_mu, valid_label = mu[observed], y[observed, j]
                rank_ic = float(spearmanr(valid_mu, valid_label).statistic) if len(valid_mu)>1 and np.std(valid_mu)>0 and np.std(valid_label)>0 else np.nan
                top = np.argsort(-f.utility.to_numpy(), kind='stable')[:5]
                top_valid = top[np.isfinite(y[top, j])]
                positive = f.utility.to_numpy() > 0
                positives = positive & observed
                rows.append(dict(signal_date=day, scope=scope, horizon=h,
                    prediction_stocks=len(f), observed_labels=int(observed.sum()),
                    positive_utility_stocks=int(positive.sum()), rank_ic=rank_ic,
                    return_mae=float(np.mean(np.abs(valid_mu-valid_label))) if observed.any() else np.nan,
                    risk_mae=float(np.mean(np.abs(risk[observed]-downside[observed, j]))) if observed.any() else np.nan,
                    top5_observed=int(len(top_valid)),
                    top5_mean_forward_return=float(np.mean(y[top_valid, j])) if len(top_valid) else np.nan,
                    positive_utility_mean_forward_return=float(np.mean(y[positives, j])) if positives.any() else np.nan))
    result = pd.DataFrame(rows)
    destination = out/'diagnostics'; destination.mkdir(exist_ok=True)
    result.to_csv(destination/'daily_forecast_quality.csv', index=False)
    result['month'] = pd.to_datetime(result.signal_date).dt.strftime('%Y-%m')
    summary = result.groupby(['month','scope','horizon']).agg(
        sessions=('signal_date','size'), mean_rank_ic=('rank_ic','mean'),
        mean_return_mae=('return_mae','mean'), mean_risk_mae=('risk_mae','mean'),
        mean_top5_forward_return=('top5_mean_forward_return','mean'),
        mean_positive_utility_stocks=('positive_utility_stocks','mean'))
    summary.to_csv(destination/'monthly_forecast_quality.csv')
    publish_manifest(dict(status='complete', source_predictions_sha256=sha256(out/'predictions.pkl'),
        source_feature_store_sha256=sha256(out/'feature_store.pkl'),
        metric='Forecast diagnostics only. Overlapping horizon returns are NOT monthly cash-account returns.',
        label_selection='Rank top five first; missing future labels masked only in metrics',
        source_sha256=sha256(Path(__file__))), destination/'quality_status.json')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(quality_report(args.out).to_string())
