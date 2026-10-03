"""Daily graph-conditioned tree heads: a distinct native-return benchmark.

The neural version is retained, including failed results. This benchmark asks
whether the bottleneck is its conditional-mean/validation collapse, rather than
daily data or the graph. No 2026 trade, stock or theme whitelist is used.
Reference: Microsoft's public Qlib LightGBM benchmark. This is not a paper's
return reproduction and is not a replacement for the four-expert neural model.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import shutil

import lightgbm as lgb
import numpy as np
import pandas as pd

from research_daily_graph_data import asof_graph, publish_frame, publish_manifest, sha256
from research_daily_opportunity import quarterly_plan, validate_prepared
from research_daily_opportunity_features import expected_utility, FEATURE_PROTOCOL


PARAMS = dict(objective='regression', metric='l2', num_leaves=15, max_depth=5,
              learning_rate=.05, min_data_in_leaf=400, lambda_l2=10.,
              feature_fraction=.8, bagging_fraction=.8, bagging_freq=1,
              num_threads=4, seed=42, verbosity=-1, deterministic=True,
              force_col_wise=True)


def daily_rows(store, edges, index, sequence_length=20):
    valid = store.valid_features[index-sequence_length+1:index+1].all(axis=0)
    indices = np.flatnonzero(valid)
    codes = tuple(store.stock_codes[i] for i in indices)
    sequence = store.features[index-sequence_length+1:index+1, indices]
    last = sequence[-1]
    graph = asof_graph(edges, store.dates[index], codes, max_age_days=None, validated=True)
    peer = graph.messages(last, exclude_self=True).astype(np.float32)
    degree = np.asarray(graph.incidence.sum(axis=1)).ravel()
    stage = store.stages[index, indices]
    x = np.column_stack((last, sequence.mean(axis=0), peer,
                         np.log1p(degree), np.eye(4)[stage])).astype(np.float32)
    return codes, x, store.label_returns[index, indices], store.label_downside[index, indices], graph.snapshot_date


def run_trees(source, out, rounds=120, start='2026-01-01', end='2026-09-24'):
    source, out = Path(source), Path(out)
    store, edges, feature_manifest = validate_prepared(source, use_graph=True)
    out.mkdir(parents=True, exist_ok=True)
    for file in ('feature_store.pkl', 'frozen_edges.pkl', 'feature_manifest.json'):
        shutil.copy2(source/file, out/file)
    checkpoint_dir = out/'checkpoints'; checkpoint_dir.mkdir(exist_ok=True)
    indices = store.signal_indices(start='2025-01-01', end=end)
    # Cache daily factors, never the 20-day sequences. All feature-valid stocks
    # enter peers even when their future labels cannot be used in fitting.
    factors = {index: daily_rows(store, edges, index) for index in indices}
    forecasts, audits = [], {}
    for quarter, plan in quarterly_plan(store, start, end).items():
        cutoff = plan['fit_cutoff']
        completed = [i for i in indices if store.dates[i] < cutoff
                     and not np.isnat(store.label_end_dates[i]).any()
                     and (store.label_end_dates[i] < np.datetime64(cutoff, 'D')).all()]
        if len(completed) < 60:
            raise ValueError('Insufficient complete historical labels')
        validation = completed[-20:]
        train_cutoff = store.dates[validation[0]]
        training = [i for i in completed if (store.label_end_dates[i] < np.datetime64(train_cutoff, 'D')).all()]
        if len(training) < 20:
            raise ValueError('Insufficient purged training days')

        def arrays(days):
            x, y, risk = [], [], []
            for i in days:
                _, features, returns, downside, _ = factors[i]
                good = np.isfinite(returns).all(axis=1) & np.isfinite(downside).all(axis=1)
                x.append(features[good]); y.append(returns[good]); risk.append(downside[good])
            return np.concatenate(x), np.concatenate(y), np.concatenate(risk)

        x, y, risk = arrays(training)
        vx, vy, vrisk = arrays(validation)
        print(json.dumps(dict(event='tree_fit_start', quarter=quarter, train_rows=len(x),
                              validation_rows=len(vx), train_sessions=len(training))), flush=True)
        heads, iterations = {}, {}
        for kind, target, vtarget in [('return', y, vy), ('risk', risk, vrisk)]:
            for j, horizon in enumerate((5, 10, 20)):
                name = f'{kind}{horizon}'
                train = lgb.Dataset(x, label=target[:, j], free_raw_data=False)
                valid = lgb.Dataset(vx, label=vtarget[:, j], reference=train, free_raw_data=False)
                model = lgb.train(PARAMS, train, num_boost_round=rounds,
                                  valid_sets=[valid], callbacks=[lgb.early_stopping(12, verbose=False)])
                heads[name] = model; iterations[name] = model.best_iteration
                print(json.dumps(dict(event='tree_head_complete', quarter=quarter,
                                      head=name, iterations=model.best_iteration)), flush=True)
        # Rally heads remain calibrated probabilities from separate binary
        # targets, not ranks re-labelled as probabilities.
        for j, (horizon, threshold) in enumerate(zip((5, 10, 20), (.10, .15, .30))):
            train = lgb.Dataset(x, label=(y[:, j]>=threshold).astype(float), free_raw_data=False)
            valid = lgb.Dataset(vx, label=(vy[:, j]>=threshold).astype(float), reference=train, free_raw_data=False)
            params = dict(PARAMS, objective='binary', metric='binary_logloss')
            model = lgb.train(params, train, num_boost_round=rounds,
                              valid_sets=[valid], callbacks=[lgb.early_stopping(12, verbose=False)])
            heads[f'rally{horizon}'] = model; iterations[f'rally{horizon}'] = model.best_iteration
        audit = dict(fit_cutoff=str(cutoff.date()), train_cutoff=str(train_cutoff.date()),
                     maximum_train_label_end=str(max(store.label_end_dates[i].max() for i in training)),
                     train_signal_dates=[str(store.dates[i].date()) for i in training],
                     validation_signal_dates=[str(store.dates[i].date()) for i in validation],
                     train_rows=len(x), validation_rows=len(vx), best_iterations=iterations,
                     configuration=dict(params=PARAMS, max_rounds=rounds, validation_sessions=20))
        checkpoint = checkpoint_dir/f'{quarter}.pt'
        with checkpoint.open('wb') as handle:
            pickle.dump(dict(heads=heads, audit=audit), handle, protocol=pickle.HIGHEST_PROTOCOL)
        with checkpoint.open('rb') as handle:
            heads = pickle.load(handle)['heads']
        audits[quarter] = dict(**audit, checkpoint_sha256=sha256(checkpoint))
        for i in plan['indices']:
            codes, features, _, _, snapshot = factors[i]
            mu = np.column_stack([heads[f'return{h}'].predict(features) for h in (5, 10, 20)])
            downside = np.maximum(0, np.column_stack([heads[f'risk{h}'].predict(features) for h in (5, 10, 20)]))
            f = pd.DataFrame(dict(signal_date=store.dates[i], ts_code=codes,
                                  utility=expected_utility(mu, downside), model_fit_cutoff=cutoff,
                                  graph_snapshot_date=snapshot))
            for j, h in enumerate((5, 10, 20)):
                f[f'mu{h}'], f[f'risk{h}'] = mu[:, j], downside[:, j]
                f[f'rally_probability{h}'] = heads[f'rally{h}'].predict(features)
            forecasts.append(f)
        publish_manifest(audits, out/'fit_audits.json')
        publish_frame(pd.concat(forecasts, ignore_index=True), out/'predictions.pkl')
    p = pd.concat(forecasts, ignore_index=True)
    p.to_csv(out/'predictions.csv', index=False, encoding='utf-8-sig')
    protocol = dict(model='daily_graph_conditioned_lightgbm_v1', features=FEATURE_PROTOCOL,
                    use_graph=True, checkpoint_format='Pickled LightGBM boosters; .pt is artifact name only',
                    train_start='2025-01-01', account_start=start, account_end=end,
                    factor_inputs='Own 26 last + own 26 twenty-session means + 26 self-free theme peer factors + graph degree + four weak past stage flags',
                    experts='This benchmark has tree heads, not the four-expert neural network',
                    score='(.2*mu5+.3*mu10+.5*mu20) - .5*(.2*risk5+.3*risk10+.5*risk20)',
                    validation='Twenty past eligible daily sessions; longest label end strictly before first validation signal',
                    hyperparameters_selected_from='Fixed before this account run, not a 2026 profit sweep',
                    research_sources=['https://github.com/microsoft/qlib/tree/main/examples/benchmarks/LightGBM'],
                    source_sha256={name: sha256(Path(__file__).parent/name) for name in (
                        'research_daily_opportunity_features.py', 'research_daily_graph_data.py',
                        'research_daily_opportunity.py', 'research_daily_opportunity_trees.py')})
    publish_manifest(protocol, out/'experiment_protocol.json')
    artifacts = {name: sha256(out/name) for name in ('predictions.pkl','fit_audits.json','experiment_protocol.json','feature_manifest.json')}
    publish_manifest(dict(status='complete',rows=len(p),sessions=int(p.signal_date.nunique()),artifacts=artifacts),out/'prediction_manifest.json')
    return p


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('results/daily_opportunity_20261002'))
    parser.add_argument('--out',type=Path,default=Path('results/daily_opportunity_trees_20261002'))
    parser.add_argument('--rounds',type=int,default=120)
    args=parser.parse_args()
    run_trees(args.source,args.out,args.rounds)
