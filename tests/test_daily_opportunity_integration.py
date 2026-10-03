"""Tiny real neural training through the daily graph/pipeline/execution seams."""
from pathlib import Path
import json
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import research_daily_opportunity as pipeline
from research_daily_graph_data import sha256
from research_daily_opportunity_execution import next_open_schedule, run_account
from research_daily_opportunity_features import LazyDailyOpportunityBatch, build_feature_store
from research_daily_opportunity_model import FitConfig, FittedOpportunityModel, eligible_label_mask


EMPTY_ACTIONS = pd.DataFrame(columns=['ts_code', 'record_date', 'ex_date',
    'pay_date', 'div_listdate', 'cash', 'stock', 'event'])


def tiny_daily():
    days = pd.bdate_range('2025-06-02', '2026-04-03')
    # Explicit actual-session gap: next prediction/execution cannot be Jan1.
    days = days[days != pd.Timestamp('2026-01-01')]
    rows = []
    for k, code in enumerate(('600001.SH', '000001.SZ', '300001.SZ')):
        for index, day in enumerate(days):
            price = round((10 + 3*k) * (1.0005 + k*.0001)**index, 2)
            rows.append(dict(date=day, ts_code=code, open=price,
                high=round(price*1.01, 2), low=round(price*.99, 2),
                close=price, amount=100000 + index*1000 + k*10000,
                adj_factor=1., volume=100.))
    return pd.DataFrame(rows), days


def tiny_edges():
    return pd.DataFrame([
        dict(snapshot_date=pd.Timestamp('2025-06-30'), theme_code='BK0001.DC', ts_code='600001.SH'),
        dict(snapshot_date=pd.Timestamp('2025-06-30'), theme_code='BK0002.DC', ts_code='600001.SH'),
        dict(snapshot_date=pd.Timestamp('2025-06-30'), theme_code='BK0001.DC', ts_code='300001.SZ'),
        dict(snapshot_date=pd.Timestamp('2026-03-31'), theme_code='BK0002.DC', ts_code='000001.SZ'),
    ])


def publish_inputs(directory, *, partial=False, unfinished=False, daily=None):
    """Generate collector-shaped metadata, not bypass prepare's provenance."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    daily = tiny_daily()[0] if daily is None else daily
    path = directory / 'adjusted_daily.pkl'
    daily.to_pickle(path)
    count = daily.date.nunique()
    adjustment = dict(status='complete', failures={}, planned_dates=count,
        completed_dates=count, expected_daily_rows=len(daily),
        adjusted_daily=dict(file=path.name, rows=len(daily), sha256=sha256(path)))
    (directory / 'adjustment_manifest.json').write_text(json.dumps(adjustment), encoding='utf-8')
    edges = tiny_edges()
    path = directory / ('monthly_edges.partial.pkl' if partial else 'monthly_edges.pkl')
    edges.to_pickle(path)
    snapshots = {}
    (directory / 'graphs').mkdir(exist_ok=True)
    for day, shard in edges.groupby('snapshot_date'):
        text = day.strftime('%Y%m%d')
        target = directory / 'graphs' / f'{text}.pkl'
        shard.to_pickle(target)
        snapshots[text] = dict(file=f'graphs/{text}.pkl', rows=len(shard),
            sha256=sha256(target), attempts_finished=not unfinished,
            status='verified_partial' if partial else 'complete')
    metadata = dict(status='running' if unfinished else 'incomplete' if partial else 'complete',
        snapshots=snapshots, partial_graph_policy='Unknown edges, not candidate filters')
    metadata['partial_aggregate' if partial else 'aggregate'] = dict(
        file=path.name, rows=len(edges), sha256=sha256(path))
    (directory / 'graph_manifest.json').write_text(json.dumps(metadata), encoding='utf-8')
    return directory


class DailyOpportunityIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def fixture(self):
        daily, days = tiny_daily()
        return daily, days, build_feature_store(daily, sessions=days), tiny_edges()

    def test_quarter_fit_is_selected_by_next_execution_not_signal_month(self):
        _, days, store, _ = self.fixture()
        plan = pipeline.quarterly_plan(store, '2026-01-01', '2026-04-03')
        self.assertEqual(set(plan), {'2026Q1', '2026Q2'})
        self.assertEqual(plan['2026Q1']['fit_cutoff'], pd.Timestamp('2025-12-31'))
        self.assertEqual(plan['2026Q2']['fit_cutoff'], pd.Timestamp('2026-03-31'))
        first_q1 = plan['2026Q1']['indices'][0]
        first_q2 = plan['2026Q2']['indices'][0]
        self.assertEqual(days[first_q1], pd.Timestamp('2025-12-31'))
        self.assertEqual(days[first_q1+1], pd.Timestamp('2026-01-02'))
        self.assertEqual(days[first_q2], pd.Timestamp('2026-03-31'))
        self.assertEqual(days[first_q2+1], pd.Timestamp('2026-04-01'))

    def test_two_quarter_real_training_reload_daily_predictions_and_account(self):
        daily, sessions, store, edges = self.fixture()
        config = FitConfig(hidden=8, epochs=1, validation_sessions=3,
                           minimum_train_sessions=2, sequence_length=20)
        with tempfile.TemporaryDirectory() as task:
            out = Path(task)
            inputs = publish_inputs(out / 'inputs')
            with patch.object(pipeline, '_progress'):
                store, edges = pipeline.prepare(inputs, out)
                predictions = pipeline.train_and_predict(store, edges, out,
                    start='2026-01-01', end='2026-04-03', train_start='2025-06-01',
                    config=config, device='cpu')
            self.assertEqual(predictions.signal_date.min(), pd.Timestamp('2025-12-31'))
            self.assertTrue((predictions.model_fit_cutoff <= predictions.signal_date).all())
            self.assertTrue(predictions.ts_code.eq('300001.SZ').any())
            self.assertTrue(np.isfinite(predictions.filter(regex='^(mu|risk|rally_probability|utility)').to_numpy()).all())
            audits = json.loads((out / 'fit_audits.json').read_text(encoding='utf-8'))
            self.assertEqual(set(audits), {'2026Q1', '2026Q2'})
            for quarter, audit in audits.items():
                self.assertLess(audit['maximum_train_label_end'], audit['train_cutoff'])
                self.assertLess(audit['maximum_train_label_end'], audit['fit_cutoff'])
                path = out / 'checkpoints' / f'{quarter}.pt'
                self.assertEqual(audit['checkpoint_sha256'], sha256(path))
                checkpoint = FittedOpportunityModel.load(path)
                first = predictions[predictions.model_fit_cutoff.eq(pd.Timestamp(audit['fit_cutoff']))]
                signal = first.signal_date.min()
                index = store.dates.get_loc(signal)
                batch = LazyDailyOpportunityBatch(store, index, edges)
                reloaded = checkpoint.predict(batch)
                saved = first[first.signal_date.eq(signal)].set_index('ts_code').loc[list(batch.stock_codes)]
                for j, horizon in enumerate((5, 10, 20)):
                    np.testing.assert_array_equal(saved[f'mu{horizon}'], reloaded['return_mu'][:, j])
                    np.testing.assert_array_equal(saved[f'risk{horizon}'], reloaded['downside'][:, j])
                self.assertLess(max(audit['validation_signal_dates']), audit['fit_cutoff'])
            # Complete predictions are persisted, including ChiNext features.
            pd.testing.assert_frame_equal(pd.read_pickle(out / 'predictions.pkl'), predictions)
            pd.testing.assert_frame_equal(pipeline.validate_predictions(out), predictions)
            # The last-day future labels are absent, without removing its graph nodes.
            batch = LazyDailyOpportunityBatch(store, len(store.dates)-1, edges)
            self.assertEqual(len(batch.stock_codes), 3)
            self.assertFalse(eligible_label_mask(batch, '2026-04-04').any())
            # Raw-price lot ledger receives these real saved-model predictions.
            account = out / 'account'
            run_account(predictions, daily, sessions, account,
                start='2026-01-01', end='2026-04-03', offline=True,
                limit_provider=lambda date, code: pd.Series(dict(up_limit=100., down_limit=1.)),
                action_provider=lambda code: EMPTY_ACTIONS.copy())
            status = json.loads((account / 'daily_run_status.json').read_text(encoding='utf-8'))
            self.assertTrue(status['account_complete'])
            trades = pd.read_csv(account / 'trades.csv', parse_dates=['date', 'signal_date'])
            if not trades.empty:
                self.assertFalse(trades.code.str.startswith('300').any())
                self.assertTrue((trades.shares % 100 == 0).all())
                self.assertTrue((trades.signal_date < trades.date).all())
            schedule = next_open_schedule(predictions, sessions)
            self.assertTrue((schedule.model_fit_cutoff < schedule.execution_date).all())

            # Even a well-formed changed prediction, checkpoint or audit is rejected.
            prediction_path = out / 'predictions.pkl'
            original_prediction_bytes = prediction_path.read_bytes()
            changed = predictions.copy()
            changed.loc[0, 'mu20'] += 1.
            changed.to_pickle(prediction_path)
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                pipeline.validate_predictions(out)
            prediction_path.write_bytes(original_prediction_bytes)
            checkpoint_path = out / 'checkpoints' / '2026Q1.pt'
            original_checkpoint_bytes = checkpoint_path.read_bytes()
            checkpoint_path.write_bytes(original_checkpoint_bytes + b'tamper')
            with self.assertRaisesRegex(ValueError, 'Checkpoint fingerprint'):
                pipeline.validate_predictions(out)
            checkpoint_path.write_bytes(original_checkpoint_bytes)
            # Rebinding the artifact SHA still cannot authorize a future checkpoint.
            changed = predictions.copy()
            changed.loc[0, 'model_fit_cutoff'] = pd.Timestamp('2030-01-01')
            changed.to_pickle(prediction_path)
            manifest_path = out / 'prediction_manifest.json'
            metadata = json.loads(manifest_path.read_text(encoding='utf-8'))
            metadata['artifacts']['predictions.pkl'] = sha256(prediction_path)
            manifest_path.write_text(json.dumps(metadata), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'future checkpoint'):
                pipeline.validate_predictions(out)

    def test_invalid_or_premature_graph_snapshot_does_not_filter_stocks(self):
        _, _, store, edges = self.fixture()
        early = store.dates.get_loc(pd.Timestamp('2025-12-31'))
        without = LazyDailyOpportunityBatch(store, early, edges.iloc[0:0])
        future_only = LazyDailyOpportunityBatch(store, early, edges[edges.snapshot_date.gt('2025-12-31')])
        self.assertEqual(without.stock_codes, future_only.stock_codes)
        self.assertEqual(tuple(without.membership.shape), (3, 0))
        self.assertEqual(tuple(future_only.membership.shape), (3, 0))
        self.assertEqual(len(without.stock_codes), 3)

    def test_prepare_partial_graph_requires_opt_in_and_finished_acquisition(self):
        with tempfile.TemporaryDirectory() as task:
            root = Path(task)
            inputs = publish_inputs(root / 'inputs', partial=True)
            with patch.object(pipeline, '_progress'):
                with self.assertRaisesRegex(ValueError, 'explicit'):
                    pipeline.prepare(inputs, root / 'rejected')
                _, edges = pipeline.prepare(inputs, root / 'accepted', allow_partial_graph=True)
                self.assertEqual(len(edges), 4)
                metadata = json.loads((root / 'accepted' / 'feature_manifest.json').read_text(encoding='utf-8'))
                self.assertEqual(metadata['graph_coverage']['status'], 'incomplete')
                publish_inputs(root / 'unfinished', partial=True, unfinished=True)
                with self.assertRaisesRegex(ValueError, 'running'):
                    pipeline.prepare(root / 'unfinished', root / 'bad', allow_partial_graph=True)

    def test_prepared_cache_rejects_mode_store_and_raw_price_tampering(self):
        with tempfile.TemporaryDirectory() as task:
            root = Path(task)
            inputs = publish_inputs(root / 'inputs')
            out = root / 'prepared'
            with patch.object(pipeline, '_progress'):
                pipeline.prepare(inputs, out)
            pipeline.validate_prepared(out, use_graph=True)
            with self.assertRaisesRegex(ValueError, 'mode'):
                pipeline.validate_prepared(out, use_graph=False)
            path = out / 'feature_store.pkl'
            original = path.read_bytes()
            store = pickle.loads(original)
            store.features[-1, 0, 0] += .01
            path.write_bytes(pickle.dumps(store))
            with self.assertRaisesRegex(ValueError, 'feature-store'):
                pipeline.validate_prepared(out)
            path.write_bytes(original)
            raw = inputs / 'adjusted_daily.pkl'
            daily = pd.read_pickle(raw)
            daily['amount'] *= 2.
            daily.to_pickle(raw)
            with self.assertRaisesRegex(ValueError, 'raw_adjusted_daily'):
                pipeline.validate_prepared(out)

    def test_cli_training_refuses_valid_but_changed_frozen_graph(self):
        with tempfile.TemporaryDirectory() as task:
            root = Path(task)
            inputs = publish_inputs(root / 'inputs')
            out = root / 'prepared'
            with patch.object(pipeline, '_progress'):
                pipeline.prepare(inputs, out)
            path = out / 'frozen_edges.pkl'
            graph = pd.read_pickle(path)
            graph.loc[0, 'theme_code'] = 'BK0099.DC'
            graph.to_pickle(path)
            with patch.object(pipeline, 'train_and_predict') as training:
                with self.assertRaisesRegex(ValueError, 'edges fingerprint'):
                    pipeline.main(['train', '--out', str(out), '--inputs', str(inputs), '--device', 'cpu'])
                training.assert_not_called()

    def test_account_refuses_other_individually_valid_price_panel(self):
        with tempfile.TemporaryDirectory() as task:
            root = Path(task)
            original_inputs = publish_inputs(root / 'inputs')
            out = root / 'prepared'
            with patch.object(pipeline, '_progress'):
                pipeline.prepare(original_inputs, out)
            different_prices = tiny_daily()[0]
            different_prices[['open', 'high', 'low', 'close']] *= 2.
            alternate_inputs = publish_inputs(root / 'other_inputs', daily=different_prices)
            # Both individual input manifests are valid. Their prices cannot be
            # substituted for those that generated the model's daily features.
            with patch('research_daily_opportunity_execution.run_account') as account:
                with self.assertRaises(ValueError):
                    pipeline.run_daily_account(pd.DataFrame(), alternate_inputs,
                        out, offline=True)
                account.assert_not_called()


if __name__ == '__main__':
    unittest.main()
