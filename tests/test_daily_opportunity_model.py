import gc
from pathlib import Path
import sys
import tempfile
import unittest
import weakref

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from research_daily_opportunity_model import (
    DailyGraphOpportunityNet, DailyOpportunityBatch, FitConfig,
    FittedOpportunityModel, build_causal_membership, concept_messages,
    eligible_label_mask, expected_risk_score, fit_walk_forward,
    opportunity_loss, purged_split,
)


class DailyOpportunityModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(12)

    def batch(self, offset=0, stocks=4, days=3):
        date = np.datetime64('2024-01-01') + np.timedelta64(offset, 'D')
        sequences = np.random.default_rng(100 + offset).normal(size=(stocks, days, 3)).astype(np.float32)
        dates = date - np.arange(days - 1, -1, -1).astype('timedelta64[D]')
        returns = np.tile(np.array([.02, .04, .08], np.float32), (stocks, 1))
        downside = np.tile(np.array([.01, .02, .03], np.float32), (stocks, 1))
        ends = date + np.array([1, 2, 3]).astype('timedelta64[D]')
        return DailyOpportunityBatch(date, sequences,
            np.ones((stocks, 2), dtype=np.float32), returns, downside,
            np.tile(ends, (stocks, 1)), [f'stock{i}' for i in range(stocks)],
            dates, date, np.arange(stocks) % 4)

    def network(self):
        return DailyGraphOpportunityNet(3, hidden=8, sequence_length=3)

    def test_multi_label_graph_exact_means_and_unmatched_stock(self):
        features = torch.tensor([[2., 0.], [4., 2.], [6., 4.], [9., 9.]])
        graph = torch.tensor([[1., 0.], [1., 1.], [0., 1.], [0., 0.]])
        expected = torch.tensor([[3., 1.], [4., 2.], [5., 3.], [0., 0.]])
        torch.testing.assert_close(concept_messages(features, graph), expected)
        torch.testing.assert_close(concept_messages(features, graph.to_sparse()), expected)

    def test_empty_and_unknown_zero_concepts_have_no_effect(self):
        values = torch.randn(4, 8)
        graph = torch.eye(4)
        before = concept_messages(values, graph)
        after = concept_messages(values, torch.cat((graph, torch.zeros(4, 3)), dim=1))
        torch.testing.assert_close(before, after)
        torch.testing.assert_close(concept_messages(values, torch.zeros(4, 0)), torch.zeros_like(values))

    def test_graph_rejects_negative_and_nonfinite_memberships(self):
        for value in (-1., float('nan')):
            with self.assertRaises(ValueError):
                concept_messages(torch.randn(2, 4), torch.tensor([[value], [1.]]))

    def test_causal_graph_discards_future_links_and_accepts_new_dated_themes(self):
        links = [dict(ts_code='a', theme_code='optics', available_date='2024-01-01'),
                 dict(ts_code='a', theme_code='optics', available_date='2024-01-01'),
                 dict(ts_code='b', theme_code='new', available_date='2024-01-03'),
                 dict(ts_code='c', theme_code='old', available_date='2023-01-01', effective_to='2024-01-02')]
        graph, themes = build_causal_membership(['a', 'b', 'c'], links, '2024-01-02')
        self.assertEqual(themes, ['optics'])
        self.assertEqual(graph._nnz(), 1)
        later, themes = build_causal_membership(['a', 'b', 'c'], links, '2024-01-03')
        self.assertEqual(themes, ['new', 'optics'])
        self.assertEqual(later._nnz(), 2)

    def test_graph_refuses_undated_current_snapshot(self):
        with self.assertRaises(ValueError):
            build_causal_membership(['a'], [dict(ts_code='a', theme_code='today')], '2024-01-01')

    def test_output_native_shapes_risk_probability_and_normalized_gate(self):
        model, batch = self.network(), self.batch()
        output = model(torch.from_numpy(batch.sequences), torch.from_numpy(batch.membership))
        for name in ('return_mu', 'downside', 'rally_logits'):
            self.assertEqual(tuple(output[name].shape), (4, 3))
            self.assertTrue(torch.isfinite(output[name]).all())
        self.assertTrue((output['downside'] >= 0).all())
        torch.testing.assert_close(output['gate'].sum(1), torch.ones(4))
        self.assertEqual(tuple(output['expert_return'].shape), (4, 4, 3))

    def test_stock_and_concept_permutation_equivariance(self):
        model, batch = self.network(), self.batch()
        x = torch.from_numpy(batch.sequences)
        graph = torch.tensor([[1., 0., 1.], [1., 1., 0.], [0., 1., 1.], [0., 0., 0.]])
        stocks, concepts = torch.tensor([2, 0, 3, 1]), torch.tensor([2, 0, 1])
        original = model(x, graph)
        shuffled = model(x[stocks], graph[stocks][:, concepts].to_sparse())
        for name in ('return_mu', 'downside', 'rally_logits', 'gate'):
            torch.testing.assert_close(shuffled[name], original[name][stocks])

    def test_temporal_order_matters_and_only_trailing_window_is_used(self):
        model, batch = self.network(), self.batch()
        x, graph = torch.from_numpy(batch.sequences), torch.from_numpy(batch.membership)
        original = model(x, graph)['return_mu']
        extended = model(torch.cat((torch.full((4, 2, 3), 999.), x), dim=1), graph)['return_mu']
        torch.testing.assert_close(original, extended)
        self.assertFalse(torch.allclose(original, model(x.flip(1), graph)['return_mu']))

    def test_future_sequence_or_membership_is_rejected(self):
        batch = self.batch()
        batch.sequence_dates[-1] += np.timedelta64(1, 'D')
        with self.assertRaises(ValueError):
            batch.validate()
        batch = self.batch()
        batch.membership_asof = '2025-01-01'
        with self.assertRaises(ValueError):
            batch.validate()

    def test_all_horizon_purge_including_equal_date_and_missing_long_label(self):
        batch = self.batch()
        batch.label_end_dates[0, 2] = np.datetime64('2024-01-05')
        batch.label_end_dates[1, 2] = np.datetime64('NaT')
        batch.returns[2, 1] = np.nan
        mask = eligible_label_mask(batch, '2024-01-05')
        np.testing.assert_array_equal(mask, [False, False, False, True])

    def test_loss_masks_labels_but_preserves_all_graph_nodes(self):
        model, batch = self.network(), self.batch()
        batch.returns[1:] = np.nan
        mask = np.array([True, False, False, False])
        output = model(torch.from_numpy(batch.sequences), torch.from_numpy(batch.membership))
        loss = opportunity_loss(output, batch.returns, batch.downside, mask=mask, stage_targets=batch.stage_targets)
        loss['total'].backward()
        self.assertTrue(torch.isfinite(loss['total']))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
        # Label-invalid node still changes the valid node's graph/market context.
        changed = torch.from_numpy(batch.sequences.copy())
        changed[1] += 5.
        other = model(changed, torch.from_numpy(batch.membership))
        self.assertFalse(torch.allclose(output['return_mu'][0], other['return_mu'][0]))

    def test_chronological_validation_has_long_label_embargo(self):
        batches = [self.batch(offset) for offset in range(15)]
        config = FitConfig(validation_sessions=3, minimum_train_sessions=2, sequence_length=3)
        training, validation, cutoff = purged_split(batches, '2024-01-17', config)
        self.assertEqual(str(cutoff), '2024-01-11')
        self.assertEqual(len(validation), 3)
        for batch, mask in training:
            self.assertTrue((batch.label_end_dates[mask] < cutoff).all())
        self.assertTrue(all(batch.signal_date >= cutoff for batch, _ in validation))

    def test_insufficient_purged_history_and_one_shot_generator_rejected(self):
        config = FitConfig(validation_sessions=3, minimum_train_sessions=2)
        with self.assertRaises(ValueError):
            purged_split([self.batch()], '2024-01-17', config)
        with self.assertRaises(ValueError):
            purged_split((self.batch(i) for i in range(15)), '2024-01-17', config)

    def test_lazy_factory_does_not_retain_stock_sequences(self):
        references = []
        def factory():
            for offset in range(15):
                batch = self.batch(offset)
                references.append(weakref.ref(batch))
                yield batch
        training, validation, _ = purged_split(factory, '2024-01-17',
            FitConfig(validation_sessions=3, minimum_train_sessions=2))
        gc.collect()
        self.assertEqual(sum(ref() is not None for ref in references), 0)
        self.assertGreater(len(list(training)), 0)
        self.assertEqual(len(list(validation)), 3)

    def test_actual_fit_calibration_checkpoint_and_native_risk_score(self):
        batches = [self.batch(offset) for offset in range(15)]
        config = FitConfig(hidden=8, sequence_length=3, epochs=2,
                           validation_sessions=3, minimum_train_sessions=2)
        events = []
        fitted = fit_walk_forward(batches, '2024-01-17', config, progress=events.append)
        self.assertEqual([event['event'] for event in events],
                         ['fit_start', 'epoch', 'epoch', 'fit_complete'])
        self.assertLess(fitted.audit['maximum_train_label_end'], fitted.audit['train_cutoff'])
        self.assertGreater(fitted.audit['train_rows'], 0)
        self.assertEqual(len(fitted.audit['history']), 2)
        batch = self.batch(18)
        expected = fitted.predict(batch)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            fitted.save(path)
            actual = FittedOpportunityModel.load(path).predict(batch)
        for name in ('return_mu', 'downside', 'rally_probability', 'gate'):
            np.testing.assert_array_equal(expected[name], actual[name])
        scores = expected_risk_score(actual, risk_weight=.5)
        np.testing.assert_allclose(scores, actual['return_mu'][:, 2] - .5 * actual['downside'][:, 2])
        self.assertTrue((actual['rally_probability'] >= 0).all())
        self.assertTrue((actual['rally_probability'] <= 1).all())
        with self.assertRaises(ValueError):
            fitted.predict(self.batch(1))

    def test_factory_can_retrain_without_materialization(self):
        calls = []
        def factory():
            calls.append(True)
            return (self.batch(offset) for offset in range(15))
        fitted = fit_walk_forward(factory, '2024-01-17', FitConfig(hidden=8,
            sequence_length=3, epochs=1, validation_sessions=3,
            minimum_train_sessions=2, calibrate=False))
        self.assertGreaterEqual(len(calls), 4)
        self.assertEqual(fitted.audit['batch_access'], 're-iterable factory')

    def test_scaler_uses_training_only_not_validation_extremes(self):
        batches = [self.batch(offset) for offset in range(15)]
        config = FitConfig(hidden=8, sequence_length=3, epochs=1,
            validation_sessions=3, minimum_train_sessions=2, calibrate=False)
        ordinary = fit_walk_forward(batches, '2024-01-17', config)
        # Chronological validation starts Jan11; train purges through Jan7.
        for batch in batches[10:]:
            batch.sequences[:] = 100000.
        changed = fit_walk_forward(batches, '2024-01-17', config)
        torch.testing.assert_close(ordinary.mean, changed.mean, rtol=0, atol=0)
        torch.testing.assert_close(ordinary.std, changed.std, rtol=0, atol=0)

    def test_future_or_incomplete_labels_do_not_change_fit_parameters(self):
        config = FitConfig(hidden=8, sequence_length=3, epochs=1,
            validation_sessions=0, minimum_train_sessions=2, calibrate=False)
        batches = [self.batch(offset) for offset in range(15)]
        # Stock1 has an unavailable 20-session target even at a past signal.
        for batch in batches:
            batch.label_end_dates[1, 2] = np.datetime64('2024-02-01')
        ordinary = fit_walk_forward(batches, '2024-01-17', config)
        for batch in batches:
            batch.returns[1] = 100.
            batch.downside[1] = 20.
        future = self.batch(25)
        future.sequences[:] = 10000.
        batches.append(future)
        changed = fit_walk_forward(batches, '2024-01-17', config)
        for key, expected in ordinary.network.state_dict().items():
            torch.testing.assert_close(expected, changed.network.state_dict()[key], rtol=0, atol=0)

    def test_invalid_loss_and_risk_score_fail_closed(self):
        model, batch = self.network(), self.batch()
        output = model(torch.from_numpy(batch.sequences), torch.from_numpy(batch.membership))
        with self.assertRaises(ValueError):
            opportunity_loss(output, batch.returns, batch.downside, mask=np.zeros(4, bool))
        with self.assertRaises(ValueError):
            expected_risk_score({'return_mu': np.ones((4, 3)), 'downside': -np.ones((4, 3))})


if __name__ == '__main__':
    unittest.main()
