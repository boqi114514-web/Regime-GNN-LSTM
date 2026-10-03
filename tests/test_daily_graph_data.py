import sys
from pathlib import Path
import unittest
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from research_daily_graph_data import (
    adjusted_prices, asof_graph, fetch_audited_pages, validate_adjustment_snapshot,
    validate_edges, validate_membership_snapshot, publish_frame, publish_manifest,
    load_graph_inputs, load_adjusted_daily,
)


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.edges = pd.DataFrame([
            ('2024-12-31', 'BK0001.DC', '600001.SH'),
            ('2024-12-31', 'BK0001.DC', '300001.SZ'),
            ('2024-12-31', 'BK0002.DC', '600001.SH'),
            ('2024-12-31', 'BK0002.DC', '688001.SH'),
            ('2025-06-30', 'BK0001.DC', '600002.SH'),
        ], columns=['snapshot_date', 'theme_code', 'ts_code'])

    def test_overlapping_labels_and_all_boards_retained(self):
        graph = asof_graph(self.edges, '2025-01-02', ['600001.SH', '300001.SZ', '688001.SH'])
        self.assertEqual(graph.incidence.shape, (3, 2))
        self.assertEqual(graph.incidence[0].sum(), 2)
        self.assertTrue(graph.known.all())

    def test_no_future_snapshot_backfill(self):
        graph = asof_graph(self.edges, '2024-12-30', ['600001.SH'])
        self.assertIsNone(graph.snapshot_date)
        self.assertFalse(graph.known[0])
        self.assertEqual(graph.messages([[7]]).tolist(), [[0]])

    def test_future_edges_cannot_change_past_messages(self):
        codes = ['600001.SH', '300001.SZ', '688001.SH']
        first = asof_graph(self.edges, '2025-01-02', codes).messages([[1], [3], [7]])
        changed = self.edges.copy()
        changed.loc[changed.snapshot_date.eq('2025-06-30'), 'ts_code'] = '600001.SH'
        second = asof_graph(changed, '2025-01-02', codes).messages([[1], [3], [7]])
        np.testing.assert_array_equal(first, second)

    def test_latest_whole_snapshot_drops_old_removed_edges(self):
        graph = asof_graph(self.edges, '2025-07-01', ['600001.SH', '600002.SH'])
        np.testing.assert_array_equal(graph.known, [False, True])
        self.assertEqual(graph.snapshot_date, pd.Timestamp('2025-06-30'))

    def test_self_excluding_multilabel_message(self):
        graph = asof_graph(self.edges, '2025-01-02', ['600001.SH', '300001.SZ', '688001.SH', '600099.SH'])
        np.testing.assert_allclose(graph.messages([[1], [3], [7], [100]]), [[5], [1], [1], [0]])

    def test_explicit_half_year_carryforward_and_default_staleness(self):
        stale = asof_graph(self.edges, '2025-05-01', ['600001.SH'])
        retained = asof_graph(self.edges, '2025-05-01', ['600001.SH'], max_age_days=None)
        self.assertIsNone(stale.snapshot_date)
        self.assertTrue(retained.known[0])

    def test_missing_peers_do_not_invent_edges(self):
        graph = asof_graph(self.edges, '2025-01-02', ['600001.SH'])
        np.testing.assert_array_equal(graph.messages([[7]]), [[0]])
        self.assertTrue(graph.known[0])

    def test_duplicate_edges_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_edges(pd.concat([self.edges, self.edges.iloc[:1]], ignore_index=True))

    def test_non_dc_and_pre_coverage_rejected(self):
        for column, value in [('theme_code', '801010.SI'), ('snapshot_date', '2024-12-19')]:
            bad = self.edges.copy(); bad.loc[0, column] = value
            with self.assertRaises(ValueError):
                validate_edges(bad)

    def test_duplicate_stocks_or_nonfinite_messages_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate requested'):
            asof_graph(self.edges, '2025-01-02', ['600001.SH']*2)
        with self.assertRaisesRegex(ValueError, 'finite'):
            asof_graph(self.edges, '2025-01-02', ['600001.SH']).messages([[np.nan]])

    def test_membership_lower_bound_is_not_relaxed(self):
        frame = pd.DataFrame(dict(trade_date=['20241231'], ts_code=['BK0001.DC'], con_code=['600001.SH']))
        with self.assertRaisesRegex(ValueError, 'below dated lower bound 2'):
            validate_membership_snapshot(frame, '20241231', 'BK0001.DC', 2)
        self.assertEqual(len(validate_membership_snapshot(frame, '20241231', 'BK0001.DC', 1)), 1)

    def test_wrong_member_date_theme_and_row_limit_rejected(self):
        frame = pd.DataFrame(dict(trade_date=['20241231'], ts_code=['BK0001.DC'], con_code=['600001.SH']))
        for day, code, limit in [('20250102', 'BK0001.DC', 5000), ('20241231', 'BK0002.DC', 5000), ('20241231', 'BK0001.DC', 1)]:
            with self.assertRaises(ValueError):
                validate_membership_snapshot(frame, day, code, 0, limit)


class AdjustmentTests(unittest.TestCase):
    def setUp(self):
        self.daily = pd.DataFrame(dict(date=pd.to_datetime(['2025-01-02', '2025-01-03']),
            ts_code=['600001.SH']*2, open=[10, 5], high=[11, 6], low=[9, 4], close=[10, 5], amount=[100, 200]))
        self.adj = pd.DataFrame(dict(trade_date=['20250102', '20250103'], ts_code=['600001.SH']*2, adj_factor=[1., 2.]))

    def test_adjusted_split_does_not_become_minus50pct(self):
        out = adjusted_prices(self.daily, self.adj)
        np.testing.assert_allclose(out.adj_close, [10, 10])
        np.testing.assert_allclose(out.close, [10, 5])
        np.testing.assert_allclose(out.amount, [100, 200])

    def test_dated_factor_coverage(self):
        one = validate_adjustment_snapshot(self.adj.iloc[:1], '20250102', ['600001.SH'])
        self.assertEqual(len(one), 1)
        with self.assertRaisesRegex(ValueError, 'Missing factors'):
            validate_adjustment_snapshot(one, '20250102', ['600002.SH'])

    def test_future_factor_does_not_change_prior_adjusted_price(self):
        before = adjusted_prices(self.daily, self.adj)
        changed = self.adj.copy(); changed.loc[1, 'adj_factor'] = 999
        after = adjusted_prices(self.daily, changed)
        self.assertEqual(before.adj_close.iloc[0], after.adj_close.iloc[0])

    def test_no_factor_filling(self):
        with self.assertRaisesRegex(ValueError, 'coverage incomplete'):
            adjusted_prices(self.daily, self.adj.iloc[:1])

    def test_nonpositive_and_duplicate_adjustments_rejected(self):
        bad = self.adj.copy(); bad.loc[0, 'adj_factor'] = 0
        with self.assertRaisesRegex(ValueError, 'Nonpositive'):
            adjusted_prices(self.daily, bad)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            adjusted_prices(self.daily, pd.concat([self.adj, self.adj.iloc[:1]], ignore_index=True))

    def test_wrong_date_nonfinite_and_duplicate_factor_snapshot(self):
        one = self.adj.iloc[:1].copy()
        with self.assertRaisesRegex(ValueError, 'Wrong factor date'):
            validate_adjustment_snapshot(one, '20250103')
        one.loc[0, 'adj_factor'] = np.inf
        with self.assertRaisesRegex(ValueError, 'Nonpositive/nonfinite'):
            validate_adjustment_snapshot(one, '20250102')
        with self.assertRaisesRegex(ValueError, 'duplicate stock'):
            validate_adjustment_snapshot(pd.concat([self.adj.iloc[:1]]*2), '20250102')


class PaginationTests(unittest.TestCase):
    class Client:
        def __init__(self, frames):
            self.frames = iter(frames)
            self.offsets = []

        def query(self, api, **params):
            self.offsets.append(params['offset'])
            return next(self.frames)

    def test_disjoint_page_then_short_page(self):
        client = self.Client([pd.DataFrame(dict(code=['a', 'b'])), pd.DataFrame(dict(code=['c']))])
        out, audit = fetch_audited_pages(client, 'x', params={}, keys=['code'], page_size=2)
        self.assertEqual(out.code.tolist(), ['a', 'b', 'c'])
        self.assertEqual(client.offsets, [0, 2])
        self.assertEqual(len(audit), 2)

    def test_ignored_offset_and_partial_overlap_rejected(self):
        for second in [['a', 'b'], ['b', 'c']]:
            client = self.Client([pd.DataFrame(dict(code=['a', 'b'])), pd.DataFrame(dict(code=second))])
            with self.assertRaisesRegex(ValueError, 'offset appears ignored'):
                fetch_audited_pages(client, 'x', params={}, keys=['code'], page_size=2)

    def test_duplicate_keys_and_unbounded_pages_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Invalid page'):
            fetch_audited_pages(self.Client([pd.DataFrame(dict(code=['a', 'a']))]), 'x', params={}, keys=['code'], page_size=2)
        with self.assertRaisesRegex(ValueError, 'bounded'):
            fetch_audited_pages(self.Client([pd.DataFrame(dict(code=['a', 'b']))]), 'x', params={}, keys=['code'], page_size=2, max_pages=1)

    def test_empty_terminal_page_is_permitted_but_not_a_census(self):
        client = self.Client([pd.DataFrame(dict(code=['a', 'b'])), pd.DataFrame(columns=['code'])])
        out, _ = fetch_audited_pages(client, 'x', params={}, keys=['code'], page_size=2)
        self.assertEqual(len(out), 2)


class ManifestTests(unittest.TestCase):
    def test_partial_graph_requires_opt_in_and_finished_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            edges = pd.DataFrame(dict(snapshot_date=pd.to_datetime(['2024-12-31']), theme_code=['BK0001.DC'], ts_code=['600001.SH']))
            artifact = publish_frame(edges, root/'monthly_edges.partial.pkl')
            shard = publish_frame(edges, root/'20241231.pkl'); shard['attempts_finished'] = True
            manifest = dict(status='incomplete', partial_aggregate=artifact, snapshots={'20241231': shard})
            publish_manifest(manifest, root/'graph_manifest.json')
            with self.assertRaisesRegex(ValueError, 'explicit allow_partial'):
                load_graph_inputs(root)
            loaded, retained_manifest = load_graph_inputs(root, allow_partial=True)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(retained_manifest['status'], 'incomplete')
            manifest['status'] = 'running'; publish_manifest(manifest, root/'graph_manifest.json')
            with self.assertRaisesRegex(ValueError, 'still running'):
                load_graph_inputs(root, allow_partial=True)
            self.assertEqual(len(load_graph_inputs(root, allow_partial=True, require_finished=False)[0]), 1)

    def test_graph_manifest_hash_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            edges = pd.DataFrame(dict(snapshot_date=pd.to_datetime(['2024-12-31']), theme_code=['BK0001.DC'], ts_code=['600001.SH']))
            artifact = publish_frame(edges, root/'monthly_edges.pkl')
            shard = publish_frame(edges, root/'20241231.pkl'); shard['attempts_finished'] = True
            publish_manifest(dict(status='complete', aggregate=artifact, snapshots={'20241231': shard}), root/'graph_manifest.json')
            edges.loc[0, 'ts_code'] = '600002.SH'; publish_frame(edges, root/'monthly_edges.pkl')
            with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
                load_graph_inputs(root)

    def test_adjustments_reject_incomplete_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publish_manifest(dict(status='incomplete', failures={'20250102': '503'}), root/'adjustment_manifest.json')
            with self.assertRaisesRegex(ValueError, 'acquisition incomplete'):
                load_adjusted_daily(root)

    def test_adjustments_reject_row_coverage_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = publish_frame(pd.DataFrame(dict(x=[1])), root/'adjusted_daily.pkl')
            publish_manifest(dict(status='complete', completed_dates=1, planned_dates=1, failures={}, adjusted_daily=artifact, expected_daily_rows=2), root/'adjustment_manifest.json')
            with self.assertRaisesRegex(ValueError, 'row coverage mismatch'):
                load_adjusted_daily(root)


if __name__ == '__main__':
    unittest.main()
