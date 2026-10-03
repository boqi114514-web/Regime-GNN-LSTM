import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from research_dc_specialist import (LIQUID_LEADER_BASE_MODES, LIQUID_LEADER_PROTOCOL,
                                    MEMBER_PROTOCOL, PROTOCOL, RELATIVE_MEMBER_PROTOCOL,
                                    protocol_for_mode, specialist_candidates)


class DCSpecialistTests(unittest.TestCase):
    def fixtures(self):
        signal = pd.Timestamp('2026-03-31')
        themes = pd.DataFrame(dict(
            signal_date=[signal] * 3, ts_code=['T1', 'T2', 'T3'],
            ret20=[.20, .10, .02], ret60=[.25, .20, .10],
            theme_qualifies=[True] * 3))
        stocks = pd.DataFrame(dict(
            signal_date=[signal] * 4, month=[signal] * 4,
            ts_code=['600001.SH', '300001.SZ', '688001.SH', '000001.SZ'],
            mom1=[.12, .08, .25, .35], mom3=[.13, .60, .30, .50],
            mom6=[-.10, 1., .50, 1.], amount_percentile=[.50, 1., .60, .70],
            liquidity_days=[20] * 4, affinity_theme_code=['T1', 'T1', 'T1', 'T2'],
            affinity_theme_score=[.80, .80, .80, .60],
            affinity_ret20=[.20, .20, .20, .10], affinity_ret60=[.25, .25, .25, .20],
            affinity_qualifies=[True] * 4, heat_excluded=[False] * 4,
            leadership_score=[.60, .95, .70, .80], eligible=[False, True, True, True]))
        return stocks, themes

    def test_log_acceleration_uses_preceding_period_not_cumulative_division(self):
        c, t = self.fixtures()
        c.loc[0, 'mom3'] = (1 + c.loc[0, 'mom1']) * 1.01 ** 2 - 1
        result = specialist_candidates(c, theme_audit=t)
        row = result[result.ts_code.eq('600001.SH')].iloc[0]
        self.assertAlmostEqual(row.specialist_acceleration, np.log1p(.12) - np.log1p(.01))
        self.assertAlmostEqual(row.specialist_relative_strength, np.log1p(.12) - np.log1p(.20))
        self.assertAlmostEqual(row.specialist_theme_acceleration,
            np.log1p(.20) - (np.log1p(.25) - np.log1p(.20)) / 2)

    def test_early_recovery_admitted_and_long_cumulative_liquidity_not_dominant(self):
        c, t = self.fixtures()
        result = specialist_candidates(c, theme_audit=t).set_index('ts_code')
        self.assertTrue(result.loc['600001.SH', 'eligible'])
        self.assertLess(result.loc['600001.SH', 'mom6'], 0)
        self.assertGreater(result.loc['600001.SH', 'leadership_score'],
                           result.loc['300001.SZ', 'leadership_score'])
        self.assertFalse(result.loc['000001.SZ', 'eligible'])
        self.assertTrue(result.phase.eq('advance').all())

    def test_early_cap_is_one_fixed_ablation_including_boundary(self):
        c, t = self.fixtures()
        c.loc[0, 'mom1'] = .30
        c.loc[2, 'mom1'] = .3001
        ordinary = specialist_candidates(c, theme_audit=t).set_index('ts_code')
        early = specialist_candidates(c, 'theme_early', t).set_index('ts_code')
        self.assertTrue(ordinary.loc['688001.SH', 'eligible'])
        self.assertFalse(early.loc['688001.SH', 'eligible'])
        self.assertTrue(early.loc['600001.SH', 'eligible'])
        self.assertIn('NOT a universal', PROTOCOL['early_ablation'])

    def test_all_boards_rank_before_execution_and_no_board_filter(self):
        c, t = self.fixtures()
        result = specialist_candidates(c, theme_audit=t).set_index('ts_code')
        self.assertTrue(result.loc['300001.SZ', 'eligible'])
        self.assertTrue(result.loc['688001.SH', 'eligible'])
        self.assertEqual(result.loc['688001.SH', 'specialist_acceleration_percentile'], .75)
        self.assertEqual(len(result), len(c))

    def test_complete_theme_universe_changes_gate_even_for_unassociated_theme(self):
        c, t = self.fixtures()
        extra = t.iloc[:1].assign(ts_code='UNASSOCIATED', ret20=.80, ret60=.81)
        full = pd.concat([t, extra], ignore_index=True)
        actual = specialist_candidates(c, theme_audit=full)
        row = actual[actual.affinity_theme_code.eq('T1')].iloc[0]
        self.assertEqual(row.specialist_theme_count, 4)
        self.assertEqual(row.specialist_theme_rank, 2)
        # ceil(4/3) is two: the associated theme is not silently ranked in a
        # stock-count-weighted or associated-theme-only population.
        self.assertAlmostEqual(row.specialist_theme_priority, .75)

    def test_theme_rank_does_not_depend_on_stock_coverage(self):
        c, t = self.fixtures()
        expected = specialist_candidates(c, theme_audit=t)
        extra = c.iloc[:1].assign(ts_code='600002.SH')
        actual = specialist_candidates(pd.concat([c, extra], ignore_index=True), theme_audit=t)
        a = actual[actual.ts_code.isin(c.ts_code)].sort_values('ts_code')
        b = expected.sort_values('ts_code')
        pd.testing.assert_series_equal(a.specialist_theme_priority.reset_index(drop=True),
                                       b.specialist_theme_priority.reset_index(drop=True))
        pd.testing.assert_series_equal(a.specialist_theme_rank.reset_index(drop=True),
                                       b.specialist_theme_rank.reset_index(drop=True))

    def test_future_rows_unknown_labels_and_order_cannot_change_current_scores(self):
        c, t = self.fixtures()
        original_c, original_t = c.copy(deep=True), t.copy(deep=True)
        expected = specialist_candidates(c, theme_audit=t)
        later = c.assign(signal_date=pd.Timestamp('2026-04-30'),
            month=pd.Timestamp('2026-04-30'), mom1=100., mom3=100., mom6=100.,
            affinity_ret20=100., affinity_ret60=100.)
        later_t = t.assign(signal_date=pd.Timestamp('2026-04-30'), ret20=100., ret60=100.)
        mixed = pd.concat([c, later], ignore_index=True).assign(future_return=1000.)
        actual = specialist_candidates(mixed.sample(frac=1, random_state=7),
            theme_audit=pd.concat([t, later_t]).sample(frac=1, random_state=9))
        actual = actual[actual.signal_date.eq(pd.Timestamp('2026-03-31'))]
        pd.testing.assert_frame_equal(expected, actual.drop(columns='future_return').reset_index(drop=True))
        pd.testing.assert_frame_equal(c, original_c)
        pd.testing.assert_frame_equal(t, original_t)
        pd.testing.assert_frame_equal(expected,
            specialist_candidates(c.iloc[::-1], theme_audit=t.iloc[::-1]))

    def test_missing_invalid_stock_features_fail_closed(self):
        for field, value in [('mom1', np.nan), ('mom3', np.inf), ('mom6', -1.),
                             ('liquidity_days', 9), ('amount_percentile', 1.1),
                             ('affinity_ret20', np.nan), ('affinity_qualifies', None),
                             ('heat_excluded', None)]:
            with self.subTest(field=field):
                c, t = self.fixtures()
                if value is None:
                    c[field] = c[field].astype(object)
                c.loc[0, field] = value
                result = specialist_candidates(c, theme_audit=t).set_index('ts_code')
                self.assertFalse(result.loc['600001.SH', 'eligible'])
        c, t = self.fixtures()
        c.loc[0, ['mom1', 'mom3']] = [.4, 1.6]
        c.loc[0, 'heat_excluded'] = False
        self.assertFalse(specialist_candidates(c, theme_audit=t).set_index('ts_code').loc['600001.SH', 'eligible'])

    def test_invalid_theme_pool_fails_closed_without_default_cash_replacement(self):
        c, t = self.fixtures()
        t['theme_qualifies'] = False
        self.assertFalse(specialist_candidates(c, theme_audit=t).eligible.any())
        c, t = self.fixtures()
        t.loc[0, 'ret60'] = np.nan
        c.loc[c.affinity_theme_code.eq('T1'), 'affinity_ret60'] = np.nan
        result = specialist_candidates(c, theme_audit=t)
        self.assertFalse(result[result.affinity_theme_code.eq('T1')].eligible.any())

    def test_theme_short_pullback_remains_in_unified_positive_midterm_pool(self):
        c, t = self.fixtures()
        t.loc[2, ['ret20', 'ret60']] = [-.02, .10]
        result = specialist_candidates(c, theme_audit=t)
        self.assertEqual(result[result.affinity_theme_code.eq('T1')].specialist_theme_count.iloc[0], 3)
        self.assertIn('advancing and recovering', PROTOCOL['theme_pool'])

    def test_nullable_unknowns_fail_closed_as_false_not_nullable_eligibility(self):
        for field in ('mom1', 'affinity_qualifies', 'heat_excluded'):
            with self.subTest(field=field):
                c, t = self.fixtures()
                c[field] = c[field].astype('Float64' if field == 'mom1' else 'boolean')
                c.loc[0, field] = pd.NA
                result = specialist_candidates(c, theme_audit=t).set_index('ts_code')
                self.assertEqual(result.loc['600001.SH', 'eligible'], False)
                self.assertFalse(result.eligible.isna().any())

    def test_complete_theme_audit_required_and_dated_link_must_match(self):
        c, t = self.fixtures()
        with self.assertRaisesRegex(ValueError, 'required'):
            specialist_candidates(c)
        with self.assertRaisesRegex(ValueError, 'missing from'):
            specialist_candidates(c, theme_audit=t.iloc[1:])
        with self.assertRaisesRegex(ValueError, 'does not match'):
            specialist_candidates(c, theme_audit=t.assign(ret20=.90))

    def test_duplicates_dates_schema_and_unknown_mode_fail(self):
        c, t = self.fixtures()
        with self.assertRaisesRegex(ValueError, 'duplicate stock'):
            specialist_candidates(pd.concat([c, c.iloc[:1]]), theme_audit=t)
        with self.assertRaisesRegex(ValueError, 'duplicate dated theme'):
            specialist_candidates(c, theme_audit=pd.concat([t, t.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, 'calendar month'):
            specialist_candidates(c.assign(month=pd.Timestamp('2026-02-28')), theme_audit=t)
        with self.assertRaisesRegex(ValueError, 'future'):
            specialist_candidates(c.assign(date=pd.Timestamp('2026-04-01')), theme_audit=t)
        with self.assertRaisesRegex(ValueError, 'missing columns'):
            specialist_candidates(c.drop(columns='mom1'), theme_audit=t)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            specialist_candidates(c, 'handpicked', t)

    def test_equal_theme_scores_have_stable_code_tie_break(self):
        c, t = self.fixtures()
        t[['ret20', 'ret60']] = [.20, .25]
        c['affinity_ret20'], c['affinity_ret60'] = .20, .25
        expected = specialist_candidates(c, theme_audit=t)
        shuffled = specialist_candidates(c.iloc[::-1], theme_audit=t.iloc[::-1])
        pd.testing.assert_frame_equal(expected, shuffled)
        self.assertTrue(expected[expected.affinity_theme_code.eq('T1')].eligible.all())
        self.assertFalse(expected[expected.affinity_theme_code.eq('T2')].eligible.any())

    def member_fixtures(self):
        c, t = self.fixtures()
        t['theme_score'] = [.80, .60, .40]
        t['name'] = ['first', 'second', 'third']
        t['theme_selected'] = [True, True, False]
        c['member_theme_code'] = ['T2', 'T1', 'T3', None]
        c['member_theme_score'] = [.60, .80, .40, np.nan]
        c['member_theme_name'] = ['second', 'first', 'third', None]
        return c, t

    def test_old_protocol_and_original_scores_remain_unchanged(self):
        self.assertEqual(PROTOCOL['modes'], ['theme_acceleration', 'theme_early'])
        self.assertEqual(protocol_for_mode('theme_acceleration'), PROTOCOL)
        self.assertEqual(protocol_for_mode('theme_early'), PROTOCOL)
        detached = protocol_for_mode('theme_acceleration')
        detached['modes'].append('changed')
        self.assertEqual(PROTOCOL['modes'], ['theme_acceleration', 'theme_early'])
        c, t = self.fixtures()
        for mode in PROTOCOL['modes']:
            result = specialist_candidates(c, mode, t)
            np.testing.assert_allclose(result.leadership_score, [.955, .3625, .50, .7275])
            self.assertEqual(result.eligible.tolist(), [False, True, True, True])
            self.assertNotIn('specialist_member_ret20', result.columns)

    def test_exact_member_uses_member_return_not_affinity_and_preserves_affinity(self):
        c, t = self.member_fixtures()
        # Price affinity is deliberately invalid: exact dated membership is
        # the independent channel, not an extra gate on the old association.
        c['affinity_qualifies'] = False
        c['affinity_theme_score'] = np.nan
        result = specialist_candidates(c, 'member_acceleration', t).set_index('ts_code')
        row = result.loc['600001.SH']
        self.assertTrue(row.eligible)
        self.assertAlmostEqual(row.specialist_relative_strength, np.log1p(.12)-np.log1p(.10))
        self.assertEqual(row.ind_code, 'T2')
        self.assertEqual(row.theme_name, 'second')
        self.assertEqual(row.theme_score, .60)
        self.assertEqual(row.classification_basis, 'exact_dated_top_five_membership')
        self.assertEqual(row.affinity_theme_code, 'T1')
        self.assertFalse(row.affinity_qualifies)
        self.assertTrue(np.isnan(row.affinity_theme_score))
        self.assertFalse(result.loc['688001.SH', 'eligible'])
        self.assertFalse(result.loc['000001.SZ', 'eligible'])

    def test_exact_top_five_member_does_not_need_top_third_theme_gate(self):
        c, t = self.member_fixtures()
        result = specialist_candidates(c, 'member_acceleration', t).set_index('ts_code')
        self.assertTrue(result.loc['600001.SH', 'eligible'])
        self.assertFalse(result.loc['600001.SH', 'structural_eligible'])
        self.assertLess(result.loc['600001.SH', 'mom6'], 0)
        self.assertNotIn('specialist_theme_selected', result.columns)
        self.assertTrue(result.loc['300001.SZ', 'eligible'])

    def test_moderate_extension_is_only_fixed_half_boundary_ablation(self):
        c, t = self.member_fixtures()
        c.loc[0, 'mom1'] = .50
        c.loc[1, ['mom1', 'mom3']] = [.5001, .80]
        ordinary = specialist_candidates(c, 'member_acceleration', t).set_index('ts_code')
        limited = specialist_candidates(c, 'member_not_extended', t).set_index('ts_code')
        self.assertTrue(ordinary.loc['300001.SZ', 'eligible'])
        self.assertFalse(limited.loc['300001.SZ', 'eligible'])
        self.assertTrue(limited.loc['600001.SH', 'eligible'])
        pd.testing.assert_series_equal(ordinary.leadership_score, limited.leadership_score)
        self.assertIn('four-account', MEMBER_PROTOCOL['moderate_extension_ablation'])
        self.assertEqual(protocol_for_mode('member_not_extended'), MEMBER_PROTOCOL)

    def test_member_dates_keys_names_and_scores_must_match_catalogue(self):
        c, t = self.member_fixtures()
        with self.assertRaisesRegex(ValueError, 'name does not match'):
            specialist_candidates(c.assign(member_theme_name='wrong'), 'member_acceleration', t)
        with self.assertRaisesRegex(ValueError, 'score does not match'):
            specialist_candidates(c.assign(member_theme_score=.1), 'member_acceleration', t)
        with self.assertRaisesRegex(ValueError, 'missing from'):
            specialist_candidates(c, 'member_acceleration', t.iloc[1:])
        with self.assertRaisesRegex(ValueError, 'duplicate dated'):
            specialist_candidates(c, 'member_acceleration', pd.concat([t, t.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, 'required'):
            specialist_candidates(c, 'member_acceleration')
        with self.assertRaisesRegex(ValueError, 'future'):
            specialist_candidates(c.assign(date=pd.Timestamp('2026-04-01')), 'member_acceleration', t)

    def test_member_missing_liquidity_heat_and_invalid_quotes_fail_closed(self):
        for field, value in [('mom1', np.nan), ('mom3', np.inf), ('mom6', -1.),
                             ('liquidity_days', 9), ('amount_percentile', 1.1),
                             ('member_theme_name', None), ('member_theme_score', np.nan),
                             ('member_theme_code', None), ('heat_excluded', None)]:
            with self.subTest(field=field):
                c, t = self.member_fixtures()
                if value is None:
                    c[field] = c[field].astype(object)
                c.loc[0, field] = value
                result = specialist_candidates(c, 'member_acceleration', t).set_index('ts_code')
                self.assertFalse(result.loc['600001.SH', 'eligible'])
        c, t = self.member_fixtures()
        c.loc[0, ['mom1', 'mom3']] = [.4, 1.6]
        self.assertFalse(specialist_candidates(c, 'member_acceleration', t).set_index('ts_code').loc['600001.SH', 'eligible'])

    def test_member_future_labels_rows_and_order_do_not_change_current_output(self):
        c, t = self.member_fixtures()
        original_c, original_t = c.copy(deep=True), t.copy(deep=True)
        expected = specialist_candidates(c, 'member_acceleration', t)
        future_c = c.assign(signal_date=pd.Timestamp('2026-04-30'), month=pd.Timestamp('2026-04-30'),
            mom1=100., mom3=100., mom6=100.)
        future_t = t.assign(signal_date=pd.Timestamp('2026-04-30'), ret20=100., ret60=100.)
        full_c = pd.concat([c, future_c], ignore_index=True).assign(future_return=1000.)
        actual = specialist_candidates(full_c.sample(frac=1, random_state=7),
            'member_acceleration', pd.concat([t, future_t]).sample(frac=1, random_state=9))
        actual = actual[actual.signal_date.eq(pd.Timestamp('2026-03-31'))]
        pd.testing.assert_frame_equal(expected, actual.drop(columns='future_return').reset_index(drop=True))
        pd.testing.assert_frame_equal(expected, specialist_candidates(c.iloc[::-1], 'member_acceleration', t.iloc[::-1]))
        pd.testing.assert_frame_equal(c, original_c)
        pd.testing.assert_frame_equal(t, original_t)

    def test_member_ranks_include_both_growth_boards_and_no_affinity_columns_needed(self):
        c, t = self.member_fixtures()
        c['member_theme_code'] = 'T1'
        c['member_theme_name'] = 'first'
        c['member_theme_score'] = .80
        c = c.drop(columns=[name for name in c if name.startswith('affinity_')])
        result = specialist_candidates(c, 'member_acceleration', t).set_index('ts_code')
        self.assertTrue(result.loc['300001.SZ', 'eligible'])
        self.assertTrue(result.loc['688001.SH', 'eligible'])
        self.assertEqual(result.loc['688001.SH', 'specialist_acceleration_percentile'], .75)

    def relative_fixtures(self):
        base, themes = self.member_fixtures()
        rows = pd.concat([base.iloc[:1]] * 10, ignore_index=True)
        rows['ts_code'] = ['600001.SH', '300001.SZ', '688001.SH', '600002.SH', '000001.SZ',
                           '600003.SH', '300002.SZ', '688002.SH', '600004.SH', '000002.SZ']
        rows['mom1'] = [.20, -.10, -.05, .05, .10, .20, .25, .30, .35, .40]
        rows['mom3'] = [.30, -.20, .03, .10, .20, .30, .50, .60, .80, .90]
        rows['mom6'] = -.10
        rows['member_theme_code'] = ['T1'] * 5 + ['T2'] * 5
        rows['member_theme_name'] = ['first'] * 5 + ['second'] * 5
        rows['member_theme_score'] = [.80] * 5 + [.60] * 5
        rows['amount_percentile'] = .50
        return rows, themes

    def test_member_scalar_golden_outputs_and_protocols_remain_unchanged(self):
        c, t = self.member_fixtures()
        for mode in MEMBER_PROTOCOL['modes']:
            result = specialist_candidates(c, mode, t)
            np.testing.assert_allclose(result.leadership_score.iloc[1:4], [13 / 30, 77 / 120, .94])
            self.assertTrue(np.isnan(result.leadership_score.iloc[0]))
            self.assertEqual(result.eligible.tolist(), [False, True, True, False])
            self.assertNotIn('specialist_member_group_count', result.columns)
            self.assertEqual(protocol_for_mode(mode), MEMBER_PROTOCOL)
        self.assertEqual(MEMBER_PROTOCOL['modes'], ['member_acceleration', 'member_not_extended'])

    def test_relative_groups_include_negative_members_before_entry_gate(self):
        c, t = self.relative_fixtures()
        result = specialist_candidates(c, 'member_relative_acceleration', t).set_index('ts_code')
        group = result[result.member_theme_code.eq('T1')]
        self.assertTrue(group.specialist_member_group_count.eq(5).all())
        self.assertAlmostEqual(group.specialist_member_median_log_mom1.iloc[0], np.log1p(.05))
        expected_accel = 1.5 * np.log1p(c.mom1.iloc[:5]) - .5 * np.log1p(c.mom3.iloc[:5])
        self.assertAlmostEqual(group.specialist_member_median_acceleration.iloc[0], expected_accel.median())
        self.assertFalse(result.loc['300001.SZ', 'eligible'])
        self.assertFalse(result.loc['688001.SH', 'eligible'])
        self.assertTrue(group.specialist_member_group_valid.all())
        # Equal own momentum can signify a clear leader in a weak peer group
        # but not in a group where most peers have already accelerated more.
        self.assertAlmostEqual(result.loc['600001.SH', 'mom1'], result.loc['600003.SH', 'mom1'])
        self.assertGreater(result.loc['600001.SH', 'leadership_score'], result.loc['600003.SH', 'leadership_score'])

    def test_relative_group_minimum_count_fails_closed_and_excludes_rank_population(self):
        c, t = self.relative_fixtures()
        short = c[~c.ts_code.eq('000001.SZ')]
        result = specialist_candidates(short, 'member_relative_acceleration', t)
        group = result[result.member_theme_code.eq('T1')]
        self.assertTrue(group.specialist_member_group_count.eq(4).all())
        self.assertFalse(group.specialist_member_group_valid.any())
        self.assertFalse(group.eligible.any())
        self.assertTrue(group.leadership_score.isna().all())
        self.assertTrue(result[result.member_theme_code.eq('T2')].specialist_member_group_valid.all())

    def test_relative_stats_include_hot_and_extended_members_but_entry_still_excludes(self):
        c, t = self.relative_fixtures()
        c.loc[1, ['mom1', 'mom3']] = [.40, 1.6]
        c.loc[2, 'mom1'] = .60
        result = specialist_candidates(c, 'member_relative_moderate', t).set_index('ts_code')
        self.assertEqual(result.loc['600001.SH', 'specialist_member_group_count'], 5)
        self.assertFalse(result.loc['300001.SZ', 'eligible'])
        self.assertFalse(result.loc['688001.SH', 'eligible'])
        self.assertTrue(result.loc['688001.SH', 'specialist_member_group_valid'])
        self.assertAlmostEqual(result.loc['600001.SH', 'specialist_member_median_log_mom1'], np.log1p(.20))

    def test_relative_moderate_reuses_half_threshold_without_censoring_group_stats(self):
        c, t = self.relative_fixtures()
        c.loc[0, 'mom1'] = .50
        c.loc[5, 'mom1'] = .5001
        ordinary = specialist_candidates(c, 'member_relative_acceleration', t).set_index('ts_code')
        limited = specialist_candidates(c, 'member_relative_moderate', t).set_index('ts_code')
        self.assertTrue(limited.loc['600001.SH', 'eligible'])
        self.assertTrue(ordinary.loc['600003.SH', 'eligible'])
        self.assertFalse(limited.loc['600003.SH', 'eligible'])
        pd.testing.assert_series_equal(ordinary.specialist_member_median_log_mom1,
                                       limited.specialist_member_median_log_mom1)
        pd.testing.assert_series_equal(ordinary.leadership_score, limited.leadership_score)
        self.assertEqual(protocol_for_mode('member_relative_moderate'), RELATIVE_MEMBER_PROTOCOL)
        self.assertIn('no further threshold search', RELATIVE_MEMBER_PROTOCOL['moderate_extension_ablation'])

    def test_relative_future_labels_rows_and_order_do_not_change_prior_output(self):
        c, t = self.relative_fixtures()
        expected = specialist_candidates(c, 'member_relative_acceleration', t)
        future_c = c.assign(signal_date=pd.Timestamp('2026-04-30'), month=pd.Timestamp('2026-04-30'),
            mom1=100., mom3=100., mom6=100.)
        future_t = t.assign(signal_date=pd.Timestamp('2026-04-30'), ret20=100., ret60=100.)
        actual = specialist_candidates(pd.concat([c, future_c], ignore_index=True).assign(future_return=1000.),
            'member_relative_acceleration', pd.concat([t, future_t], ignore_index=True))
        actual = actual[actual.signal_date.eq(pd.Timestamp('2026-03-31'))]
        pd.testing.assert_frame_equal(expected, actual.drop(columns='future_return').reset_index(drop=True))
        pd.testing.assert_frame_equal(expected, specialist_candidates(c.iloc[::-1],
            'member_relative_acceleration', t.iloc[::-1]))

    def test_relative_invalid_features_removed_from_group_without_positive_censor(self):
        c, t = self.relative_fixtures()
        c.loc[1, 'liquidity_days'] = 9
        result = specialist_candidates(c, 'member_relative_acceleration', t)
        group = result[result.member_theme_code.eq('T1')]
        self.assertFalse(group.eligible.any())
        self.assertEqual(group[group.ts_code.eq('600001.SH')].specialist_member_group_count.iloc[0], 4)
        self.assertEqual(group[group.ts_code.eq('300001.SZ')].specialist_member_group_count.iloc[0], 0)

    def test_liquid_leader_final_gate_is_inclusive_and_does_not_recompute_scores(self):
        c, t = self.relative_fixtures()
        c['amount_percentile'] = .96
        c.loc[0, 'amount_percentile'] = .95
        c.loc[5, 'amount_percentile'] = .949
        original_c, original_t = c.copy(deep=True), t.copy(deep=True)
        for mode, base_mode in LIQUID_LEADER_BASE_MODES.items():
            with self.subTest(mode=mode):
                base = specialist_candidates(c, base_mode, t).set_index('ts_code')
                actual = specialist_candidates(c, mode, t).set_index('ts_code')
                self.assertTrue(base.loc['600001.SH', 'eligible'])
                self.assertTrue(actual.loc['600001.SH', 'eligible'])
                self.assertTrue(base.loc['600003.SH', 'eligible'])
                self.assertFalse(actual.loc['600003.SH', 'eligible'])
                pd.testing.assert_series_equal(actual.specialist_base_eligible,
                    base.eligible, check_names=False)
                self.assertTrue(actual.eligible.eq(base.eligible & c.set_index('ts_code').amount_percentile.ge(.95)).all())
                # Compare every base audit column, not just the final score:
                # relative-group medians and all-board rank populations stay fixed.
                same_columns = [name for name in base if name not in
                    ('eligible', 'specialist_eligible', 'specialist_mode')]
                pd.testing.assert_frame_equal(actual[same_columns], base[same_columns])
                self.assertTrue(actual.specialist_mode.eq(mode).all())
                self.assertEqual(protocol_for_mode(mode), LIQUID_LEADER_PROTOCOL)
        pd.testing.assert_frame_equal(c, original_c)
        pd.testing.assert_frame_equal(t, original_t)

    def test_liquid_leader_does_not_bypass_base_momentum_heat_or_group_gate(self):
        c, t = self.relative_fixtures()
        c['amount_percentile'] = 1.
        c.loc[0, 'mom1'] = .5001
        c.loc[5, ['mom1', 'mom3']] = [.40, 1.6]
        for mode in LIQUID_LEADER_BASE_MODES:
            with self.subTest(mode=mode):
                result = specialist_candidates(c, mode, t).set_index('ts_code')
                self.assertFalse(result.loc['600001.SH', 'eligible'])
                self.assertFalse(result.loc['600003.SH', 'eligible'])
                self.assertFalse(result.loc['300001.SZ', 'eligible'])
        short = c[~c.ts_code.eq('000001.SZ')]
        result = specialist_candidates(short, 'member_relative_liquid_leader', t)
        self.assertFalse(result[result.member_theme_code.eq('T1')].eligible.any())

    def test_liquid_leader_future_rows_labels_and_order_do_not_change_prior_output(self):
        c, t = self.relative_fixtures()
        c['amount_percentile'] = .96
        later_c = c.assign(signal_date=pd.Timestamp('2026-04-30'),
            month=pd.Timestamp('2026-04-30'), mom1=100., mom3=100., mom6=100.)
        later_t = t.assign(signal_date=pd.Timestamp('2026-04-30'), ret20=100., ret60=100.)
        for mode in LIQUID_LEADER_BASE_MODES:
            with self.subTest(mode=mode):
                expected = specialist_candidates(c, mode, t)
                actual = specialist_candidates(pd.concat([c, later_c], ignore_index=True)
                    .assign(future_return=1000.).sample(frac=1, random_state=17), mode,
                    pd.concat([t, later_t], ignore_index=True).sample(frac=1, random_state=19))
                actual = actual[actual.signal_date.eq(pd.Timestamp('2026-03-31'))]
                pd.testing.assert_frame_equal(expected, actual.drop(columns='future_return').reset_index(drop=True))
                pd.testing.assert_frame_equal(expected, specialist_candidates(c.iloc[::-1], mode, t.iloc[::-1]))

    def test_liquid_leader_protocol_is_detached_and_old_six_modes_unchanged(self):
        self.assertEqual(PROTOCOL['modes'], ['theme_acceleration', 'theme_early'])
        self.assertEqual(MEMBER_PROTOCOL['modes'], ['member_acceleration', 'member_not_extended'])
        self.assertEqual(RELATIVE_MEMBER_PROTOCOL['modes'], ['member_relative_acceleration', 'member_relative_moderate'])
        detached = protocol_for_mode('member_liquid_leader')
        detached['base_modes']['member_liquid_leader'] = 'changed'
        self.assertEqual(LIQUID_LEADER_BASE_MODES['member_liquid_leader'], 'member_not_extended')
        self.assertEqual(LIQUID_LEADER_PROTOCOL['base_modes']['member_liquid_leader'], 'member_not_extended')
        self.assertIn('ten completed', LIQUID_LEADER_PROTOCOL['stage'])
        self.assertIn('NOT an optimal', LIQUID_LEADER_PROTOCOL['liquidity_gate'])
        self.assertIn('rank-first', LIQUID_LEADER_PROTOCOL['execution'])


if __name__ == '__main__':
    unittest.main()
