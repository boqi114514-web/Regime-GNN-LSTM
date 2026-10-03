import inspect
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import research_dc_themes as themes
from research_market_states import account_period
from verify_small_account import verify


class DatedThemeTests(unittest.TestCase):
    def write_collection(self, out, features, members, name='dc_theme_2026_retry'):
        scope = 'full' if name == 'dc_theme_retry' else '2026'
        features.to_pickle(out/'theme_features.pkl')
        members.to_pickle(out/'theme_members.pkl')
        audit = themes.rank_themes(features)
        selected = audit[audit.theme_selected].copy()
        selected.to_pickle(out/'theme_selected.pkl')
        def artifact(filename, rows):
            return dict(file=filename, sha256=hashlib.sha256((out/filename).read_bytes()).hexdigest(), rows=rows)
        feature_artifact = artifact('theme_features.pkl', len(features))
        feature = dict(status='complete', scope=scope, aggregate=feature_artifact, source_inputs={})
        membership = dict(status='complete', scope=scope,
            aggregate=artifact('theme_members.pkl', len(members)),
            selected=artifact('theme_selected.pkl', len(selected)),
            feature_sha256=feature_artifact['sha256'], source_inputs={})
        (out/'theme_feature_manifest.json').write_text(json.dumps(feature), encoding='utf-8')
        (out/'theme_membership_manifest.json').write_text(json.dumps(membership), encoding='utf-8')
        return feature, membership, selected

    def inputs(self):
        day = pd.Timestamp('2024-12-31')
        codes = ['000001.SZ', '600001.SH', '300001.SZ', '688001.SH']
        periods = pd.period_range('2024-06', '2024-12', freq='M')
        monthly = pd.DataFrame([dict(date=month.to_timestamp('M'), ts_code=code,
            close=10*(1+.02*(i+1))**j, adj_factor=1.)
            for j, month in enumerate(periods) for i, code in enumerate(codes)])
        daily = pd.DataFrame([dict(date=date, ts_code=code, volume=1., amount=100.*(i+1))
            for date in pd.bdate_range('2024-12-02', '2024-12-31') for i, code in enumerate(codes)])
        features = pd.DataFrame(dict(signal_date=day, catalogue_date=day,
            ts_code=[f'BK{i:04d}' for i in range(7)], name=[f'Theme {i}' for i in range(7)],
            ret20=np.arange(1, 8)*.01, ret60=np.arange(1, 8)*.03,
            amount_ratio=np.arange(1, 8)*.1+1, source_last_date=day, index_obs=61))
        members = pd.DataFrame([dict(trade_date='20241231', ts_code=theme, con_code=code)
            for theme in features.ts_code for code in codes])
        return monthly, daily, features, members

    def test_all_boards_rank_before_mainboard_execution_and_overlap_is_unique(self):
        monthly, daily, f, members = self.inputs()
        c, plans, audit = themes.score_candidates(monthly, daily, f, members)
        self.assertEqual(len(c), 4)
        self.assertTrue(c.eligible.all())
        self.assertFalse(c.duplicated(['month', 'ts_code']).any())
        self.assertTrue(c.ind_code.eq('BK0006').all())
        self.assertTrue(c.phase.eq('advance').all())
        self.assertEqual(set(plans.ts_code), set(f.ts_code.iloc[2:]))
        self.assertTrue(plans.risk_exposure.eq(1.).all())
        indexed = c.set_index('ts_code')
        self.assertAlmostEqual(indexed.loc['000001.SZ', 'stock_score'], .25)
        self.assertAlmostEqual(indexed.loc['688001.SH', 'stock_score'], 1.)
        self.assertAlmostEqual(indexed.loc['000001.SZ', 'leadership_score'], .7*.25+.3)
        self.assertEqual(int(audit.theme_selected.sum()), 5)

    def test_no_inherited_industry_data_can_change_decisions(self):
        monthly, daily, f, members = self.inputs()
        expected = themes.score_candidates(monthly, daily, f, members)
        poison = monthly.assign(ind_code='forbidden', l2_code='unavailable', phase='retreat',
                                size_bucket='unknown', chosen_style='small', sector_rank=0., score=-1e9)
        actual = themes.score_candidates(poison, daily, f, members)
        for left, right in zip(expected, actual):
            pd.testing.assert_frame_equal(left, right)
        module_source = inspect.getsource(themes)
        for forbidden in ('market_features.csv', 'ts_sw_', 'SW_EXCLUDE', 'size_bucket', 'l2_code', 'chosen_style'):
            self.assertNotIn(forbidden, module_source)

    def test_calendar_missing_month_is_not_skipped(self):
        monthly, _, _, _ = self.inputs()
        code = '000001.SZ'
        missing = monthly[~(monthly.ts_code.eq(code) & monthly.date.eq(pd.Timestamp('2024-11-30')))]
        c = themes.stock_momentum(missing)
        row = c[c.ts_code.eq(code) & c.month.eq(pd.Timestamp('2024-12-31'))].iloc[0]
        self.assertTrue(pd.isna(row.mom1))
        self.assertAlmostEqual(row.mom3, 1.02**3-1)
        self.assertAlmostEqual(row.mom6, 1.02**6-1)

    def test_adjustment_factors_remove_distribution_gap(self):
        monthly, _, _, _ = self.inputs()
        before = themes.stock_momentum(monthly)
        after_action = monthly.copy()
        mask = after_action.date.ge('2024-11-01')
        after_action.loc[mask, 'close'] /= 2
        after_action.loc[mask, 'adj_factor'] *= 2
        actual = themes.stock_momentum(after_action)
        for field in ('mom1', 'mom3', 'mom6'):
            np.testing.assert_allclose(before[field], actual[field], equal_nan=True)

    def test_future_theme_or_catalogue_date_is_rejected(self):
        _, _, f, _ = self.inputs()
        for field in ('source_last_date', 'catalogue_date'):
            future = f.copy()
            future.loc[0, field] += pd.Timedelta(days=1)
            with self.subTest(field=field), self.assertRaises(ValueError):
                themes.rank_themes(future)
        with self.assertRaises(ValueError):
            themes.rank_themes(f.drop(columns='index_obs'))
        with self.assertRaises(ValueError):
            themes.rank_themes(f.drop(columns='catalogue_date'))

    def test_current_membership_cannot_backfill_historical_signal(self):
        _, _, f, members = self.inputs()
        audit = themes.rank_themes(f)
        with self.assertRaises(ValueError):
            themes.dated_members(members.drop(columns='trade_date'), audit)
        with self.assertRaises(ValueError):
            themes.dated_members(members.assign(trade_date='20260930'), audit)
        with self.assertRaises(ValueError):
            themes.dated_members(members[members.ts_code.ne('BK0006')], audit)

    def test_history_threshold_and_either_positive_horizon(self):
        _, _, f, _ = self.inputs()
        f.loc[6, 'index_obs'] = 60
        f.loc[5, ['ret20', 'ret60']] = [-.1, .1]
        f.loc[4, ['ret20', 'ret60']] = [.1, -.1]
        f.loc[3, ['ret20', 'ret60']] = [0., 0.]
        audit = themes.rank_themes(f)
        self.assertFalse(audit.loc[6, 'theme_qualifies'])
        self.assertFalse(audit.loc[3, 'theme_qualifies'])
        self.assertTrue(audit.loc[5, 'theme_qualifies'])
        self.assertTrue(audit.loc[4, 'theme_qualifies'])

    def test_recent_performance_baskets_do_not_enter_or_change_theme_rank_baseline(self):
        _, _, f, _ = self.inputs()
        before = themes.rank_themes(f).set_index('ts_code')
        baskets = pd.concat([f.iloc[:1].assign(ts_code='ROLLING01', name='昨日连板_含一字',
                    ret20=400., ret60=400., amount_ratio=400.),
                f.iloc[:1].assign(ts_code='ROLLING02', name='近日打二板以上表现',
                    ret20=500., ret60=500., amount_ratio=500.),
                f.iloc[:1].assign(ts_code='ROLLING03', name='最近多板',
                    ret20=600., ret60=600., amount_ratio=600.),
                f.iloc[:1].assign(ts_code='ROLLING04', name='东方财富热股',
                    ret20=700., ret60=700., amount_ratio=700.)], ignore_index=True)
        after = themes.rank_themes(pd.concat([f, baskets], ignore_index=True)).set_index('ts_code')
        for field in ('theme_score', 'theme_rank', 'theme_selected', 'theme_qualifies'):
            pd.testing.assert_series_equal(before[field], after.loc[before.index, field])
        rolling = after.loc[baskets.ts_code]
        self.assertFalse(rolling.theme_comparable.any())
        self.assertFalse(rolling.theme_selected.any())
        self.assertFalse(rolling.theme_qualifies.any())
        self.assertTrue(rolling.history_complete.all())
        self.assertTrue(rolling.theme_score.isna().all())
        self.assertTrue(rolling.excluded_reason.eq('rolling_recent_performance_index').all())
        self.assertEqual(len(after), len(f)+4)

    def test_comparability_rule_keeps_industry_style_dividend_and_size_themes(self):
        _, _, f, _ = self.inputs()
        f['name'] = ['PCB', '算力', '红利指数', '小盘指数', '大盘价值', '高股息', '电子行业']
        audit = themes.rank_themes(f)
        self.assertTrue(audit.theme_comparable.all())
        self.assertTrue(audit.excluded_reason.eq('').all())
        self.assertEqual(audit.theme_selected.sum(), 5)

    def test_comparability_pattern_does_not_expand_to_other_price_or_valuation_groups(self):
        _, _, f, _ = self.inputs()
        f['name'] = ['最近新高', '创新高', '超跌股', '趋势反转', '市值估值', '最近突破', '东方财富热股行业']
        audit = themes.rank_themes(f)
        self.assertTrue(audit.theme_comparable.all())
        self.assertTrue(audit.excluded_reason.eq('').all())

    def test_theme_tie_break_is_stable_under_row_permutation(self):
        monthly, daily, f, members = self.inputs()
        f[['ret20', 'ret60', 'amount_ratio']] = [.1, .2, 1.2]
        one = themes.score_candidates(monthly, daily, f, members)
        two = themes.score_candidates(monthly, daily, f.sample(frac=1, random_state=1),
                                      members.sample(frac=1, random_state=2))
        self.assertTrue(one[0].ind_code.eq('BK0000').all())
        pd.testing.assert_frame_equal(one[0], two[0])

    def test_positive_liquidity_uses_actual_stock_signal_date(self):
        monthly, daily, f, members = self.inputs()
        early = pd.Timestamp('2024-12-16')
        mask = monthly.ts_code.eq('000001.SZ') & monthly.date.eq(pd.Timestamp('2024-12-31'))
        monthly.loc[mask, 'date'] = early
        daily.loc[daily.ts_code.eq('000001.SZ') & daily.date.gt(early), 'amount'] = 1e15
        c, _, _ = themes.score_candidates(monthly, daily, f, members)
        row = c[c.ts_code.eq('000001.SZ')].iloc[0]
        self.assertEqual(row.signal_amount, 100.)
        self.assertEqual(row.liquidity_last_day, early)
        self.assertEqual(row.liquidity_days, 11)
        daily.loc[daily.ts_code.eq('000001.SZ') & daily.date.le(early), 'volume'] = 0
        c, _, _ = themes.score_candidates(monthly, daily, f, members)
        self.assertFalse(c[c.ts_code.eq('000001.SZ')].eligible.iloc[0])

    def test_future_stock_observation_and_duplicate_inputs_are_rejected(self):
        monthly, daily, f, members = self.inputs()
        f['signal_date'] -= pd.Timedelta(days=1)
        f['catalogue_date'] = f.signal_date
        f['source_last_date'] = f.signal_date
        members['trade_date'] = '20241230'
        with self.assertRaises(ValueError):
            themes.score_candidates(monthly, daily, f, members)
        with self.assertRaises(ValueError):
            themes.stock_momentum(pd.concat([monthly, monthly.iloc[:1]]))

    def test_cash_only_plan_when_no_theme_qualifies(self):
        monthly, daily, f, members = self.inputs()
        f[['ret20', 'ret60']] = -.01
        c, plans, _ = themes.score_candidates(monthly, daily, f, members)
        self.assertFalse(c.eligible.any())
        self.assertEqual(plans.ts_code.tolist(), ['__CASH__'])
        self.assertEqual(plans.risk_exposure.tolist(), [1.])

    def test_stock_gate_rejects_negative_missing_or_joint_extreme_trend(self):
        monthly, daily, f, members = self.inputs()
        last = monthly.date.eq(pd.Timestamp('2024-12-31'))
        monthly.loc[last & monthly.ts_code.eq('000001.SZ'), 'close'] = 100.
        monthly.loc[last & monthly.ts_code.eq('600001.SH'), 'close'] = 8.
        monthly = monthly[~(monthly.ts_code.eq('300001.SZ') & monthly.date.eq(pd.Timestamp('2024-06-30')))]
        c, _, _ = themes.score_candidates(monthly, daily, f, members)
        c = c.set_index('ts_code')
        self.assertTrue(c.loc['000001.SZ', 'heat_excluded'])
        self.assertFalse(c.loc['000001.SZ', 'eligible'])
        self.assertFalse(c.loc['600001.SH', 'eligible'])
        self.assertTrue(pd.isna(c.loc['300001.SZ', 'mom6']))
        self.assertFalse(c.loc['300001.SZ', 'eligible'])
        self.assertTrue(c.loc['688001.SH', 'eligible'])

    def test_future_inputs_cannot_change_earlier_month_scores(self):
        monthly, daily, f, members = self.inputs()
        before = themes.score_candidates(monthly, daily, f, members)[0]
        future_monthly = monthly[monthly.date.eq(pd.Timestamp('2024-12-31'))].assign(
            date=pd.Timestamp('2025-01-31'), close=1000.)
        future_daily = daily.copy().assign(date=lambda x:x.date+pd.DateOffset(months=1), amount=1e15)
        future_features = f.assign(signal_date=pd.Timestamp('2025-01-31'),
            catalogue_date=pd.Timestamp('2025-01-31'), source_last_date=pd.Timestamp('2025-01-31'),
            ret20=100., ret60=100., amount_ratio=100.)
        future_members = members.assign(trade_date='20250131')
        after = themes.score_candidates(pd.concat([monthly, future_monthly]), pd.concat([daily, future_daily]),
            pd.concat([f, future_features]), pd.concat([members, future_members]))[0]
        earlier = after[after.month.eq(pd.Timestamp('2024-12-31'))].reset_index(drop=True)
        pd.testing.assert_frame_equal(before, earlier)

    def test_prepare_refuses_missing_signal_months_before_writing_candidates(self):
        _, _, f, members = self.inputs()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            self.write_collection(out, f, members, 'dc_theme_retry')
            with self.assertRaisesRegex(ValueError, 'must cover every 2024-12'):
                themes.prepare(out)
            self.assertFalse((Path(directory)/'candidates.pkl').exists())

    def test_account_period_and_verifier_defaults_preserve_old_variants(self):
        old = (pd.Timestamp('2023-01-01'), pd.Timestamp('2026-09-24'), pd.Timestamp('2022-12-30'), 45)
        for variant in ('baseline_replay', 'state_onset_retry', 'liquid_leader_retry'):
            self.assertEqual(account_period(variant), old)
        self.assertEqual(account_period('dc_theme_retry'),
                         (themes.START, themes.END, themes.PREVIOUS_MARK, 21))
        for variant in ('dc_theme_2026_retry', 'dc_liquid_2026_retry', 'dc_affinity_2026_retry', 'dc_structure_2026_retry'):
            self.assertEqual(account_period(variant),
                             (pd.Timestamp('2026-01-01'), themes.END, pd.Timestamp('2025-12-31'), 9))
        parameters = inspect.signature(verify).parameters
        self.assertEqual(parameters['expected_months'].default, 45)
        self.assertEqual(parameters['end_date'].default, '2026-09-24')

    def scope_inputs(self):
        _, _, original_features, original_members = self.inputs()
        codes = original_members.con_code.unique()
        periods = pd.period_range('2025-06', '2026-08', freq='M')
        monthly = pd.DataFrame([dict(date=month.to_timestamp('M'), ts_code=code,
            close=10*(1+.02*(i+1))**j, adj_factor=1.)
            for j, month in enumerate(periods) for i, code in enumerate(codes)])
        signals = themes.signal_months('dc_theme_2026_retry')
        features = pd.concat([original_features.assign(signal_date=month.to_timestamp('M'),
            catalogue_date=month.to_timestamp('M'), source_last_date=month.to_timestamp('M'))
            for month in signals], ignore_index=True)
        members = pd.concat([original_members.assign(trade_date=f'{month.to_timestamp("M"):%Y%m%d}')
                             for month in signals], ignore_index=True)
        daily = pd.DataFrame([dict(date=date, ts_code=code, volume=1., amount=100.*(i+1))
            for month in signals for date in pd.bdate_range(month.to_timestamp(), periods=10)
            for i, code in enumerate(codes)])
        return monthly, daily, features, members

    def test_period_variants_share_every_selection_and_execution_setting(self):
        old = themes.protocol_for()
        new = themes.protocol_for('dc_theme_2026_retry')
        period_keys = {'variant', 'variants', 'period', 'previous_mark', 'expected_months', 'signal_months', 'scope'}
        self.assertEqual({k:v for k,v in old.items() if k not in period_keys},
                         {k:v for k,v in new.items() if k not in period_keys})
        self.assertEqual(themes.PROTOCOL, old)
        self.assertEqual(len(themes.signal_months()), 21)
        pd.testing.assert_index_equal(themes.signal_months('dc_theme_2026_retry'),
                                      pd.period_range('2025-12', '2026-08', freq='M'))
        self.assertEqual(new['expected_months'], 9)
        self.assertEqual(new['variant'], 'dc_theme_2026_retry')

    def test_new_period_prepares_exactly_nine_signals_and_prevents_directory_mixing(self):
        monthly, daily, features, members = self.scope_inputs()
        frames = {'theme_features.pkl': features, 'stock_month_end_verified.pkl': monthly,
                  'daily.pkl': daily, 'theme_members.pkl': members}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            _, _, selected = self.write_collection(out, features, members)
            frames['theme_selected.pkl'] = selected
            with patch.object(themes.pd, 'read_pickle', side_effect=lambda path:frames[Path(path).name]):
                c, plans = themes.prepare(out, 'dc_theme_2026_retry')
            self.assertEqual(c.month.nunique(), 9)
            self.assertEqual(plans.date.nunique(), 9)
            self.assertTrue((out/'candidate_audit_dc_theme_2026_retry.pkl').exists())
            protocol = json.loads((out/'dc_theme_protocol.json').read_text(encoding='utf-8'))
            self.assertEqual(protocol, themes.protocol_for('dc_theme_2026_retry'))
            with self.assertRaises(ValueError):
                themes.prepare(out)  # The default 21-month variant cannot overwrite this directory.

    def test_new_period_rejects_any_missing_signal_month(self):
        _, _, features, members = self.scope_inputs()
        features = features[features.signal_date.ne(pd.Timestamp('2026-01-31'))]
        members = members[members.trade_date.ne('20260131')]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            self.write_collection(out, features, members)
            with self.assertRaisesRegex(ValueError, 'must cover every 2025-12'):
                themes.prepare(out, 'dc_theme_2026_retry')
            self.assertFalse((Path(directory)/'candidates.pkl').exists())

    def test_missing_running_or_wrong_scope_manifests_cannot_use_previous_artifacts(self):
        _, _, features, members = self.scope_inputs()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with self.assertRaisesRegex(ValueError, 'Missing complete collection manifest'):
                themes.collection_inputs(out, 'dc_theme_2026_retry')
            feature, membership, _ = self.write_collection(out, features, members)
            for filename, original in [('theme_feature_manifest.json', feature),
                                       ('theme_membership_manifest.json', membership)]:
                for field, value in [('status', 'running'), ('status', 'incomplete'), ('scope', 'full')]:
                    with self.subTest(filename=filename, field=field, value=value):
                        changed = {**original, field: value}
                        (out/filename).write_text(json.dumps(changed), encoding='utf-8')
                        with self.assertRaisesRegex(ValueError, 'complete and match scope'):
                            themes.collection_inputs(out, 'dc_theme_2026_retry')
                (out/filename).write_text(json.dumps(original), encoding='utf-8')

    def test_aggregate_hash_row_count_and_cross_feature_binding_are_required(self):
        _, _, features, members = self.scope_inputs()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            feature, membership, _ = self.write_collection(out, features, members)
            for filename, original, section in [('theme_feature_manifest.json', feature, 'aggregate'),
                ('theme_membership_manifest.json', membership, 'aggregate'),
                ('theme_membership_manifest.json', membership, 'selected')]:
                for field, value in [('sha256', '0'*64), ('rows', original[section]['rows']+1), ('file', 'another.pkl')]:
                    with self.subTest(filename=filename, section=section, field=field):
                        changed = {**original, section: {**original[section], field: value}}
                        (out/filename).write_text(json.dumps(changed), encoding='utf-8')
                        with self.assertRaises(ValueError):
                            themes.collection_inputs(out, 'dc_theme_2026_retry')
                (out/filename).write_text(json.dumps(original), encoding='utf-8')
            membership['feature_sha256'] = '0'*64
            (out/'theme_membership_manifest.json').write_text(json.dumps(membership), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'not bound to the current theme features'):
                themes.collection_inputs(out, 'dc_theme_2026_retry')

    def test_source_change_and_rehashed_wrong_selection_are_rejected(self):
        _, _, features, members = self.scope_inputs()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            feature, membership, selected = self.write_collection(out, features, members)
            source = out/'collector_source.py'
            source.write_text('version = 1', encoding='utf-8')
            feature['source_inputs'] = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
            (out/'theme_feature_manifest.json').write_text(json.dumps(feature), encoding='utf-8')
            themes.collection_inputs(out, 'dc_theme_2026_retry')
            source.write_text('version = 2', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Source-input fingerprint changed'):
                themes.collection_inputs(out, 'dc_theme_2026_retry')
            feature['source_inputs'] = {}
            (out/'theme_feature_manifest.json').write_text(json.dumps(feature), encoding='utf-8')
            selected = selected.iloc[:-1]
            selected.to_pickle(out/'theme_selected.pkl')
            membership['selected']['sha256'] = hashlib.sha256((out/'theme_selected.pkl').read_bytes()).hexdigest()
            membership['selected']['rows'] = len(selected)
            (out/'theme_membership_manifest.json').write_text(json.dumps(membership), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'differs from the fixed-rule'):
                themes.collection_inputs(out, 'dc_theme_2026_retry')

    def test_summary_derives_verification_period_from_saved_protocol(self):
        import verify_small_account as verifier
        original_out = verifier.OUT
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out/'dc_theme_protocol.json').write_text(
                json.dumps(themes.protocol_for('dc_theme_2026_retry')), encoding='utf-8')
            with patch.object(verifier, 'verify', side_effect=RuntimeError('verification probe')) as check:
                with self.assertRaisesRegex(RuntimeError, 'verification probe'):
                    themes.summarize(out)
                check.assert_called_once_with('dc_theme_2026_retry', expected_months=9,
                                              end_date=pd.Timestamp('2026-09-24'))
            self.assertEqual(verifier.OUT, original_out)

    def specialist_pool(self):
        day = pd.Timestamp('2026-03-31')
        codes = [f'{i:06d}.SZ' for i in range(18)]+['300001.SZ', '688001.SH']
        c = pd.DataFrame(dict(month=day, signal_date=day, date=day, ts_code=codes,
            signal_amount=np.arange(1, 21, dtype=float), liquidity_days=20,
            mom1=.1, mom3=.2, mom6=.3, stock_qualifies=True, ind_code='BK0006',
            theme_name='Theme 6', theme_score=1., stock_score=np.linspace(.1, 1., 20),
            leadership_score=.1, eligible=True, phase='advance'))
        c.loc[17:, 'mom1'] = [.3, .1, .2]
        c.loc[17:, 'mom3'] = [.2, .3, .1]
        _, _, f, _ = self.inputs()
        audit = themes.rank_themes(f.assign(signal_date=day, catalogue_date=day, source_last_date=day))
        return c, audit

    def association(self, c, audit, index=17, theme='BK0000', corr=.3, nobs=40):
        row = audit[audit.ts_code.eq(theme)].iloc[0]
        return pd.DataFrame([dict(signal_date=c.signal_date.iloc[index], ts_code=c.ts_code.iloc[index],
            affinity_theme_code=theme, affinity_theme_name=row['name'], affinity_theme_score=row.theme_score,
            affinity_corr=corr, nobs=nobs)])

    def test_liquid_specialist_ranks_all_boards_then_keeps_the_exact_membership_gate(self):
        c, _ = self.specialist_pool()
        original = c.copy(deep=True)
        scored = themes.liquid_candidates(c)
        self.assertEqual(scored.index[scored.eligible].tolist(), [17, 18, 19])
        expected = np.array([(.45+3*.35+2*.20)/3, (2*.45+.35+3*.20)/3,
                             (3*.45+2*.35+.20)/3])
        np.testing.assert_allclose(scored.loc[17:, 'liquid_score'], expected)
        np.testing.assert_allclose(scored.loc[17:, 'leadership_score'], .7*expected+.3)
        self.assertTrue(scored.loc[18:19, 'eligible'].all())  # Growth/STAR affect signal ranks.
        missing_member = c.copy()
        missing_member.loc[19, ['ind_code', 'theme_score']] = [np.nan, np.nan]
        self.assertFalse(themes.liquid_candidates(missing_member).loc[19, 'eligible'])
        pd.testing.assert_frame_equal(c, original)

    def test_liquid_specialist_requires_observation_and_positive_stock_gates_without_breadth(self):
        c, _ = self.specialist_pool()
        c.loc[19, 'stock_qualifies'] = False
        c.loc[18, 'liquidity_days'] = 9
        scored = themes.liquid_candidates(c)
        self.assertFalse(scored.loc[19, 'eligible'])
        self.assertFalse(scored.loc[18, 'eligible'])
        self.assertTrue(scored.loc[17, 'eligible'])
        poisoned = themes.liquid_candidates(c.assign(breadth=0., size_bucket='unknown', chosen_style='small'))
        for field in ('liquidity_percentile', 'liquid_score', 'eligible', 'leadership_score'):
            pd.testing.assert_series_equal(scored[field], poisoned[field])

    def test_liquid_specialist_is_prefix_invariant(self):
        c, _ = self.specialist_pool()
        before = themes.liquid_candidates(c)
        future = c.assign(month=pd.Timestamp('2026-04-30'), signal_date=pd.Timestamp('2026-04-30'),
                          signal_amount=lambda x:x.signal_amount*1e9, mom1=999., mom3=999.)
        after = themes.liquid_candidates(pd.concat([c, future]))
        pd.testing.assert_frame_equal(before, after[after.month.eq(before.month.iloc[0])].reset_index(drop=True))

    def test_affinity_can_select_nonmember_and_theme_outside_top_five(self):
        c, audit = self.specialist_pool()
        c.loc[17, ['ind_code', 'theme_name', 'theme_score']] = [np.nan, None, np.nan]
        c.loc[17, 'eligible'] = False
        f = self.association(c, audit)
        self.assertFalse(audit[audit.ts_code.eq('BK0000')].theme_selected.iloc[0])
        scored, plans = themes.affinity_candidates(c, f, audit)
        self.assertTrue(scored.loc[17, 'eligible'])
        self.assertTrue(pd.isna(scored.loc[17, 'member_theme_code']))
        self.assertEqual(scored.loc[17, 'ind_code'], 'BK0000')
        self.assertEqual(scored.loc[17, 'classification_basis'], 'price_affinity_not_membership')
        self.assertIn('BK0000', set(plans.ts_code))
        self.assertTrue(plans.risk_exposure.eq(1.).all())
        self.assertAlmostEqual(scored.loc[17, 'leadership_score'],
                               .7*scored.loc[17, 'liquid_score']+.3*f.affinity_theme_score.iloc[0])

    def test_affinity_thresholds_do_not_relax_liquidity_or_positive_stock_rules(self):
        c, audit = self.specialist_pool()
        for corr, nobs, eligible in [(.3, 40, True), (.2999, 40, False), (.3, 39, False)]:
            with self.subTest(corr=corr, nobs=nobs):
                scored, _ = themes.affinity_candidates(c, self.association(c, audit, corr=corr, nobs=nobs), audit)
                self.assertEqual(bool(scored.loc[17, 'eligible']), eligible)
        scored, _ = themes.affinity_candidates(c, self.association(c, audit, index=16), audit)
        self.assertFalse(scored.loc[16, 'eligible'])
        c.loc[17, 'stock_qualifies'] = False
        scored, _ = themes.affinity_candidates(c, self.association(c, audit), audit)
        self.assertFalse(scored.loc[17, 'eligible'])

    def test_affinity_rejects_future_duplicate_or_incorrect_theme_evidence(self):
        c, audit = self.specialist_pool()
        f = self.association(c, audit)
        for changed in [pd.concat([f, f]), f.assign(signal_date=pd.Timestamp('2026-04-01')),
                        f.assign(affinity_theme_score=.99), f.assign(affinity_theme_name='wrong'),
                        f.assign(affinity_corr=1.01), f.assign(nobs=40.5)]:
            with self.assertRaises(ValueError):
                themes.affinity_candidates(c, changed, audit)
        scored, _ = themes.affinity_candidates(c, f.iloc[:0], audit)
        self.assertFalse(scored.eligible.any())

    def test_affinity_price_aggregate_requires_feature_manifest_fingerprint(self):
        _, _, features, members = self.scope_inputs()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            out = base/'output'
            out.mkdir()
            feature, _, _ = self.write_collection(out, features, members, 'dc_affinity_2026_retry')
            price = base/'dc_theme_research'/'daily_market_2026.pkl'
            price.parent.mkdir()
            pd.DataFrame({'price': [1., 2.]}).to_pickle(price)
            with patch.object(themes, 'ROOT', base/'execution_v1'):
                with self.assertRaisesRegex(ValueError, 'not bound to the feature manifest'):
                    themes.affinity_market_path(out)
                feature['source_inputs'][str(price)] = hashlib.sha256(price.read_bytes()).hexdigest()
                (out/'theme_feature_manifest.json').write_text(json.dumps(feature), encoding='utf-8')
                self.assertEqual(themes.affinity_market_path(out), price.resolve())
                pd.DataFrame({'price': [3., 4.]}).to_pickle(price)
                with self.assertRaisesRegex(ValueError, 'Source-input fingerprint changed'):
                    themes.affinity_market_path(out)

    def test_specialist_protocols_are_explicit_exploration_without_changing_execution(self):
        baseline = themes.protocol_for('dc_theme_2026_retry')
        for name in themes.SPECIALIST_VARIANTS:
            protocol = themes.protocol_for(name)
            self.assertIn('Exploratory feedback after', protocol['stage'])
            self.assertIn('no named stocks', protocol['selection_scope'])
            self.assertEqual(protocol['expected_months'], 9)
            for field in ('exposure', 'initial_cash', 'capital_cap', 'fees', 'execution',
                          'external_topups', 'daily_supplementary_entries', 'stock_gate'):
                self.assertEqual(protocol[field], baseline[field])

    def structure_fixture(self, ret20=.15, ret60=.15, stock_index=16, corr=.3, nobs=40):
        c, audit = self.specialist_pool()
        audit.loc[audit.ts_code.eq('BK0000'), ['ret20', 'ret60']] = [ret20, ret60]
        audit = themes.rank_themes(audit)
        affinity = self.association(c, audit, index=stock_index, corr=corr, nobs=nobs)
        return c, affinity, audit

    def test_structural_advance_allows_nonliquid_stock_and_uses_original_stock_score(self):
        c, affinity, audit = self.structure_fixture()
        prior, _ = themes.affinity_candidates(c, affinity, audit)
        routed, _ = themes.structure_candidates(c, affinity, audit)
        self.assertFalse(prior.loc[16, 'eligible'])
        self.assertFalse(routed.loc[16, 'liquid_qualifies'])
        self.assertTrue(routed.loc[16, 'eligible'])
        self.assertEqual(routed.loc[16, 'theme_phase'], 'advance')
        self.assertEqual(routed.loc[16, 'ranking_basis'], 'all_board_price_volume')
        self.assertAlmostEqual(routed.loc[16, 'leadership_score'],
                               .70*c.loc[16, 'stock_score']+.30*affinity.affinity_theme_score.iloc[0])
        self.assertTrue(routed.phase.eq('advance').all())

    def test_structural_phase_boundaries_are_dated_and_inclusive(self):
        cases = [(.15, .15, 'advance'), (.1499, .15, 'range'), (.15, .1499, 'range'),
                 (0., .15, 'pullback'), (-.01, .15, 'pullback'), (.01, -.01, 'range')]
        for ret20, ret60, phase in cases:
            with self.subTest(ret20=ret20, ret60=ret60):
                c, affinity, audit = self.structure_fixture(ret20=ret20, ret60=ret60)
                routed, _ = themes.structure_candidates(c, affinity, audit)
                self.assertEqual(routed.loc[16, 'theme_phase'], phase)
                self.assertEqual(bool(routed.loc[16, 'eligible']), phase == 'advance')

    def test_structural_range_or_pullback_retains_liquid_ranking_and_all_entry_stops(self):
        for ret20, phase in [(0., 'pullback'), (.05, 'range')]:
            c, affinity, audit = self.structure_fixture(ret20=ret20, ret60=.15, stock_index=17)
            routed, _ = themes.structure_candidates(c, affinity, audit)
            self.assertTrue(routed.loc[17, 'eligible'])
            self.assertTrue(routed.loc[17, 'liquid_qualifies'])
            self.assertEqual(routed.loc[17, 'theme_phase'], phase)
            self.assertEqual(routed.loc[17, 'ranking_basis'], 'liquid_top_decile')
            self.assertEqual(routed.loc[17, 'phase'], 'advance')
            self.assertAlmostEqual(routed.loc[17, 'leadership_score'],
                .70*routed.loc[17, 'liquid_score']+.30*affinity.affinity_theme_score.iloc[0])

    def test_structural_advance_does_not_relax_affinity_or_positive_stock_qualification(self):
        for corr, nobs in [(.2999, 40), (.3, 39)]:
            c, affinity, audit = self.structure_fixture(corr=corr, nobs=nobs)
            routed, _ = themes.structure_candidates(c, affinity, audit)
            self.assertFalse(routed.loc[16, 'eligible'])
        c, affinity, audit = self.structure_fixture()
        c.loc[16, 'stock_qualifies'] = False
        routed, _ = themes.structure_candidates(c, affinity, audit)
        self.assertFalse(routed.loc[16, 'eligible'])

    def test_structural_routing_is_prefix_invariant_and_does_not_require_true_membership(self):
        c, affinity, audit = self.structure_fixture()
        c.loc[16, ['ind_code', 'theme_name', 'theme_score']] = [np.nan, None, np.nan]
        c.loc[16, 'eligible'] = False
        expected, _ = themes.structure_candidates(c, affinity, audit)
        self.assertTrue(expected.loc[16, 'eligible'])
        self.assertTrue(pd.isna(expected.loc[16, 'member_theme_code']))
        future_day = pd.Timestamp('2026-04-30')
        future_c = c.assign(month=future_day, signal_date=future_day, signal_amount=lambda x:x.signal_amount*1e9)
        future_a = affinity.assign(signal_date=future_day)
        future_t = audit.assign(month=future_day, signal_date=future_day, ret20=9., ret60=9.)
        actual, _ = themes.structure_candidates(pd.concat([c, future_c], ignore_index=True),
            pd.concat([affinity, future_a], ignore_index=True), pd.concat([audit, future_t], ignore_index=True))
        earlier = actual[actual.month.eq(expected.month.iloc[0])].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, earlier)

    def test_structural_protocol_discloses_feedback_and_preserves_account_execution(self):
        original = themes.protocol_for('dc_theme_2026_retry')
        protocol = themes.protocol_for('dc_structure_2026_retry')
        self.assertIn('liquid/affinity experiments', protocol['stage'])
        self.assertIn('ret20>=.15 AND ret60>=.15', protocol['theme_phase'])
        self.assertEqual(protocol['expected_months'], 9)
        for field in ('exposure', 'initial_cash', 'capital_cap', 'fees', 'execution',
                      'external_topups', 'daily_supplementary_entries', 'stock_gate'):
            self.assertEqual(protocol[field], original[field])


if __name__ == '__main__':
    unittest.main()
