import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from research_dc_entry_policy import admit_reversals, weekly_candidates
from research_dc_themes import protocol_for


class EntryPolicyTests(unittest.TestCase):
    def inputs(self):
        signal = pd.Timestamp('2026-03-31')
        c = pd.DataFrame(dict(signal_date=[signal]*3, ts_code=['600001.SH','300001.SZ','600002.SH'],
            month=[signal]*3, eligible=[False,False,True], leadership_score=[.5,.5,.90],
            affinity_qualifies=[True,True,True], liquidity_days=[20]*3, heat_excluded=[False]*3,
            affinity_theme_score=[.8]*3, mom1=[-.1,-.1,.1], mom3=[-.1,-.1,.2]))
        f = pd.DataFrame(dict(signal_day=[signal]*3, ts_code=c.ts_code,
            pressure5=[.04]*3,pressure20=[.08]*3,location5=[.7]*3,
            amount_ratio5_vs_prior20=[1.3]*3,mean_amount20=[100]*3,
            entry_score=[.9,.9,.7],reversal_eligible=[True]*3))
        return c,f

    def test_negative_momentum_and_growth_board_admitted_before_execution(self):
        c,f = self.inputs()
        result = admit_reversals(c,f)
        self.assertTrue(result.eligible.all())
        self.assertTrue(result.mom1.iloc[:2].lt(0).all())
        self.assertAlmostEqual(result.leadership_score.iloc[0], .87)
        self.assertAlmostEqual(result.leadership_score.iloc[2], .90)

    def test_missing_local_quote_heat_and_theme_cannot_be_bypassed(self):
        c,f = self.inputs()
        c.loc[0,'heat_excluded']=True
        c.loc[1,'affinity_qualifies']=False
        result = admit_reversals(c,f)
        self.assertEqual(result.eligible.tolist(),[False,False,True])
        result = admit_reversals(self.inputs()[0], f.iloc[2:])
        self.assertEqual(result.eligible.tolist(),[False,False,True])

    def test_weekly_keeps_previous_month_evidence_and_only_combination_adds_reversal(self):
        c,f = self.inputs()
        s = pd.Timestamp('2026-04-10'); d=pd.Timestamp('2026-04-13')
        f['signal_day']=s
        plain=weekly_candidates(c,f,{d:s},'dc_weekly_2026_retry')
        combined=weekly_candidates(c,f,{d:s},'dc_weekly_reversal_2026_retry')
        self.assertEqual(plain.eligible.tolist(),[False,False,True])
        self.assertTrue(combined.eligible.all())
        self.assertTrue(combined.theme_signal_date.eq('2026-03-31').all())
        self.assertTrue(combined.signal_day.eq(s).all())

    def test_future_monthly_evidence_fails(self):
        c,f=self.inputs()
        c['signal_date']=pd.Timestamp('2026-04-20')
        with self.assertRaisesRegex(ValueError,'future'):
            weekly_candidates(c,f,{pd.Timestamp('2026-04-13'):pd.Timestamp('2026-04-10')},'dc_weekly_2026_retry')

    def test_original_protocol_unchanged_and_three_fixed_ablations(self):
        self.assertNotIn('reversal_channel',protocol_for('dc_structure_2026_retry'))
        for name,reverse,weekly in [('dc_reversal_2026_retry',True,False),
            ('dc_weekly_2026_retry',False,True),('dc_weekly_reversal_2026_retry',True,True)]:
            p=protocol_for(name)
            self.assertEqual(p['reversal_channel'],reverse)
            self.assertEqual(p['weekly_supplementary_entries'],weekly)
            self.assertEqual(p['expected_months'],9)


if __name__ == '__main__':
    unittest.main()
