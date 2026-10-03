from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import research_dc_eligible_hold as module


class EligibleHoldTests(unittest.TestCase):
    def test_complete_eligibility_required_and_pending_exit_wins(self):
        row = pd.Series(dict(eligible=True, mom1=-.1, future_return=100))
        self.assertTrue(module.continues(row, 'advance', False))
        self.assertFalse(module.continues(row, 'advance', True))
        row['eligible'] = False
        self.assertFalse(module.continues(row, 'advance', False))
        for value in (None, np.nan, 'True', 1):
            self.assertFalse(module.continues(pd.Series(dict(eligible=value)), 'advance', False))
        self.assertFalse(module.continues(None, 'advance', False))

    def test_wrapper_restores_original_engine_even_on_failure(self):
        before = (module.states.OUT, module.states.HOLD_VARIANTS,
                  module.states.may_continue_position, module.themes.prepare)
        candidates = pd.DataFrame(dict(eligible=[True, False], ts_code=['600001.SH', '600002.SH']))
        plans = pd.DataFrame(dict(risk_exposure=[1.]))
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with module.continuation_engine(Path('fixture'), candidates, plans):
                frozen, actual_plans = module.themes.prepare(Path('fixture'), module.ENGINE_VARIANT)
                self.assertEqual(frozen.ts_code.tolist(), ['600001.SH'])
                pd.testing.assert_frame_equal(actual_plans, plans)
                self.assertIn(module.ENGINE_VARIANT, module.states.HOLD_VARIANTS)
                self.assertIs(module.states.may_continue_position, module.continues)
                raise RuntimeError('intentional')
        self.assertEqual(before, (module.states.OUT, module.states.HOLD_VARIANTS,
                                  module.states.may_continue_position, module.themes.prepare))

    def test_appreciated_retention_permitted_but_any_new_purchase_blocked(self):
        self.assertEqual(module.check_capital(50000., 1., 16000., 34000., 0.), (25000., 0.))
        with self.assertRaisesRegex(AssertionError, 'New purchases'):
            module.check_capital(50000., 1., 16000., 34000., 100.)

    def test_new_capital_is_bounded_by_both_cash_and_residual_target(self):
        self.assertEqual(module.check_capital(45000., 1., 27000., 18000., 7000.), (25000., 7000.))
        with self.assertRaises(AssertionError):
            module.check_capital(45000., 1., 27000., 18000., 7100.)
        self.assertEqual(module.check_capital(24000., 1., 2000., 18000., 2000.), (24000., 2000.))
        with self.assertRaises(AssertionError):
            module.check_capital(24000., 1., 2000., 18000., 2100.)


if __name__ == '__main__':
    unittest.main()
