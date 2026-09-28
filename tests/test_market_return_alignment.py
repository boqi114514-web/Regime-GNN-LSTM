"""Regression checks for mixed monthly-return units and signal timing."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config
from s6_etf_execution_backtest import run_backtest


class MarketReturnAlignmentTests(unittest.TestCase):
    def test_returns_use_close_and_collapse_duplicate_month(self):
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame(
                [
                    {"ts_code": "801030.SI", "date": "2025-10-31", "close": 100.0, "pct_chg": 0.0},
                    {"ts_code": "801030.SI", "date": "2025-11-28", "close": 110.0, "pct_chg": 10.0},
                    {"ts_code": "801030.SI", "date": "2025-11-30", "close": 110.0, "pct_chg": 0.10},
                    {"ts_code": "801030.SI", "date": "2025-12-31", "close": 99.0, "pct_chg": -0.10},
                ]
            ).to_csv(Path(tmp) / "ts_sw_industry_monthly.csv", index=False)
            with patch.object(config, "LEVEL", "l1"), patch.object(config, "LOCAL_DATA_RAW", tmp):
                result = config.load_industry_monthly()

        self.assertEqual(len(result), 3)
        self.assertEqual(result.date.dt.strftime("%Y-%m-%d").tolist(),
                         ["2025-10-31", "2025-11-30", "2025-12-31"])
        self.assertAlmostEqual(result.ret.iloc[1], 0.10)
        self.assertAlmostEqual(result.ret.iloc[2], -0.10)

    def test_month_end_signal_uses_next_month_return(self):
        ensemble = pd.DataFrame({
            "period": [pd.Period("2026-06")],
            "ts_code": ["801030.SI"],
            "pred_ensemble": [1.0],
        })
        sw_returns = pd.DataFrame(
            {"801030.SI": [0.05, -0.137]},
            index=pd.PeriodIndex(["2026-06", "2026-07"], freq="M"),
        )
        result = run_backtest(ensemble, sw_returns, {}, top_k=1, r2_threshold=0.85)
        self.assertEqual(result["index"].index.tolist(), [pd.Period("2026-07")])
        self.assertAlmostEqual(result["index"].iloc[0], -0.137)


if __name__ == "__main__":
    unittest.main()
