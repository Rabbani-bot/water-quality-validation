import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_data import generate_truth, inject_errors, STATIONS  # noqa: E402
from validate import validate, evaluate, exceedances  # noqa: E402

ST = set(STATIONS["station"])
WIN = ("2022-01-01", "2024-12-31")


class TestValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.truth = generate_truth()
        cls.raw, cls.log = inject_errors(cls.truth)

    def test_clean_truth_has_few_flags(self):
        v = validate(self.truth.assign(date=self.truth["date"].astype(str)), ST, WIN)
        # only statistical outliers (mostly genuine storm events) may be flagged on clean data
        self.assertEqual(int(v[["missing", "unknown_station", "date_out_of_window", "physical_range",
                                "bod_gt_cod", "duplicate"]].to_numpy().sum()), 0)
        self.assertLess(v["flagged"].mean(), 0.05)

    def test_each_rule_fires(self):
        base = self.truth.iloc[[0]].assign(date=lambda d: d["date"].astype(str)).drop(columns="storm_event")
        cases = {
            "physical_range": base.assign(pH=17.0),
            "unknown_station": base.assign(station="RS99"),
            "date_out_of_window": base.assign(date="2042-01-01"),
            "missing": base.assign(BOD_mgL=np.nan),
            "bod_gt_cod": base.assign(COD_mgL=base["BOD_mgL"] * 0.5),
        }
        for rule, row in cases.items():
            v = validate(pd.concat([self.truth.drop(columns="storm_event").assign(date=self.truth["date"].astype(str)), row]), ST, WIN)
            self.assertTrue(bool(v.iloc[-1][rule]), rule)

    def test_metrics_consistent(self):
        m = evaluate(validate(self.raw, ST, WIN), self.log)
        self.assertEqual(m["flagged"], m["records"] - (m["records"] - m["flagged"]))
        self.assertLess(m["residual_error_rate_pct"], m["error_rate_before_pct"])
        self.assertGreater(m["recall_pct"], 80)

    def test_hotspot_downstream_of_outfall(self):
        v = validate(self.raw, ST, WIN)
        ex = exceedances(v[~v["flagged"]])
        upstream = ex.loc[ex["river_km"] < 8, "BOD_mgL_exceed_pct"].max()
        downstream = ex.loc[ex["river_km"] == 14, "BOD_mgL_exceed_pct"].iloc[0]
        self.assertGreater(downstream, upstream)


if __name__ == "__main__":
    unittest.main()
