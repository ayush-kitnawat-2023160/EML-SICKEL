"""Safety checks for improved satellite/controls model candidates."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from improve_yield_models import EARLY_SAT, SAT, SPECS, engineered, pipeline_and_grid


class ImprovedModelChecks(unittest.TestCase):
    def test_satellite_feature_whitelist_has_no_yield_or_date(self):
        self.assertEqual(len(SAT), 55)
        self.assertEqual(len(EARLY_SAT), 16)
        for spec in SPECS:
            inputs = spec.numeric + spec.categorical
            self.assertFalse(set(inputs) & {"unique_id", "plot_id", "yield",
                                            "harvesting_date", "sowing_date"})
            self.assertEqual(len(inputs), len(set(inputs)))

    def test_interaction_is_row_local_and_independent_of_yield(self):
        frame = pd.DataFrame({"year": [2019, 2020], "area": [0.5, 1.0],
                              "variety": ["BPT", "CR 1009"],
                              "season": ["sep-feb", "oct-mar"],
                              "yield": [999, -999]})
        a = engineered(frame)
        other = frame.copy()
        other["yield"] = 1e9
        b = engineered(other)
        self.assertEqual(a.year_variety.tolist(), ["2019__BPT", "2020__CR 1009"])
        for field in ("year_variety", "year_season", "area_class", "area_squared"):
            pd.testing.assert_series_equal(a[field], b[field])

    def test_all_plot_grouped_outer_folds_disjoint(self):
        p = ROOT / "outputs/features/full_stage_features.csv"
        if not p.exists():
            self.skipTest("full features not extracted")
        f = pd.read_csv(p, dtype={"plot_id": str})
        train = f.loc[f.split.eq("train")].reset_index(drop=True)
        val = f.loc[f.split.eq("val")]
        self.assertFalse(set(train.plot_id) & set(val.plot_id))
        for seed in (13, 37, 79):
            splits = GroupKFold(5, shuffle=True, random_state=seed).split(
                train, train["yield"], train["plot_id"])
            for tr, te in splits:
                self.assertFalse(set(train.iloc[tr].plot_id) &
                                 set(train.iloc[te].plot_id))

    def test_fold_local_preprocessor_can_fit_without_holdout_categories(self):
        spec = next(s for s in SPECS if s.name == "original_controls_ridge")
        pipe, grid = pipeline_and_grid(spec)
        train = pd.DataFrame({"area": [0.5, 1.0, 0.75, 1.0],
                              "year": [2019, 2020, 2019, 2020],
                              "season": ["sep-feb", "sep-feb", "oct-mar", "oct-mar"],
                              "variety": ["BPT", "BPT", "CR", "CR"]})
        fitted = pipe.fit(train, np.array([1300., 1400., 1650., 1600.]))
        future = pd.DataFrame({"area": [0.75], "year": [2020],
                               "season": ["unseen"], "variety": ["unseen"]})
        self.assertTrue(np.isfinite(fitted.predict(future)).all())

    def test_candidate_oof_uses_only_train_rows(self):
        output = ROOT / "outputs/improvements/candidate_oof_predictions.csv"
        if not output.exists():
            self.skipTest("candidate experiments not run")
        oof = pd.read_csv(output, dtype={"unique_id": str, "plot_id": str})
        full = pd.read_csv(ROOT / "outputs/features/full_stage_features.csv",
                           dtype={"unique_id": str, "plot_id": str})
        train = full.loc[full.split.eq("train")]
        val = full.loc[full.split.eq("val")]
        self.assertEqual(len(oof), len(train))
        self.assertEqual(set(oof.unique_id), set(train.unique_id))
        self.assertFalse(set(oof.unique_id) & set(val.unique_id))
        for col in ("original_controls_ridge", "controls_extratrees",
                    "controls_extratrees_full_sat"):
            self.assertTrue(np.isfinite(oof[col]).all())


if __name__ == "__main__":
    unittest.main()
