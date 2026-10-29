"""Unit and leakage checks for partially penalized EML estimators."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eml_penalty_experiments import (BASE, EXPANDED, EXPERIMENTS, TRAJECTORY,
                                    TRAJECTORY_EXPERIMENTS, PreparedFit,
                                    raw_trajectory_matrix,
                                    proximal_elastic_net, scad_derivative_ratio,
                                    scad_objective)


class RegularizedEMLChecks(unittest.TestCase):
    def test_shrinkage_solver_orthogonal_solution(self):
        gram = np.eye(3)
        h = np.array([2., .5, -2.])
        obtained = proximal_elastic_net(gram, h, lam=1., weights=np.ones(3),
                                        ratio=1.)
        np.testing.assert_allclose(obtained, [1., 0., -1.], atol=1e-6)

    def test_scad_derivative_correct(self):
        lam, a = 1., 3.7
        points = np.array([0., 1., 2., a, 5., -2.])
        expected = np.array([1., 1., (a-2)/(a-1), 0., 0.,
                             (a-2)/(a-1)])
        np.testing.assert_allclose(scad_derivative_ratio(points, lam), expected)

    def test_scad_objective_scalar_branches(self):
        # 1-D: no quadratic loss means objective equals the SCAD penalty.
        for x, expected in ((.5, .5), (1., 1.), (3.7, 2.35),
                            (5., 2.35)):
            self.assertAlmostEqual(scad_objective(np.zeros((1, 1)),
                                                  np.zeros(1), np.array([x]), 1.),
                                   expected, places=8)

    def test_original_features_exclude_leakage_columns(self):
        self.assertEqual(len(BASE), 55)
        self.assertEqual(len(EXPANDED), 65)
        for name in EXPANDED:
            self.assertNotIn("yield", name)
            self.assertNotIn("harvesting_date", name)
            self.assertNotIn("plot_id", name)
        self.assertEqual(len(TRAJECTORY), 42)
        self.assertEqual(len(set(TRAJECTORY)), len(TRAJECTORY))
        self.assertEqual(len(TRAJECTORY_EXPERIMENTS), 6)

    def test_physical_trajectory_differences_and_missingness(self):
        raw = np.full((3, len(BASE)), np.nan)
        # Raw NDVI differences W2-W1=0.3, W5-W1=0.7; range=0.7.
        for j, val in enumerate([.1, .4, .5, .6, .8]):
            raw[0, BASE.index(f"{('W1_0_20','W2_20_40','W3_40_60','W4_60_80','W5_80_100')[j]}__s2_ndvi")] = val
        raw[1, BASE.index("W1_0_20__s2_ndvi")] = .2
        raw[1, BASE.index("W2_20_40__s2_ndvi")] = np.nan
        raw[1, BASE.index("W5_80_100__s2_ndvi")] = .6
        values = raw_trajectory_matrix(raw)
        self.assertEqual(values.shape, (3, 42))
        w2w1 = TRAJECTORY.index("trajectory__s2_ndvi__W2_20_40_minus_W1_0_20")
        w5w1 = TRAJECTORY.index("trajectory__s2_ndvi__W5_80_100_minus_W1_0_20")
        ndvi_range = TRAJECTORY.index("trajectory__s2_ndvi__season_range")
        self.assertAlmostEqual(values[0, w2w1], .3)
        self.assertAlmostEqual(values[0, w5w1], .7)
        self.assertAlmostEqual(values[0, ndvi_range], .7)
        self.assertTrue(np.isnan(values[1, w2w1]))
        self.assertAlmostEqual(values[1, w5w1], .4)
        self.assertTrue(np.isnan(values[1, ndvi_range]))
        self.assertTrue(np.isnan(values[2, ndvi_range]))

    def test_trajectory_preprocessing_fold_local(self):
        source = ROOT / "outputs/features/full_stage_features.csv"
        if not source.is_file():
            self.skipTest("extracted full features missing")
        frame = pd.read_csv(source).query("split == 'train'").reset_index(drop=True)
        model = PreparedFit(frame.iloc[:100],
                            frame.iloc[:100]["yield"].to_numpy(float),
                            expanded=True, trajectory=True)
        self.assertEqual(model.features_train.shape[1], len(EXPANDED) + len(TRAJECTORY))
        example = frame.iloc[[110]].copy()
        control, features = model.matrices(example)
        self.assertTrue(np.isfinite(features).all())
        # Even a wildly changed test-only pixel value cannot modify fold-fitted
        # imputation or standardization parameters.
        medians = model.trajimp.statistics_.copy()
        example.loc[:, "W2_20_40__s2_ndvi"] = 100.
        model.matrices(example)
        np.testing.assert_allclose(model.trajimp.statistics_, medians)

    def test_sensor_ablation_makes_no_use_of_excluded_input(self):
        source = ROOT / "outputs/features/full_stage_features.csv"
        if not source.is_file():
            self.skipTest("extracted full features missing")
        frame = pd.read_csv(source).query("split == 'train'").reset_index(drop=True)
        chosen = frame.iloc[:85]
        fitted = PreparedFit(chosen, chosen["yield"].to_numpy(float),
                             expanded=True, trajectory=True,
                             allowed_sensors=("s1", "s2"))
        base = fitted._raw_base(frame.iloc[[100]])
        missing_l8 = [j for j, name in enumerate(BASE) if name.rsplit("__", 1)[1].startswith("l8_")]
        self.assertTrue(np.isnan(base[:, missing_l8]).all())
        first = fitted.matrices(frame.iloc[[100]])[1]
        modified = frame.iloc[[100]].copy()
        for name in (BASE[i] for i in missing_l8):
            modified[name] = 1e8
        second = fitted.matrices(modified)[1]
        np.testing.assert_allclose(first, second)

    def test_cross_year_controls_exclude_year_and_interactions(self):
        source = ROOT / "outputs/features/full_stage_features.csv"
        if not source.is_file():
            self.skipTest("extracted full features missing")
        full = pd.read_csv(source)
        train = full.loc[full.split.eq("train")].reset_index(drop=True)
        source_year = train.loc[train.year.eq(2019)].iloc[:85].copy()
        y = source_year["yield"].to_numpy(float)
        mdl = PreparedFit(source_year, y, expanded=True, cross_year=True)
        self.assertEqual(mdl.control_numeric, ["area"])
        self.assertEqual(mdl.control_categories, ["season", "variety"])
        self.assertTrue(np.isfinite(mdl.predict(source_year, mdl.fit_coefficients(
            "adaptive", .30))).all())
        target = train.loc[train.year.eq(2020)].iloc[:1].copy()
        other_year = target.copy()
        other_year.loc[:, "year"] = 9999
        before = mdl.matrices(target)
        after = mdl.matrices(other_year)
        for m, n in zip(before, after):
            np.testing.assert_allclose(m, n, atol=0, rtol=0)

    def test_cross_year_folds_are_plot_disjoint_and_exhaustive(self):
        source = ROOT / "outputs/features/full_stage_features.csv"
        if not source.is_file():
            self.skipTest("extracted full features missing")
        from cross_year_plot_disjoint import transfer_folds
        df = pd.read_csv(source, dtype={"unique_id": str, "plot_id": str})
        train = df.loc[df.split.eq("train")]
        for source_year, target_year in ((2019, 2020), (2020, 2019)):
            seen = []
            for fold, fit, test in transfer_folds(train, source_year, target_year):
                self.assertFalse(set(fit.plot_id) & set(test.plot_id))
                self.assertEqual(set(fit.year), {source_year})
                self.assertEqual(set(test.year), {target_year})
                seen.extend(test.unique_id.tolist())
            self.assertEqual(len(seen), len(set(seen)))
            self.assertEqual(set(seen), set(train.loc[
                train.year.eq(target_year), "unique_id"]))

    def test_cross_year_saved_predictions_cover_train_only(self):
        results = ROOT / "outputs/eml_penalties/cross_year_plot_disjoint_predictions.csv"
        if not results.is_file():
            self.skipTest("cross-year experiment has not been run")
        prediction = pd.read_csv(results, dtype={"unique_id": str, "plot_id": str})
        source = pd.read_csv(ROOT / "outputs/features/full_stage_features.csv",
                             dtype={"unique_id": str, "plot_id": str})
        train = source.loc[source.split.eq("train")]
        reserved = source.loc[source.split.eq("val")]
        self.assertEqual(len(prediction), 282)
        self.assertTrue(prediction.unique_id.is_unique)
        self.assertEqual(set(prediction.unique_id), set(train.unique_id))
        self.assertFalse(set(prediction.plot_id) & set(reserved.plot_id))
        for source_year, target_year in ((2019, 2020), (2020, 2019)):
            subset = prediction.loc[prediction.target_year.eq(target_year)]
            self.assertEqual(set(subset.source_year), {source_year})
            self.assertEqual(set(subset.year), {target_year})
            self.assertEqual(subset.outer_fold.nunique(), 5)
            for name in ("ridge_controls", "extratrees_controls",
                         "extratrees_controls_satellites",
                         "nonlinear_SCAD_LLA", "nonlinear_adaptive_lasso"):
                self.assertTrue(np.isfinite(subset[name]).all())
        audit = pd.read_csv(
            ROOT / "outputs/eml_penalties/cross_year_fold_audit.csv")
        self.assertEqual(len(audit), 10)
        self.assertTrue(audit.train_test_plot_overlap.eq(0).all())
        self.assertTrue(audit.train_rows.between(90, 120).all())

    def test_matched_holdout_same_plots_no_train_overlap(self):
        from cross_year_plot_disjoint import (
            training_year_holdout, transfer_folds)
        source = ROOT / "outputs/features/full_stage_features.csv"
        if not source.exists():
            self.skipTest("extracted full features missing")
        full = pd.read_csv(source, dtype={"unique_id": str, "plot_id": str})
        df = full.loc[full.split.eq("train")]
        for src, dst in ((2019, 2020), (2020, 2019)):
            all_same = set()
            all_target = set()
            for k, fit, target in transfer_folds(df, src, dst):
                within = training_year_holdout(df, fit, target, src)
                self.assertEqual(set(within.plot_id), set(target.plot_id))
                self.assertTrue(set(fit.plot_id).isdisjoint(within.plot_id))
                self.assertTrue(set(fit.plot_id).isdisjoint(target.plot_id))
                self.assertEqual(set(within.year), {src})
                self.assertEqual(set(target.year), {dst})
                self.assertFalse(all_same & set(within.unique_id))
                self.assertFalse(all_target & set(target.unique_id))
                all_same |= set(within.unique_id)
                all_target |= set(target.unique_id)
            self.assertEqual(all_same, set(df.loc[df.year.eq(src)].unique_id))
            self.assertEqual(all_target, set(df.loc[df.year.eq(dst)].unique_id))

    def test_matched_scores_valid_on_simple_synthetic_data(self):
        from cross_year_plot_disjoint import matched_scores
        synthetic = pd.DataFrame({
            "source_year": [2019] * 4 + [2020] * 4,
            "evaluation": ["within_year"] * 2 + ["cross_year"] * 2
                          + ["within_year"] * 2 + ["cross_year"] * 2,
            "year": [2019, 2019, 2020, 2020, 2020, 2020, 2019, 2019],
            "yield": [100., 200., 110., 220., 110., 220., 100., 200.],
            "train_mean": [150.] * 8,
            "train_rows": [9] * 8,
            "plot_id": ["p1", "p2"] * 4,
        })
        from cross_year_plot_disjoint import MATCHED_MODELS
        for name in MATCHED_MODELS:
            if name != "train_mean":
                synthetic[name] = synthetic["yield"]
        result = matched_scores(synthetic)
        self.assertEqual(len(result), 4 * len(MATCHED_MODELS))
        exact = result.loc[result.model.eq("ridge_controls")]
        self.assertTrue((exact.RMSE == 0).all())
        self.assertTrue((exact.RMSE_normalized_by_target_sd == 0).all())

    def test_saved_matched_predictions_reproduce_cross_year(self):
        from cross_year_plot_disjoint import MATCHED_MODELS
        folder = ROOT / "outputs/eml_penalties"
        new_path = folder / "cross_year_matched_predictions.csv"
        old_path = folder / "cross_year_plot_disjoint_predictions.csv"
        if not new_path.exists() or not old_path.exists():
            self.skipTest("matched experiment not yet run")
        combined = pd.read_csv(new_path, dtype={"unique_id": str, "plot_id": str})
        previously = pd.read_csv(old_path, dtype={"unique_id": str, "plot_id": str})
        self.assertEqual(len(combined), 564)
        self.assertEqual(combined.groupby("evaluation").size().to_dict(),
                         {"within_year": 282, "cross_year": 282})
        for mode, subset in combined.groupby("evaluation"):
            self.assertTrue(subset.unique_id.is_unique)
            self.assertEqual(subset.plot_id.nunique(), 117)
        cross = combined.loc[combined.evaluation.eq("cross_year")].set_index(
            ["source_year", "unique_id"]).sort_index()
        previously = previously.set_index(["source_year", "unique_id"]).sort_index()
        self.assertTrue(cross.index.equals(previously.index))
        for name in MATCHED_MODELS:
            np.testing.assert_allclose(cross[name], previously[name], atol=1e-6)
        audit = pd.read_csv(folder / "cross_year_matched_fold_audit.csv")
        self.assertEqual(len(audit), 10)
        self.assertTrue(audit.identical_heldout_plot_sets.all())
        self.assertTrue(audit.training_test_plot_overlap.eq(0).all())

    def test_shift_and_matched_summary_only_training_years(self):
        folder = ROOT / "outputs/eml_penalties"
        summary_path = folder / "cross_year_matched_comparison.csv"
        if not summary_path.exists():
            self.skipTest("matched experiment not yet run")
        outcome = pd.read_csv(summary_path)
        self.assertEqual(len(outcome), 32)
        self.assertEqual(set(outcome.source_year), {2019, 2020})
        self.assertTrue(outcome.RMSE.notna().all())
        missingness = pd.read_csv(folder / "cross_year_feature_shift.csv")
        self.assertEqual(len(missingness), 57)
        self.assertTrue(missingness.missing_rate_2019.between(0, 1).all())
        self.assertTrue(missingness.missing_rate_2020.between(0, 1).all())
        coverage = pd.read_csv(folder / "cross_year_sensor_window_coverage.csv")
        self.assertEqual(len(coverage), 30)
        self.assertEqual(set(coverage.year), {2019, 2020})

    def test_predeclared_stable_eml_features_exclude_landsat(self):
        from cross_year_plot_disjoint import (
            S1S2, STABLE20, STABLE_MODELS, STABLE_PENALIZED)
        self.assertEqual(len(S1S2), 40)
        self.assertEqual(len(STABLE20), 20)
        self.assertEqual(len(STABLE_MODELS), 5)
        self.assertEqual(len(set(S1S2)), 40)
        self.assertTrue(set(STABLE20).issubset(S1S2))
        for feature in S1S2:
            self.assertNotIn("__l8_", feature)
            self.assertNotIn("yield", feature)
        self.assertEqual(set(STABLE_PENALIZED[1].allowed_features), set(STABLE20))

    def test_stable_eml_masks_every_nonselected_feature_even_on_test(self):
        from cross_year_plot_disjoint import STABLE20
        path = ROOT / "outputs/features/full_stage_features.csv"
        if not path.exists():
            self.skipTest("extracted features missing")
        frame = pd.read_csv(path).query("split == 'train'").reset_index(drop=True)
        train = frame.loc[frame.year.eq(2019)].iloc[:90]
        y = train["yield"].to_numpy(float)
        fitted = PreparedFit(train, y, expanded=False, cross_year=True,
                             allowed_features=STABLE20)
        self.assertEqual(fitted.control_numeric, ["area"])
        unseen = frame.loc[frame.year.eq(2020)].iloc[[1]].copy()
        first = fitted.matrices(unseen)[1]
        mutated = unseen.copy()
        for col in BASE:
            if col not in STABLE20:
                mutated.loc[:, col] = 1e12
        second = fitted.matrices(mutated)[1]
        np.testing.assert_array_equal(first, second)

    def test_stable_eml_saved_predictions_align_with_matched_baseline(self):
        folder = ROOT / "outputs/eml_penalties"
        path = folder / "cross_year_stable_eml_predictions.csv"
        if not path.exists():
            self.skipTest("stable EML experiments not run")
        from cross_year_plot_disjoint import STABLE_MODELS
        prediction = pd.read_csv(path, dtype={"unique_id": str, "plot_id": str})
        baseline = pd.read_csv(folder / "cross_year_matched_predictions.csv",
                               dtype={"unique_id": str, "plot_id": str})
        keys = ["source_year", "evaluation", "unique_id"]
        self.assertEqual(len(prediction), 564)
        self.assertFalse(prediction[keys].duplicated().any())
        self.assertEqual(
            set(map(tuple, prediction[keys].to_numpy())),
            set(map(tuple, baseline[keys].to_numpy())))
        self.assertTrue(np.isfinite(prediction[list(STABLE_MODELS)]).all().all())
        for col in ("ridge_controls", "ridge_controls_satellites",
                    "extratrees_controls", "nonlinear_adaptive_lasso"):
            joined = prediction[keys + [col]].merge(
                baseline[keys + [col]], on=keys, validate="one_to_one",
                suffixes=("_new", "_old"))
            np.testing.assert_allclose(joined[f"{col}_new"],
                                       joined[f"{col}_old"],
                                       rtol=0, atol=1e-9)

    def test_controls_not_penalized(self):
        source = ROOT / "outputs/features/full_stage_features.csv"
        if not source.is_file():
            self.skipTest("extracted full features missing")
        raw = pd.read_csv(source).query("split == 'train'").reset_index(drop=True)
        selected = raw.iloc[:85]
        y = selected["yield"].to_numpy(float)
        fit = PreparedFit(selected, y, expanded=True, penalize_agro_interactions=True)
        self.assertTrue(any(name.startswith("agro__") for name in fit.names))
        beta_zero = fit.fit_coefficients("controls", .6)
        self.assertEqual(int(np.count_nonzero(beta_zero)), 0)
        expected_standardized = fit.controls_train @ fit.projection[:, 0]
        expected = fit.ymean + fit.ystd * expected_standardized
        np.testing.assert_allclose(fit.predict(selected, beta_zero), expected, atol=1e-7)

    def test_post_lasso_training_projection_and_unknown_categories(self):
        source = ROOT / "outputs/features/full_stage_features.csv"
        if not source.is_file():
            self.skipTest("extracted full features missing")
        raw = pd.read_csv(source).query("split == 'train'").reset_index(drop=True)
        selected = raw.iloc[:130]
        y = selected["yield"].to_numpy(float)
        fitted = PreparedFit(selected, y, expanded=False)
        beta = fitted.fit_coefficients("post_lasso", .3)
        support = np.flatnonzero(np.abs(beta) > 1e-5)
        if support.size:
            residual = fitted.yc - fitted.xc @ beta
            np.testing.assert_allclose(fitted.xc[:, support].T @ residual,
                                       np.zeros(len(support)), atol=1e-6)
        unfamiliar = raw.iloc[[130]].copy()
        unfamiliar["variety"] = "NEVER_SEEN_VARIETY"
        unfamiliar["season"] = "NEVER_SEEN_SEASON"
        self.assertTrue(np.isfinite(fitted.predict(unfamiliar, beta)).all())

    def test_scad_local_linear_approximation_nonincreasing(self):
        source = ROOT / "outputs/features/full_stage_features.csv"
        if not source.is_file():
            self.skipTest("extracted full features missing")
        train = pd.read_csv(source).query("split == 'train'").iloc[:100]
        y = train["yield"].to_numpy(float)
        fitted = PreparedFit(train, y, expanded=True)
        for init in ("lasso", "ridge", "zero"):
            beta, history = fitted.scad_from_initialization(.6, init)
            self.assertTrue(np.isfinite(beta).all())
            self.assertLessEqual(max(np.diff(history)), 1e-6)

    def test_grouped_results_exclude_official_validation(self):
        output = ROOT / "outputs/eml_penalties/grouped_cv_predictions.csv"
        if not output.is_file():
            self.skipTest("EML experiment has not been run")
        saved = pd.read_csv(output, dtype={"plot_id": str, "unique_id": str})
        full = pd.read_csv(ROOT / "outputs/features/full_stage_features.csv",
                           dtype={"plot_id": str, "unique_id": str})
        train = full[full.split.eq("train")]
        validation = full[full.split.eq("val")]
        self.assertEqual(len(saved), 282)
        self.assertEqual(set(saved.unique_id), set(train.unique_id))
        self.assertFalse(set(saved.plot_id) & set(validation.plot_id))
        for name in ("partial_lasso", "nonlinear_SCAD_LLA",
                     "nonlinear_partial_elastic_net"):
            self.assertTrue(np.isfinite(saved[name]).all())


if __name__ == "__main__":
    unittest.main()
