#!/usr/bin/env python3
"""Cross-year / unseen-physical-plot evaluation using official TRAIN rows only.

Protocol for each year direction and each of five PLOT_ID groups:
  - train on source-year plot-seasons EXCLUDING every target-year held-out plot
  - predict target-year plot-seasons for those held-out plots
  - tune only with a 3-fold physical-plot-grouped CV on the source-year rows
  - discard year and year-by-variety effects (new year cannot be learned)

The source and target years share the same physical plots, so ordinary
train-2019/test-2020 would not be plot-disjoint. This evaluation has both
temporal transfer and new-plot constraints, but uses far fewer training
observations than earlier pooled-year experiments.

Predictions are retrospective: 0%-100% realized-season satellite windows
require the actual harvest date. This is NOT a pre-harvest forecast.

No official VAL yield is read or scored. Results are exploratory because
the model families were investigated on these same TRAIN plots earlier.
Outputs are CSV/JSON under outputs/eml_penalties/cross_year_*.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import ks_2samp

from eml_penalty_experiments import (OUT, ROOT, Experiment, PreparedFit, tune,
                                    STAGES)
from improve_yield_models import (SAT, Spec, engineered, pipeline_and_grid,
                                  score)


PENALIZED = (
    Experiment("nonlinear_adaptive_lasso", "adaptive", expanded=True),
    Experiment("nonlinear_SCAD_LLA", "scad", expanded=True),
    Experiment("nonlinear_partial_elastic_net", "elastic", expanded=True),
)
BASELINES = (
    Spec("ridge_controls", ("area",), ("season", "variety"), "ridge"),
    Spec("ridge_controls_satellites", ("area", *SAT), ("season", "variety"), "ridge"),
    Spec("extratrees_controls", ("area",), ("season", "variety"), "extratrees"),
    Spec("extratrees_controls_satellites", ("area", *SAT),
         ("season", "variety"), "extratrees"),
)
MODEL_NAMES = ("train_mean",) + tuple(s.name for s in BASELINES) + tuple(
    s.name for s in PENALIZED)

# Fixed before inspecting the new matched-sample results. The paired
# experiment uses the same training observations/model for both test years.
MATCHED_MODELS = (
    "train_mean", "ridge_controls", "ridge_controls_satellites",
    "extratrees_controls", "extratrees_controls_satellites",
    "nonlinear_adaptive_lasso", "nonlinear_SCAD_LLA",
    "nonlinear_partial_elastic_net",
)

# Five fixed, course-aligned candidates. None uses Landsat-8 (which had
# substantial year-dependent missingness in the prior shift diagnostic).
# No candidate is selected using source->target outcomes.
S1S2 = tuple(c for c in SAT if c.rsplit("__", 1)[1].startswith(("s1_", "s2_")))
STABLE20 = tuple(f"{stage}__{feature}" for stage in STAGES
                 for feature in ("s1_vv_db", "s1_vv_minus_vh_db",
                                 "s2_ndvi", "s2_ndmi"))
assert len(S1S2) == 40 and len(STABLE20) == 20
STABLE_RIDGE_SPECS = (
    Spec("ridge_s1s2_strong", ("area", *S1S2), ("season", "variety"), "ridge"),
    Spec("ridge_stable20_strong", ("area", *STABLE20), ("season", "variety"), "ridge"),
)
STABLE_PENALIZED = (
    Experiment("partial_elastic_net_s1s2", "elastic",
               allowed_sensors=("s1", "s2")),
    Experiment("partial_adaptive_lasso_stable20", "adaptive",
               allowed_features=STABLE20),
)
STABLE_MODELS = (tuple(spec.name for spec in STABLE_RIDGE_SPECS)
                 + ("pca_ridge_s1s2",)
                 + tuple(spec.name for spec in STABLE_PENALIZED))
STABLE_COMPARATORS = (
    "train_mean", "ridge_controls", "ridge_controls_satellites",
    "extratrees_controls", "nonlinear_adaptive_lasso",
)


def transfer_folds(df, source_year, target_year):
    """Return index arrays into TRAIN-only df, enforcing new-plot separation."""
    source = df.loc[df.year.eq(source_year)]
    target = df.loc[df.year.eq(target_year)]
    assert len(source) > 0 and len(target) > 0
    assert source.plot_id.nunique() == target.plot_id.nunique() == 117
    groups = GroupKFold(n_splits=5).split(target, groups=target.plot_id)
    used_target = set()
    for fold_no, (_, test_local) in enumerate(groups, start=1):
        heldout = target.iloc[test_local]
        plots = set(heldout.plot_id)
        fitting = source.loc[~source.plot_id.isin(plots)]
        assert not set(fitting.plot_id) & plots
        assert set(fitting.year) == {source_year}
        assert set(heldout.year) == {target_year}
        assert not used_target & set(heldout.unique_id)
        used_target |= set(heldout.unique_id)
        yield fold_no, fitting, heldout
    assert used_target == set(target.unique_id)


def fit_penalized(fitting, heldout, experiment):
    y = fitting["yield"].to_numpy(float)
    fraction, ratio, cv_mse = tune(fitting, y, fitting.plot_id.to_numpy(),
                                   experiment, cross_year=True)
    fitted = PreparedFit(fitting, y, expanded=experiment.expanded,
                         cross_year=True)
    assert fitted.control_numeric == ["area"]
    assert fitted.control_categories == ["season", "variety"]
    beta = fitted.fit_coefficients(experiment.kind, fraction, ratio)
    return (fitted.predict(heldout, beta),
            {"fraction_lambda_max": float(fraction),
             "l1_ratio": float(ratio), "inner_mse": float(cv_mse),
             "selected_predictors": int(np.sum(np.abs(beta) > 1e-5))})


def fit_sklearn(fitting, heldout, spec):
    x_train = engineered(fitting)
    x_test = engineered(heldout)
    columns = list(spec.numeric + spec.categorical)
    assert "year" not in columns and "year_variety" not in columns
    model, grid = pipeline_and_grid(spec)
    tuning = GridSearchCV(model, grid,
                          scoring="neg_mean_squared_error",
                          cv=GroupKFold(n_splits=3),
                          n_jobs=1, error_score="raise")
    tuning.fit(x_train[columns], fitting["yield"].to_numpy(float),
               groups=fitting.plot_id.to_numpy())
    return tuning.predict(x_test[columns]), tuning.best_params_


def training_year_holdout(df, fitting, target_rows, source_year):
    """Same held-out plot IDs as cross-year target, but source-year outcomes."""
    target_plots = set(target_rows.plot_id)
    own = df.loc[df.year.eq(source_year) & df.plot_id.isin(target_plots)].copy()
    assert set(own.plot_id) == target_plots
    assert not set(own.plot_id) & set(fitting.plot_id)
    return own


def matched_predictions(train):
    """One model fit/tuning per fold, then evaluate BOTH years on same plots.

    Matching is by identical training rows and held-out physical plot IDs, not
    by pretending the test plot-season rows are identical across seasons.
    """
    prediction = []
    choices = []
    audits = []
    for src, dst in ((2019, 2020), (2020, 2019)):
        for fold, fitting, target in transfer_folds(train, src, dst):
            same = training_year_holdout(train, fitting, target, src)
            combined = pd.concat((same, target), axis=0)
            n_same = len(same)
            assert set(same.plot_id) == set(target.plot_id)
            assert len(fitting) >= 90
            assert not set(fitting.plot_id) & set(combined.plot_id)
            row = combined[["unique_id", "plot_id", "yield", "year",
                            "season", "variety"]].copy()
            row["source_year"] = src
            row["outer_fold"] = fold
            row["evaluation"] = (["within_year"] * n_same +
                                 ["cross_year"] * len(target))
            row["train_rows"] = len(fitting)
            row["train_physical_plots"] = fitting.plot_id.nunique()
            row["train_mean"] = fitting["yield"].mean()
            audits.append({
                "source_year": src, "target_year": dst, "outer_fold": fold,
                "train_rows": len(fitting),
                "train_physical_plots": fitting.plot_id.nunique(),
                "test_same_year_rows": n_same,
                "test_other_year_rows": len(target),
                "test_plots": target.plot_id.nunique(),
                "training_test_plot_overlap": len(
                    set(fitting.plot_id) & set(target.plot_id)),
                "identical_heldout_plot_sets": set(same.plot_id) == set(target.plot_id),
            })
            for spec in BASELINES:
                scores, best = fit_sklearn(fitting, combined, spec)
                row[spec.name] = scores
                choices.append({
                    "source_year": src, "outer_fold": fold,
                    "model": spec.name, "tuned_on_source_year_only": best,
                })
            for spec in PENALIZED:
                scores, best = fit_penalized(fitting, combined, spec)
                row[spec.name] = scores
                choices.append({
                    "source_year": src, "outer_fold": fold,
                    "model": spec.name, "tuned_on_source_year_only": best,
                })
            assert np.isfinite(row[list(MATCHED_MODELS)].to_numpy(float)).all()
            prediction.append(row)
            print(f"MATCHED {src}->{dst} fold {fold}/5: train={len(fitting)} "
                  f"plots={fitting.plot_id.nunique()}, same-year={n_same}, "
                  f"cross-year={len(target)}", flush=True)
    combined = pd.concat(prediction, ignore_index=True)
    assert len(combined) == 2 * len(train)
    # Each observation is predicted exactly once for the within-year and
    # once for the cross-year direction, using *different* fitted models.
    for mode in ("within_year", "cross_year"):
        part = combined.loc[combined.evaluation.eq(mode)]
        assert len(part) == len(train)
        assert part.unique_id.is_unique
        assert set(part.unique_id) == set(train.unique_id)
    return combined, pd.DataFrame(audits), choices


def matched_scores(predictions, model_names=MATCHED_MODELS):
    """Absolute error + skill vs training-only mean, per direction and mode."""
    summaries = []
    for src in (2019, 2020):
        for mode in ("within_year", "cross_year"):
            rows = predictions.loc[
                predictions.source_year.eq(src) & predictions.evaluation.eq(mode)]
            target_std = float(rows["yield"].std(ddof=0))
            mean_error = score(rows["yield"], rows["train_mean"])["RMSE"]
            for model in model_names:
                accuracy = score(rows["yield"], rows[model])
                summaries.append({
                    "source_year": src, "evaluation": mode,
                    "target_year": int(rows.year.iloc[0]),
                    "model": model, "trained_rows_per_fold_min": int(rows.train_rows.min()),
                    "trained_rows_per_fold_max": int(rows.train_rows.max()),
                    "evaluated_rows": len(rows),
                    "evaluated_plots": rows.plot_id.nunique(),
                    "target_yield_sd": target_std,
                    "RMSE_normalized_by_target_sd": (
                        accuracy["RMSE"] / target_std if target_std else np.nan),
                    "RMSE_improvement_over_source_mean": mean_error - accuracy["RMSE"],
                    **accuracy,
                })
    return pd.DataFrame(summaries)


def matched_bootstrap(prediction, n_bootstrap=1200,
                      model_names=MATCHED_MODELS):
    """Descriptive conditional CIs for cross-year RMSE minus within-year RMSE.

    The two evaluation years each cover the *same plot IDs*; resample their
    plot IDs jointly, holding all model predictions fixed.
    """
    rng = np.random.default_rng(2026)
    report = []
    for src in (2019, 2020):
        source = prediction.loc[prediction.source_year.eq(src)]
        same = source.loc[source.evaluation.eq("within_year")]
        other = source.loc[source.evaluation.eq("cross_year")]
        assert set(same.plot_id) == set(other.plot_id)
        plots = np.sort(source.plot_id.unique())
        same_map = {p: np.flatnonzero(same.plot_id.to_numpy() == p) for p in plots}
        other_map = {p: np.flatnonzero(other.plot_id.to_numpy() == p) for p in plots}
        same_y = same["yield"].to_numpy(float)
        other_y = other["yield"].to_numpy(float)
        samples = [rng.choice(plots, len(plots), replace=True)
                   for _ in range(n_bootstrap)]
        for model in model_names:
            sv = same[model].to_numpy(float)
            ov = other[model].to_numpy(float)
            same_rmse = score(same_y, sv)["RMSE"]
            other_rmse = score(other_y, ov)["RMSE"]
            intervals = []
            for sample in samples:
                isame = np.concatenate([same_map[p] for p in sample])
                iother = np.concatenate([other_map[p] for p in sample])
                intervals.append(
                    score(other_y[iother], ov[iother])["RMSE"] -
                    score(same_y[isame], sv[isame])["RMSE"])
            low, high = np.quantile(intervals, [.025, .975])
            report.append({
                "source_year": src, "model": model,
                "within_year_RMSE": same_rmse, "cross_year_RMSE": other_rmse,
                "cross_minus_within_RMSE": other_rmse - same_rmse,
                "conditional_plot_bootstrap_low": float(low),
                "conditional_plot_bootstrap_high": float(high),
            })
    return pd.DataFrame(report)


def shift_diagnostics(train, predictions):
    """Plot-season summaries, paired physical-plot shifts and residual splits."""
    sensor_features = tuple(SAT)
    numbers = ("yield", "area") + sensor_features
    numeric = train[["plot_id", "year", *numbers]].copy()
    by_year = {}
    for year in (2019, 2020):
        by_year[year] = numeric.loc[numeric.year.eq(year)]
    shifts = []
    paired = []
    for col in numbers:
        a = pd.to_numeric(by_year[2019][col], errors="coerce").dropna()
        b = pd.to_numeric(by_year[2020][col], errors="coerce").dropna()
        pool_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        ks = ks_2samp(a, b).statistic if len(a) and len(b) else np.nan
        shifts.append({
            "variable": col,
            "sensor": col.split("__", 1)[1].split("_", 1)[0] if "__" in col else "survey",
            "window": col.split("__", 1)[0] if "__" in col else "none",
            "n_2019": len(a), "n_2020": len(b),
            "missing_rate_2019": 1 - len(a) / len(by_year[2019]),
            "missing_rate_2020": 1 - len(b) / len(by_year[2020]),
            "missing_rate_change_2020_minus_2019": (
                len(a) / len(by_year[2019]) - len(b) / len(by_year[2020])),
            "mean_2019": a.mean(), "mean_2020": b.mean(),
            "mean_shift_2020_minus_2019": b.mean() - a.mean(),
            "pooled_std_standardized_mean_shift": (
                (b.mean() - a.mean()) / pool_sd if pool_sd > 1e-12 else np.nan),
            "median_2019": a.median(), "median_2020": b.median(),
            "ks_distance": ks,  # descriptive; NO multiple-tests inference
        })
        # Repeated plot-seasons aggregated to a physical plot per year.
        plot_yr = numeric[["plot_id", "year", col]].groupby(
            ["plot_id", "year"])[col].mean().unstack("year")
        common = plot_yr[[2019, 2020]].dropna()
        delta = common[2020] - common[2019]
        paired.append({
            "variable": col, "paired_observed_plots": len(delta),
            "plot_mean_change_2020_minus_2019": delta.mean(),
            "plot_median_change_2020_minus_2019": delta.median(),
            "fraction_plots_increased": (delta.gt(0).mean() if len(delta) else np.nan),
        })

    # Counts are plot-season proportions, not physical-plot proportions.
    categories = []
    for key in ("season", "variety"):
        for year in (2019, 2020):
            group = train.loc[train.year.eq(year), key].fillna("__missing__")
            for label, count in group.value_counts().items():
                categories.append({
                    "variable": key, "year": year, "category": label,
                    "plot_seasons": int(count),
                    "fraction_of_year_rows": count / len(group),
                })

    # Availability and quality by sensor / phenological window, by year.
    stage_coverage = []
    for year in (2019, 2020):
        subset = train.loc[train.year.eq(year)]
        for window in sorted({x.split("__")[0] for x in SAT}):
            for sensor, representative in (
                ("s1", "s1_vv_db"), ("s2", "s2_ndvi"), ("l8", "l8_ndvi")):
                predictor = f"{window}__{representative}"
                accepted = f"{window}__{sensor}__accepted_scenes"
                available = f"{window}__{sensor}__available_scenes"
                stage_coverage.append({
                    "year": year, "window": window, "sensor": sensor,
                    "n_plots_seasons": len(subset),
                    "observed_feature_fraction": subset[predictor].notna().mean(),
                    "accepted_scenes_mean": subset[accepted].mean(),
                    "available_scenes_mean": subset[available].mean(),
                    "at_least_one_accepted_scene_fraction": subset[accepted].gt(0).mean(),
                })

    # Each row of prediction corresponds to an OOF observation fitted on
    # disjoint plots. Residual diagnostics are associative, not causal.
    residuals = predictions.merge(
        train[["unique_id", "area", "W1_0_20__s1_vv_db",
               "W1_0_20__s2_ndvi", "W1_0_20__l8_ndvi"]],
        on="unique_id", how="left", validate="many_to_one")
    residuals["L8_W1_available"] = np.where(
        residuals["W1_0_20__l8_ndvi"].notna(), "yes", "no")
    residuals["S2_W1_available"] = np.where(
        residuals["W1_0_20__s2_ndvi"].notna(), "yes", "no")
    residuals["S1_W1_available"] = np.where(
        residuals["W1_0_20__s1_vv_db"].notna(), "yes", "no")
    # Include both coarse year-period and agronomic/satellite categories;
    # suppress subgroup performance for n<8 to avoid misleading scores.
    subgroup_results = []
    for source in (2019, 2020):
        for evaluation in ("within_year", "cross_year"):
            rows = residuals.loc[
                residuals.source_year.eq(source) &
                residuals.evaluation.eq(evaluation)]
            for key in ("season", "variety", "L8_W1_available",
                        "S2_W1_available", "S1_W1_available"):
                for label, part in rows.groupby(key, dropna=False):
                    if len(part) < 8:
                        continue
                    for model in ("train_mean", "ridge_controls",
                                  "ridge_controls_satellites",
                                  "extratrees_controls",
                                  "extratrees_controls_satellites",
                                  "nonlinear_adaptive_lasso",
                                  "nonlinear_SCAD_LLA"):
                        actual = part["yield"].to_numpy(float)
                        predictions_values = part[model].to_numpy(float)
                        subgroup_results.append({
                            "source_year": source, "evaluation": evaluation,
                            "target_year": int(part.year.iloc[0]),
                            "dimension": key, "subgroup": str(label),
                            "model": model, "plot_seasons": len(part),
                            "unique_plots": part.plot_id.nunique(),
                            "mean_prediction_residual": float(
                                (actual - predictions_values).mean()),
                            **score(actual, predictions_values),
                        })

    # Acquisitions status is independent of outcome; use only TRAIN plot
    # scene records to avoid mixing the reserved validation cohort.
    obs = pd.read_csv(ROOT / "outputs/features/full_observation_features.csv",
                      usecols=["unique_id", "sensor", "status"],
                      dtype={"unique_id": str})
    obs = obs.merge(train[["unique_id", "year"]], on="unique_id",
                    how="inner", validate="many_to_one")
    scene = obs.groupby(["year", "sensor", "status"]).size().rename(
        "scene_count").reset_index()
    scene["all_sensor_year_scenes"] = scene.groupby(
        ["year", "sensor"])["scene_count"].transform("sum")
    scene["fraction_sensor_year_scenes"] = (
        scene.scene_count / scene.all_sensor_year_scenes)
    return {
        "cross_year_feature_shift.csv": pd.DataFrame(shifts),
        "cross_year_paired_plot_shift.csv": pd.DataFrame(paired),
        "cross_year_agronomic_composition.csv": pd.DataFrame(categories),
        "cross_year_sensor_window_coverage.csv": pd.DataFrame(stage_coverage),
        "cross_year_residual_subgroups.csv": pd.DataFrame(subgroup_results),
        "cross_year_scene_quality.csv": scene,
    }


def run_matched_diagnostics():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / "outputs/features/full_stage_features.csv",
                      dtype={"plot_id": str, "unique_id": str})
    train = raw.loc[raw.split.eq("train")].copy()
    val = raw.loc[raw.split.eq("val")]
    assert len(train) == 282 and len(val) == 37
    assert set(train.plot_id).isdisjoint(val.plot_id)
    prediction, audit, choices = matched_predictions(train)

    # Cross-year part must exactly reproduce the prior cross-year run;
    # prevents accidental alteration of training splits or model tuning.
    earlier = pd.read_csv(OUT / "cross_year_plot_disjoint_predictions.csv",
                          dtype={"plot_id": str, "unique_id": str})
    comparison = prediction.loc[
        prediction.evaluation.eq("cross_year")].set_index(
            ["source_year", "unique_id"]).sort_index()
    previously = earlier.set_index(["source_year", "unique_id"]).sort_index()
    assert comparison.index.equals(previously.index)
    for name in MATCHED_MODELS:
        np.testing.assert_allclose(
            comparison[name], previously[name], rtol=1e-8, atol=1e-6,
            err_msg=f"Mismatch with earlier cross-year run: {name}")

    prediction.to_csv(OUT / "cross_year_matched_predictions.csv", index=False)
    audit.to_csv(OUT / "cross_year_matched_fold_audit.csv", index=False)
    (OUT / "cross_year_matched_nested_choices.json").write_text(
        json.dumps(choices, indent=2) + "\n")
    result = matched_scores(prediction)
    result.to_csv(OUT / "cross_year_matched_comparison.csv", index=False)
    paired = matched_bootstrap(prediction)
    paired.to_csv(OUT / "cross_year_matched_paired_deltas.csv", index=False)
    for fname, data in shift_diagnostics(train, prediction).items():
        data.to_csv(OUT / fname, index=False)
    (OUT / "cross_year_matched_configuration.json").write_text(json.dumps({
        "training_data": "282 official TRAIN plot-seasons, 117 physical plots",
        "official_val_used": False,
        "one_fit_predicts_both_years": True,
        "train_years": [2019, 2020],
        "training_rows_matching": "Exactly identical training rows, held-out plots, models, and fitted parameters per source-year fold",
        "controls": ["area", "season", "variety"],
        "model_hyperparameter_selection": "Training-only 3-fold plot-grouped CV",
        "target_variation_warning": "Within and cross-year rows have different yields, repeated seasons and dispersion; compare target-normalized RMSE and skill vs training-mean in addition to raw RMSE",
        "not_external_validation": "All observations are reused official TRAIN plots; model choices were previously explored on them",
        "retrospective_satellite_timing": "Full-season windows use actual observed harvest date; not early forecasting",
        "shifts": "Raw distribution differences, two-sample KS distance as descriptive statistic; plot-paired deltas aggregate multiple plot-seasons",
        "bootstrap_warning": "Joint plot-cluster bootstrap uses fixed OOF predictions and is not refit or selection adjusted",
    }, indent=2) + "\n")
    print("\nMATCHED RMSE BY DIRECTION", flush=True)
    print(result.pivot_table(index="model", columns=["source_year", "evaluation"],
                             values="RMSE").round(2).to_string(), flush=True)
    print("\nMATCHED RMSE transfer minus within-year", flush=True)
    print(paired[["source_year", "model", "cross_minus_within_RMSE",
                  "conditional_plot_bootstrap_low",
                  "conditional_plot_bootstrap_high"]].round(2).to_string(
                      index=False), flush=True)
    print("\nFeature, composition, sensor-quality and residual shift CSVs saved",
          flush=True)


def fit_stable_ridge(fitting, test_rows, spec):
    """Ridge on a predeclared S1/S2 subset, strong shrinkage tuned on TRAIN."""
    columns = list(spec.numeric + spec.categorical)
    assert all(col != "year" and "__l8_" not in col
               for col in columns)
    pipeline, _ = pipeline_and_grid(spec)
    grid = {"model__alpha": [10., 100., 300., 1000., 3000.]}
    cv = GridSearchCV(pipeline, grid, scoring="neg_mean_squared_error",
                      cv=GroupKFold(n_splits=3), n_jobs=1, error_score="raise")
    cv.fit(engineered(fitting)[columns], fitting["yield"].to_numpy(float),
           groups=fitting.plot_id.to_numpy())
    return cv.predict(engineered(test_rows)[columns]), cv.best_params_


def fit_pca_ridge(fitting, test_rows):
    """PCA is unsupervised, learned strictly inside each training fold."""
    features = ["area", *S1S2, "season", "variety"]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median",
                                                 keep_empty_features=True)),
                           ("scale", StandardScaler())]), ["area"]),
        ("pca", Pipeline([("impute", SimpleImputer(strategy="median",
                                                 keep_empty_features=True)),
                           ("scale", StandardScaler()),
                           ("reduce", PCA(svd_solver="full"))]), list(S1S2)),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["season", "variety"]),
    ], sparse_threshold=0)
    model = Pipeline([("pre", pre), ("model", Ridge())])
    grid = {"pre__pca__reduce__n_components": [3, 6, 10],
            "model__alpha": [10., 100., 1000.]}
    cv = GridSearchCV(model, grid, scoring="neg_mean_squared_error",
                      cv=GroupKFold(n_splits=3), n_jobs=1, error_score="raise")
    cv.fit(engineered(fitting)[features], fitting["yield"].to_numpy(float),
           groups=fitting.plot_id.to_numpy())
    return cv.predict(engineered(test_rows)[features]), cv.best_params_


def fit_stable_penalized(fitting, test_rows, experiment):
    """Exact partial-penalty EML setup: agronomic controls always unpenalized."""
    y = fitting["yield"].to_numpy(float)
    lam, ratio, loss = tune(fitting, y, fitting.plot_id.to_numpy(),
                             experiment, cross_year=True)
    fit = PreparedFit(fitting, y, expanded=False, cross_year=True,
                      allowed_sensors=experiment.allowed_sensors,
                      allowed_features=experiment.allowed_features)
    assert fit.control_numeric == ["area"]
    assert fit.control_categories == ["season", "variety"]
    coef = fit.fit_coefficients(experiment.kind, lam, ratio)
    chosen = [name for name, value in zip(fit.names, coef) if abs(value) > 1e-5]
    permitted = set(S1S2 if experiment.allowed_sensors else STABLE20)
    assert set(chosen).issubset(permitted)
    return fit.predict(test_rows, coef), {
        "fraction_lambda_max": lam, "l1_ratio": ratio,
        "inner_mse": loss, "selected_count": len(chosen),
    }


def run_stable_eml():
    """Fixed reduced-sensor models on the existing matched cross-year protocol.

    Reuses the original held-out plot groups and *exact* training source
    samples. Benchmarks are fetched as frozen OOF predictions; never tuned
    using cross-year held-out labels.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / "outputs/features/full_stage_features.csv",
                      dtype={"unique_id": str, "plot_id": str})
    train = raw.loc[raw.split.eq("train")].copy()
    reserved = raw.loc[raw.split.eq("val")]
    assert len(train) == 282 and len(reserved) == 37
    assert set(train.plot_id).isdisjoint(reserved.plot_id)
    old = pd.read_csv(OUT / "cross_year_matched_predictions.csv",
                      dtype={"unique_id": str, "plot_id": str})
    assert len(old) == 564

    out = []
    choices = []
    for src, dst in ((2019, 2020), (2020, 2019)):
        for fold, fit_rows, target in transfer_folds(train, src, dst):
            same = training_year_holdout(train, fit_rows, target, src)
            evaluation = pd.concat([same, target], ignore_index=True)
            assert set(fit_rows.plot_id).isdisjoint(evaluation.plot_id)
            assert set(same.plot_id) == set(target.plot_id)
            row = evaluation[["unique_id", "plot_id", "yield", "year"]].copy()
            row["source_year"] = src
            row["outer_fold"] = fold
            row["evaluation"] = (["within_year"] * len(same) +
                                 ["cross_year"] * len(target))
            row["train_rows"] = len(fit_rows)
            row["train_physical_plots"] = fit_rows.plot_id.nunique()
            for spec in STABLE_RIDGE_SPECS:
                prediction, selected = fit_stable_ridge(fit_rows, evaluation, spec)
                row[spec.name] = prediction
                choices.append({"source_year": src, "outer_fold": fold,
                                "model": spec.name, "inner_selected": selected})
            prediction, selected = fit_pca_ridge(fit_rows, evaluation)
            row["pca_ridge_s1s2"] = prediction
            choices.append({"source_year": src, "outer_fold": fold,
                            "model": "pca_ridge_s1s2", "inner_selected": selected})
            for spec in STABLE_PENALIZED:
                prediction, selected = fit_stable_penalized(
                    fit_rows, evaluation, spec)
                row[spec.name] = prediction
                choices.append({"source_year": src, "outer_fold": fold,
                                "model": spec.name, "inner_selected": selected})
            assert np.isfinite(row[list(STABLE_MODELS)].to_numpy(float)).all()
            out.append(row)
            print(f"STABLE {src}->{dst} fold {fold}/5, "
                  f"training={len(fit_rows)}, same-year={len(same)}, "
                  f"other-year={len(target)}", flush=True)

    fresh = pd.concat(out, ignore_index=True)
    keys = ["source_year", "evaluation", "unique_id"]
    assert fresh[keys].duplicated().sum() == 0
    assert old[keys].duplicated().sum() == 0
    frozen = old[keys + ["outer_fold", "plot_id", "year", "yield",
                         "train_rows", *STABLE_COMPARATORS]].copy()
    joined = fresh.merge(frozen, on=keys, how="left",
                         validate="one_to_one", suffixes=("", "_old"))
    assert len(joined) == 564 and not joined[keys].duplicated().any()
    assert np.allclose(joined["yield"], joined["yield_old"])
    assert (joined["plot_id"] == joined["plot_id_old"]).all()
    assert (joined["year"] == joined["year_old"]).all()
    assert (joined["outer_fold"] == joined["outer_fold_old"]).all()
    assert (joined["train_rows"] == joined["train_rows_old"]).all()
    assert joined[list(STABLE_MODELS + STABLE_COMPARATORS)].notna().all().all()
    joined = joined.drop(columns=["yield_old", "plot_id_old", "year_old",
                                   "outer_fold_old", "train_rows_old"])

    all_models = STABLE_MODELS + STABLE_COMPARATORS
    summary = matched_scores(joined, model_names=all_models)
    diagnostics = matched_bootstrap(joined, n_bootstrap=900, model_names=all_models)
    # Plot-cluster paired comparison on cross-year target observations only.
    rng = np.random.default_rng(2026)
    differences = []
    for src in (2019, 2020):
        section = joined.loc[
            joined.source_year.eq(src) & joined.evaluation.eq("cross_year")]
        labels = section["yield"].to_numpy(float)
        plot_values = section.plot_id.to_numpy()
        plot_set = np.unique(plot_values)
        by_plot = {p: np.flatnonzero(plot_values == p) for p in plot_set}
        batches = [np.concatenate([by_plot[p] for p in rng.choice(
            plot_set, len(plot_set), replace=True)]) for _ in range(900)]
        for model in STABLE_MODELS:
            candidate = section[model].to_numpy(float)
            for comparator in ("ridge_controls", "ridge_controls_satellites"):
                baseline = section[comparator].to_numpy(float)
                values = [score(labels[ix], baseline[ix])["RMSE"] -
                          score(labels[ix], candidate[ix])["RMSE"]
                          for ix in batches]
                lo, hi = np.quantile(values, [0.025, 0.975])
                differences.append({
                    "source_year": src, "target_year": 2020 if src == 2019 else 2019,
                    "reference": comparator, "candidate": model,
                    "reference_RMSE_minus_candidate_RMSE": (
                        score(labels, baseline)["RMSE"] -
                        score(labels, candidate)["RMSE"]),
                    "conditional_plot_bootstrap_low": float(lo),
                    "conditional_plot_bootstrap_high": float(hi),
                })

    joined.to_csv(OUT / "cross_year_stable_eml_predictions.csv", index=False)
    summary.to_csv(OUT / "cross_year_stable_eml_comparison.csv", index=False)
    diagnostics.to_csv(OUT / "cross_year_stable_eml_transfer_deltas.csv", index=False)
    pd.DataFrame(differences).to_csv(
        OUT / "cross_year_stable_eml_paired_baselines.csv", index=False)
    (OUT / "cross_year_stable_eml_nested_choices.json").write_text(
        json.dumps(choices, indent=2) + "\n")
    (OUT / "cross_year_stable_eml_configuration.json").write_text(json.dumps({
        "official_val_labels_scored": False,
        "physical_plots_disjoint_per_fold": True,
        "train_years": [2019, 2020],
        "same_heldout_plot_ids_both_years": True,
        "source_year_training_rows": "Same exact 111-114 rows as cross_year_matched_predictions.csv",
        "feature_selection": "S1+S2 40, or compact20 from four predefined per-stage indices; no Landsat-8",
        "predeclared_sensor_predictors": list(S1S2),
        "predeclared_compact_predictors": list(STABLE20),
        "tuned_within_source_training_only": True,
        "ridge_alpha_grid": [10, 100, 300, 1000, 3000],
        "pca_components_grid": [3, 6, 10],
        "pca_alpha_grid": [10, 100, 1000],
        "pca_fitted_inside_each_training_fold": True,
        "eml_penalty_lambdas": [0.03, 0.08, 0.15, 0.30, 0.60],
        "cautions": [
            "Retrospective satellite windows use realized harvest date, not a live forecast",
            "Model families are selected following earlier TRAIN-only analyses; no independent external test",
            "Plot-cluster uncertainty conditions on fixed fitted predictions, not model-refits",
            "Two years only; pooled/same-year versus cross-year outcomes have different dispersion",
            "No method selection using withheld plots or the official validation labels",
        ],
    }, indent=2) + "\n")
    print("\nSTABLE EML MATCHED RMSE", flush=True)
    print(summary.pivot_table(index="model", columns=["source_year", "evaluation"],
                              values="RMSE").round(2).to_string(), flush=True)
    print("\nSTABLE EML TRANSFER COST", flush=True)
    print(diagnostics[diagnostics.model.isin(STABLE_MODELS)][
        ["source_year", "model", "cross_minus_within_RMSE",
         "conditional_plot_bootstrap_low",
         "conditional_plot_bootstrap_high"]].round(2).to_string(
             index=False), flush=True)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / "outputs/features/full_stage_features.csv",
                      dtype={"plot_id": str, "unique_id": str})
    train = raw.loc[raw.split.eq("train")].copy()
    validation = raw.loc[raw.split.eq("val")]
    assert len(train) == 282 and len(validation) == 37
    assert len(set(train.plot_id) & set(validation.plot_id)) == 0
    assert sorted(train.year.unique().tolist()) == [2019, 2020]
    assert train.unique_id.is_unique

    saved = []
    choices = []
    fold_info = []
    for source, target in ((2019, 2020), (2020, 2019)):
        target_ids = set(train.loc[train.year.eq(target)].unique_id)
        for fold, source_rows, target_rows in transfer_folds(train, source, target):
            row = target_rows[["unique_id", "plot_id", "yield", "year",
                               "season", "variety"]].copy()
            row["source_year"] = source
            row["target_year"] = target
            row["outer_fold"] = fold
            row["train_source_rows"] = len(source_rows)
            row["train_distinct_plots"] = source_rows.plot_id.nunique()
            row["test_distinct_plots"] = target_rows.plot_id.nunique()
            row["train_mean"] = source_rows["yield"].mean()
            fold_info.append({
                "source_year": source, "target_year": target,
                "outer_fold": fold, "train_rows": len(source_rows),
                "train_distinct_plots": source_rows.plot_id.nunique(),
                "test_rows": len(target_rows),
                "test_distinct_plots": target_rows.plot_id.nunique(),
                "train_test_plot_overlap": 0,
            })
            for spec in BASELINES:
                pred, tuned = fit_sklearn(source_rows, target_rows, spec)
                row[spec.name] = pred
                choices.append({"source_year": source, "target_year": target,
                                "outer_fold": fold, "model": spec.name,
                                "tuning": tuned})
            for ex in PENALIZED:
                pred, tuned = fit_penalized(source_rows, target_rows, ex)
                row[ex.name] = pred
                choices.append({"source_year": source, "target_year": target,
                                "outer_fold": fold, "model": ex.name,
                                "tuning": tuned})
            assert row[list(MODEL_NAMES)].notna().all().all()
            saved.append(row)
            print(f"FOLD {source}->{target} {fold}/5, "
                  f"training rows={len(source_rows)}, target rows={len(target_rows)}, "
                  "RMSE: " + ", ".join(
                      f"{name}={score(row['yield'], row[name])['RMSE']:.1f}"
                      for name in MODEL_NAMES), flush=True)

        this_direction = pd.concat([x for x in saved
                                    if (x.target_year == target).all()],
                                   ignore_index=True)
        assert set(this_direction.unique_id) == target_ids
        assert this_direction.unique_id.is_unique
        direction_scores = [(name, score(this_direction["yield"],
                                         this_direction[name])["RMSE"])
                            for name in MODEL_NAMES]
        print(f"DIRECTION {source}->{target} held-out new-plot scores:",
              direction_scores, flush=True)

    prediction = pd.concat(saved, ignore_index=True)
    assert len(prediction) == 282
    assert prediction.unique_id.is_unique
    assert set(prediction.unique_id) == set(train.unique_id)
    assert not set(prediction.plot_id) & set(validation.plot_id)
    prediction.to_csv(OUT / "cross_year_plot_disjoint_predictions.csv",
                      index=False)
    pd.DataFrame(fold_info).to_csv(OUT / "cross_year_fold_audit.csv", index=False)
    (OUT / "cross_year_nested_choices.json").write_text(
        json.dumps(choices, indent=2) + "\n")
    summary = []
    for yr in (2019, 2020, "pooled"):
        subset = prediction if yr == "pooled" else prediction.loc[
            prediction.target_year.eq(yr)]
        for model in MODEL_NAMES:
            summary.append({
                "target_year": str(yr), "source_year": ("other_year" if yr != "pooled"
                                                      else "both_directions"),
                "model": model, "test_plot_seasons": len(subset),
                "test_distinct_physical_plots": subset.plot_id.nunique(),
                **score(subset["yield"], subset[model]),
            })
    comparison = pd.DataFrame(summary)
    comparison.to_csv(OUT / "cross_year_grouped_comparison.csv", index=False)

    # Pooled paired plot-cluster resampling uses the same physical plot
    # as a single cluster across 2019 and 2020, preserving within-plot
    # correlation (but it does NOT refit models or correct for search).
    rng = np.random.default_rng(2026)
    plot_groups = np.unique(prediction.plot_id)
    mapping = {g: np.flatnonzero(prediction.plot_id.to_numpy() == g)
               for g in plot_groups}
    iy = prediction["yield"].to_numpy(float)
    paired = []
    for baseline, candidate in [
        ("ridge_controls", "nonlinear_adaptive_lasso"),
        ("ridge_controls", "nonlinear_SCAD_LLA"),
        ("ridge_controls", "nonlinear_partial_elastic_net"),
        ("extratrees_controls", "extratrees_controls_satellites"),
        ("ridge_controls", "extratrees_controls_satellites"),
    ]:
        ref = prediction[baseline].to_numpy(float)
        alt = prediction[candidate].to_numpy(float)
        gains = []
        for _ in range(1200):
            indices = np.concatenate([
                mapping[p] for p in rng.choice(plot_groups, len(plot_groups),
                                                replace=True)])
            gains.append(score(iy[indices], ref[indices])["RMSE"] -
                         score(iy[indices], alt[indices])["RMSE"])
        lower, upper = np.quantile(gains, [.025, .975])
        paired.append({"baseline": baseline, "candidate": candidate,
                       "baseline_rmse_minus_candidate_rmse":
                       score(iy, ref)["RMSE"] - score(iy, alt)["RMSE"],
                       "conditional_cluster_bootstrap_low": float(lower),
                       "conditional_cluster_bootstrap_high": float(upper)})
    pd.DataFrame(paired).to_csv(OUT / "cross_year_paired_deltas.csv", index=False)

    (OUT / "cross_year_configuration.json").write_text(json.dumps({
        "official_validation_split_used": False,
        "source": "official TRAIN only (2019 and 2020)",
        "distinct_physical_plots": 117,
        "cross_year_directions": ["2019->2020", "2020->2019"],
        "outer": "5 physically disjoint target-year plot groups, with heldout plots purged from source year",
        "inner": "3-fold GroupKFold within source-year remaining plots",
        "year_feature_removed_from_all_models": True,
        "year_by_variety_interaction_removed_from_all_models": True,
        "agronomic_controls": ["area", "season", "variety"],
        "retrospective_warning": "Satellite inputs use windows normalized to actual sow-to-harvest period. This is NOT a real-time forecast.",
        "interpretation": "2019->2020 is forward-year and new-plot transfer; 2020->2019 is reverse-year diagnostic only.",
        "limitations": [
            "Each fold trains on roughly 90-115 labels; scores not comparable as like-for-like training-size controls to earlier 282-row pooled-year results.",
            "Models were chosen after exploratory analyses of the same TRAIN plots; do not call this independent out-of-project validation.",
            "Plot-cluster bootstrap intervals reuse the fitted out-of-fold predictions; not model-refit or multiple-testing adjusted.",
            "Only two years; no third-year extrapolation evidence.",
        ]
    }, indent=2) + "\n")
    print("\nFINAL CROSS-YEAR SUMMARY", flush=True)
    print(comparison.pivot(index="model", columns="target_year",
                           values="RMSE").round(2).to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-diagnostics", action="store_true",
                        help="Same training plots, held-out plots in both years, plus year-wise data-shift audits")
    parser.add_argument("--stable-eml", action="store_true",
                        help="Five fixed reduced-sensor EML candidates on matching within/cross-year plot folds")
    args = parser.parse_args()
    if args.matched_diagnostics and args.stable_eml:
        parser.error("Choose at most one experiment")
    if args.stable_eml:
        run_stable_eml()
    elif args.matched_diagnostics:
        run_matched_diagnostics()
    else:
        run()
