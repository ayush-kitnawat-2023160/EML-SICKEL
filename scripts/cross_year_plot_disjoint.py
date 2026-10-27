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

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold

from eml_penalty_experiments import (OUT, ROOT, Experiment, PreparedFit, tune)
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
    run()
