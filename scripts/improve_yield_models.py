#!/usr/bin/env python3
"""Exploratory improvements over controls-only Ridge with plot-grouped nested CV.

The released TRAIN plot-seasons alone are used for model search and scoring.
The official VAL labels are never loaded into an estimator or score here.
Feature transformations learned from rows (imputation, scaling, one-hot,
hyperparameter choices) are fit only on the appropriate training folds.

Outputs: outputs/improvements/*.csv, *.json and *.png

To run: ~/miniconda3/envs/global/bin/python scripts/improve_yield_models.py
"""

from dataclasses import dataclass
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "improvements"
WINDOWS = ("W1_0_20", "W2_20_40", "W3_40_60", "W4_60_80", "W5_80_100")
S1 = ("s1_vv_db", "s1_vh_db", "s1_vv_minus_vh_db")
S2 = ("s2_ndvi", "s2_ndmi", "s2_ndre", "s2_red_reflectance", "s2_nir_reflectance")
L8 = ("l8_ndvi", "l8_red_reflectance", "l8_nir_reflectance")
SAT = [f"{w}__{f}" for w in WINDOWS for f in (*S1, *S2, *L8)]
EARLY_SAT = [f"{w}__{f}" for w in (WINDOWS[0], WINDOWS[-1]) for f in (*S1, *S2)]


def engineered(raw):
    """Only row-local transformations of agronomic features, never outcomes."""
    x = raw.copy()
    x["area"] = pd.to_numeric(x["area"], errors="coerce")
    x["year"] = pd.to_numeric(x["year"], errors="coerce")
    x["year_category"] = x["year"].fillna(-1).astype(int).astype(str)
    x["variety"] = x["variety"].fillna("UNKNOWN").astype(str)
    x["season"] = x["season"].fillna("UNKNOWN").astype(str)
    x["year_variety"] = x["year_category"] + "__" + x["variety"]
    x["year_season"] = x["year_category"] + "__" + x["season"]
    x["area_class"] = x["area"].fillna(-1).astype(str)
    # Saturation is plausible for field area, but do not fit a transform on y.
    x["area_squared"] = x["area"] ** 2
    return x


@dataclass(frozen=True)
class Spec:
    name: str
    numeric: tuple
    categorical: tuple
    family: str = "ridge"
    satellite_scale: float = 1.0


SPECS = [
    Spec("original_controls_ridge", ("area", "year"), ("season", "variety")),
    Spec("controls_year_variety", ("area", "year"), ("season", "variety", "year_variety")),
    Spec("controls_year_season", ("area", "year"), ("season", "variety", "year_season")),
    Spec("controls_both_interactions", ("area", "year"), ("season", "variety", "year_variety", "year_season")),
    Spec("controls_nonlinear_area", ("year",), ("season", "variety", "area_class")),
    Spec("controls_interactions_area", ("year",), ("season", "variety", "year_variety", "area_class")),
    Spec("controls_interactions_early_sat", ("area", "year", *EARLY_SAT), ("season", "variety", "year_variety"), satellite_scale=.3),
    Spec("controls_interactions_full_sat", ("area", "year", *SAT), ("season", "variety", "year_variety"), satellite_scale=.2),
    Spec("controls_extratrees", ("area", "year"), ("season", "variety"), "extratrees"),
    Spec("controls_histboost", ("area", "year"), ("season", "variety"), "histboost"),
    Spec("controls_extratrees_early_sat", ("area", "year", *EARLY_SAT), ("season", "variety"), "extratrees"),
    Spec("controls_extratrees_full_sat", ("area", "year", *SAT), ("season", "variety"), "extratrees"),
]


def pipeline_and_grid(spec):
    non_sat = [c for c in spec.numeric if c not in SAT]
    sat = [c for c in spec.numeric if c in SAT]
    transform = []
    if non_sat:
        transform.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler())]), non_sat))
    if sat:
        transform.append(("sat", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler())]), sat))
    if spec.categorical:
        transform.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                          list(spec.categorical)))
    weights = {"sat": spec.satellite_scale} if sat else None
    pre = ColumnTransformer(transform, transformer_weights=weights, sparse_threshold=0)
    if spec.family == "ridge":
        model = Ridge()
        # Additional shrinkage may be needed for interactions and numerous satellite inputs.
        grid = {"model__alpha": [1.0, 10.0, 30.0, 100.0, 300.0, 1000.0]}
    elif spec.family == "extratrees":
        model = ExtraTreesRegressor(n_estimators=180, random_state=2026, n_jobs=2)
        grid = {"model__min_samples_leaf": [2, 5, 10],
                "model__max_features": [0.7, 1.0]}
    elif spec.family == "histboost":
        model = HistGradientBoostingRegressor(random_state=2026, max_iter=150,
                                               early_stopping=False)
        grid = {"model__max_leaf_nodes": [3, 7],
                "model__min_samples_leaf": [10, 25],
                "model__l2_regularization": [10.0, 100.0]}
    else:
        raise ValueError(spec.family)
    return Pipeline([("pre", pre), ("model", model)]), grid


def score(y, p):
    return {"RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p)),
            "R2": float(r2_score(y, p))}


def grouped_oof(x, y, groups, spec, folds):
    columns = list(spec.numeric + spec.categorical)
    model, grid = pipeline_and_grid(spec)
    pred = np.full(len(x), np.nan)
    best = []
    for tr, te in folds:
        search = GridSearchCV(model, grid, scoring="neg_mean_squared_error",
                              cv=GroupKFold(n_splits=3), n_jobs=1,
                              error_score="raise")
        search.fit(x.iloc[tr][columns], y[tr], groups=groups[tr])
        pred[te] = search.predict(x.iloc[te][columns])
        best.append(search.best_params_)
    assert np.isfinite(pred).all()
    return pred, best


def paired_bootstrap(y, groups, base, candidate, repeats=1500):
    """Conditional on frozen OOF predictions; not model-refit uncertainty."""
    generator = np.random.default_rng(2026)
    unique = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    deltas = []
    for _ in range(repeats):
        ids = generator.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in ids])
        deltas.append(score(y[idx], base[idx])["RMSE"] -
                      score(y[idx], candidate[idx])["RMSE"])
    return list(np.quantile(deltas, [0.025, 0.975]).astype(float))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(ROOT / "outputs/features/full_stage_features.csv",
                         dtype={"unique_id": str, "plot_id": str})
    train = labels.loc[labels.split.eq("train")].reset_index(drop=True)
    reserved = labels.loc[labels.split.eq("val")]
    assert len(train) == 282 and len(reserved) == 37
    assert len(set(train.plot_id) & set(reserved.plot_id)) == 0
    x = engineered(train)
    y = train["yield"].to_numpy(float)
    groups = train["plot_id"].to_numpy()
    folds = list(GroupKFold(n_splits=5).split(x, y, groups))
    rows = []
    predictions = {"unique_id": train.unique_id,
                   "plot_id": train.plot_id, "yield": y}
    configs = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning,
                                message=".*constant.*")
        for spec in SPECS:
            p, best = grouped_oof(x, y, groups, spec, folds)
            predictions[spec.name] = p
            configs[spec.name] = best
            row = {"model": spec.name, "model_family": spec.family,
                   "predictor_count_raw": len(spec.numeric) + len(spec.categorical),
                   **score(y, p)}
            rows.append(row)
            print(spec.name, f"RMSE={row['RMSE']:.3f}, R²={row['R2']:.3f}",
                  flush=True)

    result = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    result.to_csv(OUT / "candidate_grouped_cv.csv", index=False)
    oof = pd.DataFrame(predictions)
    oof.to_csv(OUT / "candidate_oof_predictions.csv", index=False)
    baseline = oof["original_controls_ridge"].to_numpy(float)
    delta = []
    for name in result.model:
        candidate = oof[name].to_numpy(float)
        interval = paired_bootstrap(y, groups, baseline, candidate)
        delta.append({"model": name,
                      "rmse_improvement_vs_original": score(y, baseline)["RMSE"] -
                                                       score(y, candidate)["RMSE"],
                      "conditional_plot_bootstrap_low": interval[0],
                      "conditional_plot_bootstrap_high": interval[1]})
    pd.DataFrame(delta).to_csv(OUT / "paired_vs_original.csv", index=False)
    (OUT / "candidate_details.json").write_text(json.dumps({
        "train_only": True, "train_plot_seasons": len(train),
        "train_physical_plots": int(train.plot_id.nunique()),
        "val_unused": True, "official_val_rows": len(reserved),
        "grouped_outer_folds": 5, "grouped_inner_folds": 3,
        "outer_hyperparameters": configs,
        "caveats": [
            "Exploratory search over multiple model specifications: the best observed outer-CV model is selected post-hoc and its CV score is optimistically selected.",
            "Do not use official VAL for further model search. Previous baseline Ridge did use VAL historically.",
            "Bootstrap confidence limits condition on already-fitted OOF predictions; not multiple-comparison-corrected or full-refit uncertainty.",
            "Area, variety, season, year are associative control features. Satellite windows use the realized harvest date so are retrospective."
        ]}, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(10, 5))
    chosen = result.head(8).iloc[::-1]
    ax.barh(chosen.model, chosen.RMSE)
    ax.axvline(float(result.loc[result.model.eq("original_controls_ridge"), "RMSE"].iloc[0]),
               color="black", linestyle="--", label="Original controls-only Ridge")
    ax.set(xlabel="Grouped-CV RMSE (reported yield units)",
           title="Exploratory model improvements on TRAIN physical plots")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "model_improvement_comparison.png", dpi=160)
    plt.close(fig)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
