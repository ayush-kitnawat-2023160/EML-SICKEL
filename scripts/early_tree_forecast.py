#!/usr/bin/env python3
"""Early satellite forecasts using Extra Trees, with no realized harvest data.

Only date-cutoff-safe aggregated scene features from extended_experiments.early_matrix
are supplied to the learner. Nested tuning and evaluation group whole PLOT_IDs.
All experiments only use released TRAIN labels.
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from extended_experiments import early_matrix
from improve_yield_models import (ROOT, OUT, Spec, engineered, grouped_oof,
                                  paired_bootstrap, score)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / "outputs/features/full_stage_features.csv",
                        dtype={"unique_id": str, "plot_id": str})
    obs = pd.read_csv(ROOT / "outputs/features/full_observation_features.csv",
                      dtype={"unique_id": str, "plot_id": str}, low_memory=False)
    train = frame.loc[frame.split.eq("train")].reset_index(drop=True)
    val = frame.loc[frame.split.eq("val")]
    assert train.shape[0] == 282 and val.shape[0] == 37
    assert not set(train.plot_id) & set(val.plot_id)
    lengths = (pd.to_datetime(train.harvesting_date) -
               pd.to_datetime(train.sowing_date)).dt.days
    assert lengths.min() > 90
    y = train["yield"].to_numpy(float)
    groups = train.plot_id.to_numpy()
    folds = list(GroupKFold(n_splits=5).split(train, y, groups))
    baseline = pd.read_csv(OUT / "candidate_oof_predictions.csv",
                           dtype={"unique_id": str, "plot_id": str})
    assert baseline.unique_id.tolist() == train.unique_id.tolist()
    reference = baseline["controls_extratrees"].to_numpy(float)
    results = [{"cutoff_days": 0, "model": "ExtraTrees_sowing_controls_only",
                "num_satellite_features": 0, **score(y, reference)}]
    preds = {"unique_id": train.unique_id, "plot_id": groups, "yield": y,
             "controls_only": reference}
    choices = {}
    for cutoff in [15, 30, 45, 60, 75, 90]:
        scene = early_matrix(obs, train, cutoff)
        assert scene.shape[0] == len(train)
        # No normalized relative time or harvest date, only remote-sensing values
        # observed by this number of days after the known sowing date.
        cols = scene.columns.tolist()
        assert len(cols) > 0
        assert not any("harvest" in c or "relative_season" in c for c in cols)
        x = engineered(pd.concat([train.reset_index(drop=True),
                                  scene.reset_index(drop=True)], axis=1))
        spec = Spec(f"ExtraTrees_satellite_by_day_{cutoff}",
                    ("area", "year", *cols), ("season", "variety"),
                    "extratrees")
        predicted, tune = grouped_oof(x, y, groups, spec, folds)
        choices[str(cutoff)] = tune
        preds[f"day_{cutoff}"] = predicted
        row = {"cutoff_days": cutoff, "model": spec.name,
               "num_satellite_features": len(cols), **score(y, predicted)}
        row["rmse_improvement_vs_controls_only"] = (
            score(y, reference)["RMSE"] - row["RMSE"])
        interval = paired_bootstrap(y, groups, reference, predicted)
        row["conditional_plot_bootstrap_low"] = interval[0]
        row["conditional_plot_bootstrap_high"] = interval[1]
        results.append(row)
        print("Day", cutoff, row, flush=True)

    pd.DataFrame(results).to_csv(OUT / "early_extratrees_cv.csv", index=False)
    pd.DataFrame(preds).to_csv(OUT / "early_extratrees_oof_predictions.csv", index=False)
    (OUT / "early_extratrees_cv_summary.json").write_text(json.dumps({
        "train_only": True, "official_val_scored": False,
        "cutoffs": [15, 30, 45, 60, 75, 90],
        "baseline": "controls-only ExtraTrees as tuned and evaluated on identical five GroupKFold folds",
        "cutoff_invariants": "All scene observation dates at or before cutoff; cutoff < min growing-season length; actual harvest dates are not model predictors.",
        "hyperparameters_by_outer_fold": choices,
        "limitations": [
            "Exploratory multiple-cutoff search after earlier TRAIN CV; apparent best cutoff can be optimistic.",
            "Reported yield units and physical plot-mask alignment remain limitations.",
            "Conditional plot bootstrap does not refit models and is not multiple-testing corrected."
        ]}, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    r = pd.DataFrame(results)
    ax.axhline(r.loc[r.cutoff_days.eq(0), "RMSE"].iloc[0], color="black",
               ls="--", label="Extra Trees: controls at sowing")
    sub = r[r.cutoff_days.gt(0)]
    ax.plot(sub.cutoff_days, sub.RMSE, "o-", label="Extra Trees: controls + available satellite")
    ax.set(xlabel="Days after sowing", ylabel="Grouped-CV RMSE",
           title="Exploratory cutoff-safe early forecasting with Extra Trees")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "early_extratrees_cv.png", dpi=160)
    plt.close(fig)
    print(r.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
