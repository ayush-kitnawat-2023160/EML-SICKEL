#!/usr/bin/env python3
"""Secondary robustness check of a day-90 Extra Trees forecast.

Repeat plot-grouped nested CV on TRAIN with alternative shuffled physical
plot folds, using features observed no later than 90 days after sowing.
Since day 90 was chosen after exploratory results, this is NOT external
confirmation or a multiple-hypothesis-corrected estimate.
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from extended_experiments import early_matrix
from improve_yield_models import (ROOT, OUT, SPECS, Spec, engineered,
                                  grouped_oof, score)


def run(seeds):
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / "outputs/features/full_stage_features.csv",
                      dtype={"plot_id": str, "unique_id": str})
    obs = pd.read_csv(ROOT / "outputs/features/full_observation_features.csv",
                      dtype={"plot_id": str, "unique_id": str}, low_memory=False)
    train = raw.loc[raw.split.eq("train")].reset_index(drop=True)
    assert len(train) == 282
    assert not set(train.plot_id) & set(raw.loc[raw.split.eq("val")].plot_id)
    assert ((pd.to_datetime(train.harvesting_date) -
             pd.to_datetime(train.sowing_date)).dt.days > 90).all()
    early = early_matrix(obs, train, 90)
    assert early.shape[0] == len(train)
    x = engineered(pd.concat([train, early], axis=1))
    y = train["yield"].to_numpy(float)
    g = train["plot_id"].to_numpy()
    baseline = next(s for s in SPECS if s.name == "controls_extratrees")
    candidate = Spec("controls_extratrees_day90_sat",
                     ("area", "year", *early.columns), ("season", "variety"),
                     "extratrees")
    results = []
    for seed in seeds:
        folds = list(GroupKFold(n_splits=5, shuffle=True,
                               random_state=seed).split(x, y, g))
        for tr, te in folds:
            assert not set(g[tr]) & set(g[te])
        base_p, _ = grouped_oof(x, y, g, baseline, folds)
        early_p, _ = grouped_oof(x, y, g, candidate, folds)
        ref = score(y, base_p); test = score(y, early_p)
        result = {"seed": seed, "controls_only_RMSE": ref["RMSE"],
                  "day90_satellite_RMSE": test["RMSE"],
                  "improvement": ref["RMSE"] - test["RMSE"],
                  "controls_only_R2": ref["R2"],
                  "day90_satellite_R2": test["R2"]}
        results.append(result)
        print(result, flush=True)
    pd.DataFrame(results).to_csv(OUT / "early_day90_shuffled_group_cv.csv",
                                 index=False)
    (OUT / "early_day90_shuffled_group_cv.summary.json").write_text(json.dumps({
        "seed_list": seeds, "plots_grouped": True,
        "official_val_labels_scored": False,
        "early_feature_definition": "Only observations on or before 90 days after sowing; no future realized harvest timing in predictor features",
        "exploratory_warning": "Day 90 was selected after looking at TRAIN experiments and repeated splits all reuse same TRAIN plots; not independent validation."
    }, indent=2) + "\n")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 37, 79])
    args = parser.parse_args()
    run(args.seeds)
