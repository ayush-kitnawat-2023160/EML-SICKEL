#!/usr/bin/env python3
"""Repeat plot-grouped nested CV across shuffled plot partitions.

This is a secondary robustness analysis of models explored on the TRAIN data,
not a genuinely untouched confirmatory test. No official VAL labels are scored.
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from improve_yield_models import (ROOT, OUT, SPECS, engineered, grouped_oof,
                                  paired_bootstrap, score)

CHOICES = ("original_controls_ridge", "controls_year_variety",
           "controls_extratrees", "controls_extratrees_full_sat")


def run(seeds):
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / "outputs/features/full_stage_features.csv",
                      dtype={"unique_id": str, "plot_id": str})
    train = raw.loc[raw.split.eq("train")].reset_index(drop=True)
    assert train.shape[0] == 282 and raw.loc[raw.split.eq("val")].shape[0] == 37
    x = engineered(train)
    y = train["yield"].to_numpy(float)
    g = train["plot_id"].to_numpy()
    specs = {s.name: s for s in SPECS}
    rows = []
    changes = []
    for seed in seeds:
        print(f"--- shuffled GroupKFold random seed {seed} ---", flush=True)
        outer = list(GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(x, y, g))
        for tr, te in outer:
            assert not set(g[tr]) & set(g[te])
        result = {}
        for name in CHOICES:
            pred, best = grouped_oof(x, y, g, specs[name], outer)
            result[name] = pred
            item = {"seed": seed, "model": name, **score(y, pred)}
            rows.append(item)
            print(name, f"RMSE={item['RMSE']:.2f} | R2={item['R2']:.3f}", flush=True)
        ref = result["original_controls_ridge"]
        tree = result["controls_extratrees"]
        sat_tree = result["controls_extratrees_full_sat"]
        for base_name, candidate_name, base, cand in [
            ("original_controls_ridge", "controls_extratrees", ref, tree),
            ("original_controls_ridge", "controls_extratrees_full_sat", ref, sat_tree),
            ("controls_extratrees", "controls_extratrees_full_sat", tree, sat_tree),
        ]:
            changes.append({"seed": seed, "baseline": base_name, "candidate": candidate_name,
                            "rmse_improvement": score(y, base)["RMSE"] - score(y, cand)["RMSE"]})
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "shuffled_grouped_cv.csv", index=False)
    pd.DataFrame(changes).to_csv(OUT / "shuffled_grouped_cv_deltas.csv", index=False)
    stats = results.groupby("model", as_index=False).agg(
        mean_RMSE=("RMSE", "mean"), sd_RMSE=("RMSE", "std"),
        mean_MAE=("MAE", "mean"), mean_R2=("R2", "mean"))
    stats.to_csv(OUT / "shuffled_grouped_cv_summary.csv", index=False)
    (OUT / "shuffled_grouped_cv_summary.json").write_text(json.dumps({
        "seeds": seeds, "train_rows": len(train), "train_physical_plots": train.plot_id.nunique(),
        "official_val_scored": False, "outer_split": "GroupKFold(n_splits=5,shuffle=True)",
        "inner_split": "GroupKFold(n_splits=3)",
        "caveat": "These models and robustness seeds were explored after earlier TRAIN grouped CV; not independent holdout confirmation."
    }, indent=2) + "\n")
    print("--- aggregate ---")
    print(stats.sort_values("mean_RMSE").to_string(index=False), flush=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    order = stats.sort_values("mean_RMSE")
    ax.barh(order.model[::-1], order.mean_RMSE[::-1],
            xerr=order.sd_RMSE.fillna(0).to_numpy()[::-1], capsize=3)
    ax.set(xlabel="Mean shuffled grouped-CV RMSE ± split SD",
           title="Exploratory model robustness across plot-group partitions")
    fig.tight_layout()
    fig.savefig(OUT / "shuffled_grouped_cv.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 37, 79])
    args = parser.parse_args()
    run(args.seeds)
