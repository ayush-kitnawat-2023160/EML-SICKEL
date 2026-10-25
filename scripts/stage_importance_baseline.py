#!/usr/bin/env python3
"""Grouped, nested-CV Ridge baseline for SICKLE stage importance.

Uses ONLY satellite-derived features. The released validation split remains
untouched during model selection. Retrospective growing-season windows use the
true harvest date; these results are NOT real-time early-season forecasts.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from extract_features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["W1_0_20", "W2_20_40", "W3_40_60", "W4_60_80", "W5_80_100"]
ALPHAS = [0.1, 1, 10, 100, 1000]


def pipeline():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=False, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge()),
    ])


def metrics(truth, pred):
    return {"RMSE": float(np.sqrt(mean_squared_error(truth,pred))),
            "MAE": float(mean_absolute_error(truth,pred)),
            "R2": float(r2_score(truth,pred))}


def bootstrap_delta(y,full,without,groups,iterations=500,seed=7):
    """Plot-cluster bootstrap of paired OOF RMSE difference; conditional on fits."""
    rng=np.random.default_rng(seed)
    unique=np.unique(groups)
    indices={g:np.flatnonzero(groups==g) for g in unique}
    deltas=[]
    for _ in range(iterations):
        chosen=rng.choice(unique,len(unique),replace=True)
        selected=np.concatenate([indices[g] for g in chosen])
        deltas.append(metrics(y[selected],without[selected])["RMSE"] -
                      metrics(y[selected],full[selected])["RMSE"])
    return [float(v) for v in np.quantile(deltas,[.025,.975])]


def main(args):
    source=ROOT/"outputs"/"features"/f"{args.dataset}_stage_features.csv"
    df=pd.read_csv(source,dtype={"unique_id":str,"plot_id":str})
    train=df[df.split.eq("train")].reset_index(drop=True)
    val=df[df.split.eq("val")].reset_index(drop=True)
    if args.dataset=="toy":
        print("WARNING: toy subset has too few independent plots for dependable statistical conclusions.")
    assert not set(train.plot_id)&set(val.plot_id),"Plot leakage between predefined splits"
    y=train["yield"].to_numpy(dtype=float)
    group=train.plot_id.to_numpy()
    # Use only spectral/backscatter values, not acquisition-count availability
    # proxies or date/harvest information. Missing data imputed within folds.
    # Construct an exact whitelist. Prefix matching inadvertently admits
    # availability/clear-pixel-count columns; those are *not* spectral features.
    cols=[f"{w}__{feature}" for w in STAGES for feature_group in FEATURES.values()
          for feature in feature_group]
    assert len(cols)==55 and set(cols)<=set(train.columns)
    scenarios={"full":cols}
    for k,w in enumerate(STAGES):
        scenarios[f"only_{w}"]=[c for c in cols if c.startswith(w+"__")]
        scenarios[f"without_{w}"]=[c for c in cols if not c.startswith(w+"__")]
        scenarios[f"cumulative_{k+1}"]=[c for c in cols if any(c.startswith(s+"__") for s in STAGES[:k+1])]
    scenarios={key:value for key,value in scenarios.items() if len(value)>0}
    outer=GroupKFold(n_splits=5)
    folds=list(outer.split(train,y,groups=group))
    predictions={}
    report=[]
    for i,(name,columns) in enumerate(scenarios.items()):
        X=train[columns]
        predicted=np.full(len(train),np.nan)
        fold_scores=[]
        fold_alpha=[]
        for fold_id,(tr,te) in enumerate(folds):
            inner=GroupKFold(n_splits=3)
            fit=GridSearchCV(pipeline(),{"ridge__alpha":ALPHAS},
                             cv=inner,scoring="neg_mean_squared_error",n_jobs=1)
            fit.fit(X.iloc[tr],y[tr],groups=group[tr])
            estimated=fit.predict(X.iloc[te])
            predicted[te]=estimated
            fold_alpha.append(float(fit.best_params_["ridge__alpha"]))
            fold_scores.append(metrics(y[te],estimated))
        assert np.isfinite(predicted).all(), f"Missing OOF predictions in {name}"
        predictions[name]=predicted
        row={"scenario":name,"n_features":len(columns),"plots":int(train.plot_id.nunique()),
             "rows":len(train),**metrics(y,predicted),
             "fold_RMSE_mean":float(np.mean([a["RMSE"] for a in fold_scores])),
             "fold_RMSE_std":float(np.std([a["RMSE"] for a in fold_scores],ddof=1)),
             "alpha_per_fold":fold_alpha}
        report.append(row)
        print(f"{i+1}/{len(scenarios)} {name:25s} RMSE={row['RMSE']:.1f} MAE={row['MAE']:.1f} R2={row['R2']:.3f}",flush=True)
    # Fold-only training-mean benchmark, never using held-out yield labels.
    meanpred=np.full(len(train),np.nan)
    for tr,te in folds:meanpred[te]=np.mean(y[tr])
    predictions["training_fold_mean"]=meanpred
    report.append({"scenario":"training_fold_mean","n_features":0,"rows":len(train),"plots":int(train.plot_id.nunique()),**metrics(y,meanpred)})

    # Pairwise ablation uncertainty, plot-cluster bootstrap of frozen OOF fits.
    ablations=[]
    for stage in STAGES:
        key=f"without_{stage}"
        delta=metrics(y,predictions[key])["RMSE"]-metrics(y,predictions["full"])["RMSE"]
        low,hi=bootstrap_delta(y,predictions["full"],predictions[key],group)
        ablations.append({"stage":stage,"delta_RMSE_without_minus_full":delta,
                          "paired_plot_bootstrap_95pct_CI_low":low,
                          "paired_plot_bootstrap_95pct_CI_high":hi})
    out=ROOT/"outputs"/"models";out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(report).to_csv(out/f"{args.dataset}_ridge_grouped_cv.csv",index=False)
    pd.DataFrame(ablations).to_csv(out/f"{args.dataset}_ridge_stage_ablation.csv",index=False)
    pd.DataFrame({"unique_id":train.unique_id,"plot_id":train.plot_id,"yield":y,**predictions}).to_csv(
        out/f"{args.dataset}_ridge_oof_predictions.csv",index=False)

    # Model chosen via grouped-CV on train only; one validation assessment.
    fitted=GridSearchCV(pipeline(),{"ridge__alpha":ALPHAS},cv=GroupKFold(n_splits=5),
                        scoring="neg_mean_squared_error",n_jobs=1)
    fitted.fit(train[cols],y,groups=group)
    val_result=None
    if len(val):
        val_est=fitted.predict(val[cols])
        val_result={"n":len(val),"n_plots":int(val.plot_id.nunique()),
                    "chosen_ridge_alpha":float(fitted.best_params_["ridge__alpha"]),
                    **metrics(val["yield"],val_est)}
        pd.DataFrame({"unique_id":val.unique_id,"plot_id":val.plot_id,"yield":val["yield"],
                      "predicted_yield":val_est}).to_csv(out/f"{args.dataset}_ridge_holdout_predictions.csv",index=False)
    (out/f"{args.dataset}_ridge_summary.json").write_text(json.dumps({
        "train_rows":len(train),"train_unique_plots":int(train.plot_id.nunique()),
        "validation":val_result,"stage_ablation":ablations,
        "notes":["Descriptive out-of-group predictive association; not causal significance.",
                 "Stage windows use realized harvest date, so cumulative results are NOT leakage-free early forecasts.",
                 "All missing-value imputation, scaling and Ridge-alpha tuning are performed inside folds.",
                 "Bootstrap uncertainty conditions on out-of-fold fitted predictions and does not account for model refitting."]
    },indent=2)+"\n")

    plot_rows={r["scenario"]:r for r in report}
    fig,ax=plt.subplots(figsize=(9,4.8))
    labels=["Early","Early-mid","Mid","Mid-late","Late"]
    heights=[plot_rows[f"only_{stage}"]["RMSE"] for stage in STAGES]
    ax.bar(labels,heights,label="Stage alone")
    ax.axhline(plot_rows["full"]["RMSE"],ls="--",color="black",label="All stages")
    ax.set_ylabel("Out-of-group RMSE (reported yield units)")
    ax.set_title("Stage-only satellite yield prediction | Ridge / grouped CV")
    ax.legend();fig.tight_layout();fig.savefig(out/f"{args.dataset}_ridge_stage_only.png",dpi=170);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.8))
    xx=np.arange(len(STAGES));yy=np.array([r["delta_RMSE_without_minus_full"] for r in ablations]);
    ax.bar(labels,yy);ax.axhline(0,color="black",linewidth=.8)
    ax.set_ylabel("RMSE without stage − RMSE with all stages")
    ax.set_title("Added value of each observation window (positive = helpful)")
    fig.tight_layout();fig.savefig(out/f"{args.dataset}_ridge_ablation.png",dpi=170);plt.close(fig)
    print("Baseline mean RMSE",round(metrics(y,meanpred)["RMSE"],1))
    print("Official untouched validation",val_result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=["toy","full"],required=True)
    main(p.parse_args())
