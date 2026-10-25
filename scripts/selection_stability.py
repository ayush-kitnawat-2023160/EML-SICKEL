#!/usr/bin/env python3
"""Repeated plot-group subsampling for LASSO satellite-feature stability.

Fit/tune only on the official SICKLE train plot-seasons. Coefficient frequency
is associative and sensitive to correlated substitute features.
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
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from extract_features import FEATURES
from stage_importance_baseline import ROOT, STAGES


def run(repeats,seed):
    df=pd.read_csv(ROOT/"outputs"/"features"/"full_stage_features.csv",dtype={"plot_id":str})
    train=df.loc[df.split.eq('train')].reset_index(drop=True)
    cols=[f"{w}__{feature}" for w in STAGES for fs in FEATURES.values() for feature in fs]
    assert len(cols)==55
    X=train[cols];y=train['yield'].to_numpy(float);groups=train.plot_id.to_numpy()
    counts=np.zeros(len(cols),dtype=int)
    subs=GroupShuffleSplit(n_splits=repeats,train_size=.8,random_state=seed)
    for i,(train_idx,_) in enumerate(subs.split(X,y,groups)):
        model=Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=False,keep_empty_features=True)),
                        ("scale",StandardScaler()),("lasso",Lasso(max_iter=20000))])
        fit=GridSearchCV(model,{"lasso__alpha":[.1,1,5,10,25,50,100,250]},
                         cv=GroupKFold(n_splits=3),scoring="neg_mean_squared_error",n_jobs=1)
        fit.fit(X.iloc[train_idx],y[train_idx],groups=groups[train_idx])
        # Only 55 spectral/radar values; no missing-scene indicator variables.
        beta=fit.best_estimator_.named_steps["lasso"].coef_[:len(cols)]
        counts+=(np.abs(beta)>1e-5).astype(int)
        if (i+1)%10==0:print(f"Selection resamples: {i+1}/{repeats}",flush=True)
    table=pd.DataFrame({"feature":cols,"window":[c.split("__")[0] for c in cols],
                        "sensor":[c.split("__")[1].split("_")[0] for c in cols],
                        "selected_runs":counts,"selection_frequency":counts/repeats})
    table=table.sort_values("selection_frequency",ascending=False)
    by_window=table.groupby("window").agg(mean_feature_selection_frequency=("selection_frequency","mean"),
                                           features=("feature","count"),
                                           always_selected=("selection_frequency",lambda x:int((x>.75).sum())))
    output=ROOT/"outputs"/"models"
    table.to_csv(output/"full_lasso_selection_frequencies.csv",index=False)
    by_window.to_csv(output/"full_lasso_stage_selection_summary.csv")
    pivot=table.pivot_table(index="sensor",columns="window",values="selection_frequency",aggfunc="mean").reindex(columns=STAGES)
    fig,ax=plt.subplots(figsize=(9,3.8))
    img=ax.imshow(pivot.fillna(0).to_numpy(),vmin=0,vmax=1,aspect="auto",cmap="viridis")
    ax.set_xticks(range(len(STAGES)),["Early","Early-mid","Middle","Mid-late","Late"])
    ax.set_yticks(range(len(pivot.index)),pivot.index.tolist())
    ax.set_title(f"LASSO selection frequency by sensor and period ({repeats} grouped subsamples)")
    fig.colorbar(img,ax=ax,label="Mean feature selection frequency")
    fig.tight_layout();fig.savefig(output/"full_lasso_selection_heatmap.png",dpi=170);plt.close(fig)
    summary={"repeats":repeats,"seed":seed,"train_groups":len(set(groups)),"features":len(cols),
             "top_10_features":table.head(10).to_dict(orient="records"),
             "interpretation":"Selection frequency is not causal effect or formal statistical significance. Correlated features substitute for one another."}
    (output/"full_lasso_selection_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(table.head(12).to_string(index=False))
    print("Window mean frequencies:\n",by_window.to_string())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeats",type=int,default=40)
    p.add_argument("--seed",type=int,default=2026)
    a=p.parse_args();run(a.repeats,a.seed)
