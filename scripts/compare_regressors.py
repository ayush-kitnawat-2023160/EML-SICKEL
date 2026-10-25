#!/usr/bin/env python3
"""Leakage-safe grouped-CV comparison: OLS, Ridge, LASSO, Elastic Net.

This compares all-stage satellite-only yield predictors on the official
training split. Tuning is nested within plot-grouped folds. Validation
is not used to select models, windows, or hyperparameters.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stage_importance_baseline import ROOT, STAGES, metrics
from extract_features import FEATURES


def make_pipeline(estimator):
    return Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=False,keep_empty_features=True)),
                     ("scale",StandardScaler()),("model",estimator)])


def run():
    source=ROOT/"outputs"/"features"/"full_stage_features.csv"
    data=pd.read_csv(source,dtype={"unique_id":str,"plot_id":str})
    train=data[data.split.eq("train")].reset_index(drop=True)
    y=train["yield"].to_numpy(float);groups=train["plot_id"].to_numpy()
    columns=[f"{w}__{feature}" for w in STAGES for feature_group in FEATURES.values()
             for feature in feature_group]
    assert len(columns)==55 and set(columns)<=set(train.columns)
    X=train[columns]
    outer=GroupKFold(n_splits=5)
    models={
        "OLS":(LinearRegression(),None),
        "Ridge":(Ridge(),{"model__alpha":[.1,1,10,100,1000]}),
        "LASSO":(Lasso(max_iter=20000),{"model__alpha":[.1,1,5,10,25,50,100,250]}),
        "ElasticNet":(ElasticNet(max_iter=20000),{"model__alpha":[.1,1,5,10,25,50,100,250],
                                                   "model__l1_ratio":[.2,.5,.8]}),
    }
    output=[]
    preds=pd.DataFrame({"unique_id":train.unique_id,"plot_id":train.plot_id,"yield":y})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",ConvergenceWarning)
        for name,(estimator,grid) in models.items():
            pred=np.full(len(train),np.nan)
            nonzero=[];configs=[]
            for tr,te in outer.split(X,y,groups):
                pipe=make_pipeline(estimator)
                if grid is not None:
                    chosen=GridSearchCV(pipe,grid,cv=GroupKFold(n_splits=3),
                                        scoring="neg_mean_squared_error",n_jobs=1)
                    chosen.fit(X.iloc[tr],y[tr],groups=groups[tr])
                    fitted=chosen.best_estimator_
                    configs.append(chosen.best_params_)
                else:
                    fitted=pipe.fit(X.iloc[tr],y[tr]);configs.append({})
                pred[te]=fitted.predict(X.iloc[te])
                coef=fitted.named_steps["model"].coef_
                nonzero.append(int(np.count_nonzero(np.abs(coef)>1e-7)))
            preds[name]=pred
            result={"model":name,"n_train":len(train),"n_groups":len(set(groups)),
                    "n_satellite_features":len(columns),**metrics(y,pred),
                    "nonzero_coefficients_mean":float(np.mean(nonzero)),
                    "hyperparameters_outer_folds":json.dumps(configs)}
            output.append(result)
            print(name,{k:result[k] for k in ("RMSE","MAE","R2","nonzero_coefficients_mean")},flush=True)
        conv=sum(issubclass(w.category,ConvergenceWarning) for w in caught)
    out=ROOT/"outputs"/"models";out.mkdir(exist_ok=True)
    pd.DataFrame(output).to_csv(out/"full_regressor_comparison.csv",index=False)
    preds.to_csv(out/"full_regressor_oof_predictions.csv",index=False)
    (out/"full_regressor_comparison.summary.json").write_text(json.dumps({
        "model_comparison":output,"convergence_warnings":conv,
        "notes":["Only released TRAIN plot-seasons; official VAL held out from model selection.",
                 "Every outer evaluation fold holds out entire PLOT_IDs.",
                 "Hyperparameter tuning is within each training fold using inner GroupKFold.",
                 "No causal claim; no use of actual future harvest date for forecasting claim."]
    },indent=2)+"\n")


if __name__=="__main__":run()
