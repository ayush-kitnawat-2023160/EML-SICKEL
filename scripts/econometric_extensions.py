#!/usr/bin/env python3
"""EML extension: incremental satellite information beyond agronomic covariates.

Controls here are pre-harvest observable: year, planting-season category,
variety and surveyed plot area (if recorded accurately). RIVER_PART has no
nonmissing values in yield-labelled subset, so is not usable as a control.

Train-only plot-grouped outer fivefold and nested inner threefold Ridge tuning.
No formal causal identification is claimed.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from extract_features import FEATURES
from stage_importance_baseline import ROOT, STAGES, metrics, bootstrap_delta


def run():
    df=pd.read_csv(ROOT/'outputs'/'features'/'full_stage_features.csv',dtype={'plot_id':str,'unique_id':str})
    train=df[df.split.eq('train')].reset_index(drop=True)
    labels=train['yield'].to_numpy(float);plots=train.plot_id.to_numpy()
    folds=list(GroupKFold(n_splits=5).split(train,labels,groups=plots))
    satell=[f'{w}__{feat}' for w in STAGES for fs in FEATURES.values() for feat in fs]
    controls_num=['area','year'];controls_cat=['season','variety']
    assert set(satell+controls_num+controls_cat)<=set(train.columns)
    rows=[];saved={}

    specs={
        'survey_controls_only':(controls_num,controls_cat),
        'satellite_only':(satell,[]),
        'controls_plus_satellite':(controls_num+satell,controls_cat),
    }
    for name,(nums,cats) in specs.items():
        # Cast all numeric columns to float / categories to string with missing
        # sentinel; imputation/normalization and one-hot fit strictly inside CV.
        frame=train[nums+cats].copy()
        for col in nums:frame[col]=pd.to_numeric(frame[col],errors='coerce')
        for col in cats:frame[col]=frame[col].fillna('__unknown__').astype(str)
        steps=[]
        if nums:
            steps.append(('numeric',Pipeline([
                ('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
                ('scale',StandardScaler())]),nums))
        if cats:
            steps.append(('category',OneHotEncoder(handle_unknown='ignore'),cats))
        preprocess=ColumnTransformer(steps,remainder='drop')
        pred=np.full(len(train),np.nan)
        chosen=[]
        for tr,te in folds:
            model=Pipeline([('preprocess',preprocess),('ridge',Ridge())])
            fit=GridSearchCV(model,{'ridge__alpha':[0.1,1,10,100,1000]},
                             cv=GroupKFold(n_splits=3),scoring='neg_mean_squared_error',n_jobs=1)
            fit.fit(frame.iloc[tr],labels[tr],groups=plots[tr])
            pred[te]=fit.predict(frame.iloc[te]);chosen.append(float(fit.best_params_['ridge__alpha']))
        assert np.isfinite(pred).all()
        row={'model':name,'num_original_numeric':len(nums),'num_categorical':len(cats),
             'tuning_alpha_per_fold':json.dumps(chosen),**metrics(labels,pred)}
        saved[name]=pred;rows.append(row)
        print(name,row['RMSE'],row['R2'],flush=True)
    out=ROOT/'outputs'/'robustness';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'incremental_satellite_value.csv',index=False)
    pd.DataFrame({'unique_id':train.unique_id,'plot_id':train.plot_id,'yield':labels,**saved}).to_csv(
        out/'incremental_satellite_oof.csv',index=False)
    comparisons=[]
    for a,b in [('survey_controls_only','controls_plus_satellite'),('satellite_only','controls_plus_satellite')]:
        # positive means the full combined model improves OOF error
        delta=metrics(labels,saved[a])['RMSE']-metrics(labels,saved[b])['RMSE']
        low,high=bootstrap_delta(labels,saved[b],saved[a],plots,iterations=1000)
        comparisons.append({'reference':a,'combined':'controls_plus_satellite',
                            'RMSE_ref_minus_combined':delta,'conditional_bootstrap_low':low,
                            'conditional_bootstrap_high':high})
    pd.DataFrame(comparisons).to_csv(out/'incremental_satellite_value_deltas.csv',index=False)
    (out/'incremental_satellite_value.summary.json').write_text(json.dumps({
        'results':rows,'paired_comparisons':comparisons,
        'limitations':['Year, season, area and variety may encode management/geography, not causal factors.',
                       'RIVER_PART completely missing in labelled paddy subset.',
                       'This is out-of-plot associative predictive improvement, NOT causal effects.',
                       'Conditional bootstrap uses frozen grouped OOF predictions, does not refit model.']},indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(8,4.5));ax.bar([r['model'].replace('_','\n') for r in rows],[r['RMSE'] for r in rows])
    ax.set_ylabel('Grouped CV RMSE (reported-yield units)')
    ax.set_title('Do satellite features add information beyond agronomic controls?')
    fig.tight_layout();fig.savefig(out/'incremental_satellite_value.png',dpi=170);plt.close(fig)


if __name__=='__main__':run()
