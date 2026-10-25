#!/usr/bin/env python3
"""How much does *already acquired* imagery add to known agronomic inputs?

Same CV grouping and y for all horizons. Everything is constructed from
acquisitions <= specified days since sowing. No actual harvest date, variety
phenology stage, after-cutoff observation or future yield is a predictor.
"""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from extended_experiments import ROOT, early_matrix, prepare
from stage_importance_baseline import metrics, bootstrap_delta


def evaluate(frame, y, groups, numcols, catcols, folds):
    transformers=[]
    if numcols:
        transformers.append(('numeric',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
                                                  ('scale',StandardScaler())]),numcols))
    if catcols:
        transformers.append(('category',OneHotEncoder(handle_unknown='ignore'),catcols))
    fit=Pipeline([('preprocess',ColumnTransformer(transformers)),('ridge',Ridge())])
    pred=np.full(len(frame),np.nan)
    for tr,te in folds:
        search=GridSearchCV(fit,{'ridge__alpha':[0.1,1,10,100,1000]},
                            cv=GroupKFold(n_splits=3),scoring='neg_mean_squared_error',n_jobs=1)
        search.fit(frame.iloc[tr],y[tr],groups=groups[tr])
        pred[te]=search.predict(frame.iloc[te])
    assert np.isfinite(pred).all()
    return pred


def run():
    stage,obs=prepare()
    train=stage[stage.split.eq('train')].reset_index(drop=True)
    y=train['yield'].to_numpy(float); groups=train.plot_id.to_numpy()
    folds=list(GroupKFold(n_splits=5).split(train,y,groups))
    numcols=['year','area'];catcols=['variety','season']
    meta=train[numcols+catcols].copy()
    for c in numcols:meta[c]=pd.to_numeric(meta[c],errors='coerce')
    for c in catcols:meta[c]=meta[c].fillna('__unknown__').astype(str)
    base=evaluate(meta,y,groups,numcols,catcols,folds)
    rows=[{'forecast_day':0,'setting':'survey_controls_at_sowing',**metrics(y,base)}]
    predictions={'day_000_controls':base}
    for day in (15,30,45,60,75,90):
        scenes=early_matrix(obs,train,day)
        columns=scenes.columns.tolist()
        frame=pd.concat([meta.reset_index(drop=True),scenes],axis=1)
        pred=evaluate(frame,y,groups,numcols+columns,catcols,folds)
        add=metrics(y,base)['RMSE']-metrics(y,pred)['RMSE']
        low,high=bootstrap_delta(y,pred,base,groups,iterations=1000,seed=2026+day)
        result={'forecast_day':day,'setting':'controls_plus_satellites_to_day',**metrics(y,pred),
                'RMSE_controls_minus_combined':add,
                'conditional_cluster_bootstrap_low':low,'conditional_cluster_bootstrap_high':high}
        rows.append(result);predictions[f'day_{day:03d}_combined']=pred
        print(f'{day} days: RMSE {result["RMSE"]:.1f}, change vs controls {add:.1f}',flush=True)
    out=ROOT/'outputs'/'robustness'
    pd.DataFrame(rows).to_csv(out/'early_satellite_value_over_controls.csv',index=False)
    pd.DataFrame({'unique_id':train.unique_id,'plot_id':train.plot_id,'yield':y,**predictions}).to_csv(out/'early_satellite_value_oof.csv',index=False)
    (out/'early_satellite_value.summary.json').write_text(json.dumps({'rows':rows,
        'controls':['year','area','variety','season'],
        'minimum_growing_season_days':99,
        'notes':['All satellite acquisitions are <= horizon days after sowing, and all cutoffs < shortest crop season.',
                 'Known controls are observational covariates, not causal control variables.',
                 'Plot-grouped nested CV, no held-out official test set for horizon selection.',
                 'Confidence bounds condition on frozen OOF predictions; no model-refit uncertainty.']},indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(9,4.3))
    ax.plot([r['forecast_day'] for r in rows[1:]],[r['RMSE'] for r in rows[1:]],marker='o',label='Controls + available imagery')
    ax.axhline(rows[0]['RMSE'],ls='--',color='black',label='Known controls at sowing')
    ax.set_xlabel('Days since sowing');ax.set_ylabel('Plot-grouped CV RMSE')
    ax.set_title('Is early-season satellite imagery worth adding to survey covariates?')
    ax.legend();fig.tight_layout();fig.savefig(out/'early_satellite_value_over_controls.png',dpi=160);plt.close(fig)


if __name__=='__main__':run()
