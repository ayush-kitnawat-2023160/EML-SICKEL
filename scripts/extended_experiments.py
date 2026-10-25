#!/usr/bin/env python3
"""SICKLE EML robustness suite. No refitting on the official validation set.

Experiments:
  A: each satellite, satellite pairs, all satellites (five 20% windows)
  B: first-pass mask / Landsat pixel-count sensitivity
  C: ten finer, retrospective relative-season windows and leave-one-out tests
  D: true prospective information cutoffs at 15..90 days after sowing
  E: S2 cloud-probability sensitivity (requires alternate S2 NPZ extractions)

All scores are plot-grouped nested-CV OOF metrics on the 282 TRAIN rows only.
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from extract_features import FEATURES
from stage_importance_baseline import ROOT, STAGES, metrics, bootstrap_delta

SENSORS=list(FEATURES)
ALPHAS=[0.1, 1.0, 10.0, 100.0, 1000.0]
OUT=ROOT/'outputs'/'robustness'


def make_pipeline():
    return Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
                     ('scaler',StandardScaler()),('ridge',Ridge())])


def prepare():
    stage=pd.read_csv(ROOT/'outputs'/'features'/'full_stage_features.csv',
                      dtype={'unique_id':str,'plot_id':str})
    obs=pd.read_csv(ROOT/'outputs'/'features'/'full_observation_features.csv',
                    dtype={'unique_id':str,'plot_id':str})
    assert stage.unique_id.is_unique
    assert len(stage)==319
    assert not set(stage[stage.split=='train'].plot_id) & set(stage[stage.split=='val'].plot_id)
    return stage,obs


def stage_matrix(obs,labels,n_windows=5):
    """Per-scene plot medians -> per-stage plot-season medians (NO yield data)."""
    ids=pd.Index(labels.unique_id.astype(str),name='unique_id')
    result=pd.DataFrame(index=ids)
    observed=obs.loc[(obs.status=='ok') & (obs.relative_season_time>=0) &
                     (obs.relative_season_time<=1)].copy()
    observed['slot']=np.minimum((observed.relative_season_time*n_windows).astype(int),n_windows-1)
    for sensor,features in FEATURES.items():
        group=observed[observed.sensor.eq(sensor)]
        aggregate=group.groupby(['unique_id','slot'])[features].median()
        for win in range(n_windows):
            for f in features:
                name=f'W{win+1:02d}__{f}'
                if (not aggregate.empty) and (win in aggregate.index.get_level_values('slot')):
                    result[name]=aggregate.xs(win,level='slot')[f].reindex(ids)
                else:
                    result[name]=np.nan
    return result.reset_index(drop=True)


def early_matrix(obs,labels,cutoff):
    """No realized harvest date, relative-time feature or future image is used.

    Predict at fixed days after *known sowing date*. The dataset's min season
    duration is 99 days, so all 15..90 day horizons precede every harvest.
    """
    ids=pd.Index(labels.unique_id.astype(str),name='unique_id')
    result=pd.DataFrame(index=ids)
    observed=obs.loc[obs.status.eq('ok') & obs.days_since_sowing.between(0,cutoff)].copy()
    assert observed.empty or observed.days_since_sowing.max()<=cutoff
    for sensor,features in FEATURES.items():
        subset=observed[observed.sensor.eq(sensor)]
        if not subset.empty:
            group=subset.groupby('unique_id')[features]
            med=group.median()
            recent=(subset.sort_values(['days_since_sowing','npz_relative_path'])
                    .groupby('unique_id').tail(1).set_index('unique_id'))
        for f in features:
            result[f'{f}__to_day_{cutoff}__median']=med[f].reindex(ids) if not subset.empty else np.nan
            result[f'{f}__to_day_{cutoff}__latest']=recent[f].reindex(ids) if not subset.empty else np.nan
    return result.reset_index(drop=True)


def evaluate(X,labels,folds):
    assert X.shape[0]==len(labels) and X.shape[1]>0
    y=labels['yield'].to_numpy(float)
    groups=labels.plot_id.to_numpy()
    pred=np.full(len(labels),np.nan)
    alpha=[]
    for tr,te in folds:
        model=GridSearchCV(make_pipeline(),{'ridge__alpha':ALPHAS},
                           cv=GroupKFold(n_splits=3),scoring='neg_mean_squared_error',n_jobs=1)
        model.fit(X.iloc[tr],y[tr],groups=groups[tr])
        pred[te]=model.predict(X.iloc[te]);alpha.append(float(model.best_params_['ridge__alpha']))
    assert np.isfinite(pred).all()
    return pred,alpha


def run(only=None):
    OUT.mkdir(parents=True,exist_ok=True)
    stage,obs=prepare()
    train=stage[stage.split.eq('train')].reset_index(drop=True)
    assert len(train)==282
    y=train['yield'].to_numpy(float);group=train.plot_id.to_numpy()
    folds=list(GroupKFold(n_splits=5).split(train,y,group))
    rows=[];all_predictions={}

    def experiment(family,name,X,extra=None):
        if only and family not in only:return
        pred,alphas=evaluate(X,train,folds)
        result={'family':family,'experiment':name,'num_features':X.shape[1],
                'train_rows':len(train),'train_plots':len(set(group)),
                **metrics(y,pred),'selected_alpha_per_fold':json.dumps(alphas)}
        if extra:result.update(extra)
        rows.append(result);all_predictions[(family,name)]=pred
        print(f'{family:14s} {name:34s} RMSE={result["RMSE"]:.2f}  R2={result["R2"]:.3f}',flush=True)

    five_cols=[f'{w}__{feat}' for w in STAGES for fs in FEATURES.values() for feat in fs]
    original=train[five_cols]

    # A. Satellite-only and sensor-pair models. Exactly the same training groups.
    for size in (1,2,3):
        for sub in itertools.combinations(SENSORS,size):
            names=[f'{w}__{f}' for w in STAGES for s in sub for f in FEATURES[s]]
            experiment('sensor_sets','+'.join(sub),original[names])

    # B. Geometry and minimum Landsat plot-size checks on the SAME scene data.
    if not only or 'mask_l8' in only:
        base_matrix=stage_matrix(obs,train)
        experiment('mask_l8','default_resize',base_matrix)
        strict=obs.copy()
        strict.loc[strict.mask_shape!=strict.image_shape,'status']='excluded_shape_nonexact'
        experiment('mask_l8','exact_mask_shape_only',stage_matrix(strict,train))
        for pixels in (3,5):
            restrict=obs.copy()
            invalid=restrict.sensor.eq('L8') & restrict.plot_pixels.lt(pixels)
            restrict.loc[invalid,'status']='excluded_small_L8_plot'
            experiment('mask_l8',f'L8_min_{pixels}_pixels',stage_matrix(restrict,train))
        # Comparability analysis: does adding L8 help the two-sensor model?
        only_s1s2=[f'{w}__{f}' for w in STAGES for s in ('S1','S2') for f in FEATURES[s]]
        experiment('mask_l8','S1_S2_no_L8',original[only_s1s2])
        with_l8=set(stage[stage.split.eq('train')].plot_id)
        print('L8 geometry rejection scene count:',int(obs.status.eq('plot_absent_in_native_mask').sum()))

    # C. Five versus ten windows, each narrow window individually and ablated.
    if not only or 'fine_windows' in only:
        fine=stage_matrix(obs,train,10)
        experiment('fine_windows','all_ten_windows',fine)
        for window in range(10):
            token=f'W{window+1:02d}__'
            sub=[c for c in fine if c.startswith(token)]
            rest=[c for c in fine if not c.startswith(token)]
            experiment('fine_windows',f'only_{window*10:02d}_{(window+1)*10:02d}',fine[sub])
            experiment('fine_windows',f'without_{window*10:02d}_{(window+1)*10:02d}',fine[rest])

    # D. Early forecasts use fixed days since sowing: min season length >90.
    if not only or 'early_forecast' in only:
        for horizon in (15,30,45,60,75,90):
            available=obs[(obs.status=='ok') & obs.days_since_sowing.between(0,horizon)]
            covered=int(available.unique_id.isin(train.unique_id).groupby(available.unique_id).any().sum())
            X=early_matrix(obs,train,horizon)
            experiment('early_forecast',f'day_{horizon:03d}',X,
                       {'plots_with_at_least_one_scene':covered})

    # E. Cloud threshold variations: alternately-extracted S2 scenes only.
    if not only or 'cloud' in only:
        for threshold in (35,65,90):
            if threshold==65:altered=obs
            else:
                p=ROOT/'outputs'/'features'/f'full_s2cloud{threshold}_observation_features.csv'
                if not p.is_file():
                    print('SKIPPING CLOUD: missing alternate file',p,flush=True);continue
                alternative=pd.read_csv(p,dtype={'unique_id':str,'plot_id':str})
                keys=['unique_id','npz_relative_path']
                main=obs[obs.sensor!='S2'].copy()
                assert len(alternative)==len(obs[obs.sensor=='S2'])
                assert set(map(tuple,alternative[keys].to_numpy()))==set(map(tuple,obs.loc[obs.sensor=='S2',keys].to_numpy()))
                altered=pd.concat([main,alternative],ignore_index=True)
            X=stage_matrix(altered,train)
            used=int((altered.sensor.eq('S2') & altered.status.eq('ok')).sum())
            experiment('cloud',f'S2_prob_le_{threshold}',X,{'S2_accepted_scenes_all_319':used})

    report=pd.DataFrame(rows)
    report.to_csv(OUT/'train_grouped_cv_comparisons.csv',index=False)
    # Paired stage ablation intervals conditional on same grouped OOF predictions.
    deltas=[]
    if ('fine_windows','all_ten_windows') in all_predictions:
        reference=all_predictions[('fine_windows','all_ten_windows')]
        for idx in range(10):
            name=f'without_{idx*10:02d}_{(idx+1)*10:02d}'
            pred=all_predictions[('fine_windows',name)]
            delta=metrics(y,pred)['RMSE']-metrics(y,reference)['RMSE']
            ci=bootstrap_delta(y,reference,pred,group,iterations=500,seed=100+idx)
            deltas.append({'window':f'{idx*10:02d}-{(idx+1)*10:02d}%',
                           'delta_RMSE_removed_minus_full':delta,
                           'cluster_bootstrap_conditional_CI_low':ci[0],
                           'cluster_bootstrap_conditional_CI_high':ci[1]})
        pd.DataFrame(deltas).to_csv(OUT/'ten_window_ablation.csv',index=False)
    pred_out=pd.DataFrame({'unique_id':train.unique_id,'plot_id':train.plot_id,'yield':y})
    for (family,name),pred in all_predictions.items():pred_out[f'{family}__{name}']=pred
    pred_out.to_csv(OUT/'train_oof_predictions.csv',index=False)

    def plot_family(family,title,filename):
        d=report[report.family.eq(family)]
        if d.empty:return
        fig,ax=plt.subplots(figsize=(max(9,.48*len(d)),4.5))
        ax.bar(range(len(d)),d.RMSE)
        ax.set_xticks(range(len(d)),d.experiment,rotation=48,ha='right')
        ax.set_ylabel('Plot-grouped CV RMSE (yield label units)')
        ax.set_title(title);fig.tight_layout();fig.savefig(OUT/filename,dpi=160);plt.close(fig)
    plot_family('sensor_sets','Incremental information from satellite sensors','sensor_comparison.png')
    plot_family('early_forecast','Yield prediction with observations acquired by each day after sowing','early_forecasting.png')
    plot_family('mask_l8','Spatial resolution and mask-policy sensitivity','mask_landsat_sensitivity.png')
    plot_family('cloud','Sentinel-2 cloud-probability threshold sensitivity','cloud_threshold_sensitivity.png')
    if deltas:
        fig,ax=plt.subplots(figsize=(10,4.5))
        labels=[d['window'] for d in deltas];yy=[d['delta_RMSE_removed_minus_full'] for d in deltas]
        ax.bar(labels,yy);ax.axhline(0,color='black',lw=.8)
        ax.set_ylabel('RMSE without window - RMSE with all ten')
        ax.set_title('Contribution of ten narrower relative-season windows')
        fig.tight_layout();fig.savefig(OUT/'ten_window_ablation.png',dpi=160);plt.close(fig)
    summary={'train_rows':len(train),'train_physical_plots':len(set(group)),
             'validation_rows_NOT_used':len(stage[stage.split.eq('val')]),
             'minimum_season_duration_days':int((pd.to_datetime(train.harvesting_date)-pd.to_datetime(train.sowing_date)).dt.days.min()),
             'completed_experiments':len(report),'window_ablation':deltas,
             'guardrails':['Satellite features only, no acquisition-count indicators.',
                           'Ridge tuning nested in plot-grouped five-fold CV.',
                           'Early forecasts use fixed days after sowing, not future harvest-date normalization.',
                           'Cloud and mask results are observational sensitivity checks, not geometric ground-truth.',
                           'All stage importance scores are model-specific predictive associations.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Wrote',OUT/'train_grouped_cv_comparisons.csv','with',len(report),'rows')


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--only',nargs='+',choices=['sensor_sets','mask_l8','fine_windows','early_forecast','cloud'])
    arg=a.parse_args();run(arg.only)
