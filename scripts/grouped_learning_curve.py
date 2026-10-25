#!/usr/bin/env python3
"""Training-data scarcity experiment: randomly reduce *training plots*, not pixels.

Outer five grouped folds fixed across sample sizes; each training fold is
randomly subselected at the plot level, then Ridge tuned by inner grouped CV.
The outer test fold is always the same for a given fold, enabling comparisons.
"""

import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stage_importance_baseline import ROOT,STAGES,metrics
from extract_features import FEATURES


def run(repeats):
    df=pd.read_csv(ROOT/'outputs'/'features'/'full_stage_features.csv',dtype={'plot_id':str})
    train=df[df.split.eq('train')].reset_index(drop=True)
    columns=[f'{w}__{feature}' for w in STAGES for fs in FEATURES.values() for feature in fs]
    X=train[columns];y=train['yield'].to_numpy(float);group=train.plot_id.to_numpy()
    folds=list(GroupKFold(n_splits=5).split(train,y,group))
    out=[]
    for frac in (.2,.4,.6,.8,1.0):
        for repeat in range(repeats):
            predicted=np.full(len(train),np.nan)
            counts=[]
            for foldno,(full_train,test) in enumerate(folds):
                rng=np.random.default_rng(2030+repeat*101+foldno)
                all_groups=np.unique(group[full_train]);n=max(5,int(round(frac*len(all_groups))))
                selected=rng.choice(all_groups,n,replace=False) if n<len(all_groups) else all_groups
                subset=full_train[np.isin(group[full_train],selected)]
                counts.append(len(selected))
                pipe=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
                               ('scale',StandardScaler()),('ridge',Ridge())])
                fit=GridSearchCV(pipe,{'ridge__alpha':[.1,1,10,100,1000]},
                                 cv=GroupKFold(n_splits=3),scoring='neg_mean_squared_error',n_jobs=1)
                fit.fit(X.iloc[subset],y[subset],groups=group[subset])
                predicted[test]=fit.predict(X.iloc[test])
            m=metrics(y,predicted)
            out.append({'training_plot_fraction':frac,'repeat':repeat,
                        'mean_training_plots_per_fold':float(np.mean(counts)),
                        'train_only_outer_RMSE':m['RMSE'],'train_only_outer_MAE':m['MAE'],
                        'train_only_outer_R2':m['R2']})
            print(f'fraction={frac:.1f} repeat={repeat+1}/{repeats} RMSE={m["RMSE"]:.1f}',flush=True)
    output=ROOT/'outputs'/'robustness';output.mkdir(parents=True,exist_ok=True)
    data=pd.DataFrame(out)
    data.to_csv(output/'learning_curve_plot_fraction_runs.csv',index=False)
    summary=data.groupby('training_plot_fraction',as_index=False).agg(
        rmse_mean=('train_only_outer_RMSE','mean'),rmse_std=('train_only_outer_RMSE','std'),
        rmse_min=('train_only_outer_RMSE','min'),rmse_max=('train_only_outer_RMSE','max'),
        avg_training_plots=('mean_training_plots_per_fold','mean'))
    summary.to_csv(output/'learning_curve_plot_fraction.csv',index=False)
    (output/'learning_curve.summary.json').write_text(json.dumps({
        'repeat_count':repeats,'fractions':[.2,.4,.6,.8,1.0],
        'outer_folds':5,'inner_folds':3,
        'note':'Training PLOT_IDs randomly subselected within each outer training fold; outer test groups unchanged. Variability is across random plot subsamples, not a formal confidence interval.'},indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(8.5,4.5))
    xx=summary.avg_training_plots.to_numpy();yy=summary.rmse_mean.to_numpy()
    sd=summary.rmse_std.fillna(0).to_numpy()
    ax.errorbar(xx,yy,yerr=sd,marker='o',capsize=5)
    ax.set_xlabel('Average distinct training plots per outer fold')
    ax.set_ylabel('Grouped out-of-fold RMSE (mean ± SD across subsamples)')
    ax.set_title('Sensitivity to scarce annotated field-level yield labels')
    fig.tight_layout();fig.savefig(output/'learning_curve_plot_fraction.png',dpi=160);plt.close(fig)
    print(summary.to_string(index=False))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--repeats',type=int,default=4)
    run(a.parse_args().repeats)
