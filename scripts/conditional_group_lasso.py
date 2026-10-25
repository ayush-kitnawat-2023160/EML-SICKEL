#!/usr/bin/env python3
"""Conditional group LASSO for SICKLE with *unpenalized agronomic controls*.

Mathematics (training fold only):
  min_{gamma,beta} ||y - C gamma - X beta||^2/(2n)
                   + lambda * sum_{g in G} sqrt(|g|) ||beta_g||_2.
  C = intercept, scaled area/year, training-fitted categorical dummies for
  variety and season; all C coefficients are deliberately unpenalized.
  X = 55 standardized, training-median-imputed satellite features.

We eliminate gamma analytically (Frisch-Waugh-Lovell projection onto the
orthogonal complement of C), solve the satellite-only group-penalty objective
with deterministic FISTA, and recover gamma. This is the exact joint convex
optimization objective up to numerical convergence, NOT the inappropriate
two-step approach of fitting controls and then selecting on raw X residuals.

Use only released TRAIN plots for nested CV; the original VAL plots are not
used in model selection or in group-frequency estimation.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from extract_features import FEATURES
from stage_importance_baseline import ROOT, STAGES, metrics, bootstrap_delta


NUMERIC_CONTROLS = ('area', 'year')
CATEGORICAL_CONTROLS = ('season', 'variety')
SAT_NAMES = tuple(f'{w}__{f}' for w in STAGES for fs in FEATURES.values() for f in fs)
ALPHAS = (0., .03, .08, .15, .3, .5, .8, 1.)  # fractions of lambda_max
OUT = ROOT / 'outputs' / 'conditional_group_lasso'


def group_indices(scheme, feature_names=SAT_NAMES):
    """Return indices into the 55 satellite coefficients; controls excluded."""
    feature_names=tuple(feature_names)
    if scheme == 'stage':
        return {w: np.array([j for j,c in enumerate(feature_names) if c.startswith(w+'__')])
                for w in STAGES if any(c.startswith(w+'__') for c in feature_names)}
    if scheme == 'sensor_stage':
        return {w+'__'+sensor: np.array([j for j,c in enumerate(feature_names)
                if c.startswith(w+'__') and c.rsplit('__',1)[1] in features])
                for w in STAGES for sensor,features in FEATURES.items()
                if any(c.startswith(w+'__') and c.rsplit('__',1)[1] in features for c in feature_names)}
    if scheme == 'sensor':
        return {sensor: np.array([j for j,c in enumerate(feature_names)
                                  if c.rsplit('__',1)[1] in features])
                for sensor,features in FEATURES.items()
                if any(c.rsplit('__',1)[1] in features for c in feature_names)}
    raise ValueError(scheme)


def prox_groups(z, groups, penalty, step):
    beta=z.copy()
    for ix in groups.values():
        length=np.linalg.norm(z[ix]);shrink=1. - step*penalty*np.sqrt(len(ix))/length if length>0 else 0.
        beta[ix] *= max(shrink,0.)
    return beta


def group_fista(gram,linear,groups,penalty,iterations=800,tol=1e-7):
    """Solve 1/2 beta' G beta - linear' beta + weighted group penalty."""
    if not penalty:
        return np.linalg.lstsq(gram,linear,rcond=1e-9)[0]
    L=float(np.linalg.eigvalsh(gram)[-1])
    if L<1e-14:return np.zeros(len(linear))
    beta=np.zeros(len(linear));z=beta.copy();t=1.
    for _ in range(iterations):
        next_beta=prox_groups(z+(linear-gram@z)/L,groups,penalty,1./L)
        if np.linalg.norm(next_beta-beta) < tol*(1.+np.linalg.norm(beta)):
            beta=next_beta;break
        t_new=(1+np.sqrt(1+4*t*t))/2
        z=next_beta+((t-1)/t_new)*(next_beta-beta)
        beta=next_beta;t=t_new
    return beta


class ConditionalGroupLasso:
    def __init__(self, scheme='stage',alpha=.15,feature_names=SAT_NAMES):
        self.scheme=scheme;self.alpha=alpha
        self.feature_names=tuple(feature_names)
        if not self.feature_names:raise ValueError('At least one satellite feature is required')

    def fit(self,frame,y):
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        self.scaler=StandardScaler()
        self.numscale=StandardScaler()
        self.encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False)

        sat=np.asarray(self.imputer.fit_transform(frame[list(self.feature_names)]),dtype=float)
        X=self.scaler.fit_transform(sat)
        numeric=frame[list(NUMERIC_CONTROLS)].astype(float).to_numpy()
        nums=self.numscale.fit_transform(numeric)
        cats=self.encoder.fit_transform(frame[list(CATEGORICAL_CONTROLS)].fillna('__unknown__').astype(str))
        C=np.column_stack((np.ones(len(frame)),nums,cats))
        self.y_mean=float(np.mean(y));self.y_scale=float(np.std(y))
        if self.y_scale<=0:raise ValueError('Degenerate target variation')
        norm_y=(np.asarray(y,dtype=float)-self.y_mean)/self.y_scale

        # Simultaneous training-only OLS projection of y and X on controls.
        projection=np.linalg.lstsq(C,np.column_stack([norm_y,X]),rcond=None)[0]
        yc=norm_y-C@projection[:,0]
        Xc=X-C@projection[:,1:]
        groups=group_indices(self.scheme,self.feature_names)
        gram=(Xc.T@Xc)/len(frame)
        linear=(Xc.T@yc)/len(frame)
        maximum=max(float(np.linalg.norm(linear[ix])/np.sqrt(len(ix))) for ix in groups.values())
        self.lambda_max=maximum
        penalty=self.alpha*maximum
        self.beta=group_fista(gram,linear,groups,penalty)
        self.gamma=projection[:,0]-projection[:,1:]@self.beta
        self.group_norms={name:float(np.linalg.norm(self.beta[idx])) for name,idx in groups.items()}
        self.selected=tuple(name for name,norm in self.group_norms.items() if norm>1e-5)
        return self

    def predict(self,frame):
        sat=self.imputer.transform(frame[list(self.feature_names)])
        X=self.scaler.transform(sat)
        nums=self.numscale.transform(frame[list(NUMERIC_CONTROLS)].astype(float).to_numpy())
        cats=self.encoder.transform(frame[list(CATEGORICAL_CONTROLS)].fillna('__unknown__').astype(str))
        C=np.column_stack((np.ones(len(frame)),nums,cats))
        return self.y_mean+self.y_scale*(C@self.gamma+X@self.beta)


def nested_group_cv(frame,y,plot_ids,scheme,outer_folds,alphas=ALPHAS):
    """For each outer train fold, tune relative lambda on inner held-out plots."""
    predicted=np.full(len(frame),np.nan)
    choices=[];outer_selection=[];inner_records=[]
    for fold,(tr,te) in enumerate(outer_folds):
        inner=list(GroupKFold(n_splits=3).split(frame.iloc[tr],y[tr],plot_ids[tr]))
        score=[]
        for alpha in alphas:
            inner_pred=np.full(len(tr),np.nan)
            for tin,vin in inner:
                mdl=ConditionalGroupLasso(scheme,alpha).fit(frame.iloc[tr].iloc[tin],y[tr][tin])
                inner_pred[vin]=mdl.predict(frame.iloc[tr].iloc[vin])
            score.append(float(np.mean((y[tr]-inner_pred)**2)))
        best=int(np.argmin(score));chosen=alphas[best]
        final=ConditionalGroupLasso(scheme,chosen).fit(frame.iloc[tr],y[tr])
        predicted[te]=final.predict(frame.iloc[te])
        choices.append(float(chosen));outer_selection.append(final.selected)
        inner_records.append({'outer_fold':fold,'alpha_candidates':list(alphas),
                              'mse_by_alpha':score,'selected_alpha':chosen,
                              'selected_groups':list(final.selected)})
        print(f'  {scheme} fold={fold+1}/5 alpha={chosen:g}, groups={list(final.selected)}',flush=True)
    assert np.isfinite(predicted).all()
    return predicted,choices,outer_selection,inner_records


def run(repeats=25,learning_repeats=3):
    OUT.mkdir(parents=True,exist_ok=True)
    full=pd.read_csv(ROOT/'outputs'/'features'/'full_stage_features.csv',dtype={'unique_id':str,'plot_id':str})
    frame=full[full.split.eq('train')].reset_index(drop=True)
    validation=full[full.split.eq('val')]
    assert len(frame)==282 and len(validation)==37
    assert not set(frame.plot_id)&set(validation.plot_id)
    y=frame['yield'].to_numpy(float);plot_ids=frame.plot_id.to_numpy()
    folds=list(GroupKFold(n_splits=5).split(frame,y,plot_ids))

    outputs={};results=[];inner_details={}
    for scheme in ('stage','sensor_stage','sensor'):
        t=time.time()
        p,choices,selected,detail=nested_group_cv(frame,y,plot_ids,scheme,folds)
        outputs[scheme]=p;inner_details[scheme]=detail
        result={'model':'conditional_group_lasso__'+scheme,'groups_defined':len(group_indices(scheme)),
                **metrics(y,p),'outer_alpha':json.dumps(choices),
                'outer_selected_groups':json.dumps([list(s) for s in selected]),
                'elapsed_seconds':round(time.time()-t,1)}
        results.append(result)
        print('RESULT',scheme,metrics(y,p),flush=True)

    # Unpenalized controls-only uses the identical transformation/fold protocol.
    baseline=np.full(len(frame),np.nan)
    for tr,te in folds:
        fit=ConditionalGroupLasso('stage',alpha=1.1).fit(frame.iloc[tr],y[tr])
        assert not fit.selected
        baseline[te]=fit.predict(frame.iloc[te])
    outputs['unpenalized_controls_only']=baseline
    results.append({'model':'unpenalized_controls_only','groups_defined':0,**metrics(y,baseline)})
    results_table=pd.DataFrame(results)
    results_table.to_csv(OUT/'grouped_cv_comparison.csv',index=False)
    pd.DataFrame({'unique_id':frame.unique_id,'plot_id':plot_ids,'yield':y,**outputs}).to_csv(
        OUT/'grouped_cv_predictions.csv',index=False)
    (OUT/'nested_cv_choices.json').write_text(json.dumps(inner_details,indent=2)+'\n')

    paired=[]
    for scheme in ('stage','sensor_stage','sensor'):
        improvement=metrics(y,baseline)['RMSE']-metrics(y,outputs[scheme])['RMSE']
        low,high=bootstrap_delta(y,outputs[scheme],baseline,plot_ids,iterations=1200,seed=43)
        paired.append({'scheme':scheme,'controls_only_minus_group_lasso_RMSE':improvement,
                       'conditional_plot_bootstrap_ci_low':low,
                       'conditional_plot_bootstrap_ci_high':high})
    pd.DataFrame(paired).to_csv(OUT/'paired_controls_vs_group_lasso.csv',index=False)

    # Stable group selection: 80% of distinct training plots in each replicate.
    rng=GroupShuffleSplit(n_splits=repeats,train_size=.8,random_state=2026)
    frequencies=[]
    for scheme in ('stage','sensor_stage'):
        groups=group_indices(scheme)
        occurrence={name:0 for name in groups}
        tuning=[]
        for repeat,(sub,_) in enumerate(rng.split(frame,y,plot_ids)):
            ss=frame.iloc[sub];ys=y[sub];ps=plot_ids[sub]
            cv=list(GroupKFold(n_splits=3).split(ss,ys,ps))
            mse=[]
            for alpha in ALPHAS:
                pred=np.full(len(ss),np.nan)
                for tr,va in cv:
                    mdl=ConditionalGroupLasso(scheme,alpha).fit(ss.iloc[tr],ys[tr])
                    pred[va]=mdl.predict(ss.iloc[va])
                mse.append(np.mean((pred-ys)**2))
            chosen=ALPHAS[int(np.argmin(mse))]
            mdl=ConditionalGroupLasso(scheme,chosen).fit(ss,ys)
            tuning.append(chosen)
            for name in mdl.selected:occurrence[name]+=1
            if (repeat+1)%5==0:print('STABILITY',scheme,repeat+1,'/',repeats,flush=True)
        for name,n in occurrence.items():
            frequencies.append({'scheme':scheme,'group':name,'feature_count':len(groups[name]),
                                'selected_subsamples':n,'selection_frequency':n/repeats,
                                'num_repeats':repeats})
        (OUT/f'{scheme}_subsample_tuning.json').write_text(json.dumps(tuning)+'\n')
    freq_df=pd.DataFrame(frequencies)
    freq_df.to_csv(OUT/'group_selection_stability.csv',index=False)

    # Label scarcity: do candidate features get selected more consistently
    # and perform better with 30%, 60%, or 100% of TRAINING plot groups?
    # Outer folds unchanged, additional plot subsampling only in training.
    learning=[]
    for frac in (.3,.6,1.):
        for repeat in range(learning_repeats):
            predictions=np.full(len(frame),np.nan)
            groupcounts=[];num_selected=[];alphas=[]
            for k,(tr,te) in enumerate(folds):
                random=np.random.default_rng(2026+repeat*13+k)
                available=np.unique(plot_ids[tr]);num=max(8,round(frac*len(available)))
                selected=available if len(available)==num else random.choice(available,num,replace=False)
                partial=tr[np.isin(plot_ids[tr],selected)]
                inner=list(GroupKFold(n_splits=3).split(frame.iloc[partial],y[partial],plot_ids[partial]))
                errors=[]
                for alpha in ALPHAS:
                    ipred=np.full(len(partial),np.nan)
                    for ti,vi in inner:
                        mdl=ConditionalGroupLasso('stage',alpha).fit(frame.iloc[partial].iloc[ti],y[partial][ti])
                        ipred[vi]=mdl.predict(frame.iloc[partial].iloc[vi])
                    errors.append(np.mean((ipred-y[partial])**2))
                chosen=ALPHAS[int(np.argmin(errors))]
                mdl=ConditionalGroupLasso('stage',chosen).fit(frame.iloc[partial],y[partial])
                predictions[te]=mdl.predict(frame.iloc[te]);groupcounts.append(len(selected))
                num_selected.append(len(mdl.selected));alphas.append(chosen)
            learning.append({'training_plot_fraction':frac,'repeat':repeat,
                             'mean_training_plots':np.mean(groupcounts),
                             'mean_selected_windows':np.mean(num_selected),
                             'mean_cv_alpha':np.mean(alphas),**metrics(y,predictions)})
            print('LEARNING',frac,repeat+1,metrics(y,predictions)['RMSE'],flush=True)
    lc=pd.DataFrame(learning);lc.to_csv(OUT/'group_lasso_label_scarcity_runs.csv',index=False)
    summary_lc=lc.groupby('training_plot_fraction',as_index=False).agg(
        mean_training_plots=('mean_training_plots','mean'),
        rmse_mean=('RMSE','mean'),rmse_std=('RMSE','std'),
        selected_windows_mean=('mean_selected_windows','mean'))
    summary_lc.to_csv(OUT/'group_lasso_label_scarcity.csv',index=False)

    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.bar(results_table.model,results_table.RMSE)
    ax.tick_params(axis='x',rotation=24)
    ax.set_ylabel('Out-of-plot RMSE (yield-label units)')
    ax.set_title('Unpenalized agronomic controls + grouped satellite regularization')
    fig.tight_layout();fig.savefig(OUT/'group_lasso_comparison.png',dpi=170);plt.close(fig)
    sub=freq_df[freq_df.scheme=='stage']
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.bar(sub.group,sub.selection_frequency)
    ax.set_ylim(0,1);ax.set_ylabel('Selection frequency (80%-of-plots resamples)')
    ax.set_title(f'Conditional group-LASSO: growing-season window stability (n={repeats})')
    fig.tight_layout();fig.savefig(OUT/'window_selection_stability.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.errorbar(summary_lc.mean_training_plots,summary_lc.rmse_mean,yerr=summary_lc.rmse_std.fillna(0),marker='o',capsize=4)
    ax.set_xlabel('Distinct training plots per outer fold')
    ax.set_ylabel('Out-of-plot RMSE (mean ± subsampling SD)')
    ax.set_title('Label scarcity with conditional group LASSO')
    fig.tight_layout();fig.savefig(OUT/'group_lasso_learning_curve.png',dpi=170);plt.close(fig)
    (OUT/'summary.json').write_text(json.dumps({'train_rows':len(frame),'training_physical_plots':len(set(plot_ids)),
        'held_out_validation_rows_unused':len(validation),'alpha_grid_relative_lambda_max':ALPHAS,
        'group_schemes':['stage','sensor_stage','sensor'],
        'controls_unpenalized':list(NUMERIC_CONTROLS+CATEGORICAL_CONTROLS),
        'group_cv_results':results,'paired_controls_comparisons':paired,
        'stability_num_repeats':repeats,
        'scarcity_repeats':learning_repeats,
        'cautions':['All measurements derived from retrospective normalized season bins; these are not early forecasts.',
            'Group LASSO selects correlated groups jointly but does not establish causation or hypothesis-test significance.',
            'Published SICKLE yields are survey-based; local plot mask alignment approximations remain.',
            'Nominal cluster-bootstrap confidence intervals condition on already-fitted OOF predictions; no refits.',
            'The unpenalized controls-only benchmark may differ from previously reported ridge-regularized controls baseline.']},indent=2)+'\n')
    print('FINAL')
    print(results_table[['model','RMSE','MAE','R2']].to_string(index=False))
    print('PAIRED CONTROLS DELTAS')
    print(pd.DataFrame(paired).to_string(index=False))
    print('STABILITY')
    print(freq_df[freq_df.scheme=='stage'].to_string(index=False))
    print('LABEL SCARCITY')
    print(summary_lc.to_string(index=False))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--repeats',type=int,default=25)
    a.add_argument('--learning-repeats',type=int,default=3)
    opt=a.parse_args();run(opt.repeats,opt.learning_repeats)
