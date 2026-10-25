#!/usr/bin/env python3
"""Noncausal coverage diagnostics, paired CV contrasts and observation bias checks."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stage_importance_baseline import ROOT, metrics, bootstrap_delta


def run():
    out=ROOT/'outputs'/'robustness';out.mkdir(exist_ok=True,parents=True)
    stage=pd.read_csv(ROOT/'outputs'/'features'/'full_stage_features.csv',dtype={'unique_id':str,'plot_id':str})
    obs=pd.read_csv(ROOT/'outputs'/'features'/'full_observation_features.csv',dtype={'unique_id':str,'plot_id':str})
    assert stage.unique_id.is_unique
    # Positives are plot-season labels; missing radar/optical values should not
    # be coded as zero vegetation or as observations from adjacent fields.
    grouped=obs.groupby(['unique_id','sensor'])
    counts=grouped.agg(available_scenes=('status','size'),
                       accepted_scenes=('status',lambda s:int((s=='ok').sum())),
                       native_plot_absent=('status',lambda s:int((s=='plot_absent_in_native_mask').sum())),
                       size_or_geometric_mismatch=('status',lambda s:int((s=='mask_image_geometry_mismatch').sum())),
                       median_native_plot_pixels=('plot_pixels',lambda s:float(s[s>0].median()) if (s>0).any() else np.nan))
    joined=stage[['unique_id','plot_id','yield','area','year','season','variety','split']].set_index('unique_id').copy()
    for sensor in ('S1','S2','L8'):
        if sensor in counts.index.get_level_values('sensor'):
            subset=counts.xs(sensor,level='sensor')
            for col in counts.columns:
                joined[f'{sensor}_{col}']=subset[col].reindex(joined.index).fillna(0)
    joined['L8_any_clear_scene']=joined['L8_accepted_scenes'].gt(0)
    joined.to_csv(out/'plot_season_observation_coverage.csv')

    small=pd.DataFrame()
    for attr in ('area','year','season'):
        group=joined.groupby(attr,dropna=False)
        overview=group.agg(plot_seasons=('yield','size'),plots=('plot_id','nunique'),
                           yield_mean=('yield','mean'),yield_median=('yield','median'),
                           L8_any_clear_fraction=('L8_any_clear_scene','mean'),
                           L8_clear_scenes_mean=('L8_accepted_scenes','mean'),
                           S2_clear_scenes_mean=('S2_accepted_scenes','mean'))
        overview.insert(0,'grouping_variable',attr)
        overview.insert(1,'group_value',overview.index.astype(str))
        small=pd.concat([small,overview.reset_index(drop=True)],ignore_index=True)
    small.to_csv(out/'coverage_selection_diagnostics.csv',index=False)

    paired=pd.read_csv(out/'train_oof_predictions.csv',dtype={'unique_id':str,'plot_id':str})
    group=paired.plot_id.to_numpy()
    truth=paired['yield'].to_numpy(float)
    comparisons=[
        ('sensor_sets__S1+S2','sensor_sets__S1+S2+L8','Adding Landsat to S1+S2'),
        ('mask_l8__default_resize','mask_l8__exact_mask_shape_only','Exact shapes versus resize'),
        ('mask_l8__default_resize','mask_l8__L8_min_3_pixels','Require >=3 L8 plot pixels'),
        ('fine_windows__all_ten_windows','sensor_sets__S1+S2+L8','5 windows versus 10'),
        ('early_forecast__day_030','early_forecast__day_090','Day 30 versus Day 90'),
        ('cloud__S2_prob_le_35','cloud__S2_prob_le_65','S2 cloud threshold 35 versus 65'),
        ('cloud__S2_prob_le_90','cloud__S2_prob_le_65','S2 cloud threshold 90 versus 65'),
    ]
    rows=[]
    for lhs,rhs,label in comparisons:
        if lhs not in paired or rhs not in paired:continue
        a=paired[lhs].to_numpy(float);b=paired[rhs].to_numpy(float)
        delta=metrics(truth,a)['RMSE']-metrics(truth,b)['RMSE']
        low,high=bootstrap_delta(truth,b,a,group,iterations=1200,seed=42)
        rows.append({'comparison':label,'reference_model':lhs,'comparison_model':rhs,
                     'reference_RMSE':metrics(truth,a)['RMSE'],'comparison_RMSE':metrics(truth,b)['RMSE'],
                     'reference_minus_comparison_RMSE':delta,
                     'paired_plot_cluster_CI_low':low,'paired_plot_cluster_CI_high':high})
    pd.DataFrame(rows).to_csv(out/'paired_model_contrasts.csv',index=False)
    print('Coverage by area')
    print(small[small.grouping_variable=='area'].to_string(index=False))
    print('\nPaired prediction contrasts')
    print(pd.DataFrame(rows)[['comparison','reference_minus_comparison_RMSE',
                              'paired_plot_cluster_CI_low','paired_plot_cluster_CI_high']].to_string(index=False))
    summary={'n_plot_seasons':len(joined),
             'L8_any_clear_plot_seasons':int(joined.L8_any_clear_scene.sum()),
             'S2_any_clear_plot_seasons':int(joined.S2_accepted_scenes.gt(0).sum()),
             'S1_any_clear_plot_seasons':int(joined.S1_accepted_scenes.gt(0).sum()),
             'comparisons':rows,
             'warnings':['Coverage group comparisons are descriptive, not missingness-mechanism identification.',
                         'Bootstrap intervals are conditional on frozen OOF predictions and do not account for model refits.',
                         'Satellite imagery and plot yield both may differ by area/management/year, so observed correlations are not causal.']}
    (out/'coverage_and_contrasts.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(9,4.5))
    groups=small[small.grouping_variable=='area'].sort_values('group_value')
    bars=ax.bar(groups.group_value,groups.L8_any_clear_fraction)
    for b,n in zip(bars,groups.plot_seasons):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.02,str(int(n)),ha='center')
    ax.set_ylim(0,1.1);ax.set_ylabel('Fraction of plot-seasons with usable Landsat 8 scene')
    ax.set_xlabel('Reported plot area (dataset units)')
    ax.set_title('Landsat 8 observational coverage across reported plot areas (bar label = n)')
    fig.tight_layout();fig.savefig(out/'landsat_coverage_by_area.png',dpi=160);plt.close(fig)


if __name__=='__main__':run()
