"""Regression tests for cohort validity and no-future-observation features."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from extended_experiments import early_matrix,stage_matrix
from extract_features import FEATURES, valid_s2


class SmallNPZ:
    files=['SCL','MSK_CLDPRB']
    def __getitem__(self,key):
        return {'SCL':np.full((2,2),4,dtype='float32'),
                'MSK_CLDPRB':np.array([[10,40],[70,95]],dtype='float32')}[key]


class ExtendedChecks(unittest.TestCase):
    def test_cloud_prob_threshold_nesting(self):
        a=valid_s2(SmallNPZ(),(2,2),35)
        b=valid_s2(SmallNPZ(),(2,2),65)
        c=valid_s2(SmallNPZ(),(2,2),90)
        self.assertTrue((a<=b).all() and (b<=c).all())
        self.assertEqual([int(a.sum()),int(b.sum()),int(c.sum())],[1,2,3])

    def test_early_forecast_does_not_use_future_satellite(self):
        labels=pd.DataFrame({'unique_id':['101','102']})
        records=[]
        for uid in ['101','102']:
            for day,val in [(10,.2),(40,.4),(75,.8),(100,.95)]:
                r={'unique_id':uid,'sensor':'S2','days_since_sowing':day,
                   'relative_season_time':day/150,'status':'ok','npz_relative_path':f'{uid}_{day}.npz'}
                for f in FEATURES['S2']:r[f]=val
                records.append(r)
        obs=pd.DataFrame(records)
        e30=early_matrix(obs,labels,30)
        e60=early_matrix(obs,labels,60)
        e90=early_matrix(obs,labels,90)
        self.assertAlmostEqual(e30['s2_ndvi__to_day_30__median'].iloc[0],.2)
        self.assertAlmostEqual(e60['s2_ndvi__to_day_60__median'].iloc[0],.3)
        self.assertAlmostEqual(e90['s2_ndvi__to_day_90__latest'].iloc[0],.8)
        changed=obs.copy()
        changed.loc[changed.days_since_sowing>90,'s2_ndvi']=9999
        pd.testing.assert_frame_equal(e90,early_matrix(changed,labels,90))

    def test_stage_window_mapping_and_never_uses_yield(self):
        labels=pd.DataFrame({'unique_id':['1']})
        obs=pd.DataFrame({'unique_id':['1','1'],
                          'sensor':['S1','S1'],'relative_season_time':[0,.999],
                          'days_since_sowing':[0,99],'status':['ok','ok'],
                          's1_vv_db':[-15,-8],'s1_vh_db':[-25,-18],
                          's1_vv_minus_vh_db':[10,10],
                          'yield':[1_000_000,-999]})
        for sensor,features in FEATURES.items():
            for feature in features:
                if feature not in obs:obs[feature]=np.nan
        x=stage_matrix(obs,labels,5)
        self.assertEqual(x.shape,(1,55))
        self.assertEqual(x.iloc[0]['W01__s1_vv_db'],-15)
        self.assertEqual(x.iloc[0]['W05__s1_vv_db'],-8)
        self.assertNotIn('yield',x.columns)

    def test_fold_group_separation_real_data(self):
        p=ROOT/'outputs'/'features'/'full_stage_features.csv'
        if not p.exists():self.skipTest('feature extraction not completed')
        df=pd.read_csv(p,dtype={'plot_id':str})
        tr=df[df.split.eq('train')].reset_index(drop=True)
        vl=df[df.split.eq('val')]
        self.assertFalse(set(tr.plot_id)&set(vl.plot_id))
        for a,b in GroupKFold(n_splits=5).split(tr,tr['yield'],groups=tr.plot_id):
            self.assertFalse(set(tr.iloc[a].plot_id)&set(tr.iloc[b].plot_id))

    def test_all_early_cutoffs_predate_harvest(self):
        p=ROOT/'outputs'/'features'/'full_stage_features.csv'
        if not p.exists():self.skipTest('stage features not ready')
        df=pd.read_csv(p)
        days=(pd.to_datetime(df.harvesting_date)-pd.to_datetime(df.sowing_date)).dt.days
        self.assertGreater(days.min(),90)


if __name__=='__main__':unittest.main()
