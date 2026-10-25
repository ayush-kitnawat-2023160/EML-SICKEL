"""Smoke/invariant tests; run via `python -m unittest discover -s tests -v`."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from extract_features import (L8_SR_OFFSET,L8_SR_SCALE,normalized_difference,
                              plot_pixels,valid_l8,valid_s2)


class MockNPZ:
    def __init__(self,**arrays):
        self._arrays=arrays
        self.files=list(arrays)

    def __getitem__(self,key):
        return self._arrays[key]


class TestSatelliteSemantics(unittest.TestCase):
    def test_target_plot_does_not_include_neighbor(self):
        mask=np.array([[100,100,101],[0,101,0]],dtype=np.float32)
        result=plot_pixels(mask,100,(2,3))
        self.assertEqual(int(result.sum()),2)
        self.assertFalse(result[0,2])

    def test_normalized_difference(self):
        a=np.array([[.8,.6]],dtype=np.float32)
        b=np.array([[.2,.6]],dtype=np.float32)
        res=normalized_difference(a,b)
        self.assertAlmostEqual(float(res[0,0]),.6,places=6)
        self.assertAlmostEqual(float(res[0,1]),0,places=6)

    def test_sent2_scl_excludes_cloud_and_shadow_retains_water(self):
        z=MockNPZ(SCL=np.array([[4,5,6,7,3,8,9,10,11,0]],dtype=np.float32))
        v=valid_s2(z,(1,10))
        self.assertEqual(v.astype(int).tolist(),[[1,1,1,1,0,0,0,0,0,0]])

    def test_landsat_cloud_bits(self):
        z=MockNPZ(QA_PIXEL=np.array([[0,1,8,16,32,128]],dtype=np.uint16))
        self.assertEqual(valid_l8(z,(1,6)).astype(int).tolist(),[[1,0,0,0,0,1]])

    def test_landsat_c2_reflectance_scaling(self):
        # USGS scaling example: DN=18,639 should give ~0.313
        self.assertAlmostEqual(18639*L8_SR_SCALE+L8_SR_OFFSET,.31257,places=3)

    def test_extracted_output_integrity(self):
        import pandas as pd
        path=ROOT/"outputs"/"features"/"toy_stage_features.csv"
        if not path.exists():self.skipTest("toy feature extraction not run")
        d=pd.read_csv(path)
        self.assertEqual(len(d),34)
        self.assertTrue(d.unique_id.is_unique)
        self.assertTrue(d["yield"].gt(0).all())
        for name in [c for c in d if c.endswith("ndvi") or c.endswith("ndmi") or c.endswith("ndre")]:
            values=d[name].dropna()
            self.assertTrue(values.between(-1,1).all(),name)


if __name__=="__main__":unittest.main()
