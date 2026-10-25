#!/usr/bin/env python3
"""Aggregate per-acquisition SICKLE features into five stage feature groups.

Each output row is a *plot-season*. Values are medians of cloud-filtered
per-scene plot medians, not individual pixels or interpolated missing scenes.
Missing windows stay NaN for imputation *inside* model training splits.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extract_features import FEATURES

ROOT=Path(__file__).resolve().parents[1]
WINDOWS=["W1_0_20", "W2_20_40", "W3_40_60", "W4_60_80", "W5_80_100"]


def run(dataset):
    infile=ROOT/"outputs"/"features"/f"{dataset}_observation_features.csv"
    if not infile.is_file():
        raise SystemExit(f"First run: python scripts/extract_features.py --dataset {dataset}")
    data=pd.read_csv(infile,dtype={"unique_id":str,"plot_id":str})
    source="sickle_toy" if dataset=="toy" else "sickle_full"
    labels=pd.read_csv(ROOT/"data"/"raw"/source/"sickle_dataset_tabular.csv")
    labels["unique_id"]=labels["UNIQUE_ID"].astype(str)
    labels=labels[["unique_id","PLOT_ID","SPLIT","YIELD","VARIETY","STANDARD_SEASON","YEAR","AREA","SOWING_DATE","TRANSPLANTING_DATE","HARVESTING_DATE"]]
    labels=labels[labels.unique_id.isin(data.unique_id)].copy()
    assert len(labels)==data.unique_id.nunique(),"Unexpected duplicate or missing plot-season records"
    assert labels.PLOT_ID.nunique()>0
    labels["YIELD"]=pd.to_numeric(labels.YIELD,errors="coerce")
    assert labels.YIELD.gt(0).all(),"Nonpositive yield mistakenly admitted"
    rows=[]
    for _,label in labels.iterrows():
        frame=data[data.unique_id==label.unique_id]
        record={"unique_id":label.unique_id,"plot_id":str(label.PLOT_ID),"split":label.SPLIT,
                "yield":float(label.YIELD),"variety":label.VARIETY,"season":label.STANDARD_SEASON,
                "year":int(label.YEAR),"area":label.AREA,"sowing_date":label.SOWING_DATE,
                "transplanting_date":label.TRANSPLANTING_DATE,"harvesting_date":label.HARVESTING_DATE}
        for window in WINDOWS:
            subset=frame[frame.window==window]
            for sensor,featnames in FEATURES.items():
                group=subset[subset.sensor==sensor]
                prefix=f"{window}__{sensor.lower()}"
                record[f"{prefix}__available_scenes"]=len(group)
                record[f"{prefix}__accepted_scenes"]=int(group.status.eq("ok").sum())
                record[f"{prefix}__clear_pixel_fraction_median"]=float(group.loc[group.status.eq('ok'),"clear_fraction"].median()) if group.status.eq("ok").any() else np.nan
                for name in featnames:
                    record[f"{window}__{name}"]=float(group[name].median()) if group[name].notna().any() else np.nan
        rows.append(record)
    result=pd.DataFrame(rows)
    assert result.unique_id.is_unique
    assert len(result)==labels.shape[0]
    outdir=ROOT/"outputs"/"features"
    outfile=outdir/f"{dataset}_stage_features.csv"
    result.to_csv(outfile,index=False)
    cols=[f"{w}__{n}" for w in WINDOWS for fs in FEATURES.values() for n in fs]
    completeness={w:{s:{"plots_with_usable_scene":int((result[f'{w}__{s.lower()}__accepted_scenes']>0).sum()),
                         "plots_with_ndvi_or_backscatter":int(result[f'{w}__{FEATURES[s][0]}'].notna().sum())}
                     for s in FEATURES} for w in WINDOWS}
    summary={"dataset":dataset,"plot_seasons":len(result),"yield_min":float(result['yield'].min()),
             "yield_median":float(result['yield'].median()),"yield_max":float(result['yield'].max()),
             "candidate_satellite_feature_columns":len(cols),"nonmissing_satellite_cells":int(result[cols].notna().sum().sum()),
             "stage_sensor_coverage":completeness,"output_csv":str(outfile.relative_to(ROOT))}
    (outdir/f"{dataset}_stage_features.summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=["toy","full"],required=True)
    run(parser.parse_args().dataset)
