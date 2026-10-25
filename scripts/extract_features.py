#!/usr/bin/env python3
"""Extract physically interpretable plot-level features from SICKLE NPZ scenes.

Uses official mask channel 1 (plot ID), 10 m masks for S1/S2 and 30 m
for Landsat 8. Keeps an audit row for every acquisition, including failures.
Only the *target* plot is summarized; neighboring plots are never pooled.

Output: outputs/features/{dataset}_observation_features.csv and QA JSON.
Requires numpy, pandas, rasterio, opencv-python (cv2).
"""

import argparse
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio


ROOT = Path(__file__).resolve().parents[1]
FEATURES = {
    "S1": ["s1_vv_db", "s1_vh_db", "s1_vv_minus_vh_db"],
    "S2": ["s2_ndvi", "s2_ndmi", "s2_ndre", "s2_red_reflectance", "s2_nir_reflectance"],
    "L8": ["l8_ndvi", "l8_red_reflectance", "l8_nir_reflectance"],
}
ALL_FEATURES = [f for features in FEATURES.values() for f in features]
# S2 Scene Classification Layer classes 4 (vegetation), 5 (nonveg), 6 (water),
# 7 (unclassified) are retained, with cloud probability <= 65 when supplied.
S2_OK_SCL = (4, 5, 6, 7)
# Landsat Collection 2 Level-2 surface reflectance from the USGS product.
L8_SR_SCALE, L8_SR_OFFSET = 2.75e-5, -0.2


def resize_nn(array, shape):
    """Resize categorical raster to target (rows, cols), nearest neighbor."""
    return cv2.resize(np.asarray(array), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def resize_float(array, shape):
    """Resize continuous data to target raster shape, using linear resampling."""
    return cv2.resize(np.asarray(array, dtype=np.float32), (shape[1], shape[0]),
                      interpolation=cv2.INTER_LINEAR)


def plot_pixels(mask, plot_id, shape):
    """Select only the requested plot; enforce near-matching native tile geometry.

    A one-pixel size difference occurs in released NPZ cutouts. For a major
    mismatch, refuse to interpret it as the same footprint. For 20 m S2
    bands, resize the 10 m mask by design (S2 quality/index resolution).
    """
    raw = np.asarray(mask)
    return resize_nn(raw, shape) == plot_id


def safe_median(array, selection):
    values = np.asarray(array)[selection]
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


def normalized_difference(x, y):
    denom = x + y
    return np.divide(x - y, denom, out=np.full_like(x, np.nan, dtype=np.float32),
                     where=np.abs(denom) > 1e-5)


def valid_s2(z, shape, cloud_threshold=65):
    """Quality mask at a band/index's native shape.

    Missing SCL: reject scene instead of calling it cloud-free. A missing
    cloud probability is okay if SCL is present. Validity is conservative.
    """
    if "SCL" not in z.files:
        return None
    scl = resize_nn(z["SCL"], shape)
    valid = np.isin(scl, S2_OK_SCL)
    if "MSK_CLDPRB" in z.files:
        prob = resize_float(z["MSK_CLDPRB"], shape)
        valid &= np.isfinite(prob) & (prob <= cloud_threshold)
    return valid


def valid_l8(z, shape):
    if "QA_PIXEL" not in z.files:
        return None
    qa = resize_nn(z["QA_PIXEL"], shape).astype(np.uint32)
    # Bits: 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 shadow, 5 snow.
    valid = (qa & np.uint32(0b111111)) == 0
    if "QA_RADSAT" in z.files:
        radsat = resize_nn(z["QA_RADSAT"], shape).astype(np.uint32)
        # SR_B4 uses band 4 (bit 3); SR_B5 band 5 (bit 4).
        valid &= (radsat & np.uint32((1 << 3) | (1 << 4))) == 0
    return valid


@lru_cache(maxsize=1500)
def read_plot_mask(dataset, uid, sensor):
    mask_res = "30m" if sensor == "L8" else "10m"
    path = ROOT / "data" / "raw" / dataset / "masks" / mask_res / f"{uid}.tif"
    with rasterio.open(path) as ds:
        assert ds.count == 6, f"Expected official six-layer mask, got {ds.count}: {path}"
        # First of six bands is *physical* PLOT_ID, not binary cropland.
        return ds.read(1)


def base_result():
    return {**{f: np.nan for f in ALL_FEATURES}, "status": "unknown",
            "plot_pixels": 0, "valid_pixels": 0, "clear_fraction": np.nan,
            "valid_pixels_20m": 0, "mask_shape": "", "image_shape": ""}


def extract_one(row, dataset, cloud_threshold=65, mask_policy="resize"):
    out = base_result()
    sensor = row.sensor
    pid = int(float(row.plot_id))
    uid = str(row.unique_id)
    path = ROOT / row.npz_relative_path
    try:
        mask = read_plot_mask(dataset, uid, sensor)
        out["mask_shape"] = str(tuple(mask.shape))
        if not np.any(mask == pid):
            out["status"] = "plot_absent_in_native_mask"
            return out
        with np.load(path, allow_pickle=False) as z:
            if sensor == "S1":
                if not {"VV", "VH"} <= set(z.files):
                    out["status"] = "missing_bands"
                    return out
                vv, vh = z["VV"].astype(np.float32), z["VH"].astype(np.float32)
                if vv.shape != vh.shape:
                    out["status"] = "band_shape_mismatch"
                    return out
                shape = vv.shape
                out["image_shape"] = str(shape)
                if mask_policy == "exact" and shape != mask.shape:
                    out["status"] = "mask_shape_not_exact"
                    return out
                if max(abs(a-b) for a,b in zip(shape,mask.shape))>2:
                    out["status"] = "mask_image_geometry_mismatch"
                    return out
                plot = plot_pixels(mask, pid, shape)
                valid = plot & np.isfinite(vv) & np.isfinite(vh) & (vv != 0) & (vh != 0)
                valid &= (vv > -55) & (vv < 15) & (vh > -55) & (vh < 15)
                out["plot_pixels"] = int(plot.sum())
                out["valid_pixels"] = int(valid.sum())
                out["clear_fraction"] = float(valid.sum() / plot.sum()) if plot.any() else np.nan
                if not valid.any():
                    out["status"] = "no_valid_pixels"
                    return out
                out["s1_vv_db"] = safe_median(vv, valid)
                out["s1_vh_db"] = safe_median(vh, valid)
                out["s1_vv_minus_vh_db"] = safe_median(vv - vh, valid)

            elif sensor == "S2":
                if not {"B4", "B8"} <= set(z.files):
                    out["status"] = "missing_bands"
                    return out
                red, nir = z["B4"].astype(np.float32), z["B8"].astype(np.float32)
                if red.shape != nir.shape:
                    out["status"] = "band_shape_mismatch"
                    return out
                shape = red.shape
                out["image_shape"] = str(shape)
                if mask_policy == "exact" and shape != mask.shape:
                    out["status"] = "mask_shape_not_exact"
                    return out
                if max(abs(a-b) for a,b in zip(shape,mask.shape))>2:
                    out["status"] = "mask_image_geometry_mismatch"
                    return out
                plot = plot_pixels(mask, pid, shape)
                quality = valid_s2(z, shape, cloud_threshold)
                if quality is None:
                    out["status"] = "missing_quality_layer"
                    return out
                valid = plot & quality & np.isfinite(red) & np.isfinite(nir)
                valid &= (red > 0) & (nir > 0) & (red < 20000) & (nir < 20000)
                out["plot_pixels"] = int(plot.sum())
                out["valid_pixels"] = int(valid.sum())
                out["clear_fraction"] = float(valid.sum() / plot.sum()) if plot.any() else np.nan
                if not valid.any():
                    out["status"] = "no_clear_plot_pixels"
                    return out
                # Normalized indices are valid on the provided raw S2 values;
                # reflectance outputs are approximate B4/B8 DN/10000, with no
                # harmonization of post-2022 DN baselines required for 2018–21.
                out["s2_ndvi"] = safe_median(normalized_difference(nir, red), valid)
                out["s2_red_reflectance"] = safe_median(red/10000.0, valid)
                out["s2_nir_reflectance"] = safe_median(nir/10000.0, valid)
                if {"B8A", "B11", "B5"} <= set(z.files):
                    b8a = z["B8A"].astype(np.float32)
                    b11 = z["B11"].astype(np.float32)
                    b5 = z["B5"].astype(np.float32)
                    if b8a.shape == b11.shape == b5.shape:
                        plot20 = plot_pixels(mask, pid, b8a.shape)
                        clear20 = valid_s2(z, b8a.shape, cloud_threshold)
                        if clear20 is not None:
                            good20 = plot20 & clear20 & np.isfinite(b8a) & np.isfinite(b11) & np.isfinite(b5)
                            good20 &= (b8a > 0) & (b11 > 0) & (b5 > 0)
                            good20 &= (b8a < 20000) & (b11 < 20000) & (b5 < 20000)
                            out["valid_pixels_20m"] = int(good20.sum())
                            if good20.any():
                                # NDMI: (NIR narrow / SWIR1), NDRE: (NIR narrow / red-edge 1)
                                out["s2_ndmi"] = safe_median(normalized_difference(b8a, b11), good20)
                                out["s2_ndre"] = safe_median(normalized_difference(b8a, b5), good20)

            elif sensor == "L8":
                if not {"SR_B4", "SR_B5"} <= set(z.files):
                    out["status"] = "missing_bands"
                    return out
                raw_red, raw_nir = z["SR_B4"], z["SR_B5"]
                if raw_red.shape != raw_nir.shape:
                    out["status"] = "band_shape_mismatch"
                    return out
                shape = raw_red.shape
                out["image_shape"] = str(shape)
                if mask_policy == "exact" and shape != mask.shape:
                    out["status"] = "mask_shape_not_exact"
                    return out
                if max(abs(a-b) for a,b in zip(shape,mask.shape))>2:
                    out["status"] = "mask_image_geometry_mismatch"
                    return out
                plot = plot_pixels(mask, pid, shape)
                quality = valid_l8(z, shape)
                if quality is None:
                    out["status"] = "missing_quality_layer"
                    return out
                red = raw_red.astype(np.float32)*L8_SR_SCALE+L8_SR_OFFSET
                nir = raw_nir.astype(np.float32)*L8_SR_SCALE+L8_SR_OFFSET
                valid = plot & quality & np.isfinite(red) & np.isfinite(nir)
                valid &= (raw_red > 0) & (raw_nir > 0) & (red > 0) & (nir > 0)
                valid &= (red < 1.6) & (nir < 1.6)
                out["plot_pixels"] = int(plot.sum())
                out["valid_pixels"] = int(valid.sum())
                out["clear_fraction"] = float(valid.sum() / plot.sum()) if plot.any() else np.nan
                if not valid.any():
                    out["status"] = "no_clear_plot_pixels"
                    return out
                out["l8_ndvi"] = safe_median(normalized_difference(nir,red),valid)
                out["l8_red_reflectance"] = safe_median(red,valid)
                out["l8_nir_reflectance"] = safe_median(nir,valid)
            else:
                raise ValueError(f"Unexpected sensor {sensor}")
        out["status"] = "ok"
    except (OSError, ValueError, KeyError, rasterio.errors.RasterioError) as e:
        out["status"] = "io_or_decode_error"
        out["error_message"] = str(e)[:200]
    return out


def run(dataset, max_plots, max_images, cloud_threshold=65, mask_policy="resize", sensors=None, tag=""):
    dataset_dir = "sickle_toy" if dataset == "toy" else "sickle_full"
    meta_path = ROOT / "outputs" / "eda" / f"{dataset}_observation_metadata.csv"
    if not meta_path.is_file():
        raise SystemExit(f"Run `python scripts/audit_dataset.py --dataset {dataset}` first")
    metadata = pd.read_csv(meta_path, dtype={"unique_id":str,"plot_id":str})
    if sensors:
        metadata=metadata[metadata.sensor.isin(sensors)]
    if max_plots:
        ids = list(metadata.unique_id.drop_duplicates())[:max_plots]
        metadata = metadata[metadata.unique_id.isin(ids)]
    if max_images:
        metadata = metadata.head(max_images)
    records=[]
    for i, row in enumerate(metadata.itertuples(index=False)):
        records.append(extract_one(row, dataset_dir, cloud_threshold=cloud_threshold, mask_policy=mask_policy))
        if (i+1) % 1000 == 0:
            print(f"Processed {i+1}/{len(metadata)} acquisitions", flush=True)
    result = pd.concat([metadata.reset_index(drop=True),pd.DataFrame(records)],axis=1)
    out_dir = ROOT / "outputs" / "features"
    out_dir.mkdir(parents=True,exist_ok=True)
    suffix = (f"_first_{max_plots}_plots" if max_plots else "") + (f"_first_{max_images}_images" if max_images else "") + (f"_{tag}" if tag else "")
    outfile = out_dir / f"{dataset}{suffix}_observation_features.csv"
    result.to_csv(outfile,index=False)
    stats = {
        "dataset":dataset,
        "cloud_threshold":cloud_threshold,"mask_policy":mask_policy,"sensors":sensors or ["S1","S2","L8"],
        "total_acquisitions":len(result),
        "status_counts":dict(Counter(result.status)),
        "status_by_sensor":{s:dict(Counter(part.status)) for s,part in result.groupby("sensor")},
        "features_nonnull":{key:int(result[key].notna().sum()) for key in ALL_FEATURES},
        "features_range":{key:[float(result[key].min()),float(result[key].max())] if result[key].notna().any() else None for key in ALL_FEATURES},
        "output_csv":str(outfile.relative_to(ROOT)),
        "important_note":"Quality-filtered plot medians. Output is NOT a trained model. Nearest-neighbor mask alignment is an approximation for size-shifted cutouts."
    }
    json_out = outfile.with_suffix(".summary.json")
    json_out.write_text(json.dumps(stats,indent=2)+"\n")
    print(json.dumps(stats,indent=2))


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--dataset",required=True,choices=["toy","full"])
    a.add_argument("--max-plots",type=int,default=0,help="limit physical sample count for smoke tests")
    a.add_argument("--max-images",type=int,default=0,help="limit acquisitions for smoke tests")
    a.add_argument("--s2-cloud-threshold",type=int,default=65,help="S2 cloud probability threshold 0..100")
    a.add_argument("--mask-policy",choices=["resize","exact"],default="resize")
    a.add_argument("--sensors",nargs="+",choices=["S1","S2","L8"])
    a.add_argument("--tag",default="",help="Output-name suffix for alternate experiments; required when overriding defaults")
    x=a.parse_args()
    if not 0 <= x.s2_cloud_threshold <= 100: a.error("cloud threshold must be 0..100")
    if (x.s2_cloud_threshold!=65 or x.mask_policy!="resize" or x.sensors) and not x.tag:
        a.error("Specify --tag for an alternative experiment to preserve canonical baseline CSV")
    run(x.dataset,x.max_plots,x.max_images,x.s2_cloud_threshold,x.mask_policy,x.sensors,x.tag)
