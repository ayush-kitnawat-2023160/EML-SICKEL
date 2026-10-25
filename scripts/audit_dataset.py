#!/usr/bin/env python3
"""Audit SICKLE paddy yield labels and acquisition dates (stdlib only).

This is a metadata/coverage audit, NOT satellite feature extraction or a
trained yield model. It processes the NPZ file paths rather than ~3M TIFF bands.
"""

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ["W1_0_20", "W2_20_40", "W3_40_60", "W4_60_80", "W5_80_100"]
SENSORS = ["S1", "S2", "L8"]


def parse_date(value):
    try:
        return datetime.strptime(value.strip(), "%m/%d/%Y").date()
    except (TypeError, ValueError):
        return None


def image_date(filename):
    m = re.search(r"(20\d{6})", filename)
    if m is None:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def audit(dataset):
    data = ROOT / "data" / "raw" / ("sickle_toy" if dataset == "toy" else "sickle_full")
    tabular = data / "sickle_dataset_tabular.csv"
    if not tabular.is_file():
        raise SystemExit(f"Dataset is not extracted yet: {tabular}")
    with tabular.open(newline="", encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))

    paddy = [r for r in records if r["CROP"].strip().lower() == "paddy"]
    def positive_yield(r):
        try:
            return math.isfinite(float(r["YIELD"])) and float(r["YIELD"]) > 0
        except (TypeError, ValueError):
            return False

    valid = []
    for r in paddy:
        s, h = parse_date(r["SOWING_DATE"]), parse_date(r["HARVESTING_DATE"])
        if positive_yield(r) and s is not None and h is not None and h > s:
            valid.append((r, s, h))

    outdir = ROOT / "outputs" / "eda"
    outdir.mkdir(parents=True, exist_ok=True)
    observed = outdir / f"{dataset}_observation_metadata.csv"
    coverage = defaultdict(set)
    counts = Counter()
    outside = Counter()
    splits = defaultdict(set)
    for r, _, _ in valid:
        splits[r["SPLIT"]].add(r["PLOT_ID"])

    with observed.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["unique_id", "plot_id", "split", "yield_reported", "sensor", "image_date",
                         "days_since_sowing", "relative_season_time", "window", "npz_relative_path"])
        for r, sowing, harvest in valid:
            uid = r["UNIQUE_ID"]
            for sensor in SENSORS:
                image_dir = data / "images" / sensor / "npy" / uid
                if not image_dir.is_dir():
                    counts[f"{sensor}_missing_dirs"] += 1
                    continue
                for path in image_dir.glob("*.npz"):
                    date = image_date(path.name)
                    if date is None:
                        counts["unparsed_image_dates"] += 1
                        continue
                    if date < sowing or date > harvest:
                        outside[sensor] += 1
                        continue
                    elapsed = (date - sowing).days
                    relative = elapsed / (harvest - sowing).days
                    idx = min(int(relative * len(WINDOWS)), len(WINDOWS) - 1)
                    window = WINDOWS[idx]
                    writer.writerow([uid, r["PLOT_ID"], r["SPLIT"], r["YIELD"], sensor,
                                     date.isoformat(), elapsed, round(relative, 5), window,
                                     path.relative_to(ROOT)])
                    counts[sensor] += 1
                    counts[f"{sensor}_{window}"] += 1
                    coverage[(sensor, window)].add(uid)

    summary = {
        "dataset": dataset,
        "total_rows": len(records),
        "paddy_rows": len(paddy),
        "paddy_positive_yield_rows": sum(positive_yield(r) for r in paddy),
        "usable_yield_and_dates_rows": len(valid),
        "distinct_physical_plots": len({r["PLOT_ID"] for r, _, _ in valid}),
        "usable_rows_by_split": dict(Counter(r["SPLIT"] for r, _, _ in valid)),
        "plot_overlap_train_val": len(splits["train"] & splits["val"]),
        "within_season_npz_images_by_sensor": {s: counts[s] for s in SENSORS},
        "excluded_outside_season_images_by_sensor": dict(outside),
        "coverage": {s: {w: {"images": counts[f"{s}_{w}"],
                              "plots_with_images": len(coverage[(s, w)]),
                              "plot_season_fraction": round(len(coverage[(s, w)]) / len(valid), 4) if valid else None}
                          for w in WINDOWS} for s in SENSORS},
        "missing_image_directories": {s: counts[f"{s}_missing_dirs"] for s in SENSORS},
        "unparsed_image_dates": counts["unparsed_image_dates"],
        "observation_csv": str(observed.relative_to(ROOT)),
        "note": "Relative time uses known harvest date: valid for retrospective stage analysis, NOT leakage-free early prediction.",
    }
    summary_path = outdir / f"{dataset}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["toy", "full"], required=True)
    audit(p.parse_args().dataset)
