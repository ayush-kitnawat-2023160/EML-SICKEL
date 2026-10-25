# SICKLE EML: Temporal Value of Satellite Information

**Status:** Working research prototype with grouped out-of-sample experiments, not a finalized scientific result.  
**Dataset:** Local SICKLE full release, extracted in `data/raw/sickle_full/`.  
**As of:** 25 October 2026.  
**Primary research question:** Which **observation periods** and **satellites** add stable information about eventual paddy yield beyond what is already known about the field?

## Dataset and experimental design

- The downloaded CSV contains **2,370** plot-season records, of which **907** are paddy and **319** have positive yield and valid sow/harvest dates (**131 physical plots**).
- The released split has **282 labelled train** plot-seasons across **117 plots** and **37 validation** plot-seasons across **14 other plots**. No labelled physical plot occurs in both splits.
- 22,728 in-growing-season acquisition files from Sentinel-1, Sentinel-2 and Landsat-8 were identified; the default QC pipeline retained **12,252** plot-scene observations (S1 4,646, S2 6,447, L8 1,159).
- Plot ID comes from the **first of six mask-TIFF bands**, verified in the official SICKLE loader (`reference/sickle_official/utils/dataset.py`). Pixel filtering uses S2 SCL/cloud probability, L8 QA and SR scaling (2.75e-5 scale, -0.2 offset), and finite S1 VV/VH radar observations.
- Five 20%-duration bins and a refined ten 10%-duration binning are **retrospective normalized growing-season windows**. They use the *actual* harvest date and are not valid forecasts in that form.
- **All comparative CV scores** below use the **282 training rows only**, five **PLOT_ID-grouped outer folds** and inner grouped hyperparameter CV. Imputation, scaling and encoding are fit **within each training fold**. The 37-row official validation set is held out of new experiment selection.
- All RMSE/MAE values use the dataset's reported yield-label units; precision of farmer-reported survey yields is a limitation.

## A. Sensor ablation (5 relative-time windows; Ridge)

| Satellites | Grouped-CV RMSE | R² |
|---|---:|---:|
| S1 | 404.40 | -0.024 |
| S2 | 390.64 | 0.044 |
| L8 | 403.75 | -0.021 |
| S1 + S2 | **387.42** | **0.060** |
| S1 + L8 | 402.75 | -0.016 |
| S2 + L8 | 392.19 | 0.037 |
| S1 + S2 + L8 | 387.55 | 0.059 |

**Interpretation:** S1 + S2 and all three sensors perform almost identically (0.13 RMSE difference). The added predictive value of Landsat-8 is not established. This is not evidence that Landsat-8 is intrinsically useless: its 30 m footprint gives only a few plot pixels for many plots, and scene quality/coverage is limited. See `outputs/robustness/sensor_comparison.png`.

## B. Cloud, mask geometry and Landsat plot-size sensitivity

| Change to default feature processing | RMSE |
|---|---:|
| Default: resize masks, S2 cloud prob <=65 | 387.55 |
| Require exactly matching raw mask + 10/30m image dimensions | **416.04** |
| Require >=3 plot pixels for L8 | 389.97 |
| Require >=5 plot pixels for L8 | 393.40 |
| S2 cloud prob <=35 | 388.72 |
| S2 cloud prob <=90 | 387.28 |

Cloud thresholds 35/65/90 retained **6,426 / 6,447 / 6,453** S2 scenes, respectively. SCL exclusions already remove many unusable observations, so these particular probability thresholds change comparatively few accepted scenes. The RMSE differences are small.

**Critical limitation:** Original NPZ scenes are not independently georeferenced with trustworthy global offsets; TIFF transforms inspected alongside the NPZ are local/placeholder pixel coordinates. The default method nearest-neighbor-resizes a plot mask only when dimensions differ by at most 2 pixels and rejects larger differences. An *exact dimensions* policy avoids those approximations but leaves fewer observations, so the two approaches cannot be read as an isolated test of geometric precision. These differences justify further checking against the SICKLE authors' preprocessing, not a definitive claim about the "correct" alignment. The released data do not allow independent recovery of missing global scene offsets.

## C. Ten narrower timing windows

Relative-season bins divide sowing-to-harvest into ten 10%-duration periods. A 10% bin is a **relative time interval**, not an actual biological growth-stage annotation.

| Model | Grouped-CV RMSE |
|---|---:|
| All five 20% windows | **387.55** |
| All ten 10% windows | 398.02 |
| **10-20% alone** | 394.41 |
| 90-100% alone | 398.18 |

In a 10-window leave-one-window-out analysis, removing 10-20% raised error by **4.66** and removing 90-100% raised error by **2.31**, while several intermediate windows had negative ablation deltas (removing them improved prediction). Full values are in `outputs/robustness/ten_window_ablation.csv`.

**Interpretation:** This preliminary model suggests that **10-20%** of the realized season may contain useful signals, but the 10-window model itself performs *worse* than the 5-window model. Window rankings therefore cannot yet support a strong claim about "the single best observation day." Scarce labels, small bins, correlated dates and QC coverage may all contribute. The bootstrap bounds are conditional on already-generated out-of-fold predictions, and 10 related window hypotheses were examined; do **not** call individual windows statistically significant from these bounds.

## D. True early forecasting, with NO future harvest date as a predictor

At each fixed day **15, 30, 45, 60, 75 or 90 after sowing**, only satellite scenes acquired on/before that day were summarized using scene medians and latest available measurements. All released labelled paddy seasons last at least **99 days**, so these cutoffs precede every observed harvest. Sowing day is assumed known at the time of a proposed prediction.

| Available data | Grouped-CV RMSE | R² |
|---|---:|---:|
| Training-fold mean baseline | 400.8 | n/a |
| Satellite scenes acquired by day 15 | 401.27 | -0.008 |
| By day 30 | **396.31** | 0.016 |
| By day 45 | 398.04 | 0.008 |
| By day 60 | 398.62 | 0.005 |
| By day 75 | 409.28 | -0.049 |
| By day 90 | 404.17 | -0.023 |

These are genuinely cutoff-safe predictors **but not strong early yield predictions**. Accuracy does **not** improve monotonically as more images are added. The information can be noisy and regularization/model selection may vary; an acquisition's marginal value is not necessarily positive. This remains a predictive experiment, not a claim of intervention benefit.

## E. A more fundamental EML question: do satellite images help beyond farmer-known information?

Use agronomic controls expected to be available around sowing: **variety, growing-season category, year and reported plot area**. `RIVER_PART` cannot be used here because it is completely missing among the 319 positive-yield annotated paddy records. Controls are associative covariates, not causal confounder adjustment.

| Input features | Grouped-CV RMSE | R² |
|---|---:|---:|
| **Agronomic controls alone** | **351.17** | **0.228** |
| Satellite 5-window features alone | 387.55 | 0.059 |
| Agronomic controls + satellite features | 364.98 | 0.166 |

**Important negative result:** Adding the complete multi-sensor satellite feature set does **not** improve this first grouped-CV Ridge model beyond the survey controls. Controls alone achieve lower error. The controls-versus-combined RMSE delta is **-13.81** (a negative improvement), with a **conditional, plot-cluster bootstrap interval** from approximately **-38.27 to +8.88**. This is not proof satellites contain zero independent information, but an important research finding worth exploring under more parsimonious or group-regularized models.

### Early satellite observations versus controls known at sowing

| Model | RMSE |
|---|---:|
| **Controls known at sowing** | **351.17** |
| Controls + satellite scenes acquired by day 15 | 359.68 |
| Controls + satellite scenes acquired by day 30 | 355.83 |
| Controls + satellite scenes acquired by day 45 | 353.68 |
| Controls + satellite scenes acquired by day 60 | 358.01 |
| Controls + satellite scenes acquired by day 75 | 360.37 |
| Controls + satellite scenes acquired by day 90 | 360.61 |

**Conclusion for this model family and data split:** None of the evaluated early satellite cutoffs improve over the inexpensive control-only predictor. A practical project question is whether **feature selection or better temporal smoothing** can produce *incremental value of information* without increasing model variance.

## F. Econometric learning curve: benefit of more labelled physical plots

Each experiment reduces *physical training plots* in each outer fold (never individual pixels), re-tunes Ridge within the smaller training portion, and evaluates on the **same unseen plot groups**. Four random plot subsamples per fraction were run.

| Training-plot fraction | Mean training plots per outer fold | Mean grouped-CV RMSE | RMSE SD across repeats |
|---|---:|---:|---:|
| 20% | 19.0 | 416.81 | 12.03 |
| 40% | 37.6 | 408.66 | 5.48 |
| 60% | 56.0 | 393.26 | 1.91 |
| 80% | 74.6 | 390.62 | 3.79 |
| 100% | 93.6 | **387.55** | 0.00 |

This suggests **label scarcity matters**: average prediction improves as more independent fields enter training. At 100% sampling, all repeats use the same folds and thus have 0 across-repeat variation by construction. This does **not** imply sampling uncertainty is zero.

## G. Observation coverage and selection-bias diagnostics

See `outputs/robustness/plot_season_observation_coverage.csv` and
`outputs/robustness/coverage_selection_diagnostics.csv`. Coverage is not uniform
by sensor: 30 m L8 measurements often contain very few valid pixels for the
target plot. Area-stratified coverage should not be interpreted as a monotonic
size effect here: most reported areas fall between 0.5 and 1.0, and the 1.25
group has only three plot-seasons.

Missingness may be correlated with field geometry, cloud patterns, season,
sensor acquisition frequency and soil/crop state. It should not be silently
set to NDVI=0. The primary models impute *within training folds*, rather than
using unavailable observations or labels from evaluation plots.

## Research questions most appropriate to EML (next prioritization)

1. **Incremental predictive value (highest priority):** conditional on area,
   variety, year and season, which one or two **sensor-stage feature groups**
   reduce out-of-plot RMSE? Use **group LASSO / sparse-group LASSO** or stable
   grouped ablations instead of blindly concatenating all 55 predictors.
2. **Econometrics of scarce labelled plots:** interpret the learning curve;
   explicitly assess how regularization is tuned, and whether shrinking a
   correlated predictor space yields better generalization than OLS.
3. **Within-plot temporal variation:** many PLOT_IDs have repeated plot-season
   records. Explore *plot fixed effects* or first differences to control for
   time-invariant field heterogeneity **within sampled plots**. This answers
   a different question from generalizing to entirely unseen plots; do not
   confuse the two. The small survey panel may be too noisy for stable effects.
4. **Non-random satellite missingness:** do scene availability, cloud coverage,
   and plot-size patterns predict reported yield, even after controlling for
   year and variety? Test **associations** and robustness to complete-case
   selection; never assume missing at random without evidence.
5. **Temporal regularization instead of arbitrary bins:** fit a low-degree
   spline or smooth time-varying coefficient for NDVI/VV and tune smoothing
   strength inside grouped CV. Check whether the apparent 10-20% and late
   importance peaks persist across time-binning choices.
6. **Decisions under uncertainty:** calibrate out-of-plot predictive intervals
   and estimate the value of an extra satellite acquisition for reducing
   uncertainty versus acquisition/processing cost. Don't overpromise
   distribution-free coverage when repeated plots violate independence.
7. **External validity:** with only two main yield-labelled years (2019/2020),
   try **hold-one-year-out** and year-by-year error analyses. Expect unstable
   estimates. A model that merely learns crop variety/season distribution may
   fail under a new farming year or weather regime.

## Files / figures worth showing your professor

- `outputs/robustness/sensor_comparison.png`
- `outputs/robustness/mask_landsat_sensitivity.png`
- `outputs/robustness/cloud_threshold_sensitivity.png`
- `outputs/robustness/ten_window_ablation.png`
- `outputs/robustness/early_forecasting.png`
- `outputs/robustness/incremental_satellite_value.png`
- `outputs/robustness/early_satellite_value_over_controls.png`
- `outputs/robustness/learning_curve_plot_fraction.png`
- `outputs/models/full_lasso_selection_heatmap.png`

## Scientific guardrails

- These are **predictive associations**, not causal estimates of the effect of
  crop stage, field management, satellite frequency or crop type on yield.
- The released labels are **retrospective farmer survey reports**, not
  independently measured crop weights.
- The models compare many correlated hypotheses on one limited dataset. OOF
  comparisons are useful research leads; post-hoc best-window rankings and
  conditional bootstrap intervals are **not confirmatory inferential tests**.
- Spatial mask alignment is still approximate. Changes in strict-versus-resize
  coverage must be reported alongside performance; raw cutout TIFFs use
  placeholder pixel-space transforms, not recoverable global spatial offsets.
- The retrospective stage analyses know actual harvest time. **Only the
  fixed days-since-sowing early analysis is leakage-free with respect to harvest
  timing and future observations.**
- This work is distinct from the pretrained neural SICKLE models in `data/weights`.
  The SICKLE paper itself already studies actual-season versus regional-standard
  temporal sequences; that is **not** our novelty claim.
