#!/usr/bin/env python3
"""EML penalty experiments for SICKLE, TRAIN-only plot-grouped nested CV.

Models are partially penalized: the intercept and agronomic controls (area,
year, season, variety, and a year-by-variety indicator) are unpenalized.
Satellites alone get L1, Elastic Net, adaptive LASSO, or SCAD penalties.

Frisch-Waugh-Lovell eliminates controls from the joint objective, retaining
the *exact* partial-penalty convex problems (for LASSO/Elastic Net).
SCAD is nonconvex: local-linear approximation (LLA), initialized from LASSO.
Post-LASSO refits the controls and selected satellite coefficients jointly.

All learned preprocessing and all penalty selection is fold-local.
Retrospective features: observed-harvest-normalized windows, NOT real-time
early forecasting. Official VAL rows are never used for fitting or scoring.

Outputs (CSV and JSON only) are written to outputs/eml_penalties.
"""

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from extract_features import FEATURES
from stage_importance_baseline import ROOT, STAGES, metrics

OUT = ROOT / "outputs" / "eml_penalties"
BASE = tuple(f"{window}__{f}" for window in STAGES for fs in FEATURES.values() for f in fs)
assert len(BASE) == 55
FRACS = (0.03, 0.08, 0.15, 0.30, 0.60)
SCAD_A = 3.7

# All of these are fixed before running, rather than selected by correlations
# with y. Nonlinear terms are made *after fold-local scaling and imputation*.
QUADRATIC = tuple(f"{w}__s2_ndvi_squared" for w in STAGES)
INTERACTIONS = tuple(f"{w}__s2_ndvi_x_s1_vv_minus_vh" for w in STAGES)
EXPANDED = BASE + QUADRATIC + INTERACTIONS

# Predeclared physical-unit contrasts; computed BEFORE imputation and scaled
# strictly within each training fold. Requiring both scene-window values
# avoids treating a missing stage as actual zero vegetation/backscatter.
TRAJECTORY_VARIABLES = (
    "s2_ndvi", "s2_ndmi", "s1_vv_db", "s1_vv_minus_vh_db", "l8_ndvi"
)
TRAJECTORY_PAIRS = ((0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 2), (2, 4), (0, 4))
TRAJECTORY = tuple(
    f"trajectory__{feature}__{STAGES[b]}_minus_{STAGES[a]}"
    for feature in TRAJECTORY_VARIABLES for a, b in TRAJECTORY_PAIRS
) + tuple(f"trajectory__{feature}__season_range"
          for feature in TRAJECTORY_VARIABLES) + (
    "trajectory__s2_ndvi__midseason_curvature",
    "trajectory__s1_vv_db__midseason_curvature",
)
assert len(TRAJECTORY) == 42


def raw_trajectory_matrix(raw):
    """Create 42 predeclared stage-trajectory features with pairwise validity.

    Input is a *raw* float (N,55) matrix (NaNs preserved). Never use
    imputation or scaling before taking physical-unit stage differences.
    """
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(BASE):
        raise ValueError("Need raw 55-column feature matrix in BASE order")
    features = []
    ranges = []
    for name in TRAJECTORY_VARIABLES:
        indices = [BASE.index(f"{w}__{name}") for w in STAGES]
        stages = raw[:, indices]
        for a, b in TRAJECTORY_PAIRS:
            features.append(stages[:, b] - stages[:, a])
        counts = np.isfinite(stages).sum(axis=1)
        # A season's amplitude requires at least three observed stages.
        hi = np.where(np.isfinite(stages), stages, -np.inf).max(axis=1)
        lo = np.where(np.isfinite(stages), stages, np.inf).min(axis=1)
        amplitude = np.where(counts >= 3, hi - lo, np.nan)
        ranges.append(amplitude)
    # The ordering must match TRAJECTORY: all seven differences for each
    # variable, followed by one season-wide range for each variable.
    features.extend(ranges)
    for name in ("s2_ndvi", "s1_vv_db"):
        stages = raw[:, [BASE.index(f"{w}__{name}") for w in STAGES]]
        features.append(stages[:, 2] - (stages[:, 0] + stages[:, 4]) / 2.)
    return np.column_stack(features)


@dataclass(frozen=True)
class Experiment:
    name: str
    kind: str
    expanded: bool = False
    penalize_agro_interactions: bool = False
    trajectory: bool = False
    allowed_sensors: tuple | None = None


EXPERIMENTS = [
    Experiment("controls_only_OLS", "controls"),
    Experiment("partial_lasso", "lasso"),
    Experiment("post_lasso_OLS", "post_lasso"),
    Experiment("partial_elastic_net", "elastic"),
    Experiment("adaptive_lasso", "adaptive"),
    Experiment("SCAD_LLA", "scad"),
    Experiment("nonlinear_partial_elastic_net", "elastic", True),
    Experiment("nonlinear_adaptive_lasso", "adaptive", True),
    Experiment("nonlinear_SCAD_LLA", "scad", True),
    Experiment("agro_interaction_elastic_net", "elastic", False, True),
    Experiment("nonlinear_agro_interaction_elastic_net", "elastic", True, True),
    Experiment("nonlinear_agro_interaction_adaptive_lasso", "adaptive", True, True),
    Experiment("nonlinear_agro_interaction_SCAD_LLA", "scad", True, True),
]

TRAJECTORY_EXPERIMENTS = (
    Experiment("trajectory_partial_elastic_net", "elastic", trajectory=True),
    Experiment("trajectory_adaptive_lasso", "adaptive", trajectory=True),
    Experiment("trajectory_SCAD_LLA", "scad", trajectory=True),
    Experiment("trajectory_nonlinear_elastic_net", "elastic", expanded=True, trajectory=True),
    Experiment("trajectory_nonlinear_adaptive_lasso", "adaptive", expanded=True, trajectory=True),
    Experiment("trajectory_nonlinear_SCAD_LLA", "scad", expanded=True, trajectory=True),
)


def add_control_columns(frame):
    result = frame.copy()
    result["variety"] = result["variety"].fillna("__unknown__").astype(str)
    result["season"] = result["season"].fillna("__unknown__").astype(str)
    result["year_variety"] = (pd.to_numeric(result["year"], errors="coerce")
                              .fillna(-1).astype(int).astype(str)
                              + "__" + result["variety"])
    result["area_class"] = pd.to_numeric(result["area"], errors="coerce").fillna(-1).astype(str)
    return result


def shrink(z, cuts):
    return np.sign(z) * np.maximum(np.abs(z) - cuts, 0)


def proximal_elastic_net(gram, linear, lam, weights, ratio, beta=None, tol=1e-7,
                         max_iter=800):
    """Solve 0.5*b'G*b-h'b + lam*r*sum(w*abs(b)) + .5*lam*(1-r)*||b||²."""
    p = len(linear)
    if beta is None:
        beta = np.zeros(p)
    else:
        beta = beta.copy()
    if lam < 0 or not 0 < ratio <= 1:
        raise ValueError("Invalid lambda or Elastic Net l1_ratio")
    lipschitz = float(np.linalg.eigvalsh(gram).max() + lam * (1 - ratio))
    if lipschitz < 1e-12:
        return np.zeros(p)
    z = beta.copy()
    t = 1.
    for i in range(max_iter):
        grad = gram @ z - linear + lam * (1 - ratio) * z
        updated = shrink(z - grad / lipschitz, lam * ratio * weights / lipschitz)
        if np.linalg.norm(updated - beta) < tol * (1 + np.linalg.norm(beta)):
            return updated
        next_t = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = updated + ((t - 1) / next_t) * (updated - beta)
        beta = updated
        t = next_t
        if i % 100 == 99:  # periodic restart helps with near-collinearity
            z = beta.copy()
            t = 1.
    return beta


def scad_derivative_ratio(coefs, lam, a=SCAD_A):
    """p'_lambda(|beta|)/lambda for SCAD, with a=3.7."""
    abs_coefs = np.abs(coefs)
    weights = np.ones_like(abs_coefs)
    middle = (abs_coefs > lam) & (abs_coefs < a * lam)
    weights[middle] = (a * lam - abs_coefs[middle]) / ((a - 1) * lam)
    weights[abs_coefs >= a * lam] = 0.
    return weights


def scad_objective(gram, linear, beta, lam, a=SCAD_A):
    """SCAD objective up to the constant y'y/(2n); diagnostics only."""
    t = np.abs(beta)
    penalty = np.where(
        t <= lam, lam * t,
        np.where(t <= a * lam,
                 (-t*t + 2*a*lam*t - lam*lam) / (2*(a-1)),
                 .5 * (a+1) * lam*lam))
    return float(.5 * beta @ gram @ beta - linear @ beta + penalty.sum())


class PreparedFit:
    def __init__(self, train, y, expanded, penalize_agro_interactions=False,
                 trajectory=False, allowed_sensors=None, cross_year=False):
        train = add_control_columns(train)
        self.expanded = expanded
        self.penalize_agro_interactions = penalize_agro_interactions
        self.trajectory = trajectory
        self.allowed_sensors = allowed_sensors
        self.cross_year = cross_year
        self.n = len(train)
        self.numimp = SimpleImputer(strategy="median", keep_empty_features=True)
        self.numscale = StandardScaler()
        self.control_numeric = ["area"] if cross_year else ["area", "year"]
        control_num = self.numscale.fit_transform(
            self.numimp.fit_transform(train[self.control_numeric].apply(pd.to_numeric, errors="coerce")))
        self.onehot = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.control_categories = ["season", "variety"] if cross_year or penalize_agro_interactions else [
            "season", "variety", "year_variety"]
        control_cat = self.onehot.fit_transform(train[self.control_categories])
        C = np.column_stack([np.ones(self.n), control_num, control_cat])

        self.satimp = SimpleImputer(strategy="median", keep_empty_features=True)
        self.satscale = StandardScaler()
        base_raw = self._raw_base(train)
        base = self.satscale.fit_transform(self.satimp.fit_transform(base_raw))
        X = self._expanded_transform(base, fit=True)
        self.names = EXPANDED if expanded else BASE
        if trajectory:
            self.trajimp = SimpleImputer(strategy="median", keep_empty_features=True)
            self.trajscale = StandardScaler()
            contrasts = self.trajscale.fit_transform(
                self.trajimp.fit_transform(raw_trajectory_matrix(base_raw)))
            X = np.column_stack((X, contrasts))
            self.names = tuple(self.names) + TRAJECTORY
        if penalize_agro_interactions:
            # Complex survey interactions get shrinkage too, keeping
            # intercept, area, year, variety and season *unpenalized*.
            self.agro_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            keys = ["year_variety", "area_class"]
            ag = self.agro_encoder.fit_transform(train[keys])
            self.agro_scaler = StandardScaler()
            ag = self.agro_scaler.fit_transform(ag)
            X = np.column_stack((X, ag))
            self.names = tuple(self.names) + tuple(
                "agro__" + c for c in self.agro_encoder.get_feature_names_out(keys))
        assert X.shape[1] == len(self.names)
        self.ymean = float(np.mean(y))
        self.ystd = max(float(np.std(y)), 1e-12)
        standardized_y = (np.asarray(y, float) - self.ymean) / self.ystd
        projection = np.linalg.lstsq(C, np.column_stack([standardized_y, X]), rcond=1e-10)[0]
        yc = standardized_y - C @ projection[:, 0]
        xc = X - C @ projection[:, 1:]
        self.linear = xc.T @ yc / self.n
        self.gram = xc.T @ xc / self.n
        self.lambda_max = max(float(np.max(np.abs(self.linear))), 1e-12)
        self.yc = yc
        self.xc = xc
        self.projection = projection
        self.controls_train = C
        self.features_train = X
        self.ystandard_train = standardized_y
        self.init_ridge = np.linalg.solve(
            self.gram + .1 * np.eye(len(self.names)), self.linear)
        # Training-only initial Ridge coefficients define Adaptive LASSO weights.
        weights = 1. / (np.abs(self.init_ridge) + .05)
        self.adaptive_weights = np.clip(weights / np.median(weights), .2, 12.)

    def _expanded_transform(self, base, fit=False):
        if not self.expanded:
            return base
        cols = []
        for window in STAGES:
            n = BASE.index(f"{window}__s2_ndvi")
            s = BASE.index(f"{window}__s1_vv_minus_vh_db")
            cols.append(base[:, n] ** 2)
        for window in STAGES:
            n = BASE.index(f"{window}__s2_ndvi")
            s = BASE.index(f"{window}__s1_vv_minus_vh_db")
            cols.append(base[:, n] * base[:, s])
        nonlinear = np.column_stack(cols)
        if fit:
            self.extra_scaler = StandardScaler().fit(nonlinear)
        return np.column_stack([base, self.extra_scaler.transform(nonlinear)])

    def _raw_base(self, frame):
        base = frame[list(BASE)].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=float, copy=True)
        if self.allowed_sensors is not None:
            for col, name in enumerate(BASE):
                sensor = name.rsplit("__", 1)[1].split("_", 1)[0]
                if sensor not in self.allowed_sensors:
                    base[:, col] = np.nan
        return base

    def matrices(self, frame):
        frame = add_control_columns(frame)
        cn = self.numscale.transform(
            self.numimp.transform(frame[self.control_numeric].apply(pd.to_numeric, errors="coerce")))
        cc = self.onehot.transform(frame[self.control_categories])
        C = np.column_stack([np.ones(len(frame)), cn, cc])
        base_raw = self._raw_base(frame)
        base = self.satscale.transform(self.satimp.transform(base_raw))
        X = self._expanded_transform(base)
        if self.trajectory:
            contrasts = self.trajscale.transform(
                self.trajimp.transform(raw_trajectory_matrix(base_raw)))
            X = np.column_stack((X, contrasts))
        if self.penalize_agro_interactions:
            extra = self.agro_scaler.transform(
                self.agro_encoder.transform(frame[["year_variety", "area_class"]]))
            X = np.column_stack((X, extra))
        return C, X

    def fit_coefficients(self, kind, frac, ratio=1.):
        if kind == "controls":
            return np.zeros(len(self.names))
        lam = float(frac) * self.lambda_max
        weights = np.ones(len(self.names))
        if kind == "adaptive":
            weights = self.adaptive_weights
        beta = proximal_elastic_net(self.gram, self.linear, lam, weights,
                                    ratio if kind == "elastic" else 1.)
        if kind == "scad":
            # Local linear approximation, initialized at convex LASSO solution.
            # SCAD is nonconvex; this returns a local, not a global, solution.
            beta, _ = self.scad_from_initialization(frac, "lasso",
                                                    initial_lasso_beta=beta)
        if kind == "post_lasso":
            selected = np.flatnonzero(np.abs(beta) > 1e-5)
            new_beta = np.zeros(len(beta))
            if len(selected):
                # FWL joint OLS on controls and training-selected features.
                new_beta[selected] = np.linalg.lstsq(
                    self.xc[:, selected], self.yc, rcond=1e-8)[0]
            beta = new_beta
        return beta

    def scad_from_initialization(self, frac, initialization="lasso",
                                 initial_lasso_beta=None):
        """LLA iteration with explicit objective trace for multi-start checks."""
        lam = float(frac) * self.lambda_max
        if initialization == "lasso":
            beta = (initial_lasso_beta.copy() if initial_lasso_beta is not None
                    else proximal_elastic_net(
                        self.gram, self.linear, lam, np.ones(len(self.names)), 1.))
        elif initialization == "ridge":
            beta = self.init_ridge.copy()
        elif initialization == "zero":
            beta = np.zeros(len(self.names))
        else:
            raise ValueError("Unknown SCAD initialization")
        history = [scad_objective(self.gram, self.linear, beta, lam)]
        for _ in range(7):
            weights = scad_derivative_ratio(beta, lam)
            updated = proximal_elastic_net(self.gram, self.linear, lam, weights, 1., beta)
            history.append(scad_objective(self.gram, self.linear, updated, lam))
            converged = np.linalg.norm(beta - updated) <= 1e-5 * (1 + np.linalg.norm(beta))
            beta = updated
            if converged:
                break
        return beta, history

    def predict(self, frame, beta):
        C, X = self.matrices(frame)
        gamma = self.projection[:, 0] - self.projection[:, 1:] @ beta
        return self.ymean + self.ystd * (C @ gamma + X @ beta)


def candidates(experiment):
    if experiment.kind == "controls":
        return [(0., 1.)]
    if experiment.kind == "elastic":
        return [(lam, ratio) for lam in FRACS for ratio in (.35, .70)]
    return [(lam, 1.) for lam in FRACS]


def tune(frame, y, groups, experiment, cross_year=False):
    inner = list(GroupKFold(n_splits=3).split(frame, y, groups))
    prepared = []
    for itrain, ival in inner:
        prepared.append((PreparedFit(frame.iloc[itrain], y[itrain], experiment.expanded,
                                     experiment.penalize_agro_interactions,
                                     experiment.trajectory, experiment.allowed_sensors,
                                     cross_year=cross_year),
                         frame.iloc[ival], ival))
    rows = []
    for fraction, ratio in candidates(experiment):
        pred = np.full(len(frame), np.nan)
        for fitted, hold, idx in prepared:
            beta = fitted.fit_coefficients(experiment.kind, fraction, ratio)
            pred[idx] = fitted.predict(hold, beta)
        rows.append(float(np.mean((pred - y) ** 2)))
    winner = int(np.argmin(rows))
    return (*candidates(experiment)[winner], float(rows[winner]))


def trajectory_run(repeats=12):
    """Compare predeclared phenological trajectories against older EML models."""
    OUT.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(ROOT / "outputs/features/full_stage_features.csv",
                       dtype={"unique_id": str, "plot_id": str})
    train = full.loc[full.split.eq("train")].reset_index(drop=True)
    validation = full.loc[full.split.eq("val")]
    assert len(train) == 282 and len(validation) == 37
    assert not set(train.plot_id) & set(validation.plot_id)
    y, groups = train["yield"].to_numpy(float), train.plot_id.to_numpy()
    outer = list(GroupKFold(n_splits=5).split(train, y, groups))

    # The original EML models are already nested-CV-fitted on exactly these
    # plots, with the same evaluation splits. Align via identifiers to avoid
    # accidentally comparing different rows.
    previous = pd.read_csv(OUT / "grouped_cv_predictions.csv",
                           dtype={"unique_id": str, "plot_id": str})
    previous = previous.set_index("unique_id").reindex(train.unique_id)
    assert previous.plot_id.tolist() == train.plot_id.tolist()
    assert np.allclose(previous["yield"].to_numpy(float), y)
    previous_scores = pd.read_csv(OUT / "grouped_cv_comparison.csv").set_index("model")
    previous_methods = {
        "elastic": "nonlinear_partial_elastic_net",
        "adaptive": "nonlinear_adaptive_lasso",
        "scad": "nonlinear_SCAD_LLA",
    }

    predictions = {"unique_id": train.unique_id, "plot_id": groups, "yield": y}
    rows = []
    tuning = {}
    selections = {}
    for experiment in TRAJECTORY_EXPERIMENTS:
        p = np.full(len(train), np.nan)
        choices = []
        selected_by_fold = []
        for fold, (itrain, itest) in enumerate(outer):
            assert not set(groups[itrain]) & set(groups[itest])
            fraction, ratio, inner_mse = tune(train.iloc[itrain], y[itrain],
                                              groups[itrain], experiment)
            fitted = PreparedFit(train.iloc[itrain], y[itrain],
                                 expanded=experiment.expanded,
                                 trajectory=experiment.trajectory)
            beta = fitted.fit_coefficients(experiment.kind, fraction, ratio)
            p[itest] = fitted.predict(train.iloc[itest], beta)
            support = [name for name, coef in zip(fitted.names, beta)
                       if abs(coef) > 1e-5]
            selected_by_fold.append(support)
            choices.append({"fold": fold+1, "fraction_of_lambda_max": fraction,
                            "elastic_l1_ratio": ratio, "inner_MSE": inner_mse,
                            "nonzero": len(support)})
        assert np.isfinite(p).all()
        m = metrics(y, p)
        rows.append({"model": experiment.name, "family": experiment.kind,
                     "nonlinear": experiment.expanded,
                     "candidate_features": len(fitted.names),
                     "selected_features_mean": float(np.mean(
                         [len(s) for s in selected_by_fold])),
                     **m})
        predictions[experiment.name] = p
        tuning[experiment.name] = choices
        selections[experiment.name] = selected_by_fold
        print("TRAJECTORY", experiment.name, "RMSE", round(m["RMSE"], 3),
              "R2", round(m["R2"], 3), flush=True)

    scores = pd.DataFrame(rows).sort_values("RMSE")
    scores.to_csv(OUT / "trajectory_grouped_cv_comparison.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "trajectory_grouped_cv_predictions.csv",
                                   index=False)
    (OUT / "trajectory_nested_tuning.json").write_text(
        json.dumps(tuning, indent=2) + "\n")
    (OUT / "trajectory_selected_by_fold.json").write_text(
        json.dumps(selections, indent=2) + "\n")

    # Pair up fitted OOF predictions on the same physical plots. These are
    # conditional resampling intervals, not refit- or search-adjusted CIs.
    ix = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    rng = np.random.default_rng(2026)
    samples = [np.concatenate([ix[g] for g in rng.choice(list(ix), len(ix), True)])
               for _ in range(1000)]
    compare = []
    for experiment in TRAJECTORY_EXPERIMENTS:
        prior_name = previous_methods[experiment.kind]
        reference = previous[prior_name].to_numpy(float)
        current = np.asarray(predictions[experiment.name], float)
        rmse_delta = float(metrics(y, reference)["RMSE"] -
                           metrics(y, current)["RMSE"])
        intervals = [float(np.sqrt(np.mean((y[s] - reference[s])**2)) -
                           np.sqrt(np.mean((y[s] - current[s])**2)))
                     for s in samples]
        lower, upper = np.quantile(intervals, [.025, .975])
        compare.append({"trajectory_model": experiment.name,
                        "prior_nonlinear_EML": prior_name,
                        "prior_RMSE": float(previous_scores.loc[prior_name, "RMSE"]),
                        "trajectory_RMSE": float(metrics(y, current)["RMSE"]),
                        "improvement_RMSE": rmse_delta,
                        "bootstrap_conditional_low": float(lower),
                        "bootstrap_conditional_high": float(upper)})
    pd.DataFrame(compare).to_csv(OUT / "trajectory_paired_vs_prior.csv", index=False)

    # Stability on physical-plot resamples; tuning from TRAIN, not VAL.
    summary = []
    for experiment in TRAJECTORY_EXPERIMENTS:
        fraction, ratio, _ = tune(train, y, groups, experiment)
        counts = dict.fromkeys(
            (EXPANDED if experiment.expanded else BASE) + TRAJECTORY, 0)
        subs = GroupShuffleSplit(n_splits=repeats, train_size=.8,
                                 random_state=2026)
        for sub, _ in subs.split(train, y, groups):
            fitted = PreparedFit(train.iloc[sub], y[sub],
                                 experiment.expanded, trajectory=True)
            coef = fitted.fit_coefficients(experiment.kind, fraction, ratio)
            for name, b in zip(fitted.names, coef):
                counts[name] += int(abs(b) > 1e-5)
        for name, count in counts.items():
            summary.append({"model": experiment.name, "feature": name,
                            "feature_family": "trajectory" if name in TRAJECTORY
                            else "prior_satellite",
                            "selected": count, "repeats": repeats,
                            "frequency": count / repeats})
        print("STABILITY", experiment.name, repeats, "plot resamples",
              flush=True)
    pd.DataFrame(summary).to_csv(OUT / "trajectory_selection_stability.csv",
                                 index=False)
    (OUT / "trajectory_configuration.json").write_text(json.dumps({
        "new_raw_stage_contrasts": list(TRAJECTORY),
        "train_only": True, "validation_labels_scored": False,
        "train_plots": len(ix), "train_plot_seasons": len(train),
        "outer_group_folds": 5, "inner_group_folds": 3,
        "lambda_fraction_grid": FRACS,
        "method": "physical-unit contrasts before fold-local median imputation and standardization",
        "retrospective_warning": "Growing-season fractions depend on actual harvest date; NOT early forecasting.",
        "selection_warning": "Same TRAIN plots as earlier model search; all results exploratory.",
        "comparison_warning": "Intervals condition on already-fitted OOF predictions and ignore model-selection uncertainty."
    }, indent=2) + "\n")
    print("TRAJECTORY RESULTS\n", scores.to_string(index=False), flush=True)
    print("PAIRED RESULTS\n", pd.DataFrame(compare).to_string(index=False),
          flush=True)


def trajectory_checks_run():
    """Plot-partition and sensor-removal checks without touching VAL labels."""
    full = pd.read_csv(ROOT / "outputs/features/full_stage_features.csv",
                       dtype={"unique_id": str, "plot_id": str})
    train = full.loc[full.split.eq("train")].reset_index(drop=True)
    validation = full.loc[full.split.eq("val")]
    assert len(train) == 282 and len(validation) == 37
    assert not set(train.plot_id) & set(validation.plot_id)
    groups = train.plot_id.to_numpy()
    y = train["yield"].to_numpy(float)
    exp = next(ex for ex in TRAJECTORY_EXPERIMENTS
               if ex.name == "trajectory_nonlinear_adaptive_lasso")
    original = next(ex for ex in EXPERIMENTS
                    if ex.name == "nonlinear_adaptive_lasso")
    check_model = next(ex for ex in EXPERIMENTS
                       if ex.name == "nonlinear_SCAD_LLA")

    # Validate previous OOF results are on identical plot-season keys.
    previous = pd.read_csv(OUT / "trajectory_grouped_cv_predictions.csv",
                           dtype={"unique_id": str, "plot_id": str})
    assert previous.unique_id.tolist() == train.unique_id.tolist()
    assert previous.plot_id.tolist() == train.plot_id.tolist()
    assert np.allclose(previous["yield"].to_numpy(float), y)
    original_p = pd.read_csv(OUT / "grouped_cv_predictions.csv",
                             dtype={"unique_id": str, "plot_id": str})
    assert original_p.unique_id.tolist() == train.unique_id.tolist()

    # Do not tune on subgroups: these are error diagnostics on fixed OOF
    # predictions already fit without the corresponding held-out plot.
    diagnostics = []
    observed = {
        exp.name: previous[exp.name].to_numpy(float),
        original.name: original_p[original.name].to_numpy(float),
        check_model.name: original_p[check_model.name].to_numpy(float),
    }
    for key in ("year", "season"):
        for value, part in train.groupby(key):
            ix = part.index.to_numpy()
            for name, p in observed.items():
                diagnostics.append({"subgroup_field": key, "subgroup": str(value),
                                    "model": name, "n_rows": len(ix),
                                    "n_plots": int(part.plot_id.nunique()),
                                    **metrics(y[ix], p[ix])})
    pd.DataFrame(diagnostics).to_csv(OUT / "trajectory_subgroup_errors.csv",
                                     index=False)

    outer = list(GroupKFold(n_splits=5).split(train, y, groups))
    sensor_ablation = []
    sensor_predictions = {
        "unique_id": train.unique_id, "plot_id": groups, "yield": y,
        "all_sensors": previous[exp.name].to_numpy(float),
    }
    subsets = [
        ("without_S1", ("s2", "l8")),
        ("without_S2", ("s1", "l8")),
        ("without_L8", ("s1", "s2")),
        ("S1_only", ("s1",)),
        ("S2_only", ("s2",)),
    ]
    for name, allowed in subsets:
        scenario = replace(exp, name=name, allowed_sensors=allowed)
        pred = np.full(len(train), np.nan)
        for i_tr, i_te in outer:
            fraction, ratio, _ = tune(train.iloc[i_tr], y[i_tr],
                                     groups[i_tr], scenario)
            fitted = PreparedFit(train.iloc[i_tr], y[i_tr],
                                 scenario.expanded,
                                 trajectory=scenario.trajectory,
                                 allowed_sensors=allowed)
            beta = fitted.fit_coefficients(scenario.kind, fraction, ratio)
            pred[i_te] = fitted.predict(train.iloc[i_te], beta)
        assert np.isfinite(pred).all()
        sensor_predictions[name] = pred
        rm = metrics(y, pred)
        full_p = previous[exp.name].to_numpy(float)
        sensor_ablation.append({"scenario": name, "allowed_sensors": ",".join(allowed),
                                **rm, "full_minus_ablated_RMSE":
                                metrics(y, full_p)["RMSE"] - rm["RMSE"]})
        print("SENSOR CHECK", name, "RMSE", round(rm["RMSE"], 3), flush=True)
    pd.DataFrame(sensor_ablation).to_csv(OUT / "trajectory_sensor_ablation.csv",
                                         index=False)
    pd.DataFrame(sensor_predictions).to_csv(
        OUT / "trajectory_sensor_ablation_oof.csv", index=False)
    for key in ("year", "season"):
        for value, part in train.groupby(key):
            ix = part.index.to_numpy()
            diagnostics.append({
                "subgroup_field": key, "subgroup": str(value),
                "model": "trajectory_nonlinear_adaptive_without_L8",
                "n_rows": len(ix), "n_plots": int(part.plot_id.nunique()),
                **metrics(y[ix], sensor_predictions["without_L8"][ix])
            })
    pd.DataFrame(diagnostics).to_csv(OUT / "trajectory_subgroup_errors.csv",
                                     index=False)

    # Same physical plots, but shuffled between outer grouped partitions.
    # These are robustness checks only, not independent confirmation.
    new_rows = []
    no_l8 = replace(exp, name="trajectory_nonlinear_adaptive_without_L8",
                    allowed_sensors=("s1", "s2"))
    for seed in (13, 37, 79):
        shuffled = list(GroupKFold(n_splits=5, shuffle=True,
                                  random_state=seed).split(train, y, groups))
        for candidate in (exp, no_l8, original, check_model):
            pred = np.full(len(train), np.nan)
            for tr, te in shuffled:
                assert not set(groups[tr]) & set(groups[te])
                fraction, ratio, _ = tune(train.iloc[tr], y[tr],
                                          groups[tr], candidate)
                fitted = PreparedFit(train.iloc[tr], y[tr],
                                     candidate.expanded,
                                     candidate.penalize_agro_interactions,
                                     candidate.trajectory,
                                     candidate.allowed_sensors)
                beta = fitted.fit_coefficients(candidate.kind, fraction, ratio)
                pred[te] = fitted.predict(train.iloc[te], beta)
            m = metrics(y, pred)
            new_rows.append({"seed": seed, "model": candidate.name, **m})
            print("SHUFFLED", seed, candidate.name, "RMSE",
                  round(m["RMSE"], 3), flush=True)
    checks = pd.DataFrame(new_rows)
    checks.to_csv(OUT / "trajectory_shuffled_group_cv.csv", index=False)
    checks.groupby("model", as_index=False).agg(
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        R2_mean=("R2", "mean")
    ).to_csv(OUT / "trajectory_shuffled_group_summary.csv", index=False)
    (OUT / "trajectory_checks_configuration.json").write_text(json.dumps({
        "validation_labels_scored": False, "outer_group_folds": 5,
        "inner_group_folds": 3, "shuffled_group_seeds": [13, 37, 79],
        "sensor_ablation": "Blanks raw columns of omitted sensors before training-fold imputation; derived contrasts are blanked with their input sensors.",
        "subgroup_note": "Fixed OOF errors by year/season, not out-of-year validation.",
        "caution": "Same TRAIN plots reused across candidate search and shuffled partitions; results are exploratory."
    }, indent=2) + "\n")


def run(repeats):
    OUT.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(ROOT / "outputs/features/full_stage_features.csv",
                       dtype={"unique_id": str, "plot_id": str})
    train = full[full.split.eq("train")].reset_index(drop=True)
    reserved = full[full.split.eq("val")]
    assert len(train) == 282 and len(reserved) == 37
    assert not set(train.plot_id) & set(reserved.plot_id)
    y = train["yield"].to_numpy(float)
    plots = train.plot_id.to_numpy()
    outer = list(GroupKFold(n_splits=5).split(train, y, plots))
    outputs = {"unique_id": train.unique_id, "plot_id": plots, "yield": y}
    scores = []
    choices = {}
    selections = {}
    for experiment in EXPERIMENTS:
        predictions = np.full(len(train), np.nan)
        inner_choice = []
        fold_features = []
        for k, (tr, te) in enumerate(outer):
            fraction, ratio, loss = tune(train.iloc[tr], y[tr], plots[tr], experiment)
            fit = PreparedFit(train.iloc[tr], y[tr], experiment.expanded,
                              experiment.penalize_agro_interactions)
            beta = fit.fit_coefficients(experiment.kind, fraction, ratio)
            predictions[te] = fit.predict(train.iloc[te], beta)
            selected = [name for name, coef in zip(fit.names, beta) if abs(coef) > 1e-5]
            fold_features.append(selected)
            inner_choice.append({"outer_fold": k + 1, "fraction_of_lambda_max": fraction,
                                 "l1_ratio": ratio, "inner_MSE": loss,
                                 "selected_count": len(selected)})
        assert np.isfinite(predictions).all()
        outputs[experiment.name] = predictions
        choices[experiment.name] = inner_choice
        selections[experiment.name] = fold_features
        row = {"model": experiment.name, "expanded_satellite_features": experiment.expanded,
               "penalize_agro_interactions": experiment.penalize_agro_interactions,
               "candidate_penalized_features": len(fit.names),
               "selected_penalized_features_mean": float(np.mean(
                   [len(s) for s in fold_features])),
               **metrics(y, predictions)}
        scores.append(row)
        print("CV", experiment.name, "RMSE", round(row["RMSE"], 3),
              "R2", round(row["R2"], 3),
              "selected", round(row["selected_penalized_features_mean"], 1), flush=True)

    pd.DataFrame(outputs).to_csv(OUT / "grouped_cv_predictions.csv", index=False)
    comparison = pd.DataFrame(scores).sort_values("RMSE")
    comparison.to_csv(OUT / "grouped_cv_comparison.csv", index=False)
    (OUT / "nested_tuning.json").write_text(json.dumps(choices, indent=2) + "\n")
    (OUT / "fold_selected_features.json").write_text(json.dumps(selections, indent=2) + "\n")

    # Paired, fixed-OOF plot-cluster uncertainty against earlier models.
    # NEVER use validation outcomes, and never call these intervals external
    # model-refit validation or multiple-comparison-corrected inference.
    comparisons = []
    references = [
        (ROOT / "outputs/robustness/incremental_satellite_oof.csv",
         "survey_controls_only", "ridge_agronomic_controls"),
        (ROOT / "outputs/improvements/candidate_oof_predictions.csv",
         "controls_extratrees", "extratrees_agronomic_controls"),
    ]
    index_by_plot = {p: np.flatnonzero(plots == p) for p in np.unique(plots)}
    rng = np.random.default_rng(2026)
    sampled_plots = [rng.choice(np.unique(plots), size=len(index_by_plot), replace=True)
                     for _ in range(1200)]
    boot_indices = [np.concatenate([index_by_plot[p] for p in group_sample])
                    for group_sample in sampled_plots]
    for location, original, reference_name in references:
        if not location.is_file():
            continue
        earlier = pd.read_csv(location, dtype={"unique_id": str, "plot_id": str})
        earlier = earlier.set_index("unique_id").reindex(train.unique_id)
        assert earlier.plot_id.tolist() == train.plot_id.tolist()
        assert np.allclose(earlier["yield"].to_numpy(float), y)
        baseline = earlier[original].to_numpy(float)
        assert np.isfinite(baseline).all()
        rmse_ref = float(np.sqrt(np.mean((baseline - y) ** 2)))
        for target_name in ("nonlinear_SCAD_LLA", "nonlinear_adaptive_lasso",
                            "nonlinear_partial_elastic_net"):
            candidate = np.asarray(outputs[target_name], float)
            gain = rmse_ref - float(np.sqrt(np.mean((candidate - y) ** 2)))
            diffs = [float(np.sqrt(np.mean((baseline[ix] - y[ix]) ** 2)) -
                           np.sqrt(np.mean((candidate[ix] - y[ix]) ** 2)))
                     for ix in boot_indices]
            low, high = np.quantile(diffs, [.025, .975])
            comparisons.append({"reference": reference_name, "method": target_name,
                                "RMSE_reference_minus_candidate": gain,
                                "conditional_plot_bootstrap_low": float(low),
                                "conditional_plot_bootstrap_high": float(high)})
    pd.DataFrame(comparisons).to_csv(OUT / "paired_against_prior_baselines.csv", index=False)

    # Repeated subsamples for selection stability: fixed full-TRAIN-tuned
    # lambda, with no reference to unseen plot scores.
    # No external holdout is used, and this does not quantify model-refit CI.
    tracked = ("partial_lasso", "adaptive_lasso", "SCAD_LLA",
               "nonlinear_partial_elastic_net",
               "nonlinear_agro_interaction_SCAD_LLA")
    frequency = []
    for name in tracked:
        experiment = next(e for e in EXPERIMENTS if e.name == name)
        fraction, ratio, _ = tune(train, y, plots, experiment)
        complete = PreparedFit(train, y, experiment.expanded,
                               experiment.penalize_agro_interactions)
        counts = dict.fromkeys(complete.names, 0)
        splits = GroupShuffleSplit(n_splits=repeats, train_size=.8, random_state=2026)
        for idx, (sub, _) in enumerate(splits.split(train, y, plots)):
            fit = PreparedFit(train.iloc[sub], y[sub], experiment.expanded,
                              experiment.penalize_agro_interactions)
            beta = fit.fit_coefficients(experiment.kind, fraction, ratio)
            for name_, coefficient in zip(fit.names, beta):
                counts[name_] += int(abs(coefficient) > 1e-5)
        for feature, n in counts.items():
            frequency.append({"model": name, "feature": feature,
                              "selected_subsamples": n, "repeats": repeats,
                              "selection_frequency": n / repeats})
        print("STABILITY", name, "repeats", repeats,
              "tuned_fraction", fraction, "l1_ratio", ratio, flush=True)
    pd.DataFrame(frequency).to_csv(OUT / "selection_stability.csv", index=False)

    # SCAD is nonconvex; check dependence on multiple initialization points
    # on TRAIN only, at a single TRAIN-tuned lambda. Do not select a model
    # using the official VAL plot labels.
    scad_experiment = next(e for e in EXPERIMENTS if e.name == "nonlinear_SCAD_LLA")
    scad_fraction, _, _ = tune(train, y, plots, scad_experiment)
    full_fit = PreparedFit(train, y, expanded=True)
    diagnostics = []
    for initial in ("lasso", "ridge", "zero"):
        beta, history = full_fit.scad_from_initialization(scad_fraction, initial)
        diagnostics.append({
            "initialization": initial,
            "fraction_lambda_max": scad_fraction,
            "selected_satellite_features": int(np.sum(np.abs(beta) > 1e-5)),
            "objective_initial": history[0],
            "objective_final": history[-1],
            "objective_nonincreasing": bool(np.all(np.diff(history) <= 1e-6)),
            "LLA_iterations": len(history) - 1,
            "last_objective_step": float(history[-2] - history[-1]) if len(history) > 1 else 0.,
        })
    pd.DataFrame(diagnostics).to_csv(OUT / "scad_initialization_checks.csv", index=False)
    print("SCAD DIAGNOSTICS\n", pd.DataFrame(diagnostics).to_string(index=False),
          flush=True)
    (OUT / "configuration.json").write_text(json.dumps({
        "train_plot_seasons": 282, "training_physical_plots": 117,
        "official_VAL_used": False, "outer_folds": 5, "inner_folds": 3,
        "penalty_relative_lambda_max": FRACS,
        "elastic_net_l1_ratios": [.35, .70],
        "scad_a": SCAD_A, "scad_method": "LLA from LASSO warm start, up to seven iterations",
        "adaptive_weight_formula": "1/(|fold-local Ridge coefficient|+0.05), median normalized, clipped [.2,12]",
        "unpenalized": ["intercept", "area", "year", "season", "variety", "year_variety"],
        "baseline_caveat": "Controls-only OLS here is unpenalized and is not identical to the prior controls-only Ridge benchmark.",
        "feature_timing": "Retrospective, actual harvest date required to form growing-season windows.",
        "limitations": [
            "Model and tuning-grid search are exploratory on reused TRAIN plots.",
            "SCAD has only a local stationary point, not a certified global optimum.",
            "Penalized agri interactions encode year-by-variety and plot-area categories while main agronomic controls stay unpenalized.",
            "Feature selection stability on repeated TRAIN plot subsamples is descriptive.",
            "External validation and yield unit checks remain necessary."
        ]}, indent=2) + "\n")
    print("FINAL:\n", comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--repeats", type=int, default=12, help="Plot-group selection stability resamples")
    arg.add_argument("--trajectory", action="store_true",
                     help="Run phenological trajectory contrasts against existing EML results")
    arg.add_argument("--trajectory-checks", action="store_true",
                     help="Check season/year errors, sensor ablations, and shuffled plot splits")
    options = arg.parse_args()
    if options.repeats < 1:
        arg.error("--repeats must be positive")
    if options.trajectory_checks:
        trajectory_checks_run()
    elif options.trajectory:
        trajectory_run(options.repeats)
    else:
        run(options.repeats)
