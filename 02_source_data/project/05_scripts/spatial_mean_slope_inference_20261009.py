"""Bounded validation of an orthogonal mean conditional-slope estimator.

This module cannot start national inference. Existing national model artifacts
remain read-only. Spatial covariance is conditional on frozen measurements.
"""
from __future__ import annotations
from nc_paths import project_root as _nc_project_root, external_path as _nc_external_path

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from econml.grf import CausalForest
from scipy.spatial import cKDTree
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

from assemble_and_audit_lagged_support_20261008 import digest, qualification, read, require, write_json

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "06_logs"
PRIMARY = LOGS / "primary_weather_no_vpd_20261008_run02"
MODEL = LOGS / "national_primary_point_model_20261009_run01"
VALIDATION = LOGS / "national_primary_independent_validation_20261009_run02/validation.json"
MEASUREMENT = LOGS / "primary_weather_validation_20261008_run01/validation.json"
FOLDS = LOGS / "national_protocol_spatial_folds_freeze_20261008_run01/folds/national_tract_spatial_fold_map.csv"
CONFIG = ROOT / "05_scripts/primary_canopy_model_candidate_20261008.json"
MIGRATION = ROOT / "00_admin/design_state_alignment_20261009.json"
NOTEBOOK = _nc_external_path('reference_notebook')
KEY = ["state", "tract_id", "exposure_year", "outcome_year"]
T = "treatment_pp_change_from_2000"
Y = "weather_adjusted_lst_reference_mean"
COORD = ["centroid_epsg5070_x_m", "centroid_epsg5070_y_m"]
BANDWIDTHS = (100000., 200000., 400000.)


def array_hash(a):
    a = np.ascontiguousarray(a, dtype=np.float64)
    return hashlib.sha256(str(a.shape).encode("ascii") + a.tobytes()).hexdigest()


def historic_source_check(path, expected):
    """Accept only the explicitly documented administrative-state migration."""
    path = Path(path)
    if digest(path) == expected:
        return "ORIGINAL_BYTES"
    state = ROOT / "00_admin/project_state.yaml"
    require(path == state and MIGRATION.is_file(), f"Source content changed: {path}")
    record = json.loads(MIGRATION.read_text(encoding="utf-8"))
    require(record["authorized_scope"] == "ADMINISTRATIVE_STATUS_ALIGNMENT_ONLY", "Migration scope")
    require(record["old_state_sha256"] == expected, "Different historical state version")
    require(record["new_state_sha256"] == digest(state), "Current state differs from migration")
    require(digest(record["historical_snapshot"]) == expected, "Historical state snapshot changed")
    require(record["historical_audits_rewritten"] is False, "Historical audit mutation")
    return "EXPLICIT_HISTORICAL_ADMINISTRATIVE_SNAPSHOT"


def load_inputs():
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    require(validation["status"] == "PASS_INDEPENDENT_NATIONAL_MODEL_RELOAD_PREDICTIONS_RESIDUALS_ROWS", "Independent reload gate")
    require(validation["all_frozen_input_output_hashes_match"], "Historic model integrity gate")
    source_checks = []
    for p, h in validation["input_hashes"].items():
        source_checks.append({"path": p, "resolution": historic_source_check(p, h)})
    measurement = json.loads(MEASUREMENT.read_text(encoding="utf-8"))
    require(measurement["primary_measurement_gate"] == "PASS" and not measurement["vpd_included"], "Measurement gate")
    require(digest(PRIMARY / "audit.json") == measurement["input_receipt_sha256"], "Measurement receipt")
    require(digest(PRIMARY / "lagged_primary_model_sample.csv") == measurement["sample_sha256"], "Sample hash")
    receipt = json.loads((PRIMARY / "audit.json").read_text(encoding="utf-8"))
    for p, h in receipt["source_hashes"].items():
        source_checks.append({"path": p, "resolution": historic_source_check(p, h)})
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    require(cfg["seed"] == 20260912 and cfg["folds"] == 5, "Frozen seed and folds")
    require(cfg["causal_forest"]["inference"] is False, "No builtin independent-data inference")
    full = read(PRIMARY / "lagged_primary_model_sample.csv")
    require(len(full) == 105437 and not full.duplicated(KEY).any(), "Full sample identity")
    require(qualification(full).all(), "Frozen qualification")
    require(set(zip(full.exposure_year, full.outcome_year)) == {(2013, 2014), (2018, 2019), (2023, 2024)}, "Lag pairs")
    spatial = read(FOLDS)
    require(not spatial.duplicated(["state", "tract_id"]).any(), "Duplicate spatial keys")
    joined = full.merge(spatial[["state", "tract_id", *COORD]], on=["state", "tract_id"], how="left", validate="many_to_one", sort=False)
    require(len(joined) == len(full) and joined[KEY].equals(full[KEY]), "Spatial join changed rows/order")
    require(np.isfinite(joined[COORD].to_numpy(float)).all(), "Unmatched/nonfinite spatial coordinates")
    require(joined.groupby(["state", "tract_id"]).spatial_fold.nunique().eq(1).all(), "Repeated tract fold leakage")
    require(joined.groupby("spatial_block_id").spatial_fold.nunique().eq(1).all(), "Block fold leakage")
    return cfg, joined, source_checks


def orthogonal_scores(theta, residual_t, residual_y, conditional_variance, intercept=None):
    theta, residual_t, residual_y, conditional_variance = [np.asarray(a, float).reshape(-1) for a in (theta, residual_t, residual_y, conditional_variance)]
    require(len({len(a) for a in (theta, residual_t, residual_y, conditional_variance)}) == 1, "Score length mismatch")
    if intercept is None:
        intercept = np.zeros_like(theta)
    intercept = np.asarray(intercept, float).reshape(-1)
    require(len(intercept) == len(theta), "Intercept length mismatch")
    require(all(np.isfinite(a).all() for a in (theta, residual_t, residual_y, conditional_variance, intercept)), "Nonfinite score input")
    require((conditional_variance > 0).all(), "Nonpositive treatment variance: STOP; no clipping or row removal")
    correction = residual_t * (residual_y - intercept - theta * residual_t) / conditional_variance
    return theta + correction


def wendland(r):
    r = np.asarray(r, float)
    require(np.isfinite(r).all() and (r >= 0).all(), "Invalid normalized distance")
    return np.maximum(1. - r, 0.) ** 4 * (1. + 4. * r)


def spatial_covariance(frame, influence, bandwidth_m):
    """Sum repeated-year scores before spatial weighting; no block independence."""
    influence = np.asarray(influence, float).reshape(-1)
    require(len(frame) == len(influence) and len(frame) > 1, "Influence dimensions")
    require(np.isfinite(influence).all() and bandwidth_m > 0, "Covariance inputs")
    d = frame[["state", "tract_id", *COORD]].copy()
    d["influence"] = influence
    require(d.groupby(["state", "tract_id"])[COORD].nunique().le(1).all().all(), "Tract coordinate drift")
    g = d.groupby(["state", "tract_id"], sort=True).agg({COORD[0]: "first", COORD[1]: "first", "influence": "sum"}).reset_index()
    xy = g[COORD].to_numpy(float)
    require(np.isfinite(xy).all(), "Nonfinite metric coordinates")
    q = g.influence.to_numpy(float)
    pairs = cKDTree(xy).query_pairs(bandwidth_m, output_type="ndarray")
    diagonal = float(q @ q)
    pair_total = 0.
    # Chunk pair arithmetic to avoid a second large dense coordinate array.
    for start in range(0, len(pairs), 250000):
        p = pairs[start:start + 250000]
        distance = np.linalg.norm(xy[p[:, 0]] - xy[p[:, 1]], axis=1)
        pair_total += float(np.sum(wendland(distance / bandwidth_m) * q[p[:, 0]] * q[p[:, 1]]))
    numerator = diagonal + 2. * pair_total
    require(numerator >= 0, "Negative spatial variance: STOP; no truncation")
    return {"variance": numerator / len(frame) ** 2, "standard_error": float(np.sqrt(numerator) / len(frame)),
            "tracts": len(g), "rows": len(frame), "bandwidth_m": bandwidth_m,
            "within_tract_numerator": diagonal, "between_tract_numerator": 2. * pair_total,
            "unique_between_tract_pairs": len(pairs)}


def assert_split(df, train, test):
    require(len(train) > 0 and len(test) > 0, "Empty frozen split")
    require(not np.intersect1d(train, test).size, "Row leakage")
    require(not set(df.iloc[train].tract_id) & set(df.iloc[test].tract_id), "Tract leakage")
    require(not set(df.iloc[train].spatial_block_id) & set(df.iloc[test].spatial_block_id), "Block leakage")


def pilot(out):
    start = time.monotonic()
    cfg, full, source_checks = load_inputs()
    mask = full.tract_id.map(lambda s: int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big") % 20 == 0)
    df = full.loc[mask].copy().reset_index(drop=True)
    require(len(df) == 5241 and df.tract_id.nunique() == 2120, "Frozen deterministic pilot subset changed")
    require(set(df.spatial_fold.astype(int)) == set(range(5)), "Pilot folds")
    out.mkdir(parents=True, exist_ok=False)
    try:
        x = df[cfg["x_columns"]].to_numpy(float)
        year = pd.get_dummies(df.outcome_year.astype(str), prefix="outcome_year", dtype=float)
        w = np.column_stack([df[cfg["w_columns"]].to_numpy(float), year.to_numpy()])
        z = np.column_stack([x, w])
        y = df[Y].to_numpy(float); t = df[T].to_numpy(float)
        folds = df.spatial_fold.to_numpy(int)
        require(all(np.isfinite(a).all() for a in (x, w, y, t)), "Finite model inputs")
        source_hashes = {str(p): digest(p) for p in [CONFIG, VALIDATION, MEASUREMENT, PRIMARY / "audit.json", PRIMARY / "lagged_primary_model_sample.csv", FOLDS, NOTEBOOK, Path(__file__)]}
        write_json(out / "run_started.json", {"mode": "DIAGNOSTIC_REAL_DATA_PILOT_NOT_NATIONAL_INFERENCE", "rows": len(df), "seed": cfg["seed"], "expected_fits": 60,
                   "source_hashes": source_hashes, "started_local": dt.datetime.now().isoformat(), "formal_claim_allowed": False})
        df[KEY + ["spatial_block_id", "spatial_fold", *COORD]].to_csv(out / "pilot_sample_manifest.csv", index=False)
        write_json(out / "executed_config.json", {"frozen_point_config": cfg, "inference_target": "ROW_EQUAL_WEIGHT_MEAN_CONDITIONAL_RESIDUAL_PROJECTION_SLOPE",
                   "variance_forest_parameters": cfg["nuisance_forest"], "outer_folds": 5, "inner_folds": "FOUR_REMAINING_ORIGINAL_FOLDS",
                   "spatial_bandwidths_m": BANDWIDTHS, "bandwidth_role": "PREDECLARED_DIAGNOSTIC_NO_SIGNIFICANCE_SELECTION"})
        events = []; predictions = []; reload_errors = []; split_manifest = []

        def fit(estimator, features, target, train, test, outer, stage, inner=None):
            assert_split(df, train, test)
            key = f"outer{outer}_{stage}_inner{inner}"
            require(key not in {e["fit_id"] for e in events}, "Duplicate fit")
            event = {"fit_id": key, "outer_fold": outer, "inner_fold": inner, "stage": stage,
                     "train_rows": len(train), "test_rows": len(test), "train_index_sha256": array_hash(train), "test_index_sha256": array_hash(test),
                     "train_design_sha256": array_hash(features), "train_target_sha256": array_hash(target), "parameters": estimator.get_params(),
                     "seed": cfg["seed"], "train_test_tract_overlap": 0, "train_test_block_overlap": 0,
                     "reused": False, "skipped": False}
            split_manifest.append({k: v for k, v in event.items() if k != "parameters"})
            with (out / "fit_events.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({**event, "status": "STARTED"}) + "\n")
            s = time.monotonic()
            if stage == "conditional_forest":
                estimator.fit(features, target[:, 0].reshape(-1, 1), target[:, 1])
            else:
                estimator.fit(features, target)
            event.update(status="COMPLETED", elapsed_seconds=time.monotonic() - s)
            with (out / "fit_events.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
            events.append(event)
            return estimator

        for outer in range(5):
            print(f"OUTER {outer + 1}/5 START; only deterministic pilot", flush=True)
            train = np.flatnonzero(folds != outer); test = np.flatnonzero(folds == outer)
            assert_split(df, train, test)
            ru = np.full(len(df), np.nan); rv = np.full(len(df), np.nan); visits = np.zeros(len(df), int)
            saved = {"inner": [], "outer_train_index": train, "outer_test_index": test}
            for inner in sorted(set(range(5)) - {outer}):
                inner_train = np.flatnonzero((folds != outer) & (folds != inner))
                inner_test = np.flatnonzero(folds == inner)
                require(not np.intersect1d(inner_train, test).size and not np.intersect1d(inner_test, test).size, "Outer fold contaminated inner nuisance")
                my = fit(RandomForestRegressor(**cfg["nuisance_forest"]), z[inner_train], y[inner_train], inner_train, inner_test, outer, "inner_y", inner)
                mt = fit(RandomForestRegressor(**cfg["nuisance_forest"]), z[inner_train], t[inner_train], inner_train, inner_test, outer, "inner_t", inner)
                ru[inner_test] = y[inner_test] - my.predict(z[inner_test])
                rv[inner_test] = t[inner_test] - mt.predict(z[inner_test])
                visits[inner_test] += 1
                saved["inner"].append({"fold": inner, "train": inner_train, "test": inner_test, "y_model": my, "t_model": mt})
            require((visits[train] == 1).all() and (visits[test] == 0).all(), "Inner visits")
            require(np.isfinite(ru[train]).all() and np.isfinite(rv[train]).all(), "Inner residual completeness")
            outer_y = fit(RandomForestRegressor(**cfg["nuisance_forest"]), z[train], y[train], train, test, outer, "outer_y")
            outer_t = fit(RandomForestRegressor(**cfg["nuisance_forest"]), z[train], t[train], train, test, outer, "outer_t")
            forest = fit(CausalForest(n_jobs=1, random_state=cfg["seed"], **cfg["causal_forest"]), x[train], np.column_stack([rv[train], ru[train]]), train, test, outer, "conditional_forest")
            varforest = fit(RandomForestRegressor(**cfg["nuisance_forest"]), x[train], rv[train] ** 2, train, test, outer, "conditional_variance")
            pred = forest.predict_full(x[test])
            require(pred.shape == (len(test), 2), "Continuous slope/intercept dimensions")
            require(np.allclose(pred[:, 0], forest.predict(x[test]).reshape(-1), rtol=0, atol=1e-12), "Final-wrapper slope equivalence")
            u = y[test] - outer_y.predict(z[test]); v = t[test] - outer_t.predict(z[test]); variance = varforest.predict(x[test])
            scores = orthogonal_scores(pred[:, 0], v, u, variance, pred[:, 1])
            r = df.iloc[test][KEY + ["spatial_block_id", "spatial_fold", *COORD]].copy()
            for name, values in {"pilot_row": test, "theta": pred[:, 0], "intercept": pred[:, 1], "residual_y": u, "residual_t": v, "conditional_variance": variance, "orthogonal_score": scores}.items():
                r[name] = values
            r.to_csv(out / f"outer_fold_{outer}_predictions.csv", index=False)
            saved.update(outer_y=outer_y, outer_t=outer_t, forest=forest, variance_forest=varforest, inner_residual_y=ru[train], inner_residual_t=rv[train])
            model_path = out / f"outer_fold_{outer}_models.joblib"
            joblib.dump(saved, model_path, compress=0)
            del saved
            reloaded = joblib.load(model_path)
            pred2 = reloaded["forest"].predict_full(x[test])
            u2 = y[test] - reloaded["outer_y"].predict(z[test]); v2 = t[test] - reloaded["outer_t"].predict(z[test]); vvar2 = reloaded["variance_forest"].predict(x[test])
            score2 = orthogonal_scores(pred2[:, 0], v2, u2, vvar2, pred2[:, 1])
            errors = {name: float(np.max(np.abs(a-b))) for name, a, b in [("parameters", pred, pred2), ("residual_y", u, u2), ("residual_t", v, v2), ("variance", variance, vvar2), ("scores", scores, score2)]}
            require(max(errors.values()) == 0, "Independent pilot reload mismatch")
            reload_errors.append({"outer_fold": outer, **errors})
            del reloaded, forest, varforest, outer_y, outer_t
            predictions.append(r)
            print(f"OUTER {outer + 1}/5 DONE; completed fits={len(events)}/60", flush=True)
        result = pd.concat(predictions, ignore_index=True).sort_values("pilot_row").reset_index(drop=True)
        require(np.array_equal(result.pilot_row.to_numpy(), np.arange(len(df))), "Score row coverage/order")
        require(result[KEY].equals(df[KEY]), "Score identity mismatch")
        require(len(events) == 60 and len({e["fit_id"] for e in events}) == 60, "Fit inventory")
        require(sum(e["stage"] == "conditional_forest" for e in events) == 5, "Final forests inventory")
        result.to_csv(out / "crossfit_scores.csv", index=False)
        pd.DataFrame(split_manifest).to_csv(out / "split_manifest.csv", index=False)
        pd.DataFrame(reload_errors).to_csv(out / "independent_reload_errors.csv", index=False)
        mu = float(result.orthogonal_score.mean())
        influence = result.orthogonal_score.to_numpy(float) - mu
        covariance = []
        for h in BANDWIDTHS:
            cov = spatial_covariance(result, influence, h)
            covariance.append({**cov, "estimate": mu, "normal_reference_lower": mu - norm.ppf(.975) * cov["standard_error"],
                               "normal_reference_upper": mu + norm.ppf(.975) * cov["standard_error"],
                               "status": "PILOT_DIAGNOSTIC_NOT_A_NATIONAL_CONFIDENCE_INTERVAL"})
        pd.DataFrame(covariance).to_csv(out / "spatial_covariance_diagnostic.csv", index=False)
        centered_sq = influence ** 2
        per_state = result.assign(score_sq=centered_sq).groupby("state").score_sq.sum()
        inverse_var = 1. / result.conditional_variance.to_numpy(float)
        require(all(digest(p) == h for p, h in source_hashes.items()), "Inputs changed during pilot")
        output_hashes = {p.name: digest(p) for p in out.iterdir() if p.is_file()}
        audit = {"status": "PASS_BOUNDED_ORTHOGONAL_SPATIAL_IMPLEMENTATION_PILOT", "mode": "DIAGNOSTIC_REAL_DATA_PILOT",
                 "rows": len(df), "tracts": int(df.tract_id.nunique()), "blocks": int(df.spatial_block_id.nunique()), "states": int(df.state.nunique()),
                 "expected_fits": 60, "executed_fits": len(events), "failed_fits": 0, "duplicate_fits": 0, "reused_fits": 0, "skipped_fits": 0,
                 "maximum_reload_error": max(max(v for k, v in e.items() if k != "outer_fold") for e in reload_errors),
                 "pilot_crossfit_plugin_mean": float(result.theta.mean()), "pilot_orthogonal_mean": mu,
                 "pilot_correction_mean": float((result.orthogonal_score-result.theta).mean()),
                 "conditional_variance_quantiles": {str(q): float(result.conditional_variance.quantile(q)) for q in [0, .01, .5, .99, 1]},
                 "inverse_variance_maximum_weight_share": float(inverse_var.max()/inverse_var.sum()),
                 "maximum_state_influence_square_share": float(per_state.max()/per_state.sum()),
                 "folds_regenerated": False, "old_national_model_retrained": False, "old_national_results_changed": False,
                 "spatial_independence_proven": False, "national_inference": "NOT_RUN", "bandwidth_sufficiency": "NOT_ESTABLISHED_BY_PILOT",
                 "causal_identification": "NOT_ESTABLISHED_BY_COVARIANCE", "full_pipeline_coverage": "NOT_VERIFIED",
                 "measurement_uncertainty": "CONDITIONAL_ON_FROZEN_WEATHER_STANDARDIZATION_AND_RASTER_MEASUREMENTS",
                 "estimand": "ELIGIBLE_COMPLETE_CASE_TRACT_YEAR_EQUAL_WEIGHT_MEAN_CONDITIONAL_RESIDUAL_PROJECTION_SLOPE",
                 "historical_source_resolutions": source_checks, "source_hashes": source_hashes, "output_hashes": output_hashes,
                 "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__},
                 "elapsed_seconds": time.monotonic()-start, "scientific_claims_allowed": False}
        write_json(out / "audit.json", audit)
        print(json.dumps({k: audit[k] for k in ["status", "rows", "executed_fits", "pilot_crossfit_plugin_mean", "pilot_orthogonal_mean", "elapsed_seconds"]}), flush=True)
    except BaseException:
        (out / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preflight", "pilot"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "pilot":
        pilot(args.out)
    else:
        cfg, df, source_checks = load_inputs()
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(args.out / "preflight.json", {"status": "PASS_INFERENCE_INPUTS_ONLY_NO_MODEL_RUN", "rows": len(df), "seed": cfg["seed"], "source_checks": source_checks,
                   "formal_national_inference": "NOT_RUN", "model_fits": 0})
        print("PASS input preflight; no estimator fitted", flush=True)


if __name__ == "__main__":
    main()
