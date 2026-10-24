"""Requalified nested five-fold mean-slope estimation with explicit checkpoints.

First fit only treatment/variance models and verify numerical support. Then
fit outcome/slope models. No clipping, new folds, significance selection,
measurement replacement, or overwriting historical experiment evidence.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from econml.grf import CausalForest
from scipy.spatial import cKDTree
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

from assemble_and_audit_lagged_support_20261008 import digest, read, require, write_json, qualification
from spatial_mean_slope_inference_20261009 import (
    ROOT, LOGS, MODEL, CONFIG, KEY, T, Y, COORD, BANDWIDTHS,
    array_hash, assert_split, orthogonal_scores, wendland,
)

# This is an independent geographic-nuisance version.  The historical model
# configuration remains the gate for the frozen sample; this configuration is
# a new, separately audited analysis input.
BASE_CONFIG = ROOT / "05_scripts/primary_canopy_model_candidate_20261008.json"
CONFIG = ROOT / "05_scripts/primary_canopy_model_geo_adjusted_20261009.json"


def append_json(path, event):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+"\n")


def load_requalified(source, validation):
    a = json.loads((source / "audit.json").read_text(encoding="utf-8"))
    v = json.loads(validation.read_text(encoding="utf-8"))
    require(v["status"] == "PASS_INDEPENDENT_GEOGRAPHIC_MEASUREMENT_REPAIR_AND_REQUALIFICATION", "Independent geographic gate")
    require(v["input_audit_sha256"] == digest(source / "audit.json"), "Geographic receipt changed")
    require(all(digest(p) == h for p, h in a["source_hashes"].items()), "Requalification source changed")
    require(all(digest(source / p) == h for p, h in a["outputs"].items()), "Requalification output changed")
    require(v["sample_sha256"] == digest(source / "revised_analysis_sample.csv"), "Revised sample hash")
    require(v["config_sha256"] == digest(BASE_CONFIG), "Frozen sample/model gate")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    df = read(source / "revised_analysis_sample.csv")
    require(len(df) == v["revised_rows"] and not df.duplicated(KEY).any(), "Revised identity")
    require(qualification(df).all() and set(df.spatial_fold.astype(int)) == set(range(5)), "Revised observation/fold qualification")
    require(np.isfinite(df[COORD].to_numpy(float)).all(), "Missing metric coordinates")
    return cfg, df


def streamed_covariance(frame, scores, radii=BANDWIDTHS):
    mu = float(np.mean(scores))
    d = frame[["state", "tract_id", *COORD]].copy()
    d["influence"] = np.asarray(scores)-mu
    require(d.groupby(["state", "tract_id"])[COORD].nunique().le(1).all().all(), "Coordinate drift")
    g = d.groupby(["state", "tract_id"], sort=True).agg({COORD[0]: "first", COORD[1]: "first", "influence": "sum"}).reset_index()
    xy = g[COORD].to_numpy(float); q = g.influence.to_numpy(float)
    diagonal = float(q@q)
    require(diagonal > 0, "No influence variation")
    tree = cKDTree(xy); pair_sums = np.zeros(len(radii)); counts = np.zeros(len(radii), dtype=np.int64)
    # Same all-pair estimator as the validated pilot, with bounded memory.
    for start in range(0, len(g), 128):
        neighbors = tree.query_ball_point(xy[start:start+128], max(radii), workers=1)
        for offset, items in enumerate(neighbors):
            i = start+offset
            j = np.asarray([item for item in items if item > i], dtype=np.int64)
            if not len(j):
                continue
            distance = np.linalg.norm(xy[j]-xy[i], axis=1)
            products = q[i]*q[j]
            for k, h in enumerate(radii):
                inside = distance <= h
                pair_sums[k] += float(np.sum(wendland(distance[inside]/h)*products[inside]))
                counts[k] += int(inside.sum())
    rows = []
    for k, h in enumerate(radii):
        numerator = diagonal+2*pair_sums[k]
        require(numerator >= 0, "Negative spatial variance; no truncation")
        se = float(np.sqrt(numerator)/len(frame))
        rows.append({"bandwidth_m": h, "rows": len(frame), "tracts": len(g),
                     "estimate": mu, "variance": numerator/len(frame)**2, "standard_error": se,
                     "within_tract_numerator": diagonal, "between_tract_numerator": 2*pair_sums[k],
                     "unique_between_tract_pairs": int(counts[k]),
                     "normal_reference_lower": mu-norm.ppf(.975)*se,
                     "normal_reference_upper": mu+norm.ppf(.975)*se,
                     "interval_scope": "CONDITIONAL_SPATIAL_REFERENCE_COVERAGE_NOT_PROVEN"})
    shares = q*q/diagonal
    g["diagonal_share"] = shares
    g["delete_one_tract_mean_change_NO_REFIT"] = -q/(len(frame)-frame.groupby(["state", "tract_id"]).size().reindex(
        pd.MultiIndex.from_frame(g[["state", "tract_id"]])).to_numpy())
    return pd.DataFrame(rows), g.sort_values("diagonal_share", ascending=False).reset_index(drop=True)


def spatial_tail_diagnostic(influence, seed):
    """Uniform ordered-pair sampling for descriptive remote mark dependence."""
    xy = influence[COORD].to_numpy(float)
    q = influence.influence.to_numpy(float)
    denominator = float(np.mean(q*q))
    require(denominator > 0, "Degenerate tail marks")
    rng = np.random.default_rng(seed)
    edges = np.array([0., 100000., 200000., 400000., 800000., 1200000., 2000000., np.inf])
    totals = np.zeros(7); squares = np.zeros(7); counts = np.zeros(7, dtype=np.int64)
    for _ in range(20):
        left = rng.integers(0, len(q), 100000)
        right = rng.integers(0, len(q), 100000)
        use = left != right
        left, right = left[use], right[use]
        distance = np.linalg.norm(xy[left]-xy[right], axis=1)
        marks = q[left]*q[right]/denominator
        bin_id = np.searchsorted(edges, distance, side="right")-1
        for k in range(7):
            values = marks[bin_id == k]
            counts[k] += len(values); totals[k] += values.sum(); squares[k] += values@values
    rows = []
    for k in range(7):
        require(counts[k] > 0, "Empty spatial-tail diagnostic bin")
        mean = totals[k]/counts[k]
        rows.append({"lower_distance_km": edges[k]/1000, "upper_distance_km": edges[k+1]/1000 if k < 6 else None,
                     "sampled_ordered_pair_count": int(counts[k]), "normalized_mark_product_mean": float(mean),
                     "normalized_mark_product_second_moment": float(squares[k]/counts[k]),
                     "role": "DESCRIPTIVE_PAIR_SAMPLING_NOT_INDEPENDENT_PAIRS_NO_P_VALUE_NO_AUTOMATIC_BANDWIDTH_SELECTION"})
    return pd.DataFrame(rows)


def execute(args):
    started = time.monotonic()
    cfg, full = load_requalified(args.source, args.validation)
    df = full.copy()
    if args.mode == "pilot":
        import hashlib
        keep = df.tract_id.map(lambda t: int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "big") % 20 == 0)
        df = df.loc[keep].reset_index(drop=True)
    require(set(df.spatial_fold.astype(int)) == set(range(5)), "All folds required")
    require(cfg["seed"] == 20260912 and cfg["folds"] == 5 and cfg["causal_forest"]["inference"] is False, "Frozen parameters")
    if args.mode == "estimate":
        require(args.authorize_national and args.pilot_validation is not None, "National run needs authorization and independent pilot validation")
        p = json.loads(args.pilot_validation.read_text(encoding="utf-8"))
        require(p["status"] == "PASS_INDEPENDENT_REQUALIFIED_NESTED_MODELS_SCORES_AND_EXECUTION" and p["mode"] == "pilot", "Independent new-version pilot gate")
        require(p["script_sha256"] == digest(Path(__file__)) and p["revised_sample_sha256"] == digest(args.source / "revised_analysis_sample.csv"), "Pilot code/sample identity")
    source_hashes = {str(p): digest(p) for p in [Path(__file__), CONFIG, args.validation,
                     args.source / "audit.json", args.source / "revised_analysis_sample.csv", ROOT / "00_admin/project_state.yaml"]}
    identity = {"mode": args.mode, "rows": len(df), "source_hashes": source_hashes,
                "seed": cfg["seed"], "expected_support_fits": 30, "expected_total_fits": 60}
    out = args.out
    if args.resume:
        require((out / "run_identity.json").is_file(), "No resumable run")
        require(json.loads((out / "run_identity.json").read_text(encoding="utf-8")) == identity, "Resume source/code/sample mismatch")
        if (out / "audit.json").is_file():
            a = json.loads((out / "audit.json").read_text(encoding="utf-8"))
            require(all(digest(out / p) == h for p, h in a["outputs"].items()), "Completed run changed")
            print("COMPLETED run verified; no training repeated", flush=True)
            return
        append_json(out / "resume_events.jsonl", {"at": dt.datetime.now().isoformat(), "event": "EXPLICIT_RESUME"})
    else:
        require(not out.exists(), "Refuse run overwrite; use explicit resume for the same code/inputs")
        out.mkdir(parents=True)
        (out / "checkpoints").mkdir()
        write_json(out / "run_identity.json", identity)
        write_json(out / "executed_config.json", cfg)
        df[KEY+[T, Y, "spatial_fold", "spatial_block_id", *COORD]].to_csv(out / "sample_manifest.csv", index=False)
    x = df[cfg["x_columns"]].to_numpy(float)
    years = pd.get_dummies(df.outcome_year.astype(str), prefix="outcome_year", dtype=float)
    xy = df[COORD].to_numpy(float) / 1_000_000.0
    geo = np.column_stack([xy[:, 0], xy[:, 1], xy[:, 0] ** 2,
                           xy[:, 1] ** 2, xy[:, 0] * xy[:, 1]])
    z = np.column_stack([x, df[cfg["w_columns"]].to_numpy(float), geo, years.to_numpy()])
    t = df[T].to_numpy(float); y = df[Y].to_numpy(float); folds = df.spatial_fold.to_numpy(int)
    expected_ids = set(); newly_executed = 0; reused = 0

    def fit(stage, outer, features, target, train, test, inner=None):
        nonlocal newly_executed, reused
        assert_split(df, train, test)
        fit_id = f"outer{outer}_{stage}_inner{inner}"
        require(fit_id not in expected_ids, "Duplicate logical fit")
        expected_ids.add(fit_id)
        estimator = (CausalForest(n_jobs=1, random_state=cfg["seed"], **cfg["causal_forest"])
                     if stage == "conditional_forest" else RandomForestRegressor(**cfg["nuisance_forest"]))
        meta = {"fit_id": fit_id, "stage": stage, "outer_fold": outer, "inner_fold": inner,
                "train_rows": len(train), "test_rows": len(test), "train_index_sha256": array_hash(train),
                "test_index_sha256": array_hash(test), "train_design_sha256": array_hash(features),
                "train_target_sha256": array_hash(target), "parameters": estimator.get_params()}
        model_path = out / "checkpoints" / f"{fit_id}.joblib"
        meta_path = out / "checkpoints" / f"{fit_id}.json"
        if model_path.exists() or meta_path.exists():
            require(args.resume and model_path.exists() and meta_path.exists(), "Incomplete/unexpected checkpoint; STOP for explicit repair")
            receipt = json.loads(meta_path.read_text(encoding="utf-8"))
            require(receipt["identity"] == meta and receipt["model_sha256"] == digest(model_path), "Checkpoint identity/hash mismatch")
            estimator = joblib.load(model_path)
            require(estimator.get_params() == meta["parameters"], "Checkpoint parameters changed")
            reused += 1
            append_json(out / "resume_events.jsonl", {"fit_id": fit_id, "event": "REUSED_VERIFIED_CHECKPOINT"})
            return estimator
        print(f"FIT {fit_id} START rows={len(train)} trees={estimator.n_estimators}", flush=True)
        begin = time.monotonic()
        append_json(out / "fit_events.jsonl", {**meta, "status": "STARTED", "at": dt.datetime.now().isoformat()})
        if stage == "conditional_forest":
            estimator.fit(features, target[:, 0].reshape(-1, 1), target[:, 1])
        else:
            estimator.fit(features, target)
        temporary = model_path.with_suffix(".joblib.tmp")
        joblib.dump(estimator, temporary, compress=0)
        os.replace(temporary, model_path)
        write_json(meta_path, {"identity": meta, "model_sha256": digest(model_path)})
        elapsed = time.monotonic()-begin
        append_json(out / "fit_events.jsonl", {**meta, "status": "COMPLETED", "elapsed_seconds": elapsed})
        newly_executed += 1
        print(f"FIT {fit_id} DONE seconds={elapsed:.1f}", flush=True)
        return estimator

    support_rows = []; inner_t_by_outer = {}
    for outer in range(5):
        train = np.flatnonzero(folds != outer); test = np.flatnonzero(folds == outer)
        residual = np.full(len(df), np.nan); visits = np.zeros(len(df), dtype=int)
        for inner in sorted(set(range(5))-{outer}):
            a = np.flatnonzero((folds != outer) & (folds != inner)); b = np.flatnonzero(folds == inner)
            m = fit("inner_t", outer, z[a], t[a], a, b, inner)
            residual[b] = t[b]-m.predict(z[b]); visits[b] += 1
            del m
        require((visits[train] == 1).all() and (visits[test] == 0).all(), "Nested treatment leakage/coverage")
        mt = fit("outer_t", outer, z[train], t[train], train, test)
        v = t[test]-mt.predict(z[test]); del mt
        mv = fit("conditional_variance", outer, x[train], residual[train]**2, train, test)
        variance = mv.predict(x[test]); del mv
        require(np.isfinite(v).all() and np.isfinite(variance).all() and (variance > 0).all(), "Nonfinite/nonpositive conditional treatment variance; STOP")
        inner_t_by_outer[outer] = residual[train]
        r = df.iloc[test][KEY+["spatial_fold", "spatial_block_id", *COORD]].copy()
        r["analysis_row"] = test; r["residual_t"] = v; r["conditional_variance"] = variance
        r["standardized_residual_square"] = v*v/variance
        support_rows.append(r)
        print(f"SUPPORT outer={outer+1}/5 complete; no outcome/slope fit yet", flush=True)
    support = pd.concat(support_rows).sort_values("analysis_row").reset_index(drop=True)
    require(support[KEY].equals(df[KEY]) and len(expected_ids) == 30, "Support phase execution inventory")
    support.to_csv(out / "nested_treatment_support.csv", index=False)
    write_json(out / "support_gate.json", {"status": "PASS_FINITE_POSITIVE_CROSSFIT_CONDITIONAL_VARIANCE",
               "expected_support_fits": 30, "rows": len(df), "variance_min": float(support.conditional_variance.min()),
               "variance_quantiles": {str(q): float(support.conditional_variance.quantile(q)) for q in [0, .01, .5, .99, 1]},
               "standardized_residual_square_max": float(support.standardized_residual_square.max()),
               "no_variance_clipping": True, "unconfoundedness_proven": False,
               "continuous_density_support_proven": False, "qualification_role": "COMPUTATIONAL_INFERENCE_ELIGIBILITY_NOT_CAUSAL_IDENTIFICATION_PROOF"})
    print("PASS nested support phase; start authorized outcome/slope refits", flush=True)
    results = []
    for outer in range(5):
        train = np.flatnonzero(folds != outer); test = np.flatnonzero(folds == outer)
        residual_y = np.full(len(df), np.nan); visits = np.zeros(len(df), dtype=int)
        for inner in sorted(set(range(5))-{outer}):
            a = np.flatnonzero((folds != outer) & (folds != inner)); b = np.flatnonzero(folds == inner)
            m = fit("inner_y", outer, z[a], y[a], a, b, inner)
            residual_y[b] = y[b]-m.predict(z[b]); visits[b] += 1
            del m
        require((visits[train] == 1).all() and (visits[test] == 0).all(), "Nested outcome leakage/coverage")
        my = fit("outer_y", outer, z[train], y[train], train, test)
        u = y[test]-my.predict(z[test]); del my
        targets = np.column_stack([inner_t_by_outer[outer], residual_y[train]])
        forest = fit("conditional_forest", outer, x[train], targets, train, test)
        parameters = forest.predict_full(x[test])
        require(parameters.shape == (len(test), 2), "Slope/intercept dimensions")
        del forest
        r = support.iloc[test].copy()
        require(np.array_equal(r.analysis_row.to_numpy(), test), "Support/output row mismatch")
        r["theta"] = parameters[:, 0]; r["intercept"] = parameters[:, 1]; r["residual_y"] = u
        r["orthogonal_score"] = orthogonal_scores(parameters[:, 0], r.residual_t, u, r.conditional_variance, parameters[:, 1])
        r.to_csv(out / f"outer_fold_{outer}_predictions.csv", index=False)
        results.append(r)
        print(f"SCORES outer={outer+1}/5 DONE", flush=True)
    require(len(expected_ids) == 60, "Nested total fit inventory")
    result = pd.concat(results).sort_values("analysis_row").reset_index(drop=True)
    require(result[KEY].equals(df[KEY]) and np.array_equal(result.analysis_row, np.arange(len(df))), "National score row/order/coverage")
    result.to_csv(out / "crossfit_scores.csv", index=False)
    print("SPATIAL covariance START; bounded memory; all declared radii", flush=True)
    covariance, influence = streamed_covariance(result, result.orthogonal_score.to_numpy(float))
    covariance.to_csv(out / "spatial_covariance_reference.csv", index=False)
    influence.to_csv(out / "tract_influence_concentration.csv", index=False)
    spatial_tail_diagnostic(influence, cfg["seed"]).to_csv(out / "spatial_mark_tail_DIAGNOSTIC_ONLY.csv", index=False)
    events = [json.loads(line) for line in (out / "fit_events.jsonl").read_text(encoding="utf-8").splitlines()]
    completed = [e for e in events if e["status"] == "COMPLETED"]
    require(len(completed) == 60 and len({e["fit_id"] for e in completed}) == 60, "Executed fit inventory not 60 unique completed fits")
    require(set(e["fit_id"] for e in completed) == expected_ids, "Missing/extra fit events")
    require(all(digest(p) == h for p, h in source_hashes.items()), "Sources changed during inference")
    old = json.loads((MODEL / "audit.json").read_text(encoding="utf-8"))
    require(all(digest(MODEL / p) == h for p, h in old["outputs"].items()), "Historical national model changed")
    write_json(out / "audit.json", {
        "status": "COMPLETED_REQUALIFIED_NESTED_ORTHOGONAL_ESTIMATION_PENDING_INDEPENDENT_VALIDATION",
        "mode": args.mode, "rows": len(df), "tracts": int(df.tract_id.nunique()), "expected_fits": 60,
        "executed_unique_fits": 60, "newly_executed_this_invocation": newly_executed, "reused_this_invocation": reused,
        "crossfit_plugin_mean": float(result.theta.mean()), "orthogonal_mean": float(result.orthogonal_score.mean()),
        "maximum_tract_diagonal_share": float(influence.diagonal_share.max()),
        "top10_tract_diagonal_share": float(influence.diagonal_share.head(10).sum()),
        "frozen_folds_or_model_parameters_changed": False, "historic_sources_changed": False,
        "full_national_refit": args.mode == "estimate", "geometry_revision_is_post_result": True,
        "population_health_effect_estimated": False, "causal_identification_proven": False,
        "spatial_interval_coverage": "CONDITIONAL_REFERENCE_NOT_EMPIRICALLY_OR_THEORETICALLY_PROVEN",
        "source_hashes": source_hashes, "script_sha256": digest(Path(__file__)),
        "outputs": {p.name: digest(p) for p in out.iterdir() if p.is_file()},
        "checkpoints": {str(p.relative_to(out)): digest(p) for p in (out / "checkpoints").iterdir() if p.is_file()},
        "elapsed_seconds": time.monotonic()-started})
    print(f"COMPLETED {args.mode} rows={len(df)} executed=60; independent validation required", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pilot", "estimate"], required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--authorize-national", action="store_true")
    parser.add_argument("--pilot-validation", type=Path)
    args = parser.parse_args()
    try:
        execute(args)
    except BaseException:
        if args.out.is_dir():
            path = args.out / f"failure_{dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.txt"
            path.write_text(traceback.format_exc(), encoding="utf-8")
        raise
