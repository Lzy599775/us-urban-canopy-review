"""Independent reload and actual 60-fit execution-chain verification. No fits."""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from assemble_and_audit_lagged_support_20261008 import digest, read, require, write_json
from spatial_mean_slope_inference_20261009 import KEY, T, Y, COORD, BANDWIDTHS, array_hash, assert_split
from run_geo_adjusted_national_orthogonal_20261009 import load_requalified, COORD


def independent_covariance(result):
    mean = float(result.orthogonal_score.mean())
    a = result[["state", "tract_id", *COORD]].copy()
    a["mark"] = result.orthogonal_score.to_numpy()-mean
    g = a.groupby(["state", "tract_id"], sort=True).agg({COORD[0]: "first", COORD[1]: "first", "mark": "sum"})
    xy = g[COORD].to_numpy(); q = g.mark.to_numpy(); tree = cKDTree(xy)
    sums = np.zeros(3); counts = np.zeros(3, dtype=np.int64)
    for i, point in enumerate(xy):
        js = np.asarray([j for j in tree.query_ball_point(point, max(BANDWIDTHS)) if j > i], dtype=np.int64)
        if not len(js):
            continue
        distance = np.sqrt(((xy[js]-point)**2).sum(axis=1))
        for k, h in enumerate(BANDWIDTHS):
            use = distance <= h
            ratio = distance[use]/h
            sums[k] += float(np.sum((1-ratio)**4*(1+4*ratio)*q[i]*q[js[use]]))
            counts[k] += int(use.sum())
    return (q@q+2*sums)/len(result)**2, counts


def execute(args):
    start = time.monotonic()
    run = args.run
    a = json.loads((run / "audit.json").read_text(encoding="utf-8"))
    require(a["status"] == "COMPLETED_REQUALIFIED_NESTED_ORTHOGONAL_ESTIMATION_PENDING_INDEPENDENT_VALIDATION", "Completed nested run gate")
    require(all(digest(p) == h for p, h in a["source_hashes"].items()), "Run source changed")
    require(all(digest(run / p) == h for p, h in a["outputs"].items()), "Run output changed")
    require(all(digest(run / p) == h for p, h in a["checkpoints"].items()), "Run checkpoint changed")
    cfg, df = load_requalified(args.source, args.validation)
    if a["mode"] == "pilot":
        import hashlib
        mask = df.tract_id.map(lambda t: int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "big") % 20 == 0)
        df = df.loc[mask].reset_index(drop=True)
    result = read(run / "crossfit_scores.csv")
    require(result[KEY].equals(df[KEY]) and np.array_equal(result.analysis_row.to_numpy(), np.arange(len(df))), "Validation row/order/coverage")
    x = df[cfg["x_columns"]].to_numpy(float)
    year = pd.get_dummies(df.outcome_year.astype(str), prefix="outcome_year", dtype=float)
    xy = df[COORD].to_numpy(float) / 1_000_000.0
    geo = np.column_stack([xy[:, 0], xy[:, 1], xy[:, 0] ** 2,
                           xy[:, 1] ** 2, xy[:, 0] * xy[:, 1]])
    z = np.column_stack([x, df[cfg["w_columns"]].to_numpy(float), geo, year.to_numpy()])
    t = df[T].to_numpy(float); y = df[Y].to_numpy(float); folds = df.spatial_fold.to_numpy(int)
    ledger = [json.loads(s) for s in (run / "fit_events.jsonl").read_text(encoding="utf-8").splitlines()]
    events = [e for e in ledger if e["status"] == "COMPLETED"]
    require(len(events) == 60 and len({e["fit_id"] for e in events}) == 60, "Incomplete/duplicate actual fits")
    by_id = {e["fit_id"]: e for e in events}; ids = set(); errors = []

    def model(stage, outer, features, target, train, test, inner=None):
        assert_split(df, train, test)
        fit_id = f"outer{outer}_{stage}_inner{inner}"
        require(fit_id not in ids and fit_id in by_id, "Duplicate/missing executed model")
        ids.add(fit_id)
        e = by_id[fit_id]
        for key, value in {"train_rows": len(train), "test_rows": len(test),
                           "train_index_sha256": array_hash(train), "test_index_sha256": array_hash(test),
                           "train_design_sha256": array_hash(features), "train_target_sha256": array_hash(target)}.items():
            require(e[key] == value, f"Actual execution mismatch: {fit_id} {key}")
        m = joblib.load(run / "checkpoints" / f"{fit_id}.joblib")
        require(m.get_params() == e["parameters"] and len(m.estimators_) == m.n_estimators, "Saved estimator/trees differ")
        expected = cfg["causal_forest"] if stage == "conditional_forest" else cfg["nuisance_forest"]
        require(all(m.get_params()[k] == v for k, v in expected.items()), "Frozen model parameter mismatch")
        return m

    for outer in range(5):
        train = np.flatnonzero(folds != outer); test = np.flatnonzero(folds == outer)
        ru = np.full(len(df), np.nan); rv = np.full(len(df), np.nan)
        for inner in sorted(set(range(5))-{outer}):
            itrain = np.flatnonzero((folds != outer) & (folds != inner)); itest = np.flatnonzero(folds == inner)
            require(not np.intersect1d(itrain, test).size and not np.intersect1d(itest, test).size, "Outer test contaminates inner stage")
            m = model("inner_t", outer, z[itrain], t[itrain], itrain, itest, inner)
            rv[itest] = t[itest]-m.predict(z[itest]); del m
            m = model("inner_y", outer, z[itrain], y[itrain], itrain, itest, inner)
            ru[itest] = y[itest]-m.predict(z[itest]); del m
        m = model("outer_t", outer, z[train], t[train], train, test)
        v = t[test]-m.predict(z[test]); del m
        m = model("outer_y", outer, z[train], y[train], train, test)
        u = y[test]-m.predict(z[test]); del m
        m = model("conditional_variance", outer, x[train], rv[train]**2, train, test)
        variance = m.predict(x[test]); del m
        m = model("conditional_forest", outer, x[train], np.column_stack([rv[train], ru[train]]), train, test)
        p = m.predict_full(x[test]); del m
        score = p[:, 0]+v*(u-p[:, 1]-p[:, 0]*v)/variance
        for name, value in {"theta": p[:, 0], "intercept": p[:, 1], "residual_t": v,
                            "residual_y": u, "conditional_variance": variance, "orthogonal_score": score}.items():
            err = float(np.max(np.abs(value-result.iloc[test][name].to_numpy(float))))
            require(err == 0, f"Independent reload mismatch: {outer} {name}")
            errors.append({"outer_fold": outer, "quantity": name, "maximum_absolute_error": err})
        print(f"VALIDATE outer={outer+1}/5; 12 actual fits independently checked; no new fits", flush=True)
    require(ids == set(by_id), "Missing/extra stages in actual execution")
    computed, counts = independent_covariance(result)
    saved = read(run / "spatial_covariance_reference.csv")
    for k, h in enumerate(BANDWIDTHS):
        row = saved.loc[saved.bandwidth_m.eq(h)]
        require(len(row) == 1 and float(row.iloc[0].estimate) == float(result.orthogonal_score.mean()), "Wrong covariance/estimator pair")
        require(np.isclose(computed[k], row.iloc[0].variance, rtol=1e-11, atol=1e-14), "Independent spatial covariance mismatch")
        require(counts[k] == row.iloc[0].unique_between_tract_pairs, "Missing spatial pairs")
    require(not args.out.exists(), "Refuse overwriting independent model validation")
    args.out.mkdir(parents=True)
    pd.DataFrame(errors).to_csv(args.out / "independent_reload_errors.csv", index=False)
    write_json(args.out / "validation.json", {
        "status": "PASS_INDEPENDENT_REQUALIFIED_NESTED_MODELS_SCORES_AND_EXECUTION", "mode": a["mode"],
        "rows": len(df), "tracts": int(df.tract_id.nunique()), "verified_actual_unique_fits": len(ids),
        "started_fit_attempts": sum(e["status"] == "STARTED" for e in ledger),
        "new_model_fits": 0, "maximum_recompute_error": 0., "spatial_covariance_verified": True,
        "script_sha256": a["script_sha256"], "revised_sample_sha256": digest(args.source / "revised_analysis_sample.csv"),
        "input_audit_sha256": digest(run / "audit.json"), "validator_sha256": digest(Path(__file__)),
        "coverage_or_causal_identification_proven": False, "elapsed_seconds": time.monotonic()-start})
    print(f"PASS independent {a['mode']} nested-model reload, row order and all 60 actual fits", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True); p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    execute(p.parse_args())
