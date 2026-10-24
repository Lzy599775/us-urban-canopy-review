"""Join frozen lagged treatment pairs and diagnose support using frozen folds.

Existing adjusted outcomes are retained as supplied, with their VPD/protocol
conflict explicitly recorded. No outcome/effect model or project-state update.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "06_logs"
LAG = LOGS / "national_lagged_canopy_treatment_readiness_20260915_run01"
MATCH = LOGS / "national_gridmet_date_matching_20261008_run01"
ADJUST = LOGS / "weather_adjustment_support_qualification_20261008_run01"
FREEZE = LOGS / "national_protocol_spatial_folds_freeze_20261008_run01"
KEY = ["state", "tract_id", "outcome_year"]
PAIRS = {(2013, 2014), (2018, 2019), (2023, 2024)}
SEED = 20260912
T = "treatment_pp_change_from_2000"
Y = "weather_adjusted_lst_reference_mean"
XCOLS = ["age65plus_share_2000", "poverty_share_2000", "income_point_usd_2000", "population_2000"]
WCOLS = ["baseline_canopy_percent_fixed_land_2000", "tmmx_c_warm_season_mean_available_land",
         "tmmn_c_warm_season_mean_available_land", "pr_mm_warm_season_mean_available_land"]
REQUIRED = [T, Y, *XCOLS, *WCOLS, "spatial_block_id", "spatial_fold"]
CONFLICT = "ADJUSTED_OUTCOME_INCLUDES_VPD_EXCLUDED_BY_PRIMARY_CANDIDATE_FREEZE"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path, **kwargs):
    return pd.read_csv(path, dtype={"state": str, "tract_id": str, "county_code": str},
                       float_precision="round_trip", **kwargs)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def qualification(frame):
    return ((frame.valid_temperature_dates >= 6) & (frame.valid_temperature_months >= 4)
            & (frame.mean_valid_fraction_geometric >= .25)
            & (frame.maximum_gap_including_season_edges_days <= 64)
            & frame.weather_spatial_support_status.eq("FULL_FIXED_LAND_SUPPORT"))


def assemble(out):
    started = time.monotonic()
    paths = [LAG / "national_lagged_canopy_treatment_readiness.csv", LAG / "audit.json",
             MATCH / "national_tract_year_lst_weather_unqualified.csv",
             ADJUST / "annual_weather_adjusted_valid_only_diagnostic.csv", ADJUST / "audit.json",
             FREEZE / "folds/national_tract_spatial_fold_map.csv", FREEZE / "audit.json",
             FREEZE / "冻结方案_天气调整_观测资格_空间折_20261008.md", ROOT / "00_admin/project_state.yaml"]
    hashes = {str(p): digest(p) for p in paths}
    lag_audit = json.loads((LAG / "audit.json").read_text(encoding="utf-8"))
    require(hashes[str(paths[0])] == lag_audit["output_sha256_file_bytes"], "Lag source hash mismatch")
    freeze_audit = json.loads((FREEZE / "audit.json").read_text(encoding="utf-8"))
    for path, expected in freeze_audit["source_hashes"].items():
        require(digest(path) == expected, f"Frozen source changed: {path}")
    require(hashes[str(paths[7])] == freeze_audit["protocol_sha256_file_bytes"], "Frozen protocol changed")
    adjustment_audit = json.loads((ADJUST / "audit.json").read_text(encoding="utf-8"))
    require("vpd_kpa_mean_available_land" in adjustment_audit["weather_model"]["features"],
            "Reinspect changed adjusted-outcome provenance")
    lag = read(paths[0])
    require(len(lag) == 194997 and not lag.duplicated(KEY).any(), "Lag source keys")
    require(set(zip(lag.exposure_year, lag.outcome_year)) == PAIRS, "Lag exposure/outcome pairs")
    require((lag.outcome_year == lag.exposure_year + 1).all(), "Temporal alignment")
    numeric = lag[[T, "exposure_canopy_percent_fixed_land", "baseline_canopy_percent_fixed_land_2000"]].notna().all(axis=1)
    require(np.allclose(lag.loc[numeric, T], lag.loc[numeric, "exposure_canopy_percent_fixed_land"]
                        - lag.loc[numeric, "baseline_canopy_percent_fixed_land_2000"], atol=1e-10, rtol=0), "Treatment identity")
    require(lag[T].notna().equals(lag.treatment_readiness_status.eq("AVAILABLE_CUMULATIVE_CHANGE")), "Treatment status mismatch")
    panel_columns = ["state", "tract_id", "year", "treatment_canopy_change_pp", "year_canopy_percent_fixed_land",
                     "baseline_canopy_percent_fixed_land_2000", "urban_fixed_land_m2", *XCOLS, *WCOLS[1:],
                     "valid_temperature_dates", "valid_temperature_months", "mean_valid_fraction_geometric",
                     "maximum_gap_including_season_edges_days", "weather_spatial_support_status"]
    panel = read(paths[2], usecols=panel_columns).rename(columns={
        "year": "outcome_year", "treatment_canopy_change_pp": "contemporaneous_canopy_change_pp_diagnostic_only",
        "year_canopy_percent_fixed_land": "outcome_year_canopy_percent_diagnostic_only",
        "baseline_canopy_percent_fixed_land_2000": "panel_baseline_canopy_check",
        "urban_fixed_land_m2": "panel_fixed_land_check"})
    adjusted = read(paths[3]).rename(columns={"year": "outcome_year"})
    folds = read(paths[5])
    require(len(panel) == 140181 and len(adjusted) == adjustment_audit["valid_adjusted_rows"]
            and not panel.duplicated(KEY).any()
            and not adjusted.duplicated(KEY).any(), "Outcome keys")
    require(set(map(tuple, adjusted[KEY].to_numpy())).issubset(set(map(tuple, panel[KEY].to_numpy()))),
            "Adjusted outcome has unknown keys")
    require(not folds.duplicated(["state", "tract_id"]).any(), "Fold duplicate keys")
    require(folds.groupby("spatial_block_id").spatial_fold.nunique().eq(1).all(), "Frozen group leakage")
    joined = lag.merge(panel, on=KEY, how="left", validate="one_to_one", indicator="temperature_join_status")
    joined = joined.merge(adjusted, on=KEY, how="left", validate="one_to_one", indicator="adjustment_join_status")
    joined = joined.merge(folds[["state", "tract_id", "spatial_block_id", "spatial_fold"]],
                          on=["state", "tract_id"], how="left", validate="many_to_one")
    positive = joined.urban_fixed_land_m2.gt(0)
    matched = joined.temperature_join_status.eq("both")
    require(len(joined) == len(lag) and positive.equals(matched), "Fixed-land/outcome join mismatch")
    adjustment_matched = joined.adjustment_join_status.eq("both")
    missing_adjusted = matched & ~adjustment_matched
    require(not (adjustment_matched & ~matched).any(), "Adjusted outcome outside temperature units")
    require(int(missing_adjusted.sum()) == len(panel) - len(adjusted)
            and joined.loc[missing_adjusted, "valid_temperature_dates"].eq(0).all(),
            "Unexplained missing adjusted outcomes")
    require(joined.loc[matched, "spatial_fold"].notna().all(), "Missing frozen fold")
    comparisons = {"baseline_canopy_percent_fixed_land_2000": "panel_baseline_canopy_check",
                   "urban_fixed_land_m2": "panel_fixed_land_check", "age65plus_share_sf3_p8": XCOLS[0],
                   "poverty_share": XCOLS[1], "income_point_usd": XCOLS[2], "population_2000_100_percent": XCOLS[3]}
    for left, right in comparisons.items():
        require(np.allclose(joined.loc[matched, left], joined.loc[matched, right], equal_nan=True,
                            rtol=1e-10, atol=1e-8), f"Baseline attribute mismatch: {left}/{right}")
    joined["observation_qualified_diagnostic"] = matched & qualification(joined)
    joined["complete_required_diagnostic"] = joined[REQUIRED].notna().all(axis=1)
    joined["support_sample_selected_diagnostic"] = joined.observation_qualified_diagnostic & joined.complete_required_diagnostic
    reason = np.full(len(joined), "SUPPORT_DIAGNOSTIC_INCLUDED", dtype=object)
    reason[~joined.complete_required_diagnostic] = "REQUIRED_FIELD_MISSING"
    reason[~joined.observation_qualified_diagnostic] = "OBSERVATION_QUALIFICATION_FAILED"
    reason[~matched] = "ZERO_FIXED_URBAN_LAND_NO_TEMPERATURE_UNIT"
    joined["sample_flow_reason"] = reason
    joined["adjusted_outcome_protocol_status"] = CONFLICT
    joined["formal_model_status"] = "NOT_RUN"
    require(not joined.duplicated(KEY).any(), "Joined duplicate keys")
    joined.to_csv(out / "lagged_canopy_weather_panel_all_records.csv", index=False)
    sample = joined.loc[joined.support_sample_selected_diagnostic].copy().reset_index(drop=True)
    require(set(sample.spatial_fold.astype(int)) == set(range(5)), "Support sample lacks a frozen fold")
    require(sample.groupby(["state", "tract_id"]).spatial_fold.nunique().eq(1).all(), "Year fold leakage")
    require(np.isfinite(sample[REQUIRED[:-2]].to_numpy(float)).all(), "Nonfinite diagnostic sample")
    sample.to_csv(out / "lagged_support_sample_diagnostic.csv", index=False)
    yearflow = []
    for (exposure, outcome), group in joined.groupby(["exposure_year", "outcome_year"]):
        row = {"exposure_year": int(exposure), "outcome_year": int(outcome), "all_rows": len(group),
               "temperature_matched": int(group.temperature_join_status.eq("both").sum()),
               "observation_qualified": int(group.observation_qualified_diagnostic.sum()),
               "support_complete": int(group.support_sample_selected_diagnostic.sum())}
        row.update({str(k): int(v) for k, v in group.sample_flow_reason.value_counts().items()})
        yearflow.append(row)
    pd.DataFrame(yearflow).fillna(0).to_csv(out / "sample_flow_by_pair.csv", index=False)
    pd.DataFrame({"field": REQUIRED, "missing_in_positive_land_rows": joined.loc[positive, REQUIRED].isna().sum().values}).to_csv(
        out / "required_field_missing_counts.csv", index=False)
    require(all(digest(p) == h for p, h in hashes.items()), "Inputs changed during assembly")
    audit = {"status": "PASS_LAGGED_JOIN_DIAGNOSTIC_WITH_OUTCOME_PROTOCOL_CONFLICT", "input_rows": len(lag),
             "output_rows": len(joined), "unique_tracts": int(joined.tract_id.nunique()), "duplicate_keys": 0,
             "dropped_from_full_panel": 0, "temperature_matched_rows": int(matched.sum()),
             "unmatched_zero_land_rows_retained": int((~matched).sum()),
             "missing_adjusted_zero_valid_date_rows_retained": int(missing_adjusted.sum()),
             "observation_qualified_rows": int(joined.observation_qualified_diagnostic.sum()),
             "support_sample_rows": len(sample), "support_sample_tracts": int(sample.tract_id.nunique()),
             "support_sample_blocks": int(sample.spatial_block_id.nunique()), "pairs": sorted(PAIRS),
             "fold_source": str(paths[5]), "folds_regenerated": False, "seed": SEED,
             "outcome_protocol_conflict": CONFLICT, "formal_primary_outcome_ready": False,
             "ten_fold_requested": False, "ten_fold_reason": "USER_ACCEPTED_5_FOLD_ONLY_NO_CLEAR_EXISTING_ADVANTAGE",
             "formal_model": "NOT_RUN", "effect_estimated": False, "authority_modified": False,
             "elapsed_seconds": time.monotonic() - started, "input_hashes": hashes,
             "output_hashes": {p.name: digest(p) for p in out.glob("*.csv")},
             "script_sha256": digest(Path(__file__)), "created_local": dt.datetime.now().isoformat()}
    write_json(out / "assembly_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


def diagnose(assembly, out):
    started = time.monotonic()
    receipt = json.loads((assembly / "assembly_audit.json").read_text(encoding="utf-8"))
    sample_path = assembly / "lagged_support_sample_diagnostic.csv"
    require(digest(sample_path) == receipt["output_hashes"][sample_path.name], "Assembly sample changed")
    require(all(digest(p) == h for p, h in receipt["input_hashes"].items()), "Assembly sources changed")
    frame = read(sample_path)
    require(len(frame) == receipt["support_sample_rows"] and not frame.duplicated(KEY).any(), "Diagnostic keys")
    require(qualification(frame).all() and frame[REQUIRED].notna().all().all(), "Diagnostic qualification")
    years = pd.get_dummies(frame.outcome_year.astype(str), prefix="outcome_year", dtype=float)
    design = pd.concat([frame[XCOLS + WCOLS].astype(float), years], axis=1)
    x = design.to_numpy(); t = frame[T].to_numpy(float)
    foldid = frame.spatial_fold.to_numpy(int); groups = frame.spatial_block_id.to_numpy()
    require(np.isfinite(x).all() and np.isfinite(t).all(), "Nonfinite design")
    require(frame.groupby("spatial_block_id").spatial_fold.nunique().eq(1).all(), "Group leakage")
    predictions = np.full(len(frame), np.nan); visits = np.zeros(len(frame), dtype=int)
    rows = []
    for fold in range(5):
        fold_start = time.monotonic()
        train = np.flatnonzero(foldid != fold); test = np.flatnonzero(foldid == fold)
        require(len(train) > 0 and len(test) > 0 and not set(groups[train]) & set(groups[test]), "Split leakage")
        print(f"FOLD {fold + 1}/5 START train={len(train)} test={len(test)}", flush=True)
        model = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                      random_state=SEED, n_jobs=1)
        model.fit(x[train], t[train]); pred = model.predict(x[test]); residual = t[test] - pred
        require(np.isfinite(pred).all(), "Nonfinite predictions")
        predictions[test] = pred; visits[test] += 1
        lo, hi = np.quantile(t[train], [.005, .995])
        variance = np.sum((t[test] - t[test].mean()) ** 2)
        require(variance > 0, "Zero held-out treatment variance")
        row = {"held_out_fold": fold, "train_rows": len(train), "test_rows": len(test),
               "train_blocks": int(np.unique(groups[train]).size), "test_blocks": int(np.unique(groups[test]).size),
               "residual_mean": float(residual.mean()), "residual_sd": float(residual.std(ddof=1)),
               "residual_p01": float(np.quantile(residual, .01)), "residual_p99": float(np.quantile(residual, .99)),
               "treatment_test_sd": float(t[test].std(ddof=1)),
               "test_r2": float(1 - np.sum(residual ** 2) / variance),
               "outside_train_minmax_n": int(((t[test] < t[train].min()) | (t[test] > t[train].max())).sum()),
               "outside_train_q005_q995_n": int(((t[test] < lo) | (t[test] > hi)).sum()),
               "outside_train_q005_q995_fraction": float(((t[test] < lo) | (t[test] > hi)).mean()),
               "train_treatment_min": float(t[train].min()), "train_treatment_max": float(t[train].max()),
               "test_treatment_min": float(t[test].min()), "test_treatment_max": float(t[test].max()),
               "executed_fits": 1, "elapsed_seconds": time.monotonic() - fold_start}
        rows.append(row)
        write_json(out / f"fold_{fold}_receipt.json", row)
        print(f"FOLD {fold + 1}/5 DONE seconds={row['elapsed_seconds']:.1f} residual_sd={row['residual_sd']:.4f}", flush=True)
    require((visits == 1).all() and np.isfinite(predictions).all(), "Missing or duplicate held-out predictions")
    fold_table = pd.DataFrame(rows)
    fold_table.to_csv(out / "frozen_5fold_treatment_support.csv", index=False)
    oof = frame[["state", "tract_id", "exposure_year", "outcome_year", T, "spatial_block_id", "spatial_fold"]].copy()
    oof["held_out_prediction_pp"] = predictions; oof["treatment_residual_pp"] = t - predictions
    oof["held_out_visits"] = visits
    oof.to_csv(out / "held_out_treatment_predictions.csv", index=False)
    require(all(digest(p) == h for p, h in receipt["input_hashes"].items()), "Sources changed during fit")
    require(digest(sample_path) == receipt["output_hashes"][sample_path.name], "Sample changed during fit")
    summary = {"min_residual_sd_pp": float(fold_table.residual_sd.min()),
               "max_residual_sd_pp": float(fold_table.residual_sd.max()),
               "min_test_r2": float(fold_table.test_r2.min()), "max_test_r2": float(fold_table.test_r2.max()),
               "outside_training_extrema_total_rows": int(fold_table.outside_train_minmax_n.sum()),
               "max_outside_training_q005_q995_fraction": float(fold_table.outside_train_q005_q995_fraction.max())}
    audit = {"status": "COMPLETED_FROZEN_5FOLD_LAGGED_TREATMENT_SUPPORT_DIAGNOSTIC", "summary": summary,
             "rows": len(frame), "tracts": int(frame.tract_id.nunique()), "blocks": int(frame.spatial_block_id.nunique()),
             "expected_fits": 5, "executed_fits": len(rows), "failed_fits": 0, "reused_fits": 0,
             "skipped_fits": 0, "duplicate_fits": 0, "held_out_visits_per_row": 1, "folds_regenerated": False,
             "model": "RandomForestRegressor", "model_parameters": {"n_estimators": 200, "max_depth": 12,
                 "min_samples_leaf": 5, "random_state": SEED, "n_jobs": 1}, "design_columns": list(design.columns),
             "treatment_field": T, "outcome_used_for_eligibility_only": Y,
             "outcome_protocol_conflict": CONFLICT, "formal_primary_outcome_ready": False,
             "formal_model": "NOT_RUN", "effect_estimated": False, "authority_modified": False,
             "ten_fold": "NOT_RUN_PER_USER_DECISION", "elapsed_seconds": time.monotonic() - started,
             "assembly_receipt_sha256": digest(assembly / "assembly_audit.json"), "sample_sha256": digest(sample_path),
             "source_hashes_verified_unchanged": True, "output_hashes": {p.name: digest(p) for p in out.glob("*.csv")}}
    write_json(out / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["assemble", "diagnose"], required=True)
    parser.add_argument("--assembly", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Refusing existing output directory")
    if args.mode == "diagnose":
        require(args.assembly is not None, "Diagnosis requires explicit assembly directory")
    args.out.mkdir()
    if args.mode == "assemble":
        assemble(args.out)
    else:
        diagnose(args.assembly, args.out)
