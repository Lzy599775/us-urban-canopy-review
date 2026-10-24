"""Primary weather measurement repair; full-season reference, no VPD.

Writes new derived outputs only. Frozen support/folds and lagged treatments
are retained; measurement acceptance does not grant causal identification.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from assemble_and_audit_lagged_support_20261008 import digest, read, require, qualification, REQUIRED, write_json

ROOT = Path(__file__).resolve().parents[1]
MATCH = ROOT / "06_logs/national_gridmet_date_matching_20261008_run01"
PREVIOUS = ROOT / "06_logs/lagged_canopy_weather_join_20261008_run02"
FREEZE = ROOT / "06_logs/national_protocol_spatial_folds_freeze_20261008_run01"
LABELS = ["tmmx_c", "tmmn_c", "pr_mm"]
WEATHER = [v + "_mean_available_land" for v in LABELS]
FEATURES = [*WEATHER, "sin_doy", "cos_doy"]
YRAW = "temperature_c_mean_valid"
Y = "weather_adjusted_lst_reference_mean"
KEY = ["state", "tract_id", "outcome_year"]


def daily(path):
    f = read(path, usecols=["state", "year", "tract_id", "observation_date", "support_status", YRAW, *WEATHER])
    require(not f.duplicated(["state", "year", "tract_id", "observation_date"]).any(), f"Duplicate daily keys: {path.name}")
    d = pd.to_datetime(f.observation_date)
    require(d.dt.month.between(5, 9).all() and d.dt.year.eq(f.year).all(), "Daily dates")
    f["sin_doy"] = np.sin(2 * np.pi * d.dt.dayofyear.to_numpy() / 365.25)
    f["cos_doy"] = np.cos(2 * np.pi * d.dt.dayofyear.to_numpy() / 365.25)
    valid = f[YRAW].notna() & f.support_status.eq("VALID_TEMPERATURE")
    require(f.loc[valid, FEATURES].notna().all().all(), "Missing primary weather on a valid temperature date")
    return f.loc[valid].copy(), len(f)


def rebuild(out):
    start = time.monotonic()
    paths = sorted((MATCH / "daily").glob("state_*_observation_weather.csv.gz"))
    warm = sorted((MATCH / "warm_season").glob("state_*_fixed_land_weather.npz"))
    require(len(paths) == len(warm) == 147, "Expected 147 state-year tasks")
    base_sources = [MATCH / "national_tract_year_lst_weather_unqualified.csv", MATCH / "output_files.json",
                    PREVIOUS / "lagged_canopy_weather_panel_all_records.csv", PREVIOUS / "assembly_audit.json",
                    FREEZE / "audit.json", FREEZE / "冻结方案_天气调整_观测资格_空间折_20261008.md",
                    FREEZE / "folds/national_tract_spatial_fold_map.csv", ROOT / "00_admin/project_state.yaml"]
    previous = json.loads((PREVIOUS / "assembly_audit.json").read_text(encoding="utf-8"))
    require(all(digest(p) == h for p, h in previous["input_hashes"].items()), "Previous sources changed")
    sources = {str(p): digest(p) for p in [*base_sources, *paths, *warm]}
    upstream = json.loads((MATCH / "output_files.json").read_text(encoding="utf-8"))
    for p in [*paths, *warm]:
        rel = str(p.relative_to(MATCH))
        require(sources[str(p)] == upstream[rel]["sha256_file_bytes"], f"Daily/warm source changed: {rel}")
    refs = {}; ref_rows = []
    for path in warm:
        _, state, year, *_ = path.stem.split("_"); year = int(year)
        with np.load(path, allow_pickle=False) as a:
            ids = a["tract_id"]; dates = pd.DatetimeIndex(a["date"])
            require(dates.equals(pd.date_range(f"{year}-05-01", f"{year}-09-30")), "Incomplete full-season dates")
            require(len(set(ids)) == len(ids), "Duplicate reference tracts")
            columns = [list(a["variables"]).index(v) for v in LABELS]
            values = a["values_mean_available_land"][:, :, columns]
            require(values.shape == (153, len(ids), 3) and np.isfinite(values).all(), "Reference weather missing")
            doy = dates.dayofyear.to_numpy()
            mx = np.r_[values.mean(axis=(0, 1)), np.sin(2 * np.pi * doy / 365.25).mean(),
                       np.cos(2 * np.pi * doy / 365.25).mean()]
            refs[(state, year)] = mx
            ref_rows.append({"state": state, "year": year, "tracts": len(ids), "reference_days": 153,
                             "reference_tract_days": len(ids) * 153, **dict(zip(FEATURES, mx))})
    pd.DataFrame(ref_rows).to_csv(out / "full_warm_season_reference.csv", index=False)
    # Match the existing within-state-year common-slope specification. Only the
    # excluded feature and the erroneous reference distribution are changed.
    xx = np.zeros((5, 5)); xy = np.zeros(5); yy = 0.; daily_n = 0; valid_n = 0; groups = set()
    for i, path in enumerate(paths):
        f, n = daily(path); daily_n += n; valid_n += len(f)
        keys = f[["state", "year"]].drop_duplicates()
        require(len(keys) == 1, "Mixed state-year task")
        key = (str(keys.iloc[0].state), int(keys.iloc[0].year)); groups.add(key)
        x = f[FEATURES].to_numpy(float); y = f[YRAW].to_numpy(float)
        xc = x - x.mean(axis=0); yc = y - y.mean()
        xx += xc.T @ xc; xy += xc.T @ yc; yy += yc @ yc
        if (i + 1) % 30 == 0:
            print(f"FIT_STATISTICS {i + 1}/147", flush=True)
    require(groups == set(refs) and daily_n == 7106498, "Task/daily record accounting")
    require(np.linalg.matrix_rank(xx) == 5, "Rank-deficient weather design")
    beta = np.linalg.solve(xx, xy)
    require(np.isfinite(beta).all(), "Nonfinite weather coefficients")
    pd.DataFrame({"feature": FEATURES, "coefficient": beta}).to_csv(out / "primary_weather_coefficients.csv", index=False)
    parts = []
    for i, path in enumerate(paths):
        f, _ = daily(path)
        state = f.state.iloc[0]; year = int(f.year.iloc[0])
        f["daily_adjusted_c"] = f[YRAW].to_numpy(float) - (f[FEATURES].to_numpy(float) - refs[(state, year)]) @ beta
        part = f.groupby(["state", "year", "tract_id"], sort=True).agg(
            valid_lst_dates=(YRAW, "size"), raw_lst_valid_date_mean=(YRAW, "mean"),
            weather_adjusted_lst_reference_mean=("daily_adjusted_c", "mean")).reset_index()
        part["weather_adjustment_difference_c"] = part[Y] - part.raw_lst_valid_date_mean
        parts.append(part)
        if (i + 1) % 30 == 0:
            print(f"ADJUST_ANNUAL {i + 1}/147", flush=True)
    adjusted = pd.concat(parts, ignore_index=True).sort_values(["state", "tract_id", "year"], kind="stable")
    require(len(adjusted) == 140177 and not adjusted.duplicated(["state", "tract_id", "year"]).any(), "Annual output accounting")
    adjusted.to_csv(out / "annual_primary_weather_standardized.csv", index=False)
    old = read(PREVIOUS / "lagged_canopy_weather_panel_all_records.csv")
    require(len(old) == 194997 and not old.duplicated(KEY).any(), "Previous lagged keys")
    full = old.drop(columns=[Y, "valid_lst_dates", "raw_lst_valid_date_mean", "weather_adjustment_difference_c",
                             "adjustment_join_status"]).merge(adjusted.rename(columns={"year": "outcome_year"}),
                     on=KEY, how="left", validate="one_to_one", indicator="adjustment_join_status")
    full = full.sort_values(["state", "tract_id", "outcome_year"], kind="stable").reset_index(drop=True)
    require(len(full) == len(old), "Records lost during primary join")
    full["adjusted_outcome_protocol_status"] = "PRIMARY_NO_VPD_FULL_STATE_YEAR_WARM_SEASON_REFERENCE"
    full["complete_required_diagnostic"] = full[REQUIRED].notna().all(axis=1)
    full["support_sample_selected_diagnostic"] = qualification(full) & full.complete_required_diagnostic
    require(np.array_equal(full.observation_qualified_diagnostic.to_numpy(), qualification(full).to_numpy()), "Observation qualification changed")
    old = old.sort_values(["state", "tract_id", "outcome_year"], kind="stable").reset_index(drop=True)
    require(full[KEY].equals(old[KEY]), "Joined key ordering")
    changed = full.support_sample_selected_diagnostic != old.support_sample_selected_diagnostic
    full.loc[qualification(full), "sample_flow_reason"] = np.where(full.loc[qualification(full), "complete_required_diagnostic"],
                                                               "SUPPORT_DIAGNOSTIC_INCLUDED", "REQUIRED_FIELD_MISSING")
    full.to_csv(out / "lagged_primary_panel_all_records.csv", index=False)
    sample = full.loc[full.support_sample_selected_diagnostic].copy().reset_index(drop=True)
    sample.to_csv(out / "lagged_primary_model_sample.csv", index=False)
    full.loc[changed, [*KEY, "support_sample_selected_diagnostic"]].to_csv(out / "sample_membership_changes.csv", index=False)
    comparison = full[KEY].copy(); comparison["previous_adjusted_c"] = old[Y]; comparison["primary_adjusted_c"] = full[Y]
    comparison["primary_minus_previous_c"] = full[Y] - old[Y]
    comparison.to_csv(out / "primary_vs_previous_outcome.csv", index=False)
    delta = comparison.primary_minus_previous_c.dropna()
    require(all(digest(p) == h for p, h in sources.items()), "Source files changed during rebuild")
    audit = {"status": "COMPLETED_PRIMARY_OUTCOME_REBUILD_PENDING_INDEPENDENT_VALIDATION",
             "features": FEATURES, "vpd_included": False, "reference": "FULL_153_DAYS_EQUAL_TRACT_DAY_WITHIN_STATE_YEAR",
             "coefficient_specification": "POOLED_COMMON_SLOPE_WITH_STATE_YEAR_FIXED_INTERCEPTS",
             "reference_traces": len(refs), "daily_source_rows": daily_n, "valid_temperature_daily_rows": valid_n,
             "weather_within_group_r2": float(beta @ xy / yy), "weather_matrix_condition_number": float(np.linalg.cond(xx)),
             "annual_adjusted_rows": len(adjusted), "full_lagged_rows": len(full), "sample_rows": len(sample),
             "sample_tracts": int(sample.tract_id.nunique()), "sample_blocks": int(sample.spatial_block_id.nunique()),
             "sample_added": int((changed & full.support_sample_selected_diagnostic).sum()),
             "sample_removed": int((changed & old.support_sample_selected_diagnostic).sum()),
             "outcome_change_c_quantiles": {str(q): float(delta.quantile(q)) for q in [0, .01, .5, .99, 1]},
             "outcome_change_mean_c": float(delta.mean()), "seed_or_fold_changed": False,
             "formal_model": "NOT_RUN", "primary_measurement_gate": "PENDING_INDEPENDENT_VALIDATION",
             "causal_identification_gate": "NOT_ESTABLISHED_BY_MEASUREMENT_VALIDATION",
             "project_authority_modified": False, "elapsed_seconds": time.monotonic() - start,
             "source_hashes": sources, "output_hashes": {p.name: digest(p) for p in out.glob("*.csv")},
             "script_sha256": digest(Path(__file__)), "created_local": dt.datetime.now().isoformat()}
    write_json(out / "audit.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k not in ["source_hashes", "output_hashes"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); require(not args.out.exists(), "Refusing existing output directory")
    args.out.mkdir(); rebuild(args.out)
