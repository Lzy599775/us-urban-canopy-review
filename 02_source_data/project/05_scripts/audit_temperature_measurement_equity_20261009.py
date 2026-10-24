"""Read-only product applicability audit; no temperature correction or refit."""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "06_logs"
SAMPLE = LOG / "geographic_measurement_repair_20261009_run01/revised_analysis_sample.csv"
MANIFEST = LOG / "lst_targeted_recovery_final_20261008_run02/daily_table_manifest.csv"
CATALOG = LOG / "national_lst_candidate_inventory_20260915_run01"
KEY = ["state", "tract_id", "outcome_year"]
DAILY = ["state", "tract_id", "year", "observation_date", "source_item_ids", "status",
         "valid_temperature_area_m2", "temperature_c_mean_valid", "st_emis_mean",
         "st_emsd_mean", "st_qa_mean_c", "st_cdist_mean_km"]
URLS = {
    "known_issues": "https://www.usgs.gov/landsat-missions/landsat-collection-2-known-issues",
    "surface_temperature": "https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature",
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path, **kwargs):
    return pd.read_csv(path, dtype={"state": str, "tract_id": str},
                       float_precision="round_trip", **kwargs)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def select_points(sample):
    """Select before looking at emissivity, temperature, scores or effect size."""
    rows = []
    for state in ["04", "06", "17", "48"]:
        f = sample.loc[sample.state.eq(state) & sample.outcome_year.eq(2024)].copy()
        require(len(f) > 0, "Target state has no 2024 observations")
        pr = "pr_mm_warm_season_mean_available_land"
        threshold = float(f[pr].median())
        for wetter in [False, True]:
            g = f.loc[f[pr].gt(threshold).eq(wetter)].copy()
            require(len(g) > 0, "Empty within-state precipitation stratum")
            t = "treatment_pp_change_from_2000"
            choices = {
                "largest_increase": g.loc[g[t] > 0].sort_values([t, "tract_id"], ascending=[False, True]),
                "largest_decrease": g.loc[g[t] < 0].sort_values([t, "tract_id"]),
                "closest_to_zero": g.assign(abs_change=g[t].abs()).sort_values(["abs_change", "tract_id"]),
            }
            for label, ordered in choices.items():
                require(len(ordered) > 0, f"Empty treatment stratum {state}/{label}")
                row = ordered.iloc[0][["state", "tract_id", t, "baseline_canopy_percent_fixed_land_2000", pr]].to_dict()
                row.update(selection=label, precipitation_stratum="above_state_median" if wetter else "at_or_below_state_median",
                           precipitation_median=threshold, selection_uses_temperature_or_effect=False)
                rows.append(row)
    return pd.DataFrame(rows)


def pixel_audit(out, hashes):
    local = ROOT / "04_pilot/landsat_c2_l2/full_pilot_20260912_run01"
    choices = {}
    for directory in sorted(local.iterdir()):
        if directory.is_dir():
            parts = directory.name.split("_")
            day = parts[3]
            if day[4:6] in ("06", "07", "08"):
                choices.setdefault(parts[2], directory)
    require(len(choices) > 0, "No local summer scenes")
    rows = []
    for directory in choices.values():
        assets = {}
        for name in ["ST_B10", "ST_EMIS", "ST_EMSD", "ST_QA", "QA_PIXEL", "QA_RADSAT"]:
            matches = list(directory.glob(f"*_{name}.TIF"))
            require(len(matches) == 1, f"Missing/duplicate local asset {directory.name}/{name}")
            assets[name] = matches[0]
            # Avoid hashing multi-gigabyte scenes; exact audited windows are hashed below.
            hashes[str(matches[0])] = {"bytes": matches[0].stat().st_size,
                                      "mtime_ns": matches[0].stat().st_mtime_ns,
                                      "basis": "STAT_PLUS_WINDOW_CONTENT_SHA256_NOT_FULL_FILE_SHA256"}
        handles = {k: rasterio.open(p) for k, p in assets.items()}
        try:
            base = handles["ST_B10"]
            require(base.crs.is_projected and base.res == (30., 30.), "Local pixel CRS/grid")
            require(all((s.crs, s.transform, s.shape) == (base.crs, base.transform, base.shape)
                        for s in handles.values()), "Ancillary grid mismatch")
            for fraction in [(0.35, 0.35), (0.35, 0.65), (0.5, 0.5), (0.65, 0.35), (0.65, 0.65)]:
                window = Window(int(base.width*fraction[0]), int(base.height*fraction[1]), 128, 128)
                arrays = {k: s.read(1, window=window) for k, s in handles.items()}
                t = arrays["ST_B10"]
                clear = (t > 0) & ((arrays["QA_PIXEL"] & 63) == 0) & ((arrays["QA_RADSAT"] & (1 << 11)) == 0)
                e = arrays["ST_EMIS"]
                valid_e = clear & (e != handles["ST_EMIS"].nodata) & (e > 0)
                vals = e[valid_e].astype(float)*.0001
                temps = t[valid_e].astype(float)*.00341802+149.-273.15
                row = {"item_id": directory.name, "crs": str(base.crs), "output_grid_m": 30,
                       "thermal_native_m": 100, "col_off": int(window.col_off), "row_off": int(window.row_off),
                       "window_pixels": int(t.size), "clear_valid_temperature_pixels": int(clear.sum()),
                       "paired_emissivity_pixels": int(valid_e.sum()),
                       "invalid_emissivity_on_valid_temperature": int((clear & ~valid_e).sum()),
                       "emissivity_min": float(vals.min()) if len(vals) else None,
                       "emissivity_max": float(vals.max()) if len(vals) else None,
                       "emissivity_mean": float(vals.mean()) if len(vals) else None,
                       "temperature_mean_c": float(temps.mean()) if len(vals) else None,
                       "outside_physical_emissivity_range": int(((vals <= 0) | (vals > 1)).sum()),
                       "zero_emissivity_sd_pixels_on_clear": int((clear & (arrays["ST_EMSD"] == 0)).sum()),
                       "independent_vegetation_reference_available": False,
                       "anomaly_truth_status": "NOT_IDENTIFIABLE_FROM_EMISSIVITY_AND_TEMPERATURE_ALONE"}
                for key, array in arrays.items():
                    row[f"{key}_window_sha256"] = hashlib.sha256(array.tobytes()).hexdigest()
                rows.append(row)
        finally:
            for handle in handles.values():
                handle.close()
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "local_native_pixel_checks.csv", index=False)
    return frame


def execute(out):
    require(not out.exists(), "Refuse existing output directory")
    out.mkdir()
    started = time.monotonic()
    sources = {str(p): digest(p) for p in [Path(__file__), SAMPLE, MANIFEST]}
    official = []
    session = requests.Session()
    session.trust_env = False
    for label, url in URLS.items():
        response = session.get(url, timeout=(15, 60))
        response.raise_for_status()
        require("emissiv" in response.text.lower(), "Unexpected official webpage")
        p = out / f"official_{label}.html"
        p.write_bytes(response.content)
        official.append({"url": url, "status_code": response.status_code,
                        "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "saved_file": p.name, "sha256_file_bytes": digest(p)})
    require("Vegetation Adjustment Anomaly" in (out / "official_known_issues.html").read_text(encoding="utf-8"),
            "Official anomaly statement not found")
    sample = read(SAMPLE)
    require(len(sample) == 104371 and not sample.duplicated(KEY).any(), "Frozen sample identity")
    targets = select_points(sample)
    targets.to_csv(out / "target_selection_before_measurement.csv", index=False)
    manifest = read(MANIFEST)
    require(len(manifest) == 147 and not manifest.duplicated(["state", "year"]).any(), "Daily task identity")
    parts, point_parts, task_rows = [], [], []
    for i, row in enumerate(manifest.itertuples(index=False)):
        p = Path(row.daily_path)
        h = digest(p)
        require(h == row.daily_sha256_file_bytes, f"Released daily data changed: {p}")
        sources[str(p)] = h
        f = read(p, usecols=DAILY)
        require(len(f) == row.rows and not f.duplicated(["state", "tract_id", "year", "observation_date"]).any(), "Daily accounting")
        valid = f.temperature_c_mean_valid.notna() & f.valid_temperature_area_m2.gt(0)
        v = f.loc[valid].copy()
        require(np.isfinite(v.temperature_c_mean_valid).all(), "Nonfinite valid temperature")
        v["emis_missing"] = v.st_emis_mean.isna()
        v["emis_outside_range"] = v.st_emis_mean.notna() & ~v.st_emis_mean.between(0, 1, inclusive="right")
        v["emis_rounded"] = v.st_emis_mean.round(4)
        g = v.groupby(["state", "tract_id", "year"], sort=True).agg(
            valid_dates=("observation_date", "size"), mean_emissivity=("st_emis_mean", "mean"),
            min_emissivity=("st_emis_mean", "min"), max_emissivity=("st_emis_mean", "max"),
            sd_emissivity=("st_emis_mean", "std"), unique_rounded_emissivity=("emis_rounded", "nunique"),
            missing_emissivity_dates=("emis_missing", "sum"), outside_range_dates=("emis_outside_range", "sum"),
            raw_temperature_mean_c=("temperature_c_mean_valid", "mean"),
            reported_temperature_uncertainty_mean_c=("st_qa_mean_c", "mean"),
            reported_emissivity_sd_mean=("st_emsd_mean", "mean")).reset_index()
        g = g.rename(columns={"year": "outcome_year"})
        parts.append(g)
        ids = set(targets.loc[targets.state.eq(str(row.state)), "tract_id"])
        point_parts.append(f.loc[f.tract_id.isin(ids)])
        task_rows.append({"state": row.state, "year": row.year, "input_rows": len(f), "valid_rows": int(valid.sum()),
                          "missing_emissivity_on_valid_rows": int(v.emis_missing.sum()),
                          "outside_range_on_valid_rows": int(v.emis_outside_range.sum())})
        if (i+1) % 20 == 0:
            print(f"EMISSIVITY_DAILY_AUDIT {i+1}/147", flush=True)
    all_annual = pd.concat(parts, ignore_index=True)
    main = sample[KEY + ["treatment_pp_change_from_2000", "baseline_canopy_percent_fixed_land_2000",
                         "income_point_usd_2000", "poverty_share_2000", "age65plus_share_2000"]].merge(
        all_annual, on=KEY, how="left", validate="one_to_one", indicator=True, sort=False)
    require(len(main) == len(sample) and main[KEY].equals(sample[KEY]) and main._merge.eq("both").all(), "Measurement sample join")
    require(main.missing_emissivity_dates.eq(0).all() and main.outside_range_dates.eq(0).all(), "Invalid emissivity in frozen model sample")
    main.drop(columns="_merge").to_csv(out / "main_sample_emissivity_diagnostic.csv", index=False)
    pd.concat(point_parts, ignore_index=True).to_csv(out / "target_tract_daily_measurement.csv", index=False)
    pd.DataFrame(task_rows).to_csv(out / "daily_task_accounting.csv", index=False)
    correlation = main[["treatment_pp_change_from_2000", "baseline_canopy_percent_fixed_land_2000",
                        "income_point_usd_2000", "poverty_share_2000", "age65plus_share_2000",
                        "mean_emissivity", "sd_emissivity", "raw_temperature_mean_c"]].corr(method="spearman")
    correlation.to_csv(out / "measurement_rank_associations_DIAGNOSTIC.csv")
    # Reprocessing information is catalog evidence, not an assertion of anomaly correction.
    records = []
    for p in sorted(CATALOG.glob("*_page*.json.gz")):
        sources[str(p)] = digest(p)
        with gzip.open(p, "rt", encoding="utf-8") as stream:
            for item in json.load(stream)["features"]:
                prop = item["properties"]
                records.append({"item_id": item["id"], "catalog_created": prop.get("created"),
                                "collection_number": prop.get("landsat:collection_number"),
                                "temperature_asset": item["assets"]["lwir11"]["href"],
                                "catalog_page": p.name})
    inventory = pd.DataFrame(records)
    require(not inventory.item_id.duplicated().any(), "Duplicate catalog scenes")
    inventory.to_csv(out / "product_processing_inventory.csv", index=False)
    local_hashes = {}
    pixels = pixel_audit(out, local_hashes)
    require(all(digest(p) == h for p,h in sources.items()), "Inputs changed during audit")
    require(all(Path(p).stat().st_size == h["bytes"] and Path(p).stat().st_mtime_ns == h["mtime_ns"]
                for p,h in local_hashes.items()), "Local image changed during audit")
    audit = {"status": "COMPLETED_TARGETED_MEASUREMENT_APPLICABILITY_ADJUDICATION",
             "execution_class": "DIAGNOSTIC_ONLY_NO_MEASUREMENT_REPLACEMENT",
             "main_rows": len(main), "main_tracts": int(main.tract_id.nunique()), "tasks": len(manifest),
             "daily_rows": int(sum(v["input_rows"] for v in task_rows)),
             "valid_daily_rows": int(sum(v["valid_rows"] for v in task_rows)),
             "main_invalid_emissivity_dates": int(main.missing_emissivity_dates.sum()+main.outside_range_dates.sum()),
             "target_selection_rows": len(targets), "target_unique_tracts": int(targets.tract_id.nunique()),
             "local_scene_count": int(pixels.item_id.nunique()), "local_windows": len(pixels),
             "local_paired_pixels": int(pixels.paired_emissivity_pixels.sum()),
             "local_invalid_emissivity_on_valid_temperature": int(pixels.invalid_emissivity_on_valid_temperature.sum()),
             "independent_aster_reference_vegetation_available": False,
             "independent_temperature_truth_available": False,
             "canopy_change_equals_vegetation_fraction_change": False,
             "product_anomaly_ruled_out": False,
             "measurement_applicability": "CONDITIONAL_DESCRIPTIVE_AND_PROVISIONAL_EFFECT_ANALYSIS",
             "formal_causal_equity_release": False,
             "reason": "Official vegetation-emissivity anomaly remains relevant; quality flags and physical emissivity range do not identify or remove it.",
             "official_sources": official, "input_file_sha256": sources, "local_image_stat_and_window_basis": local_hashes,
             "new_model_fits": 0, "raw_or_formal_inputs_modified": False, "elapsed_seconds": time.monotonic()-started}
    audit["output_file_sha256"] = {p.name: digest(p) for p in out.iterdir() if p.is_file()}
    write_json(out / "audit.json", audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ["input_file_sha256", "output_file_sha256", "local_image_stat_and_window_basis"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    execute(p.parse_args().out)
