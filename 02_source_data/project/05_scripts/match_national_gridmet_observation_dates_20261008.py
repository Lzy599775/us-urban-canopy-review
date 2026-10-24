"""Match immutable warm-season observation support to fixed-land weather.

Descriptive measurement diagnostics only. Raw files, temperature outcomes,
qualification rules, and project authority are never changed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pyproj import Transformer
from scipy import sparse
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "06_logs"
SUPPORT = LOGS / "lst_observation_opportunity_20261008_run03"
ANNUAL = LOGS / "lst_annual_outcome_analysis_ready_unqualified_20261008_run04"
CATALOG = LOGS / "national_urban_weights_summary_20260914_run01/national_weight_catalog.csv"
RAW = ROOT / "04_pilot/gridMET/raw"
YEARS = (2014, 2019, 2024)
VARIABLES = {
    "tmmx_c": ("tmmx", "air_temperature", "K", -273.15),
    "tmmn_c": ("tmmn", "air_temperature", "K", -273.15),
    "pr_mm": ("pr", "precipitation_amount", "mm", 0.0),
    "vpd_kpa": ("vpd", "mean_vapor_pressure_deficit", "kPa", 0.0),
}
STAGES = ("geometric", "product", "valid")
RULE = "frozen_30m_land_area_weights_by_grid_center_to_weather_cell_v1"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_csv(path, **kwargs):
    return pd.read_csv(path, dtype={"state": str, "tract_id": str, "county_code": str},
                       float_precision="round_trip", **kwargs)


def say(message):
    print(f"{dt.datetime.now().isoformat(timespec='seconds')} {message}", flush=True)


def scalar(attrs, name):
    return np.asarray(attrs[name]).item()


def text_attr(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def cell_codes(longitude, latitude, lon, lat):
    """Half-open cells; outside the product domain gets an explicit sentinel."""
    dx, dy = lon[1] - lon[0], lat[0] - lat[1]
    require(dx > 0 and dy > 0, "Expected eastward longitude and southward latitude")
    require(np.allclose(np.diff(lon), dx, rtol=0, atol=1e-10), "Irregular longitude grid")
    require(np.allclose(np.diff(lat), -dy, rtol=0, atol=1e-10), "Irregular latitude grid")
    require(np.isfinite(longitude).all() and np.isfinite(latitude).all(), "Invalid transformed coordinate")
    col = np.floor((longitude - (lon[0] - dx / 2)) / dx).astype(np.int64)
    row = np.floor(((lat[0] + dy / 2) - latitude) / dy).astype(np.int64)
    inside = (col >= 0) & (col < len(lon)) & (row >= 0) & (row < len(lat))
    return np.where(inside, row * len(lon) + col, len(lat) * len(lon))


def decode(dataset, start, stop):
    packed = dataset[start:stop]
    fill = scalar(dataset.attrs, "_FillValue")
    valid = packed != fill
    if "missing_value" in dataset.attrs:
        valid &= packed != scalar(dataset.attrs, "missing_value")
    if text_attr(dataset.attrs.get("_Unsigned", "false")).lower() == "true" and packed.dtype.kind == "i":
        packed = packed.view(np.dtype(f"u{packed.dtype.itemsize}"))
    values = packed.astype(np.float64) * float(scalar(dataset.attrs, "scale_factor"))
    values += float(scalar(dataset.attrs, "add_offset"))
    return values, valid


def warm_dates(year):
    return np.arange(np.datetime64(f"{year}-05-01"), np.datetime64(f"{year}-10-01"))


def preflight(out):
    support_audit = load_json(SUPPORT / "audit.json")
    require(support_audit["status"] ==
            "PASS_LST_OBSERVATION_OPPORTUNITY_CORRECTED_AFTER_RECOVERY_NOT_FORMAL_OUTCOME", "Support acceptance")
    require(load_json(LOGS / "lst_annual_support_independent_validation_20261008_run02/audit.json")["status"] ==
            "PASS_FULL_INDEPENDENT_ANNUAL_RECONSTRUCTION", "Independent temperature acceptance")
    annual = read_csv(ANNUAL / "national_tract_year_lst_outcome_analysis_ready_unqualified.csv")
    receipt = load_json(ANNUAL / "audit.json")
    require(digest(ANNUAL / "national_tract_year_lst_outcome_analysis_ready_unqualified.csv") ==
            receipt["output_sha256_file_bytes"], "Annual input fingerprint")
    require(len(annual) == 140181 and not annual.duplicated(["tract_id", "year"]).any(), "Annual keys")
    catalog = read_csv(CATALOG)
    require(len(catalog) == 49 and not catalog.state.duplicated().any(), "49 state/DC spatial tasks expected")
    previous = load_json(SUPPORT / "input_files.json")
    inputs = {}

    def track(path, expected=None):
        path = Path(path).resolve()
        key = str(path)
        if key not in inputs:
            actual = digest(path)
            require(expected is None or actual == expected, f"Changed source: {path}")
            inputs[key] = {"bytes": path.stat().st_size, "sha256_file_bytes": actual}

    track(CATALOG, previous[str(CATALOG)]["sha256_file_bytes"])
    track(ANNUAL / "national_tract_year_lst_outcome_analysis_ready_unqualified.csv", receipt["output_sha256_file_bytes"])
    weather_prior = pd.read_csv(LOGS / "gridmet_internal_20260910_run01/file_summary.csv").set_index("filename")
    lon, lat = None, None
    weather_metadata = []
    for year in YEARS:
        for label, (code, name, unit, offset) in VARIABLES.items():
            path = RAW / f"{code}_{year}.nc"
            track(path, weather_prior.loc[path.name, "sha256"])
            with h5py.File(path, "r") as f:
                x, y = f["lon"][:], f["lat"][:]
                if lon is None:
                    lon, lat = x, y
                require(np.array_equal(x, lon) and np.array_equal(y, lat), "Weather grid mismatch")
                ds = f[name]
                require(ds.shape[1:] == (len(lat), len(lon)), "Weather dimension order")
                require(text_attr(ds.attrs["units"]) == unit, f"Unexpected unit: {path}")
                require(text_attr(f["day"].attrs["units"]) == "days since 1900-01-01 00:00:00", "Time origin")
                require(text_attr(f["day"].attrs["calendar"]) == "gregorian", "Calendar")
                dates = np.datetime64("1900-01-01") + f["day"][:].astype("timedelta64[D]")
                expected = np.arange(np.datetime64(f"{year}-01-01"), np.datetime64(f"{year+1}-01-01"))
                require(np.array_equal(dates, expected), "Missing, duplicate, or shifted weather date")
                weather_metadata.append({"file": str(path), "shape": list(ds.shape), "chunks": list(ds.chunks),
                    "variable": label, "unit": unit, "scale": float(scalar(ds.attrs, "scale_factor")),
                    "offset": float(scalar(ds.attrs, "add_offset")),
                    "day_definition": text_attr(f.attrs["note5"]),
                    "release_slice_metadata": {k: text_attr(v) for k, v in f.attrs.items() if "slice" in k}})
    daily_hashes = {Path(r["daily_output"]).name: r["daily_output_sha256"] for r in support_audit["task_audits"]}
    for task in catalog.to_dict("records"):
        state = task["state"]
        paths = [Path(task["grid_path"])]
        paths += sorted(Path(task["weights_dir"]).glob(f"urban_weights_{state}_*.npz"))
        paths += sorted(Path(task["weights_dir"]).glob(f"urban_area_{state}_*.csv"))
        require(len(paths) == 1 + 2 * int(task["matrix_count"]), f"Spatial shard count {state}")
        for path in paths:
            track(path, previous[str(path)]["sha256_file_bytes"])
        for year in YEARS:
            path = SUPPORT / "daily" / f"state_{state}_{year}_tract_date_observation_support.csv.gz"
            track(path, daily_hashes[path.name])
    write_json(out / "input_files.json", inputs)
    write_json(out / "weather_metadata.json", weather_metadata)
    return annual, catalog, lon, lat


def build_weights(out, annual, catalog, lon, lat):
    global_ids, global_fixed, matrices, receipts = [], [], [], []
    ncell = len(lon) * len(lat)
    for task in catalog.to_dict("records"):
        state = task["state"]
        grid = load_json(task["grid_path"])
        grid_ids = grid["tract_order"]
        transform = grid["transform"][:6]
        require(transform[1] == 0 and transform[3] == 0 and transform[0] == 30 and transform[4] == -30,
                "Unexpected frozen 30m grid")
        height, width = grid["shape"]
        converter = Transformer.from_crs(grid["crs_wkt"], "EPSG:4326", always_xy=True)
        expected = annual[(annual.state == state) & (annual.year == YEARS[0])].set_index("tract_id")
        pieces, ids, fixed_parts = [], [], []
        cursor = 0
        for path in sorted(Path(task["weights_dir"]).glob(f"urban_weights_{state}_*.npz")):
            table = read_csv(path.with_name(path.name.replace("urban_weights_", "urban_area_").replace(".npz", ".csv")))
            require(table.tract_id.tolist() == grid_ids[cursor:cursor + len(table)], "Frozen shard row order")
            cursor += len(table)
            original = sparse.load_npz(path).tocsr()
            require(original.shape == (len(table), height * width), "Frozen shard shape")
            keep = table.tract_id.isin(expected.index).to_numpy()
            if not keep.any():
                continue
            matrix = original[keep]
            require(np.isfinite(matrix.data).all() and (matrix.data > 0).all(), "Invalid frozen area weights")
            fixed = np.asarray(matrix.sum(axis=1)).ravel()
            shard_ids = table.tract_id[keep].tolist()
            require(np.allclose(fixed, expected.loc[shard_ids, "urban_fixed_land_m2"], rtol=1e-10, atol=.01),
                    "Frozen fixed-land area mismatch")
            # Centers assign the already frozen intersection areas; no new land mask.
            active, inverse = np.unique(matrix.indices, return_inverse=True)
            rr, cc = np.divmod(active, width)
            xx = transform[0] * (cc + .5) + transform[2]
            yy = transform[4] * (rr + .5) + transform[5]
            longitude, latitude = converter.transform(xx, yy)
            codes = cell_codes(longitude, latitude, lon, lat)
            row = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
            coarse = sparse.coo_matrix((matrix.data, (row, codes[inverse])), shape=(matrix.shape[0], ncell + 1)).tocsr()
            closure = np.asarray(coarse.sum(axis=1)).ravel() - fixed
            require(np.allclose(closure, 0, rtol=0, atol=.01), "Weather-weight area closure")
            pieces.append(coarse)
            ids.extend(shard_ids)
            fixed_parts.append(fixed)
        require(cursor == len(grid_ids) and ids == sorted(expected.index), f"State tract accounting {state}")
        stacked = sparse.vstack(pieces, format="csr")
        fixed = np.concatenate(fixed_parts)
        sparse.save_npz(out / "weights" / f"state_{state}_fixed_land_weather_weights.npz", stacked)
        pd.DataFrame({"state": state, "tract_id": ids, "fixed_land_area_m2": fixed,
            "outside_weather_grid_area_m2": stacked[:, ncell].toarray().ravel()}).to_csv(
                out / "weights" / f"state_{state}_weather_weight_rows.csv", index=False)
        receipts.append({"state": state, "tracts": len(ids), "nnz": stacked.nnz,
                         "outside_grid_area_m2": float(stacked[:, ncell].sum())})
        global_ids.extend(ids)
        global_fixed.extend(fixed)
        matrices.append(stacked)
        say(f"SPATIAL_WEIGHTS {len(receipts)}/49 state={state} tracts={len(ids)}")
    require(len(global_ids) == 46727 and len(set(global_ids)) == 46727, "National positive-land keys")
    weights = sparse.vstack(matrices, format="csr")
    selected = np.unique(weights.indices)
    weights = weights[:, selected].tocsr()
    write_json(out / "spatial_receipt.json", {"rule": RULE, "rows": len(global_ids), "states": receipts,
        "coarse_cells_including_outside_sentinel": len(selected),
        "boundary_approximation": "assign frozen 30m cell-intersection area at cell center; not exact weather-cell boundary clipping",
        "weather_grid_edges": "longitude [west,east), latitude (south,north]; outside-domain sentinel, never nearest-neighbor fill",
        "weather_resolution_interpretation": "approximately 4km background weather; never interpreted as 30m climate"})
    return weights, selected, np.array(global_ids), np.array(global_fixed), receipts


def read_weather(out, year, lon, lat, selected):
    dates = warm_dates(year)
    ncell = len(lon) * len(lat)
    data = np.zeros((len(dates), len(selected), len(VARIABLES)), dtype=np.float64)
    masks = np.zeros((len(selected), len(VARIABLES)), dtype=bool)
    metadata = []
    inside = selected < ncell
    for vi, (label, (code, name, unit, offset)) in enumerate(VARIABLES.items()):
        with h5py.File(RAW / f"{code}_{year}.nc", "r") as f:
            all_dates = np.datetime64("1900-01-01") + f["day"][:].astype("timedelta64[D]")
            first = int(np.flatnonzero(all_dates == dates[0])[0])
            stop = first + len(dates)
            ds = f[name]
            base_mask = None
            minimum, maximum = np.inf, -np.inf
            for start in range(first, stop, ds.chunks[0]):
                end = min(stop, start + ds.chunks[0])
                values, valid = decode(ds, start, end)
                if base_mask is None:
                    base_mask = valid[0].copy()
                require(np.array_equal(valid, np.broadcast_to(base_mask, valid.shape)), "Time-varying weather missingness")
                values += offset
                minimum = min(minimum, float(values[valid].min()))
                maximum = max(maximum, float(values[valid].max()))
                values[~valid] = 0.0
                data[start-first:end-first, inside, vi] = values.reshape(end-start, -1)[:, selected[inside]]
            masks[inside, vi] = base_mask.ravel()[selected[inside]]
            metadata.append({"variable": label, "decoded_min": minimum, "decoded_max": maximum,
                "selected_missing_cells": int((~masks[:, vi]).sum()), "time_varying_missingness": False})
    require(np.array_equal(masks, np.broadcast_to(masks[:, :1], masks.shape)), "Variable masks differ")
    inverted = int(((data[:, :, 0] < data[:, :, 1]) & masks[:, 0]).sum())
    require((data[:, :, 2][np.broadcast_to(masks[:, 2], data[:, :, 2].shape)] >= 0).all(), "Negative precipitation")
    require((data[:, :, 3][np.broadcast_to(masks[:, 3], data[:, :, 3].shape)] >= 0).all(), "Negative vapor pressure deficit")
    write_json(out / f"weather_decode_{year}.json", {"year": year, "variables": metadata,
        "selected_cell_days_tmax_below_tmin_retained": inverted})
    say(f"WEATHER_DECODE year={year} cells={len(selected)}")
    return dates, data, masks


def masked_mean(values, mask, count):
    total = np.zeros(values.shape[1], dtype=np.float64)
    np.add.at(total, mask["tract_position"], values[mask["day_position"], mask["tract_position"]])
    return np.divide(total, count, out=np.full(len(count), np.nan), where=count > 0)


def state_match(out, state, year, ids, fixed, values, coverage, dates, annual_input):
    path = SUPPORT / "daily" / f"state_{state}_{year}_tract_date_observation_support.csv.gz"
    daily = read_csv(path)
    require(not daily.duplicated(["tract_id", "observation_date"]).any(), "Duplicate daily keys")
    daypos = pd.Index(dates.astype(str)).get_indexer(daily.observation_date)
    tractpos = pd.Index(ids).get_indexer(daily.tract_id)
    require((daypos >= 0).all() and (tractpos >= 0).all(), "Unmatched weather date/tract")
    require(daily.state.eq(state).all() and daily.year.eq(year).all(), "State/year identity")
    require(set(daily.tract_id) == set(ids), "Daily tract universe")
    require(np.allclose(daily.fixed_land_area_m2, fixed[tractpos], rtol=1e-10, atol=.01), "Daily fixed area mismatch")
    months = pd.DatetimeIndex(dates).month.to_numpy()
    rows = pd.DataFrame({"state": state, "tract_id": ids, "year": year,
                         "fixed_land_area_m2": fixed, "weather_valid_land_fraction": coverage})
    complete = np.isclose(coverage, 1.0, rtol=0, atol=1e-10)
    rows["weather_spatial_support_status"] = np.where(complete, "FULL_FIXED_LAND_SUPPORT",
        np.where(coverage > 0, "PARTIAL_FIXED_LAND_SUPPORT", "NO_WEATHER_LAND_SUPPORT"))
    daily["weather_valid_land_fraction"] = coverage[tractpos]
    daily["weather_spatial_support_status"] = rows.weather_spatial_support_status.to_numpy()[tractpos]
    np.savez_compressed(out / "warm_season" / f"state_{state}_{year}_fixed_land_weather.npz",
                        tract_id=ids, date=dates.astype(str), variables=np.array(list(VARIABLES)),
                        values_mean_available_land=values, weather_valid_land_fraction=coverage,
                        fixed_land_area_m2=fixed)
    geo = daily.geometric_opportunity_area_m2.to_numpy() > 0
    product = daily.product_available_area_m2.to_numpy() > 0
    valid = daily.valid_temperature_area_m2.to_numpy() > 0
    require((~product | geo).all() and (~valid | product).all(), "Non-nested observation stages")
    require(np.array_equal(valid, daily.temperature_c_mean_valid.notna().to_numpy()), "Valid temperature state")
    masks = dict(zip(STAGES, (geo, product, valid)))
    counts = {stage: np.bincount(tractpos[mask], minlength=len(ids)) for stage, mask in masks.items()}
    source = annual_input.set_index("tract_id").loc[ids]
    for stage, field in zip(STAGES, ("geometric_opportunity_dates", "product_available_dates", "valid_temperature_dates")):
        require(np.array_equal(counts[stage], source[field]), "Opportunity denominator changed")
    for vi, label in enumerate(VARIABLES):
        weather = values[:, :, vi]
        daily[f"{label}_mean_available_land"] = weather[daypos, tractpos]
        rows[f"{label}_warm_season_mean_available_land"] = np.mean(weather, axis=0)
        anomaly = weather.copy()
        for month in range(5, 10):
            positions = months == month
            anomaly[positions] -= np.mean(weather[positions], axis=0)
        for stage, mask in masks.items():
            pos = {"day_position": daypos[mask], "tract_position": tractpos[mask]}
            rows[f"{label}_{stage}_dates_mean_available_land"] = masked_mean(weather, pos, counts[stage])
            rows[f"{label}_{stage}_dates_tract_month_anomaly"] = masked_mean(anomaly, pos, counts[stage])
        for left, right in (("geometric_dates", "warm_season"), ("product_dates", "geometric_dates"),
                            ("valid_dates", "product_dates"), ("valid_dates", "geometric_dates"),
                            ("valid_dates", "warm_season")):
            rows[f"{label}_{left}_minus_{right}"] = (
                rows[f"{label}_{left}_mean_available_land"] - rows[f"{label}_{right}_mean_available_land"])
        rows[f"{label}_valid_minus_geometric_within_tract_month"] = (
            rows[f"{label}_valid_dates_tract_month_anomaly"] - rows[f"{label}_geometric_dates_tract_month_anomaly"])
    daily.to_csv(out / "daily" / f"state_{state}_{year}_observation_weather.csv.gz", index=False,
                 compression={"method": "gzip", "mtime": 0})
    rows.to_csv(out / "annual_parts" / f"state_{state}_{year}_weather_sampling.csv", index=False)
    return rows, {"state": state, "year": year, "tracts": len(ids), "daily_rows": len(daily),
        "daily_weather_rows_all_variables_present": int(daily[[f"{v}_mean_available_land" for v in VARIABLES]].notna().all(axis=1).sum()),
        "warm_season_tract_days": len(ids) * len(dates), "unmatched_rows": 0,
        "duplicate_rows": 0, "date_counts_verified": True,
        "statuses": {str(k): int(v) for k, v in daily.support_status.value_counts().items()}}


def correlations(frame, out):
    features = ["baseline_canopy_percent_fixed_land_2000", "year_canopy_percent_fixed_land",
                "treatment_canopy_change_pp", "income_point_usd_2000", "poverty_share_2000",
                "age65plus_share_2000", "tmmx_c_warm_season_mean_available_land", "vpd_kpa_warm_season_mean_available_land"]
    outcomes = ["geometric_opportunity_dates", "valid_temperature_dates", "valid_date_fraction_geometric",
                "mean_valid_fraction_geometric", "weather_valid_land_fraction", "tmmx_c_valid_dates_minus_warm_season",
                "tmmx_c_valid_minus_geometric_within_tract_month"]
    records = []
    for year, group in frame.groupby("year", sort=True):
        for feature in features:
            for outcome in outcomes:
                pair = group[["state", feature, outcome]].dropna(subset=[feature, outcome])
                x, y = pair[feature].to_numpy(), pair[outcome].to_numpy()
                rho = float(spearmanr(x, y).statistic) if len(x) > 1 and np.ptp(x) > 0 and np.ptp(y) > 0 else np.nan
                centered = pair[[feature, outcome]] - pair.groupby("state")[[feature, outcome]].transform("mean")
                a, b = centered[feature].to_numpy(), centered[outcome].to_numpy()
                denominator = np.sqrt(np.dot(a, a) * np.dot(b, b))
                r = float(np.dot(a, b) / denominator) if denominator > 0 else np.nan
                records.append({"year": year, "feature": feature, "outcome": outcome,
                    "pairwise_complete_rows": len(pair), "missing_feature_or_outcome_rows": len(group) - len(pair),
                    "national_spearman_rho_descriptive": rho, "within_state_pearson_r_descriptive": r,
                    "interpretation": "descriptive association only; no causal or no-bias claim; pairwise missingness retained in main table"})
    pd.DataFrame(records).to_csv(out / "missingness_covariate_associations.csv", index=False)


def summaries(frame, out):
    records = []
    for (year, support), group in frame.groupby(["year", "weather_spatial_support_status"], sort=True):
        for label in VARIABLES:
            for contrast in ("geometric_dates_minus_warm_season", "valid_dates_minus_geometric_dates",
                             "valid_dates_minus_warm_season", "valid_minus_geometric_within_tract_month"):
                values = group[f"{label}_{contrast}"].dropna()
                row = {"year": year, "spatial_support": support, "variable": label,
                       "contrast": contrast, "rows": len(group), "nonmissing_rows": len(values)}
                for q in (.05, .25, .5, .75, .95):
                    row[f"p{int(q*100):02d}"] = float(values.quantile(q)) if len(values) else np.nan
                row["mean"] = float(values.mean()) if len(values) else np.nan
                records.append(row)
    pd.DataFrame(records).to_csv(out / "weather_sampling_contrast_summary.csv", index=False)
    # Continuous attributes remain primary; these five rank bands are diagnostic,
    # NOT frozen income/age effect-modification groups or formal sample choices.
    records = []
    for year, group in frame.groupby("year"):
        for feature in ("income_point_usd_2000", "poverty_share_2000", "age65plus_share_2000",
                        "baseline_canopy_percent_fixed_land_2000", "tmmx_c_warm_season_mean_available_land"):
            rank = group[feature].rank(method="average", pct=True)
            bands = np.ceil(rank * 5).clip(1, 5)
            for band in (1, 2, 3, 4, 5, None):
                selected = group.loc[bands.isna() if band is None else bands.eq(band)]
                rec = {"year": year, "feature": feature, "diagnostic_rank_band": "MISSING" if band is None else band,
                    "rows": len(selected), "feature_min": selected[feature].min(), "feature_max": selected[feature].max(),
                    "band_status": "DIAGNOSTIC_ONLY_NOT_FORMAL_GROUP"}
                for field in ("valid_temperature_dates", "valid_date_fraction_geometric", "mean_valid_fraction_geometric",
                              "tmmx_c_valid_dates_minus_warm_season", "tmmx_c_valid_minus_geometric_within_tract_month"):
                    rec[f"median_{field}"] = selected[field].median()
                records.append(rec)
    pd.DataFrame(records).to_csv(out / "missingness_diagnostic_rank_bands.csv", index=False)


def run(args):
    started = time.monotonic()
    out = args.out.resolve()
    require(not out.exists(), "Output exists; refusal to overwrite. Use a new run directory.")
    out.mkdir()
    for name in ("weights", "warm_season", "daily", "annual_parts"):
        (out / name).mkdir()
    write_json(out / "run_record.json", {"status": "RUNNING", "scope": "date-matched weather and descriptive missingness",
        "command": sys.argv, "script_sha256": digest(__file__), "python": sys.executable,
        "started_local": dt.datetime.now().isoformat(), "source_mutations": "NONE", "formal_model": "NOT_RUN",
        "estimated_runtime_minutes_provisional": [20, 60], "mode": "EMPIRICAL",
        "raw_temperature_changed": False, "qualification_changed": False})
    response = requests.get("https://www.climatologylab.org/gridmet.html", timeout=45)
    response.raise_for_status()
    (out / "official_gridmet_page.html").write_bytes(response.content)
    source_text = BeautifulSoup(response.text, "html.parser").get_text(" ", strip=True)
    require("Mountain Standard" in source_text and "4-km" in source_text, "Official product description")
    (out / "official_gridmet_page_text.txt").write_text(source_text, encoding="utf-8")
    write_json(out / "source_receipt.json", {"url": response.url, "http_status": response.status_code,
        "retrieved_local": dt.datetime.now().isoformat(), "sha256_file_bytes": digest(out / "official_gridmet_page.html"),
        "fact_boundary": "official product definition checked; remote individual file checksums not compared"})
    say("PREFLIGHT hashing accepted inputs and checking weather metadata")
    annual, catalog, lon, lat = preflight(out)
    say("PREFLIGHT PASS")
    weights, selected, ids, fixed, spatial_receipts = build_weights(out, annual, catalog, lon, lat)
    position = {state: np.flatnonzero(np.char.startswith(ids, state)) for state in catalog.state}
    chunks, receipts = [], []
    for year in YEARS:
        dates, weather, masks = read_weather(out, year, lon, lat, selected)
        valid_area = weights @ masks[:, 0].astype(float)
        coverage = valid_area / fixed
        require(np.all((coverage >= 0) & (coverage <= 1 + 1e-10)), "Invalid weather coverage fraction")
        for state in catalog.state:
            pos = position[state]
            sw = weights[pos]
            values = np.empty((len(dates), len(pos), len(VARIABLES)), dtype=np.float64)
            for vi in range(len(VARIABLES)):
                sums = (sw @ weather[:, :, vi].T).T
                values[:, :, vi] = np.divide(sums, valid_area[pos], out=np.full_like(sums, np.nan), where=valid_area[pos] > 0)
            part, receipt = state_match(out, state, year, ids[pos], fixed[pos], values, coverage[pos], dates,
                annual[(annual.state == state) & (annual.year == year)])
            chunks.append(part)
            receipts.append(receipt)
            write_json(out / "progress.json", {"tasks_completed": len(receipts), "tasks_expected": 147,
                "last_state": state, "last_year": year, "elapsed_seconds": time.monotonic() - started})
            say(f"DATE_MATCH {len(receipts)}/147 state={state} year={year} rows={receipt['daily_rows']}")
    weather_annual = pd.concat(chunks, ignore_index=True)
    require(len(weather_annual) == len(annual) and not weather_annual.duplicated(["tract_id", "year"]).any(), "Weather annual keys")
    merged = annual.merge(weather_annual.drop(columns=["fixed_land_area_m2"]), on=["state", "tract_id", "year"],
                          how="outer", validate="one_to_one", indicator=True)
    require(merged._merge.eq("both").all() and len(merged) == len(annual), "Annual merge accounting")
    merged = merged.drop(columns="_merge").sort_values(["state", "tract_id", "year"])
    for field in annual.columns:
        expected = annual.sort_values(["state", "tract_id", "year"])[field].reset_index(drop=True)
        pd.testing.assert_series_equal(merged[field].reset_index(drop=True), expected, check_names=False)
    require(sum(r["daily_rows"] for r in receipts) == 7106498, "National daily rows")
    # Existing status columns are deliberately preserved; new columns describe this task.
    merged["weather_date_matching_status"] = "MATCHED_DIAGNOSTIC_NOT_TEMPERATURE_ADJUSTED"
    merged["weather_missingness_diagnostic_status"] = "DESCRIPTIVE_DIAGNOSTICS_COMPLETED_NOT_FORMAL_QUALIFICATION"
    final = out / "national_tract_year_lst_weather_unqualified.csv"
    merged.to_csv(final, index=False)
    correlations(merged, out)
    summaries(merged, out)
    status_counts = merged.weather_spatial_support_status.value_counts().to_dict()
    audit = {"status": "PASS_DATE_MATCHED_WEATHER_AND_DESCRIPTIVE_MISSINGNESS_NOT_FORMAL_OUTCOME",
        "tasks": len(receipts), "states_including_dc": len(catalog), "years": list(YEARS),
        "tracts": len(ids), "annual_rows": len(merged), "daily_rows": sum(r["daily_rows"] for r in receipts),
        "warm_season_tract_days": sum(r["warm_season_tract_days"] for r in receipts),
        "daily_weather_all_variables_present": sum(r["daily_weather_rows_all_variables_present"] for r in receipts),
        "annual_weather_spatial_support_counts": {str(k): int(v) for k, v in status_counts.items()},
        "unmatched_rows": 0, "duplicate_annual_rows": 0, "dropped_rows": 0,
        "all_existing_annual_columns_unchanged": True, "raw_data_modified": False,
        "temperature_adjusted": False, "formal_sample_selected": False, "formal_model": "NOT_RUN",
        "income_age_formal_groups_assigned": False, "weather_boundary_rule": RULE,
        "weather_day_definition": "approximately midnight-to-midnight Mountain Standard Time (UTC-7); not satellite overpass temperature",
        "daily_weather_units": {"tmmx_c": "degrees_Celsius", "tmmn_c": "degrees_Celsius", "pr_mm": "millimetres", "vpd_kpa": "kilopascals"},
        "missingness_inference": "descriptive selection diagnostics; correlation neither proves ignorability nor identifies its correction",
        "source": "https://www.climatologylab.org/gridmet.html",
        "elapsed_seconds": time.monotonic() - started, "annual_output_sha256_file_bytes": digest(final),
        "task_receipts": receipts}
    write_json(out / "audit.json", audit)
    record = load_json(out / "run_record.json")
    record.update(status="COMPLETED", finished_local=dt.datetime.now().isoformat(), elapsed_seconds=time.monotonic()-started)
    write_json(out / "run_record.json", record)
    outputs = {str(path.relative_to(out)): {"bytes": path.stat().st_size, "sha256_file_bytes": digest(path)}
               for path in sorted(out.rglob("*")) if path.is_file() and path.name != "output_files.json"}
    write_json(out / "output_files.json", outputs)
    say(f"PASS tasks=147 annual_rows={len(merged)} seconds={time.monotonic()-started:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.exists() and not (args.out / "audit.json").exists():
            (args.out / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
