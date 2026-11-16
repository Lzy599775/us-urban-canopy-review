"""Compare same-version Science and NLCD TCC on frozen 2000 urban fixed-land weights.

Inputs are read only. Pixel weights and baseline groups are the existing study
objects; no resampling, imputation, model fitting, or group redefinition occurs.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import rasterio
from scipy.sparse import load_npz


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "05_scripts"))
from aggregate_national_canopy_state import row_sum, source_window  # noqa: E402

CATALOG = ROOT / "06_logs/national_urban_weights_summary_20260914_run01/national_weight_catalog.csv"
FROZEN = ROOT / "06_logs/canopy_cooling_equity_20261009_run01/annual_canopy_equity_row_manifest.csv.gz"
SOURCES = {
    ("nlcd", 2000): ROOT / "04_pilot/tcc_annual/nlcd_tcc_conus_2000_v2025-6_wgs84.zip",
    ("nlcd", 2025): ROOT / "04_pilot/tcc_annual/nlcd_tcc_conus_2025_v2025-6_wgs84.zip",
    ("science", 2000): OUT / "inputs/science_tcc_conus_2000_v2025-6_wgs84.zip",
    ("science", 2025): ROOT / "04_pilot/science_tcc_conus_2025_v2025-6_wgs84.zip",
}
PRODUCTS = ("nlcd", "science")
SOCIAL = ("income", "poverty", "age65plus")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_uri(path: Path) -> str:
    require(path.is_file(), f"Product archive missing: {path}")
    with ZipFile(path) as z:
        members = [n for n in z.namelist() if n.lower().endswith(".tif")]
    require(len(members) == 1, f"Expected one TIF in {path}")
    return f"/vsizip/{path.as_posix()}/{members[0]}"


def load_study_ids(years: tuple[int, ...]) -> tuple[pd.DataFrame, dict]:
    cols = ["state", "tract_id", "year", "balanced_26years", "single_city_id",
            *(f"{v}_city_quartile" for v in SOCIAL), "urban_canopy_percent_fixed_land"]
    m = pd.read_csv(FROZEN, usecols=cols, low_memory=False)
    base = m.loc[m.year.eq(2000) & m.balanced_26years].drop(columns="year")
    require(len(base) == 46424 and not base.duplicated(["state", "tract_id"]).any(), "Study baseline differs")
    frozen = m.loc[m.year.isin(years) & m.balanced_26years,
                   ["state", "tract_id", "year", "urban_canopy_percent_fixed_land"]]
    require(len(frozen) == 46424 * len(years), "Frozen endpoint coverage differs")
    lookup = {(int(r.state), int(r.tract_id), int(r.year)): float(r.urban_canopy_percent_fixed_land)
              for r in frozen.itertuples()}
    return base, lookup


def load_state_weights(catalog_row: dict[str, str]):
    audit = json.loads(Path(catalog_row["audit_path"]).read_text(encoding="utf-8"))
    grid = json.loads(Path(catalog_row["grid_path"]).read_text(encoding="utf-8"))
    require(audit["status"] in {"PASS_DIAGNOSTIC_STRUCTURE", "PASS_DIAGNOSTIC_STRUCTURE_AND_REGRESSION"},
            f"Weight gate not passed for state {catalog_row['state']}")
    d = Path(catalog_row["weights_dir"])
    matrices = sorted(d.glob(f"urban_weights_{catalog_row['state']}_*.npz"))
    tables = {p.name.replace("urban_area_", "").replace(".csv", ""): p
              for p in d.glob(f"urban_area_{catalog_row['state']}_*.csv")}
    require(len(matrices) == len(tables) == int(catalog_row["matrix_count"]), "Weight shard inventory differs")
    shards = []
    ids = []
    for p in matrices:
        suffix = p.stem.replace("urban_weights_", "")
        require(suffix in tables, f"Missing area table: {suffix}")
        with tables[suffix].open(encoding="utf-8-sig", newline="") as f:
            table = list(csv.DictReader(f))
        matrix = load_npz(p).tocsr()
        require(matrix.has_sorted_indices and matrix.shape == (len(table), int(np.prod(grid["shape"]))),
                f"Invalid weight matrix: {p}")
        expected_ids = grid["tract_order"][len(ids):len(ids) + len(table)]
        require([row["tract_id"] for row in table] == expected_ids, "Weight and tract order differ")
        fixed = np.asarray(matrix.sum(axis=1)).ravel()
        area = np.array([float(row["urban_fixed_land_m2"]) for row in table])
        require(np.allclose(fixed, area, atol=.01, rtol=1e-10), "Fixed-land weight sums differ")
        ids.extend(expected_ids)
        shards.append((matrix, table, fixed, suffix))
    require(len(ids) == len(set(ids)) == int(audit["rows"]), "Weight tract count differs")
    return grid, shards


def load_state_pixels(grid: dict, shards: list, uris: dict[tuple[str, int], str]) -> dict:
    pixels = {}
    shape = tuple(grid["shape"])
    for key, uri in uris.items():
        with rasterio.open(uri) as source:
            require(source.count == 1 and source.dtypes == ("uint8",), f"Unexpected TCC encoding: {key}")
            require(source.res == (30., 30.) and source.nodata == 255, f"Unexpected TCC resolution/nodata: {key}")
            require(source.crs == rasterio.crs.CRS.from_wkt(grid["crs_wkt"]), f"TCC CRS differs: {key}")
            window = source_window(source, grid)
            values = source.read(1, window=window)
        require(values.shape == shape, f"TCC state window shape differs: {key}")
        flat = values.ravel()
        pixels[key] = [flat[matrix.indices].copy() for matrix, *_ in shards]
        del flat, values
        gc.collect()
    return pixels


def aggregate_state(catalog_row: dict[str, str], base: pd.DataFrame, lookup: dict,
                    uris: dict[tuple[str, int], str], years: tuple[int, ...]) -> tuple[list[dict], dict]:
    state = int(catalog_row["state"])
    grid, shards = load_state_weights(catalog_row)
    pixels = load_state_pixels(grid, shards, uris)
    state_base = base.loc[base.state.eq(state)].set_index("tract_id")
    rows = []
    max_nlcd_diff = 0.
    n_source_tracts = n_balanced = n_complete_all = n_any_common = 0
    for index, (matrix, table, fixed, suffix) in enumerate(shards):
        p = {key: values[index] for key, values in pixels.items()}
        n_source_tracts += len(table)
        for key, x in p.items():
            require(np.all((x <= 100) | (x == 254) | (x == 255)), f"Unexpected TCC code: {state}/{suffix}/{key}")
        by_product = {}
        for key, x in p.items():
            valid = x <= 100
            area = row_sum(matrix, valid.astype("float64"))
            numerator = row_sum(matrix, np.where(valid, x, 0).astype("float64"))
            percent = np.divide(numerator, area, out=np.full(len(area), np.nan), where=area > 0)
            by_product[key] = {"valid": valid, "area": area, "percent": percent}
        by_year_common = {}
        for year in years:
            common = by_product[("science", year)]["valid"] & by_product[("nlcd", year)]["valid"]
            area = row_sum(matrix, common.astype("float64"))
            by_year_common[year] = {
                "area": area,
                **{prod: np.divide(row_sum(matrix, np.where(common, p[(prod, year)], 0).astype("float64")),
                                   area, out=np.full(len(area), np.nan), where=area > 0)
                   for prod in PRODUCTS},
            }
        fourway = None
        if len(years) == 2:
            valid4 = np.logical_and.reduce([by_product[(prod, year)]["valid"]
                                           for prod in PRODUCTS for year in years])
            area4 = row_sum(matrix, valid4.astype("float64"))
            fourway = {
                "area": area4,
                **{f"{prod}_{year}": np.divide(
                    row_sum(matrix, np.where(valid4, p[(prod, year)], 0).astype("float64")),
                    area4, out=np.full(len(area4), np.nan), where=area4 > 0)
                   for prod in PRODUCTS for year in years},
            }
        for j, item in enumerate(table):
            tract = int(item["tract_id"])
            if tract not in state_base.index:
                continue
            n_balanced += 1
            known = state_base.loc[tract]
            land = float(fixed[j])
            require(land > 0 and float(item["urban_landcover_nodata_m2"]) == 0,
                    f"Balanced tract fixed land invalid: {state}/{tract}")
            row = {"state": state, "tract_id": tract, "single_city_id": known.single_city_id,
                   **{f"{v}_city_quartile": known[f"{v}_city_quartile"] for v in SOCIAL},
                   "fixed_land_m2": land, "weight_shard": suffix}
            for year in years:
                for prod in PRODUCTS:
                    result = by_product[(prod, year)]
                    area = float(result["area"][j])
                    complete = abs(area - land) <= max(.01, land * 1e-10)
                    row[f"{prod}_{year}_valid_fraction"] = area / land
                    row[f"{prod}_{year}_full_cover"] = float(result["percent"][j]) if complete else np.nan
                    row[f"{prod}_{year}_common_cover"] = float(by_year_common[year][prod][j])
                row[f"common_{year}_valid_fraction"] = float(by_year_common[year]["area"][j] / land)
                frozen_value = lookup[(state, tract, year)]
                measured = row[f"nlcd_{year}_full_cover"]
                require(np.isfinite(measured), f"Frozen NLCD full cover missing: {state}/{tract}/{year}")
                max_nlcd_diff = max(max_nlcd_diff, abs(measured - frozen_value))
                require(abs(measured - frozen_value) <= 1e-9,
                        f"NLCD raster/weight aggregation differs from frozen panel: {state}/{tract}/{year}")
            if fourway is not None:
                area = float(fourway["area"][j])
                row["common_4way_valid_fraction"] = area / land
                n_any_common += int(area > 0)
                n_complete_all += int(abs(area - land) <= max(.01, land * 1e-10))
                for prod in PRODUCTS:
                    for year in years:
                        row[f"{prod}_{year}_fourway_cover"] = float(fourway[f"{prod}_{year}"][j])
                row["nlcd_change_4way_pp"] = row["nlcd_2025_fourway_cover"] - row["nlcd_2000_fourway_cover"]
                row["science_change_4way_pp"] = row["science_2025_fourway_cover"] - row["science_2000_fourway_cover"]
                row["processing_change_difference_pp"] = row["nlcd_change_4way_pp"] - row["science_change_4way_pp"]
                row["nlcd_change_full_pp"] = row["nlcd_2025_full_cover"] - row["nlcd_2000_full_cover"]
                row["science_change_full_pp"] = row["science_2025_full_cover"] - row["science_2000_full_cover"]
                row["processing_change_full_pp"] = row["nlcd_change_full_pp"] - row["science_change_full_pp"]
            rows.append(row)
    require(n_balanced == len(state_base) and n_source_tracts == len(grid["tract_order"]),
            f"State tract accounting differs: {state}")
    report = {"state": state, "source_tracts": n_source_tracts, "balanced_tracts": n_balanced,
              "max_frozen_nlcd_difference_pp": max_nlcd_diff,
              "complete_fourway_tracts": n_complete_all if fourway is not None else None,
              "positive_common_fourway_tracts": n_any_common if fourway is not None else None}
    return rows, report


def summarize(rows: pd.DataFrame, output: Path) -> dict:
    require(len(rows) == 46424 and not rows.duplicated(["state", "tract_id"]).any(),
            "Full national comparison must contain all balanced tracts")
    report = {"balanced_tracts": len(rows),
              "science_full_fixed_land_2000": int(rows.science_2000_full_cover.notna().sum()),
              "science_full_fixed_land_2025": int(rows.science_2025_full_cover.notna().sum()),
              "science_full_fixed_land_both": int(rows.science_change_full_pp.notna().sum()),
              "positive_fourway_support": int(rows.common_4way_valid_fraction.gt(0).sum()),
              "complete_fourway_support": int(np.isclose(rows.common_4way_valid_fraction, 1, atol=1e-10, rtol=0).sum()),
              "common_fraction_quantiles": rows.common_4way_valid_fraction.quantile([0,.01,.05,.25,.5,.75,.95,1]).to_dict()}
    summaries = []
    fourway_columns = ["nlcd_2000_fourway_cover", "science_2000_fourway_cover",
                       "nlcd_2025_fourway_cover", "science_2025_fourway_cover",
                       "nlcd_change_4way_pp", "science_change_4way_pp", "processing_change_difference_pp",
                       "nlcd_change_full_pp", "common_4way_valid_fraction"]
    full_columns = ["nlcd_2000_full_cover", "science_2000_full_cover",
                    "nlcd_2025_full_cover", "science_2025_full_cover",
                    "nlcd_change_full_pp", "science_change_full_pp", "processing_change_full_pp"]
    scopes = [("original_all_fixed_land", rows, full_columns),
              ("any_common_4way", rows.loc[rows.common_4way_valid_fraction.gt(0)], fourway_columns),
              ("complete_4way_fixed_land", rows.loc[np.isclose(rows.common_4way_valid_fraction, 1, atol=1e-10, rtol=0)], fourway_columns)]
    for scope, d, columns in scopes:
        require(len(d) > 0, f"No comparison units in {scope}")
        for col in columns:
            available = d.loc[d[col].notna()]
            summaries.append({"support": scope, "target": "national_tract_mean", "variable": "all",
                              "quantity": col, "value": float(available[col].mean()) if len(available) else np.nan,
                              "tracts": len(available), "cities": int(available.single_city_id.nunique())})
        for variable in SOCIAL:
            q = f"{variable}_city_quartile"
            selected = d.loc[d[q].isin([1, 4])]
            for quartile, subset in selected.groupby(q):
                for col in columns:
                    available = subset.loc[subset[col].notna()]
                    summaries.append({"support": scope, "target": f"q{int(quartile)}_tract_mean",
                                      "variable": variable, "quantity": col,
                                      "value": float(available[col].mean()) if len(available) else np.nan,
                                      "tracts": len(available), "cities": int(available.single_city_id.nunique())})
            for col in columns:
                city = selected.groupby(["single_city_id", q])[col].mean().unstack(q)
                city = city.dropna(subset=[1.0, 4.0])
                summaries.append({"support": scope, "target": "q4_minus_q1_equal_city",
                                  "variable": variable, "quantity": col,
                                  "value": float((city[4.0] - city[1.0]).mean()) if len(city) else np.nan,
                                  "tracts": int(selected.loc[selected.single_city_id.isin(city.index) & selected[col].notna()].shape[0]),
                                  "cities": len(city)})
    pd.DataFrame(summaries).to_csv(output / "product_difference_summary.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", nargs="+", required=True, help="Use 'all' for the 49-state full comparison")
    parser.add_argument("--years", nargs="+", type=int, choices=[2000, 2025], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    years = tuple(sorted(args.years))
    require(len(set(years)) == len(years), "Duplicate years")
    require(args.output.resolve().is_relative_to(OUT), "Output must remain inside this run folder")
    args.output.mkdir(exist_ok=False)
    with CATALOG.open(encoding="utf-8-sig", newline="") as f:
        catalog = {row["state"]: row for row in csv.DictReader(f)}
    states = sorted(catalog) if args.states == ["all"] else args.states
    require(len(states) == len(set(states)) and all(state in catalog for state in states), "Invalid state list")
    base, lookup = load_study_ids(years)
    uris = {(prod, year): source_uri(SOURCES[(prod, year)]) for prod in PRODUCTS for year in years}
    rows = []; state_results = []
    for position, state in enumerate(states, 1):
        data, result = aggregate_state(catalog[state], base, lookup, uris, years)
        rows.extend(data); state_results.append(result)
        print(f"state {state}: {position}/{len(states)}; balanced {result['balanced_tracts']}; NLCD diff {result['max_frozen_nlcd_difference_pp']:.3g}", flush=True)
    frame = pd.DataFrame(rows).sort_values(["state", "tract_id"])
    frame.to_csv(args.output / "tract_product_comparison.csv", index=False)
    pd.DataFrame(state_results).to_csv(args.output / "state_support.csv", index=False)
    summary = summarize(frame, args.output) if states == sorted(catalog) and years == (2000, 2025) else None
    report = {"status": "PASS_FULL_PRODUCT_COMPARISON" if summary else "PASS_BOUNDED_PREFLIGHT",
              "states": states, "years": years, "product_version": "v2025-6", "grid_metres": 30,
              "study_tracts": len(frame), "max_frozen_nlcd_difference_pp": max(x["max_frozen_nlcd_difference_pp"] for x in state_results),
              "summary": summary, "source_archives": {f"{prod}_{year}": str(SOURCES[(prod, year)])
                                                  for prod in PRODUCTS for year in years},
              "source_data_read_only": True, "formal_results_modified": False}
    (args.output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"states", "source_archives"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
