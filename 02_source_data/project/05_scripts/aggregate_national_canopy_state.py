"""Aggregate annual canopy for one explicitly requested audited state.

The program retains every tract represented by the state weight catalog. It
does not choose a formal sample, impute missing canopy, or fit a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import traceback
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.sparse import csr_matrix, load_npz


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SUMMARY = ROOT / "06_logs/national_urban_weights_summary_20260914_run01"
SOURCE_READINESS = ROOT / "06_logs/national_tcc_receipt_readiness_20260914_run01/audit.json"
VALID_STATES = set("01 04 05 06 08 09 10 11 12 13 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56".split())
VALID_YEARS = set(range(1999, 2026))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    require(bool(rows), f"No rows to write: {path}")
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def row_sum(matrix: csr_matrix, multiplier: np.ndarray) -> np.ndarray:
    require(len(multiplier) == matrix.nnz, "Sparse multiplier length differs")
    counts = np.diff(matrix.indptr)
    result = np.zeros(matrix.shape[0], dtype="float64")
    nonempty = counts > 0
    if np.any(nonempty):
        starts = matrix.indptr[:-1][nonempty]
        weighted = matrix.data * multiplier
        result[nonempty] = np.add.reduceat(weighted, starts)
    return result


def source_window(dataset: rasterio.DatasetReader, grid: dict[str, object]) -> Window:
    transform = rasterio.Affine(*grid["transform"][:6])
    width, height = int(grid["shape"][1]), int(grid["shape"][0])
    col = (transform.c - dataset.transform.c) / dataset.transform.a
    row = (transform.f - dataset.transform.f) / dataset.transform.e
    require(abs(col - round(col)) <= 1e-8 and abs(row - round(row)) <= 1e-8, "Canopy grid offset is not integral")
    window = Window(int(round(col)), int(round(row)), width, height)
    require(window.col_off >= 0 and window.row_off >= 0, "Canopy window starts outside raster")
    require(window.col_off + width <= dataset.width and window.row_off + height <= dataset.height, "Canopy window ends outside raster")
    require(
        np.allclose(list(dataset.window_transform(window)), list(transform), rtol=0, atol=1e-8),
        "Canopy window transform differs from weight grid",
    )
    return window


def expected_matrix_hash(audit: dict[str, object], name: str) -> str:
    hashes = audit["weights_sha256"]
    require(isinstance(hashes, dict), "State weight hash ledger missing")
    matches = [value for key, value in hashes.items() if Path(key).name == name]
    require(len(matches) == 1, f"Missing or ambiguous state matrix hash: {name}")
    return str(matches[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    state = args.state.zfill(2)
    years = sorted(set(args.years))
    require(state in VALID_STATES, f"State is outside frozen scope: {state}")
    require(len(years) == len(args.years) and set(years).issubset(VALID_YEARS), "Years must be unique and within 1999-2025")
    args.out.mkdir(exist_ok=False)
    started = time.time()
    evidence: dict[str, dict[str, object]] = {}
    source_stats: dict[int, tuple[int, int]] = {}

    def append_progress(event: dict[str, object]) -> None:
        with (args.out / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**event, "at_epoch_seconds": time.time()}, ensure_ascii=False) + "\n")

    def track_hash(path: Path) -> Path:
        path = path.resolve()
        stat = path.stat()
        evidence[str(path)] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256_file_bytes": sha256(path),
            "verification": "FULL_FILE_SHA256",
        }
        return path

    def track_receipted_archive(path: Path, year: int) -> Path:
        path = path.resolve()
        stat = path.stat()
        source_stats[year] = (stat.st_size, stat.st_mtime_ns)
        evidence[str(path)] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "verification": "PRIOR_FULL_MEMBER_HASH_AND_CURRENT_CENTRAL_DIRECTORY_RECEIPT_PLUS_RUN_TIME_STAT",
            "year": year,
        }
        return path

    try:
        track_hash(Path(__file__))
        summary_audit_path = track_hash(WEIGHT_SUMMARY / "audit.json")
        catalog_path = track_hash(WEIGHT_SUMMARY / "national_weight_catalog.csv")
        summary_audit = json.loads(summary_audit_path.read_text(encoding="utf-8"))
        require(
            summary_audit["status"] == "PASS_NATIONAL_URBAN_WEIGHT_STRUCTURE_WITH_COVERAGE_REVIEW",
            "National weight summary gate not passed",
        )
        require(int(summary_audit["rows"]) == 64999 and int(summary_audit["states_and_dc"]) == 49, "National weight scope differs")
        catalogs = {row["state"]: row for row in read_csv(catalog_path)}
        require(len(catalogs) == 49 and state in catalogs, "National state catalog differs")
        catalog = catalogs[state]

        state_audit_path = track_hash(Path(catalog["audit_path"]))
        grid_path = track_hash(Path(catalog["grid_path"]))
        weights_dir = Path(catalog["weights_dir"]).resolve()
        state_audit = json.loads(state_audit_path.read_text(encoding="utf-8"))
        grid = json.loads(grid_path.read_text(encoding="utf-8"))
        require(state_audit["status"] in {"PASS_DIAGNOSTIC_STRUCTURE", "PASS_DIAGNOSTIC_STRUCTURE_AND_REGRESSION"}, "State weight gate")
        require(state_audit["state"] == state, "State audit identifier differs")
        matrices = sorted(weights_dir.glob(f"urban_weights_{state}_*.npz"))
        tables = sorted(weights_dir.glob(f"urban_area_{state}_*.csv"))
        require(len(matrices) == len(tables) == int(state_audit["shards"]), "State shard inventory differs")
        table_by_suffix = {
            table.name.replace("urban_area_", "").replace(".csv", ""): table for table in tables
        }
        shards: list[tuple[csr_matrix, list[dict[str, str]], str]] = []
        tract_ids: list[str] = []
        total_nnz = 0
        for matrix_path in matrices:
            matrix_path = track_hash(matrix_path)
            require(
                evidence[str(matrix_path)]["sha256_file_bytes"] == expected_matrix_hash(state_audit, matrix_path.name),
                f"Matrix hash differs: {matrix_path.name}",
            )
            suffix = matrix_path.name.replace("urban_weights_", "").replace(".npz", "")
            table_path = table_by_suffix.get(suffix)
            require(table_path is not None, f"Area table missing: {suffix}")
            table_path = track_hash(table_path)
            table = read_csv(table_path)
            matrix = load_npz(matrix_path).tocsr()
            require(matrix.shape == (len(table), int(grid["shape"][0]) * int(grid["shape"][1])), f"Matrix shape differs: {suffix}")
            require(matrix.has_sorted_indices, f"Matrix indices not sorted: {suffix}")
            require([row["tract_id"] for row in table] == grid["tract_order"][len(tract_ids):len(tract_ids) + len(table)], f"Tract order differs: {suffix}")
            fixed = np.asarray(matrix.sum(axis=1)).ravel()
            reported = np.array([float(row["urban_fixed_land_m2"]) for row in table])
            require(np.allclose(fixed, reported, rtol=1e-10, atol=0.01), f"Fixed-land row sums differ: {suffix}")
            tract_ids.extend(row["tract_id"] for row in table)
            total_nnz += matrix.nnz
            shards.append((matrix, table, suffix))
        require(len(tract_ids) == len(set(tract_ids)) == int(state_audit["rows"]), "State tract accounting differs")
        require(total_nnz == int(state_audit["stored_weights"]), "State stored-weight count differs")

        readiness_path = track_hash(SOURCE_READINESS)
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        require(readiness["status"] == "PASS_RECEIPT_AND_CENTRAL_DIRECTORY_CHECK_ONLY", "Canopy source readiness gate")
        source_by_year = {int(row["year"]): row for row in readiness["years"]}
        require(set(years).issubset(source_by_year), "Requested canopy year lacks receipt")
        sources: dict[int, tuple[Path, str]] = {}
        for year in years:
            receipt = source_by_year[year]
            require(
                receipt["recorded_pass_status"] is True
                and receipt["archive_bytes_match"] is True
                and receipt["member_count_match"] is True
                and all(
                    member["recorded_validation_passed"] is True
                    and member["recorded_hash_present"] is True
                    and member["directory_size_matches"] is True
                    and member["directory_crc_matches"] is True
                    for member in receipt["members"]
                ),
                f"Canopy source receipt incomplete: {year}",
            )
            track_hash(Path(receipt["receipt_path"]))
            source = track_receipted_archive(Path(receipt["source_path"]), year)
            with zipfile.ZipFile(source) as archive:
                members = [name for name in archive.namelist() if name.lower().endswith(".tif")]
            require(len(members) == 1, f"Expected one canopy raster member: {year}")
            sources[year] = (source, members[0])

        all_rows: list[dict[str, object]] = []
        state_year_rows: list[dict[str, object]] = []
        status_counts = Counter()
        publication_status_counts = Counter()
        max_closure = 0.0
        grid_transform = rasterio.Affine(*grid["transform"][:6])
        expected_shape = (int(grid["shape"][0]), int(grid["shape"][1]))
        for position, year in enumerate(years, 1):
            append_progress({"event": "YEAR_START", "state": state, "year": year, "position": position, "year_count": len(years)})
            source, member = sources[year]
            uri = f"/vsizip/{source.as_posix()}/{member}"
            year_rows: list[dict[str, object]] = []
            with rasterio.open(uri) as dataset:
                require(dataset.count == 1 and dataset.dtypes == ("uint8",), f"Canopy encoding differs: {year}")
                require(dataset.res == (30.0, 30.0) and dataset.nodata == 255, f"Canopy resolution/nodata differs: {year}")
                require(dataset.crs == rasterio.crs.CRS.from_wkt(grid["crs_wkt"]), f"Canopy coordinate system differs: {year}")
                window = source_window(dataset, grid)
                values = dataset.read(1, window=window)
            require(values.shape == expected_shape, f"Canopy state window shape differs: {year}")
            flat = values.ravel()
            del values
            year_valid = year_missing = year_254 = year_255 = year_tree_area = year_fixed = 0.0
            complete_tracts = incomplete_tracts = zero_fixed_tracts = 0
            landmask_missing_tracts = 0
            for matrix, table, suffix in shards:
                fixed = np.array([float(row["urban_fixed_land_m2"]) for row in table])
                pixel_values = flat[matrix.indices]
                require(np.all((pixel_values <= 100) | (pixel_values == 254) | (pixel_values == 255)), f"Unexpected canopy code: {year}/{suffix}")
                valid = row_sum(matrix, (pixel_values <= 100).astype("float64"))
                nonprocessing = row_sum(matrix, (pixel_values == 254).astype("float64"))
                background = row_sum(matrix, (pixel_values == 255).astype("float64"))
                numerator = row_sum(matrix, np.where(pixel_values <= 100, pixel_values, 0).astype("float64"))
                closure = valid + nonprocessing + background - fixed
                tolerance = np.maximum(0.01, fixed * 1e-10)
                require(np.all(np.abs(closure) <= tolerance), f"Annual area closure failed: {year}/{suffix}")
                max_closure = max(max_closure, float(np.max(np.abs(closure), initial=0.0)))
                for index, source_row in enumerate(table):
                    fixed_area = float(fixed[index])
                    valid_area = float(valid[index])
                    missing_area = float(nonprocessing[index] + background[index])
                    landmask_missing_area = float(source_row["urban_landcover_nodata_m2"])
                    require(landmask_missing_area >= 0, f"Negative land-mask missing area: {source_row['tract_id']}")
                    if landmask_missing_area > 0:
                        landmask_missing_tracts += 1
                    if fixed_area == 0:
                        status = "ZERO_FIXED_LAND_RETAINED"
                        zero_fixed_tracts += 1
                    elif valid_area == 0:
                        status = "NO_VALID_ANNUAL_CANOPY_NO_IMPUTATION"
                        incomplete_tracts += 1
                    elif missing_area > 0:
                        status = "INCOMPLETE_ANNUAL_CANOPY_NO_IMPUTATION"
                        incomplete_tracts += 1
                    else:
                        status = "COMPLETE_FIXED_LAND"
                        complete_tracts += 1
                    observed_percent: object = ""
                    fixed_percent: object = ""
                    valid_fraction: object = ""
                    tree_area: object = ""
                    if fixed_area > 0:
                        valid_fraction = valid_area / fixed_area
                    if valid_area > 0:
                        observed_percent = float(numerator[index] / valid_area)
                        tree_area = float(numerator[index] / 100.0)
                    if fixed_area == 0:
                        publication_status = "UNDEFINED_ZERO_FIXED_LAND"
                    elif landmask_missing_area > 0:
                        publication_status = "OBSERVED_DOMAIN_ONLY_LAND_MASK_MISSING"
                    elif status == "COMPLETE_FIXED_LAND":
                        publication_status = "FULL_FIXED_LAND_ESTIMATE_AVAILABLE"
                    elif valid_area > 0:
                        publication_status = "OBSERVED_DOMAIN_ONLY_ANNUAL_CANOPY_MISSING"
                    else:
                        publication_status = "NO_ANNUAL_CANOPY_ESTIMATE"
                    if publication_status == "FULL_FIXED_LAND_ESTIMATE_AVAILABLE":
                        fixed_percent = observed_percent
                    status_counts[status] += 1
                    publication_status_counts[publication_status] += 1
                    year_rows.append({
                        "state": state,
                        "county_code": source_row["tract_id"][:5],
                        "tract_id": source_row["tract_id"],
                        "year": year,
                        "urban_geometry_m2": float(source_row["urban_geometry_m2"]),
                        "urban_fixed_land_m2": fixed_area,
                        "urban_landcover_nodata_m2": landmask_missing_area,
                        "landmask_status": "INCOMPLETE_NO_IMPUTATION" if landmask_missing_area > 0 else "COMPLETE_CLASSIFICATION",
                        "urban_valid_canopy_land_m2": valid_area,
                        "urban_missing_canopy_land_m2": missing_area,
                        "urban_nonprocessing_254_land_m2": float(nonprocessing[index]),
                        "urban_background_255_land_m2": float(background[index]),
                        "urban_valid_fraction": valid_fraction,
                        "urban_canopy_percent_observed_land": observed_percent,
                        "urban_canopy_percent_fixed_land": fixed_percent,
                        "urban_canopy_area_observed_m2": tree_area,
                        "urban_change_from_2000_pp": "",
                        "canopy_status": status,
                        "publication_status": publication_status,
                        "formal_sample_selected": "NOT_ASSESSED",
                    })
                year_valid += float(valid.sum())
                year_missing += float((nonprocessing + background).sum())
                year_254 += float(nonprocessing.sum())
                year_255 += float(background.sum())
                year_tree_area += float(numerator.sum() / 100.0)
                year_fixed += float(fixed.sum())
            require(len(year_rows) == len(tract_ids), f"State-year row count differs: {year}")
            all_rows.extend(year_rows)
            state_year_rows.append({
                "state": state,
                "year": year,
                "tract_rows": len(year_rows),
                "fixed_land_m2": year_fixed,
                "valid_canopy_land_m2": year_valid,
                "missing_canopy_land_m2": year_missing,
                "nonprocessing_254_land_m2": year_254,
                "background_255_land_m2": year_255,
                "canopy_area_observed_m2": year_tree_area,
                "complete_tracts": complete_tracts,
                "incomplete_tracts": incomplete_tracts,
                "zero_fixed_land_tracts": zero_fixed_tracts,
                "landmask_missing_tracts": landmask_missing_tracts,
            })
            del flat
            append_progress({"event": "YEAR_COMPLETE", **state_year_rows[-1], "position": position, "year_count": len(years)})

        baseline = {
            row["tract_id"]: row["urban_canopy_percent_fixed_land"]
            for row in all_rows if row["year"] == 2000
        }
        if 2000 in years:
            require(len(baseline) == len(tract_ids), "2000 baseline accounting differs")
            for row in all_rows:
                current = row["urban_canopy_percent_fixed_land"]
                base = baseline[row["tract_id"]]
                row["urban_change_from_2000_pp"] = (
                    float(current) - float(base) if current != "" and base != "" else ""
                )
        write_csv(args.out / f"tract_year_canopy_{state}.csv", all_rows)
        write_csv(args.out / f"state_year_canopy_{state}.csv", state_year_rows)

        for path, record in evidence.items():
            stat = Path(path).stat()
            require((stat.st_size, stat.st_mtime_ns) == (record["bytes"], record["mtime_ns"]), f"Input changed during run: {path}")
            if record["verification"] == "FULL_FILE_SHA256":
                require(sha256(Path(path)) == record["sha256_file_bytes"], f"Input hash changed during run: {path}")
        for year, (size, mtime_ns) in source_stats.items():
            source = sources[year][0]
            stat = source.stat()
            require((stat.st_size, stat.st_mtime_ns) == (size, mtime_ns), f"Canopy archive changed during run: {year}")

        audit = {
            "status": "PASS_STATE_CANOPY_AGGREGATION_WITH_COVERAGE_REVIEW_NOT_SAMPLE_AUTHORITY",
            "scope": "FIXED_2000_URBAN_INTERSECTION_AND_2001_FIXED_LAND_MASK",
            "state": state,
            "years": years,
            "year_count": len(years),
            "tracts": len(tract_ids),
            "rows": len(all_rows),
            "unique_tract_year_keys": len({(row["tract_id"], row["year"]) for row in all_rows}),
            "duplicates": 0,
            "dropped": 0,
            "stored_weights": total_nnz,
            "canopy_status_counts": dict(status_counts),
            "publication_status_counts": dict(publication_status_counts),
            "max_abs_annual_area_closure_m2": max_closure,
            "baseline_2000_change_derived": 2000 in years,
            "missing_imputed": False,
            "formal_sample_selected": False,
            "formal_model": "NOT_RUN",
            "raw_data_modified": False,
            "elapsed_seconds": time.time() - started,
        }
        require(audit["rows"] == audit["unique_tract_year_keys"] == len(tract_ids) * len(years), "Final key accounting")
        (args.out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.out / "input_files.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        append_progress({"event": "RUN_COMPLETE", "status": audit["status"], "elapsed_seconds": audit["elapsed_seconds"]})
        print(json.dumps(audit, ensure_ascii=False), flush=True)
    except BaseException:
        (args.out / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        (args.out / "input_files_partial.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
