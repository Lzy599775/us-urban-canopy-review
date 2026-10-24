"""Build tract-date Landsat LST diagnostics with pixel-level same-day compositing.

The treatment, sample, annual outcome, and causal model remain unset. Pixels
observed by multiple same-day scenes are selected without using temperature.
"""
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from affine import Affine
from scipy.sparse import load_npz
from shapely.geometry import shape

import run_state_lst_window_diagnostic_blocked as core


FIELDS = [
    "state", "tract_id", "date_composite_id", "source_item_ids", "source_scene_count",
    "observation_date", "year", "fixed_land_area_m2", "valid_temperature_area_m2",
    "valid_fraction", "temperature_c_mean_valid", "temperature_c_sd_valid",
    "st_qa_mean_c", "st_cdist_mean_km", "st_emis_mean", "st_emsd_mean",
    "qa_pixel_any_source_cloud_or_shadow_or_fill_area_m2",
    "qa_radsat_any_source_terrain_occlusion_area_m2",
    "temperature_all_sources_fill_area_m2", "status",
]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validated_resume_rows(
    source_manifest_path: Path,
    source_progress_path: Path,
    source_output_path: Path,
    manifest: list[dict[str, object]],
    dates: list[str],
    by_date: dict[str, list[dict[str, object]]],
    positive_ids: list[str],
    state: str,
    year: int,
) -> tuple[list[dict[str, str]], list[str], int]:
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    core.require(source_manifest == manifest, "Resume candidate manifest differs")

    events = []
    with source_progress_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"Invalid resume progress JSON at line {line_number}") from error
    import_events = [event for event in events if event.get("event") == "RESUME_IMPORT_COMPLETE"]
    core.require(len(import_events) <= 1, "Resume source has multiple import events")
    imported_dates = [] if not import_events else [str(value) for value in import_events[0].get("imported_complete_dates", [])]
    if import_events:
        core.require(
            int(import_events[0].get("imported_rows", -1)) == len(imported_dates) * len(positive_ids),
            "Resume import event row count mismatch",
        )
    complete_events = [event for event in events if event.get("event") == "DATE_COMPLETE"]
    newly_completed_dates = [str(event.get("date")) for event in complete_events]
    completed_dates = imported_dates + newly_completed_dates
    core.require(completed_dates, "Resume source has no completed dates")
    core.require(len(completed_dates) == len(set(completed_dates)), "Resume source repeats a completed date")
    core.require(completed_dates == dates[: len(completed_dates)], "Resume dates are not a consecutive prefix")
    for position, event in enumerate(complete_events, len(imported_dates) + 1):
        core.require(int(event.get("position", -1)) == position, "Resume date position mismatch")
        core.require(int(event.get("date_count", -1)) == len(dates), "Resume date count mismatch")
        core.require(int(event.get("tract_rows", -1)) == len(positive_ids), "Resume tract row count mismatch")

    with source_output_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        core.require(reader.fieldnames == FIELDS, "Resume output fields differ")
        source_rows = list(reader)
    rows_by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        observation_date = row.get("observation_date", "")
        core.require(observation_date in dates, f"Unexpected resume observation date: {observation_date}")
        rows_by_date[observation_date].append(row)

    imported_rows: list[dict[str, str]] = []
    for date in completed_dates:
        date_rows = rows_by_date.get(date, [])
        core.require(len(date_rows) == len(positive_ids), f"Incomplete resume rows for {date}")
        core.require([row["tract_id"] for row in date_rows] == positive_ids, f"Resume tract order differs for {date}")
        expected_items = sorted(item["id"] for item in by_date[date])
        expected_source_ids = ";".join(expected_items)
        expected_date_id = f"DATE_COMPOSITE_{date.replace('-', '')}"
        for row in date_rows:
            core.require(row["state"] == state, f"Resume state differs for {date}")
            core.require(int(row["year"]) == year, f"Resume year differs for {date}")
            core.require(row["date_composite_id"] == expected_date_id, f"Resume composite id differs for {date}")
            core.require(row["source_item_ids"] == expected_source_ids, f"Resume source scenes differ for {date}")
            core.require(int(row["source_scene_count"]) == len(expected_items), f"Resume scene count differs for {date}")
            core.require(row["status"] in {"VALID_TEMPERATURE", "NO_VALID_TEMPERATURE"}, f"Resume status differs for {date}")
        imported_rows.extend(date_rows)

    uncommitted_rows = [row for row in source_rows if row["observation_date"] not in set(completed_dates)]
    if uncommitted_rows:
        next_date_index = len(completed_dates)
        core.require(next_date_index < len(dates), "Rows remain after all dates were completed")
        core.require(
            {row["observation_date"] for row in uncommitted_rows} == {dates[next_date_index]},
            "Uncommitted resume rows are not confined to the next date",
        )
    return imported_rows, completed_dates, len(uncommitted_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--year", required=True, type=int, choices=core.YEARS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, default=None)
    args = parser.parse_args()
    state = args.state.zfill(2)
    core.require(state in core.VALID_STATES, "State outside frozen scope")
    args.out.mkdir(exist_ok=False)
    started = time.time()
    evidence: dict[str, dict[str, object]] = {}
    output_stream = None

    def track(path: Path) -> Path:
        resolved = path.resolve()
        stat = resolved.stat()
        evidence[str(resolved)] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256_file_bytes": sha256(resolved),
        }
        return resolved

    def progress(event: dict[str, object]) -> None:
        with (args.out / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**event, "at_epoch_seconds": time.time()}, ensure_ascii=False) + "\n")

    try:
        track(Path(__file__))
        baseline_audit = json.loads(track(core.BASELINE_DIR / "audit.json").read_text(encoding="utf-8"))
        canopy_audit = json.loads(track(core.ROOT / "06_logs/national_canopy_streaming_summary_20260915_run01/audit.json").read_text(encoding="utf-8"))
        candidate_audit = json.loads(track(core.CATALOG_DIR / "audit.json").read_text(encoding="utf-8"))
        weight_audit = json.loads(track(core.WEIGHT_SUMMARY / "audit.json").read_text(encoding="utf-8"))
        core.require(baseline_audit["status"] == "PASS_NATIONAL_ANALYSIS_BASELINE_JOIN_NOT_SAMPLE_AUTHORITY", "Baseline gate")
        core.require(canopy_audit["status"] == "PASS_NATIONAL_CANOPY_STREAMING_SUMMARY_NOT_SAMPLE_AUTHORITY", "Canopy gate")
        core.require(candidate_audit["status"] == "PASS_NATIONAL_LST_CANDIDATE_INVENTORY_NOT_DOWNLOAD_AUTHORITY", "Candidate gate")
        core.require(weight_audit["status"] == "PASS_NATIONAL_URBAN_WEIGHT_STRUCTURE_WITH_COVERAGE_REVIEW", "Weight gate")

        baseline_path = track(core.BASELINE_DIR / "national_analysis_baseline_readiness.csv")
        baseline = {r["tract_id"]: r for r in read_csv(baseline_path) if r["state"] == state and float(r["urban_fixed_land_m2"]) > 0}
        core.require(baseline, "No positive fixed-land tracts")
        boundary_path = track(core.BOUNDARY_DIR / f"tl_2010_{state}_tract00.zip")
        targets = gpd.read_file("zip://" + boundary_path.as_posix())
        core.require(targets.crs is not None and targets.geometry.is_valid.all(), "Invalid boundary")
        targets = targets[targets["CTIDFP00"].isin(baseline)].copy().sort_values("CTIDFP00").reset_index(drop=True)
        core.require(set(targets["CTIDFP00"]) == set(baseline), "Boundary/baseline mismatch")

        catalog_path = track(core.WEIGHT_SUMMARY / "national_weight_catalog.csv")
        catalog = {r["state"]: r for r in read_csv(catalog_path)}[state]
        state_audit_path = track(Path(catalog["audit_path"]))
        grid_path = track(Path(catalog["grid_path"]))
        weights_dir = Path(catalog["weights_dir"])
        state_audit = json.loads(state_audit_path.read_text(encoding="utf-8"))
        core.require(state_audit["status"] in {"PASS_DIAGNOSTIC_STRUCTURE", "PASS_DIAGNOSTIC_STRUCTURE_AND_REGRESSION"}, "Weight gate")
        grid = json.loads(grid_path.read_text(encoding="utf-8"))
        target_shape = tuple(int(v) for v in grid["shape"])
        target_transform = Affine(*grid["transform"][:6])
        target_crs = grid["crs_wkt"]
        full_ids = list(grid["tract_order"])
        matrix_paths = sorted(weights_dir.glob(f"urban_weights_{state}_*.npz"))
        table_paths = sorted(weights_dir.glob(f"urban_area_{state}_*.csv"))
        table_by_suffix = {p.name.replace("urban_area_", "").replace(".csv", ""): p for p in table_paths}
        core.require(len(matrix_paths) == len(table_paths) == int(state_audit["shards"]), "Shard inventory mismatch")

        shards = []
        positive_ids: list[str] = []
        active_parts = []
        full_position = 0
        for matrix_path in matrix_paths:
            matrix_path = track(matrix_path)
            core.require(sha256(matrix_path) == core.expected_matrix_hash(state_audit, matrix_path.name), "Matrix hash mismatch")
            suffix = matrix_path.name.replace("urban_weights_", "").replace(".npz", "")
            table_path = track(table_by_suffix[suffix])
            table = read_csv(table_path)
            matrix = load_npz(matrix_path).tocsr()
            ids = [r["tract_id"] for r in table]
            core.require(ids == full_ids[full_position:full_position + len(ids)], "Tract order mismatch")
            full_position += len(ids)
            keep = np.array([x in baseline for x in ids], dtype=bool)
            if np.any(keep):
                selected = matrix[keep].tocsr()
                selected_ids = [x for x, retain in zip(ids, keep) if retain]
                fixed = np.asarray(selected.sum(axis=1)).ravel()
                expected = np.array([float(baseline[x]["urban_fixed_land_m2"]) for x in selected_ids])
                core.require(np.allclose(fixed, expected, rtol=1e-10, atol=0.01), "Fixed area mismatch")
                shards.append((selected, selected_ids, fixed))
                positive_ids.extend(selected_ids)
                active_parts.append(selected.indices.copy())
        core.require(full_position == len(full_ids) and positive_ids == sorted(baseline), "Tract accounting mismatch")
        active_indices = np.unique(np.concatenate(active_parts))
        active_rows, active_cols = np.divmod(active_indices, target_shape[1])
        shard_positions = []
        for matrix, _, _ in shards:
            positions = np.searchsorted(active_indices, matrix.indices)
            core.require(np.array_equal(active_indices[positions], matrix.indices), "Active-pixel mapping mismatch")
            shard_positions.append(positions)

        target_union = targets.to_crs(4326).geometry.union_all()
        items = {}
        for page in sorted(core.CATALOG_DIR.glob(f"{args.year}_page*.json.gz")):
            track(page)
            with gzip.open(page, "rt", encoding="utf-8") as stream:
                data = json.load(stream)
            for item in data["features"]:
                item_id = item["id"]
                if not item_id.startswith("LC08_L2SP_") or not item_id.endswith("_T1"):
                    continue
                if not shape(item["geometry"]).intersects(target_union):
                    continue
                core.require(item_id not in items and all(k in item.get("assets", {}) for k in core.REQUIRED_ASSET_KEYS), "Candidate item invalid")
                items[item_id] = item
        by_date: dict[str, list[dict[str, object]]] = defaultdict(list)
        for item in items.values():
            by_date[item["properties"]["datetime"][:10]].append(item)
        dates = sorted(by_date)
        manifest = [{"observation_date": d, "item_ids": sorted(x["id"] for x in by_date[d])} for d in dates]
        (args.out / "date_candidate_items.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        output_path = args.out / f"state_{state}_tract_date_temperature_diagnostic.csv"
        output_stream = output_path.open("x", encoding="utf-8-sig", newline="")
        writer = csv.DictWriter(output_stream, fieldnames=FIELDS)
        writer.writeheader()
        session = requests.Session()
        # Use the direct local network path requested for the public blob store.
        session.trust_env = False
        token_cache: dict[tuple[str, str], tuple[str, float]] = {}
        retry_events = []
        status_counts = Counter()
        rows_written = 0
        max_destination_cells = 0
        completed_dates: list[str] = []
        resume_info = None
        if args.resume_from is not None:
            resume_dir = args.resume_from.resolve()
            core.require(resume_dir.is_dir(), f"Resume directory is absent: {resume_dir}")
            core.require(resume_dir != args.out.resolve(), "Resume source and output directory are identical")
            core.require(not (resume_dir / "audit.json").exists(), "Resume source already has a final audit")
            source_manifest_path = track(resume_dir / "date_candidate_items.json")
            source_progress_path = track(resume_dir / "progress.jsonl")
            source_output_path = track(resume_dir / f"state_{state}_tract_date_temperature_diagnostic.csv")
            source_failure_path = resume_dir / "failure.txt"
            if source_failure_path.exists():
                track(source_failure_path)
            resume_rows, completed_dates, uncommitted_row_count = validated_resume_rows(
                source_manifest_path,
                source_progress_path,
                source_output_path,
                manifest,
                dates,
                by_date,
                positive_ids,
                state,
                args.year,
            )
            writer.writerows(resume_rows)
            output_stream.flush()
            rows_written = len(resume_rows)
            status_counts.update(row["status"] for row in resume_rows)
            resume_info = {
                "source_directory": str(resume_dir),
                "imported_complete_dates": completed_dates,
                "imported_rows": rows_written,
                "ignored_uncommitted_rows": uncommitted_row_count,
                "prior_segment_retry_events_available": False,
            }
            progress({"event": "RESUME_IMPORT_COMPLETE", **resume_info})

        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".TIF", GDAL_HTTP_MAX_RETRY="0"):
            for date_position, date in enumerate(dates[len(completed_dates):], len(completed_dates) + 1):
                date_items = sorted(by_date[date], key=lambda x: x["id"])
                progress({"event": "DATE_START", "position": date_position, "date_count": len(dates), "date": date, "source_scenes": len(date_items)})
                n = len(active_indices)
                selected = np.zeros(n, dtype=bool)
                best_cdist = np.full(n, -1, dtype="int32")
                best_qa = np.full(n, np.iinfo("int32").max, dtype="int32")
                selected_rank = np.full(n, np.iinfo("int16").max, dtype="int16")
                selected_raw = {k: np.zeros(n, dtype="uint16") for k in ("lwir11", "qa", "cdist", "emis", "emsd")}
                any_cloud = np.zeros(n, dtype=bool)
                any_terrain = np.zeros(n, dtype=bool)
                all_fill = np.ones(n, dtype=bool)

                for rank, item in enumerate(date_items):
                    item_id = item["id"]
                    asset_paths = {k: core.signed_url(session, item["assets"][k]["href"], token_cache) for k in core.ASSETS}
                    metadata = session.get(core.signed_url(session, item["assets"]["mtl.json"]["href"], token_cache), timeout=(15, 60))
                    metadata.raise_for_status()
                    product_id = metadata.json()["LANDSAT_METADATA_FILE"]["PRODUCT_CONTENTS"]["LANDSAT_PRODUCT_ID"]
                    core.require(product_id.startswith(item_id.removesuffix("_02_T1") + "_") and product_id.endswith("_02_T1"), "Metadata mismatch")
                    raw = {}
                    reference = None
                    for key in core.ASSETS:
                        try:
                            values, signature, destination_shape = core.read_active_asset(asset_paths[key], active_rows, active_cols, target_crs, target_transform, target_shape, retry_events)
                        except rasterio.errors.RasterioError as error:
                            # Refresh the signed URL once after a remote tile read failure.
                            token_cache.clear()
                            refreshed = core.signed_url(session, item["assets"][key]["href"], token_cache)
                            retry_events.append({"asset": item["assets"][key]["href"].rsplit("/", 1)[-1], "error_type": type(error).__name__, "action": "refresh_signed_url_and_retry"})
                            values, signature, destination_shape = core.read_active_asset(refreshed, active_rows, active_cols, target_crs, target_transform, target_shape, retry_events)
                        max_destination_cells = max(max_destination_cells, destination_shape[0] * destination_shape[1])
                        if reference is None:
                            reference = signature
                        core.require(signature == reference, f"Asset grid mismatch: {item_id}/{key}")
                        raw[key] = values.astype("uint16", copy=False)
                    valid = (raw["lwir11"] > 0) & ((raw["qa_pixel"] & 63) == 0) & ((raw["qa_radsat"] & (1 << 11)) == 0)
                    better = valid & ((~selected) | (raw["cdist"].astype("int32") > best_cdist) | ((raw["cdist"].astype("int32") == best_cdist) & (raw["qa"].astype("int32") < best_qa)) | ((raw["cdist"].astype("int32") == best_cdist) & (raw["qa"].astype("int32") == best_qa) & (rank < selected_rank)))
                    for key in selected_raw:
                        selected_raw[key][better] = raw[key][better]
                    best_cdist[better] = raw["cdist"][better]
                    best_qa[better] = raw["qa"][better]
                    selected_rank[better] = rank
                    selected[better] = True
                    any_cloud |= (raw["qa_pixel"] & 63) != 0
                    any_terrain |= (raw["qa_radsat"] & (1 << 11)) != 0
                    all_fill &= raw["lwir11"] == 0
                    del raw, valid, better
                    gc.collect()

                temp = selected_raw["lwir11"].astype("float64") * 0.00341802 + 149.0 - 273.15
                fixed_area = np.concatenate([fixed for _, _, fixed in shards])
                tract_count = len(positive_ids)
                valid_area = np.zeros(tract_count); temp_sum = np.zeros(tract_count); temp_sq = np.zeros(tract_count)
                cloud_area = np.zeros(tract_count); terrain_area = np.zeros(tract_count); fill_area = np.zeros(tract_count)
                numerators = {k: np.zeros(tract_count) for k in ("qa", "cdist", "emis", "emsd")}
                denominators = {k: np.zeros(tract_count) for k in numerators}
                scales = {"qa": 0.01, "cdist": 0.01, "emis": 0.0001, "emsd": 0.0001}
                offset = 0
                for (matrix, ids, _), positions in zip(shards, shard_positions):
                    current = slice(offset, offset + len(ids)); local_valid = selected[positions]
                    valid_area[current] = core.row_sum(matrix, local_valid.astype("float64"))
                    temp_sum[current] = core.row_sum(matrix, np.where(local_valid, temp[positions], 0.0))
                    temp_sq[current] = core.row_sum(matrix, np.where(local_valid, temp[positions] ** 2, 0.0))
                    cloud_area[current] = core.row_sum(matrix, any_cloud[positions].astype("float64"))
                    terrain_area[current] = core.row_sum(matrix, any_terrain[positions].astype("float64"))
                    fill_area[current] = core.row_sum(matrix, all_fill[positions].astype("float64"))
                    for key, scale in scales.items():
                        available = local_valid & (selected_raw[key][positions] > 0)
                        denominators[key][current] = core.row_sum(matrix, available.astype("float64"))
                        numerators[key][current] = core.row_sum(matrix, np.where(available, selected_raw[key][positions].astype("float64") * scale, 0.0))
                    offset += len(ids)
                core.require(offset == tract_count, "Aggregation offset mismatch")

                source_ids = ";".join(x["id"] for x in date_items)
                date_id = f"DATE_COMPOSITE_{date.replace('-', '')}"
                rows = []
                for index, tract_id in enumerate(positive_ids):
                    area = float(valid_area[index])
                    if area > 0:
                        status = "VALID_TEMPERATURE"
                        mean = float(temp_sum[index] / area)
                        sd = math.sqrt(max(0.0, float(temp_sq[index] / area - mean ** 2)))
                        means = {k: (float(numerators[k][index] / denominators[k][index]) if denominators[k][index] > 0 else None) for k in numerators}
                    else:
                        status = "NO_VALID_TEMPERATURE"; mean = sd = None; means = {k: None for k in numerators}
                    status_counts[status] += 1
                    rows.append({"state": state, "tract_id": tract_id, "date_composite_id": date_id, "source_item_ids": source_ids,
                        "source_scene_count": len(date_items), "observation_date": date, "year": args.year,
                        "fixed_land_area_m2": float(fixed_area[index]), "valid_temperature_area_m2": area,
                        "valid_fraction": area / fixed_area[index], "temperature_c_mean_valid": mean,
                        "temperature_c_sd_valid": sd, "st_qa_mean_c": means["qa"], "st_cdist_mean_km": means["cdist"],
                        "st_emis_mean": means["emis"], "st_emsd_mean": means["emsd"],
                        "qa_pixel_any_source_cloud_or_shadow_or_fill_area_m2": float(cloud_area[index]),
                        "qa_radsat_any_source_terrain_occlusion_area_m2": float(terrain_area[index]),
                        "temperature_all_sources_fill_area_m2": float(fill_area[index]), "status": status})
                writer.writerows(rows); output_stream.flush(); rows_written += len(rows)
                progress({"event": "DATE_COMPLETE", "position": date_position, "date_count": len(dates), "date": date, "tract_rows": len(rows)})
                del selected, selected_raw, best_cdist, best_qa, selected_rank, temp
                gc.collect()

        output_stream.close(); output_stream = None
        core.require(rows_written == len(dates) * len(positive_ids), "Date/tract row accounting mismatch")
        for path, record in evidence.items():
            stat = Path(path).stat()
            core.require((stat.st_size, stat.st_mtime_ns) == (record["bytes"], record["mtime_ns"]), f"Input changed: {path}")
            core.require(sha256(Path(path)) == record["sha256_file_bytes"], f"Input hash changed: {path}")
        audit = {"status": "PASS_STATE_LST_SAME_DAY_PIXEL_COMPOSITE_NOT_FORMAL_OUTCOME", "state": state, "year": args.year,
            "source_scenes": len(items), "composite_dates": len(dates), "same_day_multi_scene_dates": sum(len(v) > 1 for v in by_date.values()),
            "target_positive_fixed_land_tracts": len(positive_ids), "rows": rows_written,
            "same_day_selection": "valid first; larger ST_CDIST; smaller ST_QA; lexicographically smaller product id; temperature excluded",
            "quality_rule": "ST_B10>0; QA_PIXEL bits 0-5 zero; QA_RADSAT terrain-occlusion bit 11 zero",
            "status_counts": dict(status_counts), "controlled_single_asset_retry_count": len(retry_events),
            "controlled_single_asset_retry_events": retry_events, "max_single_asset_destination_cells": max_destination_cells,
            "resume": resume_info,
            "controlled_single_asset_retry_count_scope": "CURRENT_SEGMENT_ONLY" if resume_info else "FULL_RUN",
            "max_single_asset_destination_cells_scope": "CURRENT_SEGMENT_ONLY" if resume_info else "FULL_RUN",
            "annual_outcome_constructed": False, "coverage_threshold_applied": False, "formal_sample_selected": False,
            "formal_model": "NOT_RUN", "raw_data_modified": False, "signed_urls_persisted": False,
            "output_sha256_file_bytes": sha256(output_path), "elapsed_seconds": time.time() - started}
        (args.out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.out / "input_files.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        progress({"event": "RUN_COMPLETE", "status": audit["status"], "rows": rows_written})
        print(json.dumps(audit, ensure_ascii=False), flush=True)
    except BaseException:
        if output_stream is not None:
            output_stream.close()
        (args.out / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
