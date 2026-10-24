"""Versioned fixed-land spatial qualification, without changing measurements.

The 120 m square/majority rule is a declared geometric screen, not an official
accuracy guarantee. Local treatment extrema are diagnostics, never exclusions.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.sparse import load_npz
from scipy.spatial import cKDTree

from aggregate_national_canopy_state import expected_matrix_hash
from assemble_and_audit_lagged_support_20261008 import digest, read, require, write_json, qualification
from spatial_mean_slope_inference_20261009 import ROOT, LOGS, PRIMARY, CONFIG, KEY, T, Y, load_inputs

RULE = ROOT / "03_protocol/geographic_measurement_support_revision_20261009.json"
CATALOG = LOGS / "national_urban_weights_summary_20260914_run01/national_weight_catalog.csv"
PRIOR = LOGS / "national_tail_support_adjudication_20261009_run01"


def square_support(indices, areas, width, size=4, tolerance=.0001):
    indices = np.asarray(indices, dtype=np.int64)
    areas = np.asarray(areas, dtype=float)
    require(len(indices) == len(areas) and np.isfinite(areas).all(), "Invalid sparse support")
    require((areas > 0).all() and (areas <= 900+tolerance).all(), "Invalid cell overlap area")
    require(len(np.unique(indices)) == len(indices), "Duplicate raster indices")
    order = np.argsort(indices)
    indices, areas = indices[order], areas[order]
    full = indices[np.abs(areas-900.) <= tolerance]
    starts = full[full % width <= width-size]
    for dy in range(size):
        for dx in range(size):
            if dx or dy:
                starts = np.intersect1d(starts, full-(dy*width+dx), assume_unique=True)
                if not len(starts):
                    break
        if not len(starts):
            break
    covered = (np.unique(np.concatenate([starts+dy*width+dx for dy in range(size) for dx in range(size)]))
               if len(starts) else np.empty(0, dtype=np.int64))
    mass = float(areas[np.searchsorted(indices, covered)].sum()) if len(covered) else 0.
    total = float(areas.sum())
    fraction = mass/total if total else 0.
    require(0 <= fraction <= 1+1e-12, "Spatial support fraction outside physical range")
    return {"fixed_land_weight_sum_m2": total, "stored_land_cells": len(indices),
            "complete_land_cells": len(full), "complete_120m_square_windows": len(starts),
            "square_supported_land_cells": len(covered), "square_supported_land_m2": mass,
            "square_supported_land_fraction": fraction}


def local_support(df, cfg, out):
    values = df[cfg["x_columns"]+cfg["w_columns"]].to_numpy(float)
    folds = df.spatial_fold.to_numpy(int)
    treatment = df[T].to_numpy(float)
    rows = []
    for fold in range(5):
        for year in (2014, 2019, 2024):
            train = np.flatnonzero((folds != fold) & df.outcome_year.eq(year).to_numpy())
            test = np.flatnonzero((folds == fold) & df.outcome_year.eq(year).to_numpy())
            require(len(train) >= 200 and len(test) > 0, "Empty/undersized revised support split")
            scale = np.quantile(values[train], .75, axis=0)-np.quantile(values[train], .25, axis=0)
            require((scale > 0).all(), "Degenerate support feature; no scale substitution")
            center = np.median(values[train], axis=0)
            tree = cKDTree((values[train]-center)/scale)
            # Only a bounded query array is materialized at a time.
            for start in range(0, len(test), 512):
                ix = test[start:start+512]
                distance, neighbors = tree.query((values[ix]-center)/scale, k=200, workers=1)
                neighbor_t = treatment[train[neighbors]]
                lo, hi = neighbor_t.min(axis=1), neighbor_t.max(axis=1)
                sd = neighbor_t.std(axis=1, ddof=1)
                require(np.isfinite(sd).all() and (sd > 0).all(), "No treatment variation in a diagnostic neighborhood")
                for j, i in enumerate(ix):
                    rows.append({**df.iloc[i][KEY].to_dict(), "spatial_fold": fold,
                                 "treatment_pp": float(treatment[i]), "local_min_pp": float(lo[j]),
                                 "local_max_pp": float(hi[j]), "local_treatment_sd_pp": float(sd[j]),
                                 "outside_local_extrema_diagnostic": bool(treatment[i] < lo[j] or treatment[i] > hi[j]),
                                 "farthest_scaled_distance": float(distance[j, -1]),
                                 "qualification_exclusion": False})
            print(f"LOCAL fold={fold+1}/5 year={year} rows={len(test)}", flush=True)
    result = pd.DataFrame(rows)
    require(len(result) == len(df) and not result.duplicated(KEY).any(), "Local diagnostic row loss")
    result.to_csv(out / "revised_local_treatment_support_DIAGNOSTIC_ONLY.csv", index=False)
    return result


def execute(out):
    started = time.monotonic()
    require(not out.exists(), "Refuse overwriting repair evidence")
    out.mkdir(parents=True)
    try:
        cfg, full, _ = load_inputs()
        rule = json.loads(RULE.read_text(encoding="utf-8"))
        require(rule["contiguous_square_cells"] == 4 and rule["minimum_square_supported_land_fraction"] == .5,
                "Declared geographic rule changed")
        hashes = {str(p): digest(p) for p in [Path(__file__), RULE, CONFIG, CATALOG,
                  PRIMARY / "lagged_primary_model_sample.csv", ROOT / "00_admin/project_state.yaml",
                  PRIOR / "audit.json", LOGS / "national_tail_support_validation_20261009_run01/validation.json"]}
        catalog = read(CATALOG)
        tract = full[["state", "tract_id", "urban_fixed_land_m2"]].drop_duplicates()
        require(len(tract) == 42805 and not tract.duplicated(["state", "tract_id"]).any(), "Original tract inventory")
        wanted = set(tract.tract_id)
        metrics = []
        for _, c in catalog.iterrows():
            grid = json.loads(Path(c.grid_path).read_text(encoding="utf-8"))
            hashes[c.grid_path] = digest(c.grid_path)
            hashes[c.audit_path] = digest(c.audit_path)
            source_audit = json.loads(Path(c.audit_path).read_text(encoding="utf-8"))
            transform = rasterio.Affine(*grid["transform"][:6])
            crs = rasterio.crs.CRS.from_wkt(grid["crs_wkt"])
            require(crs.is_projected and np.allclose([transform.a, transform.e, transform.b, transform.d],
                    [30., -30., 0., 0.], rtol=0, atol=1e-12), "Unsupported grid CRS/resolution/orientation")
            width = int(grid["shape"][1])
            processed = 0
            for table in sorted(Path(c.weights_dir).glob(f"urban_area_{c.state}_*_*.csv")):
                area = read(table)
                use = np.flatnonzero(area.tract_id.isin(wanted).to_numpy())
                if not len(use):
                    continue
                matrix_path = table.with_name(table.name.replace("urban_area_", "urban_weights_")).with_suffix(".npz")
                hashes[str(table)] = digest(table)
                hashes[str(matrix_path)] = digest(matrix_path)
                require(hashes[str(matrix_path)] == expected_matrix_hash(source_audit, matrix_path.name), "Frozen weights changed")
                matrix = load_npz(matrix_path).tocsr()
                require(matrix.shape == (len(area), int(np.prod(grid["shape"]))), "Sparse matrix shape mismatch")
                for row in use:
                    weights = matrix.getrow(row)
                    result = square_support(weights.indices, weights.data, width)
                    require(abs(result["fixed_land_weight_sum_m2"]-area.iloc[row].urban_fixed_land_m2)
                            <= max(.01, result["fixed_land_weight_sum_m2"]*1e-10), "Weight/table area mismatch")
                    metrics.append({"state": c.state, "tract_id": area.iloc[row].tract_id, **result,
                                    "measurement_support_pass": bool(result["complete_120m_square_windows"] > 0 and
                                      result["square_supported_land_fraction"] >= .5),
                                    "weight_table": str(table), "weight_matrix": str(matrix_path), "matrix_row": int(row)})
                    processed += 1
                del matrix
            print(f"GEOGRAPHIC state={c.state} tracts={processed}; no model fits", flush=True)
        metrics = pd.DataFrame(metrics)
        require(len(metrics) == len(tract) and not metrics.duplicated(["state", "tract_id"]).any(), "Measurement metric join loss/duplicates")
        joined = full.merge(metrics, on=["state", "tract_id"], how="left", sort=False, validate="many_to_one", indicator="support_join")
        require(len(joined) == len(full) and joined[KEY].equals(full[KEY]) and joined.support_join.eq("both").all(), "Repair row/order/unmatched check")
        require(np.allclose(joined.urban_fixed_land_m2, joined.fixed_land_weight_sum_m2, rtol=1e-10, atol=.01), "Frozen support extent changed")
        keep = joined.measurement_support_pass.to_numpy(bool)
        revised = full.loc[keep].copy().reset_index(drop=True)
        require(qualification(revised).all() and not revised.duplicated(KEY).any(), "Original qualification no longer satisfied")
        require(set(revised.spatial_fold.astype(int)) == set(range(5)), "Revised sample lost a frozen fold")
        require(revised.groupby("spatial_block_id").spatial_fold.nunique().eq(1).all(), "Block leakage")
        require(revised.groupby("tract_id").spatial_fold.nunique().eq(1).all(), "Tract leakage")
        require(np.isfinite(revised[[T, Y, *cfg["x_columns"], *cfg["w_columns"]]].to_numpy(float)).all(), "Nonfinite revised input")
        require(not revised.tract_id.eq("16055000200").any(), "Negligible boundary fragment still qualifies")
        metrics.to_csv(out / "tract_measurement_support.csv", index=False)
        joined[KEY+["spatial_fold", "spatial_block_id", "urban_fixed_land_m2", "measurement_support_pass",
                    "complete_120m_square_windows", "square_supported_land_fraction"]].to_csv(out / "requalification_manifest_all_original_rows.csv", index=False)
        revised.to_csv(out / "revised_analysis_sample.csv", index=False)
        full.loc[~keep].to_csv(out / "excluded_measurement_records_NOT_DELETED_FROM_SOURCE.csv", index=False)
        flow = joined.groupby(["outcome_year", "spatial_fold"]).measurement_support_pass.agg(["size", "sum"]).reset_index()
        flow.columns = ["outcome_year", "spatial_fold", "original_rows", "requalified_rows"]
        flow["excluded_geographic_support"] = flow.original_rows-flow.requalified_rows
        flow.to_csv(out / "sample_flow_year_fold.csv", index=False)
        social = []
        for field in cfg["x_columns"]:
            for label, mask in [("retained", keep), ("not_measurement_qualified", ~keep)]:
                a = full.loc[mask, field]
                social.append({"field": field, "sample": label, "rows": len(a), "mean": float(a.mean()),
                               "median": float(a.median()), "p10": float(a.quantile(.1)), "p90": float(a.quantile(.9))})
        pd.DataFrame(social).to_csv(out / "geographic_selection_social_context.csv", index=False)
        local = local_support(revised, cfg, out)
        require(all(digest(p) == h for p, h in hashes.items()), "Inputs changed during requalification")
        write_json(out / "audit.json", {
            "status": "PASS_GEOGRAPHIC_REPAIR_AND_COMPUTATIONAL_REQUALIFICATION_PENDING_INDEPENDENT_VALIDATION",
            "original_rows": len(full), "original_tracts": len(tract), "revised_rows": len(revised),
            "revised_tracts": int(revised.tract_id.nunique()), "excluded_rows_in_new_version": int((~keep).sum()),
            "excluded_tracts_in_new_version": int((~metrics.measurement_support_pass).sum()),
            "duplicates": 0, "unmatched": 0, "original_source_rows_deleted": 0, "new_model_fits": 0,
            "folds_regenerated": False, "seed_changed": False, "measurements_changed": False,
            "local_outside_extrema_rows_diagnostic_not_exclusion": int(local.outside_local_extrema_diagnostic.sum()),
            "minimum_diagnostic_local_treatment_sd_pp": float(local.local_treatment_sd_pp.min()),
            "support_gate_scope": "GEOMETRY_AND_NONCOLLAPSED_LOCAL_VARIATION_NOT_UNCONFOUNDEDNESS_OR_DENSITY_PROOF",
            "formal_inference": "NOT_RUN", "model_configuration_sha256": digest(CONFIG),
            "source_hashes": hashes, "outputs": {p.name: digest(p) for p in out.iterdir() if p.is_file()},
            "elapsed_seconds": time.monotonic()-started})
        print(f"PASS geographic repair rows={len(revised)} tracts={revised.tract_id.nunique()}; independent validation required", flush=True)
    except BaseException:
        (out / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    execute(parser.parse_args().out)
