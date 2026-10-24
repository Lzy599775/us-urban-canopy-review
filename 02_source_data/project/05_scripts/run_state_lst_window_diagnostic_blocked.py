"""Memory-bounded state Landsat LST window diagnostic.

Only active fixed-land pixels inside each Landsat scene bounding window are
materialized. Scientific rules match run_state_lst_window_diagnostic.py.
This diagnostic does not create an annual outcome, select a formal sample, or
fit a causal model.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import geopandas as gpd
import numpy as np
import rasterio
import requests
from affine import Affine
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy.sparse import csr_matrix, load_npz
from shapely.geometry import shape


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "06_logs/national_analysis_baseline_join_20260914_run01"
BOUNDARY_DIR = ROOT / "04_national/boundaries_2000/tract_by_state"
CATALOG_DIR = ROOT / "06_logs/national_lst_candidate_inventory_20260915_run01"
WEIGHT_SUMMARY = ROOT / "06_logs/national_urban_weights_summary_20260914_run01"
VALID_STATES = set(
    "01 04 05 06 08 09 10 11 12 13 16 17 18 19 20 21 22 23 24 25 26 27 "
    "28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 "
    "51 53 54 55 56".split()
)
YEARS = (2014, 2019, 2024)
ASSETS = {
    "lwir11": ("ST_B10", "temperature"),
    "qa": ("ST_QA", "temperature_uncertainty"),
    "cdist": ("ST_CDIST", "cloud_distance"),
    "emis": ("ST_EMIS", "emissivity"),
    "emsd": ("ST_EMSD", "emissivity_sd"),
    "qa_pixel": ("QA_PIXEL", "pixel_quality"),
    "qa_radsat": ("QA_RADSAT", "radiometric_quality"),
}
REQUIRED_ASSET_KEYS = tuple(ASSETS) + ("mtl.json",)
OUTPUT_FIELDS = [
    "state",
    "tract_id",
    "item_id",
    "observation_datetime",
    "observation_date",
    "year",
    "fixed_land_area_m2",
    "valid_temperature_area_m2",
    "valid_fraction",
    "temperature_c_mean_valid",
    "temperature_c_sd_valid",
    "st_qa_mean_c",
    "st_cdist_mean_km",
    "st_emis_mean",
    "st_emsd_mean",
    "qa_pixel_cloud_or_shadow_or_fill_area_m2",
    "qa_radsat_terrain_occlusion_area_m2",
    "temperature_fill_area_m2",
    "status",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


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


# ---------------------------------------------------------------------------
# Network-outage retry support (added; existing logic untouched).
# While the network is down, wait with exponential backoff and reconnect;
# give up only after NETWORK_RETRY_TOTAL_BUDGET_SECONDS so the date-level
# checkpoint stays intact and a restart resumes cleanly.
NETWORK_RETRY_BASE_DELAY_SECONDS = 15.0
NETWORK_RETRY_MAX_DELAY_SECONDS = 60.0
NETWORK_RETRY_TOTAL_BUDGET_SECONDS = 1800.0
# Bounded retries for a single asset that fails to open (e.g. "not recognized
# as being in a supported file format"). Such faults on a remote /vsicurl read
# are usually a transient bad response from the blob endpoint, not a permanent
# defect; a fresh token plus a retry typically recovers. After this many
# attempts the error is raised so the caller can skip the scene.
OPEN_FAIL_MAX_ATTEMPTS = 4


def network_retry_delay(attempt: int) -> float:
    return min(NETWORK_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)), NETWORK_RETRY_MAX_DELAY_SECONDS)


def log_network_retry(operation: str, attempt: int, error: BaseException, delay: float) -> None:
    msg = str(error)
    if len(msg) > 300:
        msg = msg[:300] + "..."
    print(
        f"NETWORK_INTERRUPTED operation={operation} attempt={attempt} "
        f"error={type(error).__name__} retry_in_seconds={delay:g} msg={msg!r}",
        file=sys.stderr,
        flush=True,
    )


def is_network_rasterio_error(error: BaseException) -> bool:
    """Heuristic: was this RasterioError caused by a transient remote-read
    failure (worth retrying) versus a persistent geometry/warp/data defect
    (retrying cannot help)?"""
    text = str(error).lower()
    markers = (
        "vsicurl", "curl", "http", "connection", "timeout", "timed out",
        "ireadblock", "read block", "reset", "broken pipe", "econn",
        "socket", "network", "503", "502", "504", "500", "429", "404",
    )
    return any(marker in text for marker in markers)


def fetch_with_network_retry(session: requests.Session, url: str, timeout, label: str) -> requests.Response:
    """GET with raise_for_status, tolerating temporary network outages."""
    attempt = 0
    budget_started: float | None = None
    while True:
        attempt += 1
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as error:
            status = error.response.status_code if error.response is not None else None
            if status is not None and 400 <= status < 500 and status not in (429, 408):
                # Client errors such as 404 never recover by waiting; fail fast.
                raise
            now = time.monotonic()
            if budget_started is None:
                budget_started = now
            elif now - budget_started > NETWORK_RETRY_TOTAL_BUDGET_SECONDS:
                raise
            delay = network_retry_delay(attempt)
            log_network_retry(label, attempt, error, delay)
            time.sleep(delay)
        except requests.exceptions.RequestException as error:
            now = time.monotonic()
            if budget_started is None:
                budget_started = now
            elif now - budget_started > NETWORK_RETRY_TOTAL_BUDGET_SECONDS:
                raise
            delay = network_retry_delay(attempt)
            log_network_retry(label, attempt, error, delay)
            time.sleep(delay)


STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"


def stac_refresh_item(session: requests.Session, item: dict) -> dict | None:
    """Find a live replacement for a cached STAC candidate item.

    USGS reprocessing replaces Landsat C2 item IDs (the processing-date
    segment changes or is dropped) and deletes the old blobs, so cached
    inventory entries can start returning 404. Re-query the catalog by
    WRS path/row + platform + acquisition datetime and return a candidate
    that carries every required asset, or None when nothing matches.
    """
    props = item.get("properties", {})
    query: dict[str, object] = {}
    if props.get("landsat:wrs_path") is not None:
        query["landsat:wrs_path"] = {"eq": str(props["landsat:wrs_path"]).zfill(3)}
    if props.get("landsat:wrs_row") is not None:
        query["landsat:wrs_row"] = {"eq": str(props["landsat:wrs_row"]).zfill(3)}
    if props.get("platform"):
        query["platform"] = {"in": [props["platform"]]}
    if not query or not props.get("datetime"):
        return None
    body = {
        "collections": ["landsat-c2-l2"],
        "datetime": props["datetime"],
        "query": query,
        "limit": 20,
    }
    response = session.post(STAC_SEARCH_URL, json=body, timeout=(15, 60))
    response.raise_for_status()
    required = set(ASSETS) | {"mtl.json", "mtl.txt"}
    for candidate in response.json().get("features", []):
        if candidate["id"] == item["id"]:
            continue
        if not required.issubset(candidate.get("assets", {}).keys()):
            continue
        return candidate
    return None


def mtl_product_id(
    session: requests.Session,
    item: dict,
    cache: dict[tuple[str, str], tuple[str, float]],
) -> str:
    """Return LANDSAT_PRODUCT_ID for a scene, tolerating missing mtl.json.

    Reprocessed USGS scenes ship only the text-format metadata (mtl.txt);
    parse LANDSAT_PRODUCT_ID out of it when the JSON asset is gone.
    """
    for key, kind in (("mtl.json", "json"), ("mtl.txt", "text")):
        asset = item["assets"].get(key)
        if asset is None:
            continue
        response = session.get(signed_url(session, asset["href"], cache), timeout=(15, 60))
        if response.status_code == 404:
            continue
        response.raise_for_status()
        if kind == "json":
            return response.json()["LANDSAT_METADATA_FILE"]["PRODUCT_CONTENTS"]["LANDSAT_PRODUCT_ID"]
        match = re.search(r'LANDSAT_PRODUCT_ID\s*=\s*"?([^"\r\n]+)"?', response.text)
        require(match is not None, f"MTL.txt lacks LANDSAT_PRODUCT_ID: {item['id']}")
        return match.group(1).strip()
    require(False, f"No usable MTL asset for scene: {item['id']}")
    raise RuntimeError("Unreachable")


def token_retry_delay(response: requests.Response) -> float:
    retry_after = response.headers.get("Retry-After")
    require(retry_after is not None, "Token service rate limit omitted Retry-After")
    try:
        delay = float(retry_after)
    except ValueError:
        retry_at = parsedate_to_datetime(retry_after)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        delay = retry_at.astimezone(timezone.utc).timestamp() - time.time()
    require(delay >= 0, f"Token service returned invalid Retry-After: {retry_after}")
    return delay


def signed_url(
    session: requests.Session,
    url: str,
    cache: dict[tuple[str, str], tuple[str, float]],
    force: bool = False,
) -> str:
    parsed = urlparse(url)
    require(parsed.netloc.endswith(".blob.core.windows.net"), f"Unexpected asset host: {parsed.netloc}")
    account = parsed.netloc.split(".")[0]
    container = parsed.path.strip("/").split("/")[0]
    key = (account, container)
    cached = None if force else cache.get(key)
    # Refresh before expiry because one state diagnostic can run for nearly an hour.
    if cached is None or time.time() >= cached[1] - 120:
        max_attempts = 5
        rate_limit_attempts = 0
        network_attempt = 0
        budget_started: float | None = None
        while True:
            try:
                response = session.get(
                    f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}?cache_bust={time.time_ns()}",
                    headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
                    timeout=(15, 60),
                )
            except requests.exceptions.RequestException as error:
                # Network outage: wait, reconnect, and retry without consuming
                # the rate-limit attempt budget.
                now = time.monotonic()
                if budget_started is None:
                    budget_started = now
                elif now - budget_started > NETWORK_RETRY_TOTAL_BUDGET_SECONDS:
                    raise
                network_attempt += 1
                delay = network_retry_delay(network_attempt)
                log_network_retry("sas_token", network_attempt, error, delay)
                time.sleep(delay)
                continue
            if response.status_code in (408, 500, 502, 503, 504):
                # Transient server-side fault (e.g. 504 Gateway Time-out from
                # the SAS issuer): the connection works but the endpoint is
                # momentarily unhealthy. Retry with the same backoff + total
                # budget as a network outage instead of aborting the run.
                now = time.monotonic()
                if budget_started is None:
                    budget_started = now
                elif now - budget_started > NETWORK_RETRY_TOTAL_BUDGET_SECONDS:
                    response.raise_for_status()
                network_attempt += 1
                delay = network_retry_delay(network_attempt)
                log_network_retry(
                    "sas_token_http",
                    network_attempt,
                    requests.exceptions.HTTPError(f"{response.status_code} {response.reason} for SAS token endpoint"),
                    delay,
                )
                time.sleep(delay)
                continue
            network_attempt = 0
            budget_started = None
            if response.status_code != 429:
                response.raise_for_status()
                break
            rate_limit_attempts += 1
            if rate_limit_attempts == max_attempts:
                response.raise_for_status()
            delay = token_retry_delay(response)
            print(
                f"TOKEN_RATE_LIMIT attempt={rate_limit_attempts}/{max_attempts} retry_after_seconds={delay:g}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
        token = response.json()["token"]
        token_query = parse_qs(token)
        require("se" in token_query and token_query["se"], "Signed token has no expiry")
        expiry_text = token_query["se"][0]
        expiry = datetime.fromisoformat(expiry_text.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
        require(expiry > time.time(), "Signed token is already expired")
        cache[key] = (token, expiry)
    return url + "?" + cache[key][0]


def refresh_signed_url(
    session: requests.Session,
    url: str,
    cache: dict[tuple[str, str], tuple[str, float]],
    reason: str,
) -> str:
    """Force-fetch a fresh SAS token and rebuild the signed URL.

    HTTP 403 on a blob read means the SAS token embedded in the URL is no
    longer valid (typically expired mid-scene); backoff alone cannot help,
    but a fresh token can."""
    base = url.split("?", 1)[0]
    parsed = urlparse(base)
    if not parsed.netloc.endswith(".blob.core.windows.net"):
        return url
    refreshed = signed_url(session, base, cache, force=True)
    print(
        f"TOKEN_REFRESHED reason={reason} asset={base.rsplit('/', 1)[-1]}",
        file=sys.stderr,
        flush=True,
    )
    return refreshed


def expected_matrix_hash(audit: dict[str, object], name: str) -> str:
    hashes = audit["weights_sha256"]
    require(isinstance(hashes, dict), "State weight hash ledger missing")
    matches = [value for key, value in hashes.items() if Path(key).name == name]
    require(len(matches) == 1, f"Missing or ambiguous matrix hash: {name}")
    return str(matches[0])


def destination_window(
    source: rasterio.DatasetReader,
    target_crs: str,
    target_transform: Affine,
    target_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    left, bottom, right, top = transform_bounds(
        source.crs,
        target_crs,
        *source.bounds,
        densify_pts=21,
    )
    inverse = ~target_transform
    corners = [
        inverse * (left, bottom),
        inverse * (left, top),
        inverse * (right, bottom),
        inverse * (right, top),
    ]
    columns = [point[0] for point in corners]
    rows = [point[1] for point in corners]
    col_start = max(0, math.floor(min(columns)) - 1)
    col_stop = min(target_shape[1], math.ceil(max(columns)) + 1)
    row_start = max(0, math.floor(min(rows)) - 1)
    row_stop = min(target_shape[0], math.ceil(max(rows)) + 1)
    if col_stop <= col_start or row_stop <= row_start:
        return None
    return row_start, row_stop, col_start, col_stop


def read_active_asset(
    url: str,
    active_rows: np.ndarray,
    active_cols: np.ndarray,
    target_crs: str,
    target_transform: Affine,
    target_shape: tuple[int, int],
    retry_events: list[dict[str, object]] | None = None,
    session: requests.Session | None = None,
    token_cache: dict[tuple[str, str], tuple[str, float]] | None = None,
) -> tuple[np.ndarray, tuple[object, ...], tuple[int, int]]:
    attempt = 0
    token_refreshes = 0
    open_fail_attempts = 0
    retry_budget_started: float | None = None
    while True:
        attempt += 1
        try:
            with rasterio.open(url) as source:
                signature = (source.crs, source.transform, source.width, source.height, source.count)
                window = destination_window(source, target_crs, target_transform, target_shape)
                output = np.zeros(len(active_rows), dtype=source.dtypes[0])
                if window is None:
                    return output, signature, (0, 0)
                row_start, row_stop, col_start, col_stop = window
                active_mask = (
                    (active_rows >= row_start)
                    & (active_rows < row_stop)
                    & (active_cols >= col_start)
                    & (active_cols < col_stop)
                )
                positions = np.flatnonzero(active_mask)
                if positions.size == 0:
                    return output, signature, (row_stop - row_start, col_stop - col_start)
                destination = np.zeros(
                    (row_stop - row_start, col_stop - col_start),
                    dtype=source.dtypes[0],
                )
                reproject(
                    rasterio.band(source, 1),
                    destination,
                    src_transform=source.transform,
                    src_crs=source.crs,
                    dst_transform=target_transform * Affine.translation(col_start, row_start),
                    dst_crs=target_crs,
                    src_nodata=source.nodata,
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )
                output[positions] = destination[
                    active_rows[positions] - row_start,
                    active_cols[positions] - col_start,
                ]
                return output, signature, destination.shape
        except rasterio.errors.RasterioIOError as error:
            text = str(error)
            # 403 means the SAS token in the URL expired (or was revoked)
            # mid-scene. Backoff cannot help; refresh the token and retry.
            if "403" in text and session is not None and token_cache is not None:
                if token_refreshes >= 5:
                    raise
                token_refreshes += 1
                if retry_events is not None:
                    retry_events.append({"asset": url.split("?", 1)[0].rsplit("/", 1)[-1], "error_type": type(error).__name__, "attempt": attempt, "action": "token_refresh_403", "error_msg": text[:300]})
                url = refresh_signed_url(session, url, token_cache, "asset_read_403")
                time.sleep(2.0)
                continue
            # Any other open failure (e.g. "not recognized as being in a
            # supported file format") on a remote /vsicurl read is almost always
            # a transient bad response from the blob endpoint, not a permanent
            # defect in the asset. Retry with backoff and a fresh SAS token; only
            # after the attempts are exhausted do we raise, so the caller can
            # skip the scene instead of aborting the whole state/year run.
            open_fail_attempts += 1
            if open_fail_attempts > OPEN_FAIL_MAX_ATTEMPTS:
                raise
            delay = network_retry_delay(open_fail_attempts)
            if retry_events is not None:
                retry_events.append({"asset": url.split("?", 1)[0].rsplit("/", 1)[-1], "error_type": type(error).__name__, "attempt": attempt, "action": "open_failed_retry", "error_msg": text[:300]})
            if session is not None and token_cache is not None:
                url = refresh_signed_url(session, url, token_cache, "asset_open_failed")
            time.sleep(delay)
            continue
        except rasterio.errors.RasterioError as error:
            # A persistent geometry/warp/data defect will never succeed on
            # retry; fail fast so the date is recorded and the run moves on.
            if not is_network_rasterio_error(error):
                raise
            now = time.monotonic()
            if retry_budget_started is None:
                retry_budget_started = now
            elif now - retry_budget_started > NETWORK_RETRY_TOTAL_BUDGET_SECONDS:
                raise
            # Remote tile reads over /vsicurl surface network outages as
            # RasterioError: wait for the connection to come back, then retry.
            if retry_events is not None:
                retry_events.append({"asset": url.split("?", 1)[0].rsplit("/", 1)[-1], "error_type": type(error).__name__, "attempt": attempt, "action": "network_backoff_retry", "error_msg": str(error)[:300]})
            delay = network_retry_delay(attempt)
            log_network_retry("asset_read", attempt, error, delay)
            time.sleep(delay)
    raise RuntimeError("Unreachable asset-read retry state")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    parser.add_argument("--scene-start", type=int, default=1)
    parser.add_argument("--scene-end", type=int, default=None)
    args = parser.parse_args()
    state = args.state.zfill(2)
    require(state in VALID_STATES, f"State is outside frozen scope: {state}")
    selected_years = tuple(sorted(set(args.years)))
    require(selected_years and set(selected_years).issubset(YEARS), "Years are outside the frozen diagnostic scope")
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
        baseline_audit = json.loads(track(BASELINE_DIR / "audit.json").read_text(encoding="utf-8"))
        canopy_audit = json.loads(
            track(ROOT / "06_logs/national_canopy_streaming_summary_20260915_run01/audit.json").read_text(encoding="utf-8")
        )
        candidate_audit = json.loads(track(CATALOG_DIR / "audit.json").read_text(encoding="utf-8"))
        weight_summary_audit = json.loads(track(WEIGHT_SUMMARY / "audit.json").read_text(encoding="utf-8"))
        require(baseline_audit["status"] == "PASS_NATIONAL_ANALYSIS_BASELINE_JOIN_NOT_SAMPLE_AUTHORITY", "Baseline gate")
        require(canopy_audit["status"] == "PASS_NATIONAL_CANOPY_STREAMING_SUMMARY_NOT_SAMPLE_AUTHORITY", "Canopy gate")
        require(candidate_audit["status"] == "PASS_NATIONAL_LST_CANDIDATE_INVENTORY_NOT_DOWNLOAD_AUTHORITY", "Candidate gate")
        require(weight_summary_audit["status"] == "PASS_NATIONAL_URBAN_WEIGHT_STRUCTURE_WITH_COVERAGE_REVIEW", "Weight gate")

        baseline_path = track(BASELINE_DIR / "national_analysis_baseline_readiness.csv")
        baseline = {
            row["tract_id"]: row
            for row in read_csv(baseline_path)
            if row["state"] == state and float(row["urban_fixed_land_m2"]) > 0
        }
        require(baseline, "No positive fixed-land tracts for state")
        boundary_path = track(BOUNDARY_DIR / f"tl_2010_{state}_tract00.zip")
        targets = gpd.read_file("zip://" + boundary_path.as_posix())
        require(targets.crs is not None and targets.geometry.is_valid.all(), "Invalid state boundary")
        targets = targets[targets["CTIDFP00"].isin(baseline)].copy().sort_values("CTIDFP00").reset_index(drop=True)
        require(set(targets["CTIDFP00"]) == set(baseline), "State boundary and baseline sets differ")

        catalog_path = track(WEIGHT_SUMMARY / "national_weight_catalog.csv")
        catalog = {row["state"]: row for row in read_csv(catalog_path)}[state]
        state_audit_path = track(Path(catalog["audit_path"]))
        grid_path = track(Path(catalog["grid_path"]))
        weights_dir = Path(catalog["weights_dir"])
        state_audit = json.loads(state_audit_path.read_text(encoding="utf-8"))
        require(state_audit["status"] in {"PASS_DIAGNOSTIC_STRUCTURE", "PASS_DIAGNOSTIC_STRUCTURE_AND_REGRESSION"}, "State weight gate")
        require(state_audit["state"] == state, "State weight identifier differs")
        grid = json.loads(grid_path.read_text(encoding="utf-8"))
        target_shape = tuple(int(value) for value in grid["shape"])
        target_transform = Affine(*grid["transform"][:6])
        target_crs = grid["crs_wkt"]
        full_tract_ids = list(grid["tract_order"])
        matrix_paths = sorted(weights_dir.glob(f"urban_weights_{state}_*.npz"))
        table_paths = sorted(weights_dir.glob(f"urban_area_{state}_*.csv"))
        require(len(matrix_paths) == len(table_paths) == int(state_audit["shards"]), "State shard inventory differs")
        table_by_suffix = {path.name.replace("urban_area_", "").replace(".csv", ""): path for path in table_paths}

        shards: list[tuple[csr_matrix, list[str], np.ndarray]] = []
        positive_tract_ids: list[str] = []
        full_position = 0
        all_active_parts: list[np.ndarray] = []
        for matrix_path in matrix_paths:
            matrix_path = track(matrix_path)
            require(sha256(matrix_path) == expected_matrix_hash(state_audit, matrix_path.name), f"Matrix hash differs: {matrix_path.name}")
            suffix = matrix_path.name.replace("urban_weights_", "").replace(".npz", "")
            table_path = track(table_by_suffix[suffix])
            table = read_csv(table_path)
            matrix = load_npz(matrix_path).tocsr()
            require(matrix.shape == (len(table), target_shape[0] * target_shape[1]), f"Matrix shape differs: {suffix}")
            ids = [row["tract_id"] for row in table]
            require(ids == full_tract_ids[full_position : full_position + len(ids)], f"Tract order differs: {suffix}")
            full_position += len(ids)
            keep = np.array([tract_id in baseline for tract_id in ids], dtype=bool)
            if np.any(keep):
                selected = matrix[keep].tocsr()
                selected_ids = [tract_id for tract_id, retain in zip(ids, keep) if retain]
                fixed = np.asarray(selected.sum(axis=1)).ravel()
                expected = np.array([float(baseline[tract_id]["urban_fixed_land_m2"]) for tract_id in selected_ids])
                require(np.allclose(fixed, expected, rtol=1e-10, atol=0.01), f"Fixed area differs: {suffix}")
                all_active_parts.append(selected.indices.copy())
                shards.append((selected, selected_ids, fixed))
                positive_tract_ids.extend(selected_ids)
        require(full_position == len(full_tract_ids), "Full tract accounting differs")
        require(positive_tract_ids == sorted(baseline), "Positive tract order differs")
        require(len(positive_tract_ids) == len(set(positive_tract_ids)) == len(baseline), "Positive tract accounting differs")

        active_indices = np.unique(np.concatenate(all_active_parts))
        active_rows, active_cols = np.divmod(active_indices, target_shape[1])
        shard_active_positions: list[np.ndarray] = []
        for matrix, _, _ in shards:
            positions = np.searchsorted(active_indices, matrix.indices)
            require(np.array_equal(active_indices[positions], matrix.indices), "Sparse active-pixel mapping differs")
            shard_active_positions.append(positions)

        target_union = targets.to_crs(4326).geometry.union_all()
        items: dict[str, dict[str, object]] = {}
        for page in sorted(CATALOG_DIR.glob("*_page*.json.gz")):
            year = int(page.name.split("_", 1)[0])
            if year not in selected_years:
                continue
            track(page)
            with gzip.open(page, "rt", encoding="utf-8") as stream:
                data = json.load(stream)
            for item in data["features"]:
                item_id = item["id"]
                if not item_id.startswith("LC08_L2SP_") or not item_id.endswith("_T1"):
                    continue
                if not shape(item["geometry"]).intersects(target_union):
                    continue
                require(item_id not in items, f"Duplicate catalog item: {item_id}")
                require(all(key in item.get("assets", {}) for key in REQUIRED_ASSET_KEYS), f"Missing asset: {item_id}")
                items[item_id] = item
        state_items = sorted(items.values(), key=lambda item: (item["properties"]["datetime"], item["id"]))
        require(state_items, "No state-level candidate items")
        scene_start = args.scene_start
        scene_end = args.scene_end if args.scene_end is not None else len(state_items)
        require(1 <= scene_start <= scene_end <= len(state_items), "Scene range outside candidate inventory")
        state_items = state_items[scene_start - 1 : scene_end]
        item_manifest = [
            {
                "item_id": item["id"],
                "datetime": item["properties"]["datetime"],
                "asset_keys": sorted(item["assets"]),
            }
            for item in state_items
        ]
        (args.out / "candidate_items.json").write_text(json.dumps(item_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        output_path = args.out / f"state_{state}_tract_observation_temperature_diagnostic.csv"
        output_stream = output_path.open("x", encoding="utf-8-sig", newline="")
        writer = csv.DictWriter(output_stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        rows_written = 0
        quality_reason_counts = Counter()
        retry_events: list[dict[str, object]] = []
        max_destination_cells = 0
        session = requests.Session()
        token_cache: dict[tuple[str, str], tuple[str, float]] = {}
        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".TIF",
            # Resilient remote reads: let GDAL retry transient HTTP failures
            # itself instead of surfacing them instantly as WarpOperationError.
            GDAL_HTTP_MAX_RETRY="5",
            GDAL_HTTP_RETRY_DELAY="2",
            GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
            CPL_VSIL_CURL_USE_HEAD="NO",
        ):
            for position, item in enumerate(state_items, 1):
                scene_rows: list[dict[str, object]] = []
                item_id = item["id"]
                progress({"event": "SCENE_START", "position": position, "scene_count": len(state_items), "item_id": item_id})
                asset_paths = {}
                for key, (suffix, _) in ASSETS.items():
                    href = item["assets"][key]["href"]
                    require(href.rsplit("/", 1)[-1].endswith(f"_{suffix}.TIF"), f"Asset filename differs: {item_id}/{key}")
                    asset_paths[key] = signed_url(session, href, token_cache)
                try:
                    metadata = session.get(signed_url(session, item["assets"]["mtl.json"]["href"], token_cache), timeout=(15, 60))
                    metadata.raise_for_status()
                    product_id = metadata.json()["LANDSAT_METADATA_FILE"]["PRODUCT_CONTENTS"]["LANDSAT_PRODUCT_ID"]
                except requests.exceptions.HTTPError as error:
                    status = error.response.status_code if error.response is not None else None
                    if status != 404:
                        # A stale SAS token surfaces as 403 here; refresh once.
                        if status != 403:
                            raise
                        refreshed = refresh_signed_url(session, signed_url(session, item["assets"]["mtl.json"]["href"], token_cache), token_cache, "metadata_403")
                        metadata = session.get(refreshed, timeout=(15, 60))
                        metadata.raise_for_status()
                        product_id = metadata.json()["LANDSAT_METADATA_FILE"]["PRODUCT_CONTENTS"]["LANDSAT_PRODUCT_ID"]
                    else:
                        # Reprocessed USGS scenes drop mtl.json (only mtl.txt
                        # ships); if the whole scene vanished from the catalog,
                        # find the live replacement via STAC as a last resort.
                        print(f"MTL_REFRESH_404 item={item_id} mtl.json missing, trying mtl.txt fallback", flush=True)
                        product_id = None
                        try:
                            product_id = mtl_product_id(session, item, token_cache)
                        except (requests.exceptions.HTTPError, RuntimeError) as inner:
                            inner_status = inner.response.status_code if isinstance(inner, requests.exceptions.HTTPError) and inner.response is not None else None
                            if inner_status != 404:
                                raise
                        if product_id is None:
                            print(f"SCENE_REFRESH_404 item={item_id} searching live STAC for replacement", flush=True)
                            replacement = stac_refresh_item(session, item)
                            if replacement is None:
                                print(f"SKIP_SCENE item={item_id} no catalog replacement found", flush=True)
                                if retry_events is not None:
                                    retry_events.append({"item": item_id, "action": "scene_skipped_missing_in_catalog"})
                                continue
                            item = replacement
                            item_id = item["id"]
                            token_cache.clear()
                            asset_paths = {key: signed_url(session, item["assets"][key]["href"], token_cache) for key, _ in ASSETS.items()}
                            product_id = mtl_product_id(session, item, token_cache)
                item_prefix = item_id.removesuffix("_02_T1") + "_"
                require(product_id == item_id or (product_id.startswith(item_prefix) and product_id.endswith("_02_T1")), f"Metadata mismatch: {item_id}")

                reference_grid = None
                active_assets: dict[str, np.ndarray] = {}
                for key in ("lwir11", "qa_pixel", "qa_radsat"):
                    values, signature, destination_shape = read_active_asset(
                        asset_paths[key], active_rows, active_cols, target_crs, target_transform, target_shape, retry_events
                    )
                    max_destination_cells = max(max_destination_cells, destination_shape[0] * destination_shape[1])
                    if reference_grid is None:
                        reference_grid = signature
                    require(signature == reference_grid, f"Asset grid metadata differs: {item_id}/{key}")
                    active_assets[key] = values
                temp_dn = active_assets["lwir11"].astype("uint16", copy=False)
                qa_pixel = active_assets["qa_pixel"].astype("uint16", copy=False)
                qa_radsat = active_assets["qa_radsat"].astype("uint16", copy=False)
                valid_active = (temp_dn > 0) & ((qa_pixel & 63) == 0) & ((qa_radsat & (1 << 11)) == 0)
                temp_c_active = temp_dn.astype("float64") * 0.00341802 + 149.0 - 273.15

                tract_count = len(positive_tract_ids)
                valid_area = np.zeros(tract_count, dtype="float64")
                temp_sum = np.zeros(tract_count, dtype="float64")
                temp_sq_sum = np.zeros(tract_count, dtype="float64")
                cloud_flag_area = np.zeros(tract_count, dtype="float64")
                terrain_flag_area = np.zeros(tract_count, dtype="float64")
                temp_fill_area = np.zeros(tract_count, dtype="float64")
                ancillary_numerators = {name: np.zeros(tract_count) for name in ("uncertainty", "cloud_distance", "emissivity", "emissivity_sd")}
                ancillary_denominators = {name: np.zeros(tract_count) for name in ancillary_numerators}
                row_offset = 0
                for (matrix, ids, _), active_positions in zip(shards, shard_active_positions):
                    current = slice(row_offset, row_offset + len(ids))
                    local_valid = valid_active[active_positions]
                    local_temp = temp_c_active[active_positions]
                    valid_area[current] = row_sum(matrix, local_valid.astype("float64"))
                    temp_sum[current] = row_sum(matrix, np.where(local_valid, local_temp, 0.0))
                    temp_sq_sum[current] = row_sum(matrix, np.where(local_valid, local_temp**2, 0.0))
                    cloud_flag_area[current] = row_sum(matrix, ((qa_pixel[active_positions] & 63) != 0).astype("float64"))
                    terrain_flag_area[current] = row_sum(matrix, ((qa_radsat[active_positions] & (1 << 11)) != 0).astype("float64"))
                    temp_fill_area[current] = row_sum(matrix, (temp_dn[active_positions] == 0).astype("float64"))
                    row_offset += len(ids)
                require(row_offset == tract_count, "Tract aggregation offset differs")

                for key, name, scale in (
                    ("qa", "uncertainty", 0.01),
                    ("cdist", "cloud_distance", 0.01),
                    ("emis", "emissivity", 0.0001),
                    ("emsd", "emissivity_sd", 0.0001),
                ):
                    raw_values, signature, destination_shape = read_active_asset(
                        asset_paths[key], active_rows, active_cols, target_crs, target_transform, target_shape, retry_events,
                        session=session, token_cache=token_cache,
                    )
                    max_destination_cells = max(max_destination_cells, destination_shape[0] * destination_shape[1])
                    require(signature == reference_grid, f"Asset grid metadata differs: {item_id}/{key}")
                    values = raw_values.astype("float64") * scale
                    available = valid_active & (raw_values > 0)
                    row_offset = 0
                    for (matrix, ids, _), active_positions in zip(shards, shard_active_positions):
                        current = slice(row_offset, row_offset + len(ids))
                        local_available = available[active_positions]
                        ancillary_denominators[name][current] = row_sum(matrix, local_available.astype("float64"))
                        ancillary_numerators[name][current] = row_sum(
                            matrix,
                            np.where(local_available, values[active_positions], 0.0),
                        )
                        row_offset += len(ids)

                fixed_area = np.concatenate([fixed for _, _, fixed in shards])
                scene_date = item["properties"]["datetime"][:10]
                for index, tract_id in enumerate(positive_tract_ids):
                    observed_area = float(valid_area[index])
                    if observed_area > 0:
                        status = "VALID_TEMPERATURE"
                        mean_temp = float(temp_sum[index] / observed_area)
                        variance = max(0.0, float(temp_sq_sum[index] / observed_area - mean_temp**2))
                        sd_temp = math.sqrt(variance)
                        ancillary_means = {
                            name: (
                                float(ancillary_numerators[name][index] / ancillary_denominators[name][index])
                                if ancillary_denominators[name][index] > 0
                                else None
                            )
                            for name in ancillary_numerators
                        }
                    else:
                        status = "NO_VALID_TEMPERATURE"
                        mean_temp = sd_temp = None
                        ancillary_means = {name: None for name in ancillary_numerators}
                    quality_reason_counts[status] += 1
                    scene_rows.append(
                        {
                            "state": state,
                            "tract_id": tract_id,
                            "item_id": item_id,
                            "observation_datetime": item["properties"]["datetime"],
                            "observation_date": scene_date,
                            "year": int(scene_date[:4]),
                            "fixed_land_area_m2": float(fixed_area[index]),
                            "valid_temperature_area_m2": observed_area,
                            "valid_fraction": observed_area / fixed_area[index],
                            "temperature_c_mean_valid": mean_temp,
                            "temperature_c_sd_valid": sd_temp,
                            "st_qa_mean_c": ancillary_means["uncertainty"],
                            "st_cdist_mean_km": ancillary_means["cloud_distance"],
                            "st_emis_mean": ancillary_means["emissivity"],
                            "st_emsd_mean": ancillary_means["emissivity_sd"],
                            "qa_pixel_cloud_or_shadow_or_fill_area_m2": float(cloud_flag_area[index]),
                            "qa_radsat_terrain_occlusion_area_m2": float(terrain_flag_area[index]),
                            "temperature_fill_area_m2": float(temp_fill_area[index]),
                            "status": status,
                        }
                    )
                writer.writerows(scene_rows)
                rows_written += len(scene_rows)
                output_stream.flush()
                progress({"event": "SCENE_COMPLETE", "position": position, "scene_count": len(state_items), "item_id": item_id, "tract_rows": tract_count})

        output_stream.close()
        output_stream = None
        require(rows_written == len(state_items) * len(positive_tract_ids), "Scene/tract output accounting")
        for path, record in evidence.items():
            stat = Path(path).stat()
            require((stat.st_size, stat.st_mtime_ns) == (record["bytes"], record["mtime_ns"]), f"Input changed: {path}")
            require(sha256(Path(path)) == record["sha256_file_bytes"], f"Input hash changed: {path}")
        audit = {
            "status": "PASS_STATE_LST_MEMORY_BOUNDED_DIAGNOSTIC_NOT_FORMAL_OUTCOME",
            "state": state,
            "years": list(selected_years),
            "candidate_scenes": len(state_items),
            "target_positive_fixed_land_tracts": len(positive_tract_ids),
            "rows": rows_written,
            "active_fixed_land_grid_cells": len(active_indices),
            "full_state_grid_cells_not_materialized": target_shape[0] * target_shape[1],
            "max_single_asset_destination_cells": max_destination_cells,
            "required_assets_read": list(REQUIRED_ASSET_KEYS),
            "quality_rule": "ST_B10>0; QA_PIXEL bits 0-5 zero; QA_RADSAT terrain-occlusion bit 11 zero; 2001 fixed land excludes NLCD 11 and 250",
            "temperature_scale": "ST_B10*0.00341802 + 149 - 273.15",
            "temperature_uncertainty_scale": "ST_QA*0.01 degrees C",
            "cloud_distance_scale": "ST_CDIST*0.01 km",
            "emissivity_scale": "ST_EMIS and ST_EMSD * 0.0001",
            "fixed_weight_grid_cell_area_m2": 900,
            "spatial_weighting": "prevalidated exact tract-by-urban-intersection-by-2001-fixed-land area weights",
            "processing_layout": "scene-intersection destination window plus sparse active fixed-land pixel vector",
            "landsat_to_fixed_grid_resampling": "nearest for temperature and all quality fields; diagnostic before sensitivity review",
            "quality_reason_counts": dict(quality_reason_counts),
            "controlled_single_asset_retry_count": len(retry_events),
            "controlled_single_asset_retry_events": retry_events,
            "candidate_selection": "ALL_CATALOG_CANDIDATES_NO_CLOUD_OR_TEMPERATURE_FILTER",
            "annual_outcome_constructed": False,
            "formal_sample_selected": False,
            "formal_model": "NOT_RUN",
            "signed_urls_persisted": False,
            "raw_data_modified": False,
            "output_sha256_file_bytes": sha256(output_path),
            "elapsed_seconds": time.time() - started,
        }
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
