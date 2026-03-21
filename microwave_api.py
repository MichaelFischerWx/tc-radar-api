"""
microwave_api.py — TC-PRIMED Microwave Satellite Overlay Endpoints
===================================================================
FastAPI APIRouter that provides passive microwave satellite imagery
from the TC-PRIMED dataset (NOAA/CSU) hosted on public AWS S3.

Products:
    - 89 GHz PCT (Polarization-Corrected Temperature):
        PCT = 1.818 * TB_V − 0.818 * TB_H
      Available from: GMI, SSMIS, AMSR2, SSM/I, TMI
    - 37 GHz H-pol brightness temperature:
      Available from: GMI, SSMIS, AMSR2, SSM/I, TMI

Data source:
    s3://noaa-nesdis-tcprimed-pds/v01r01/final/{season}/{basin}/{ATCF_ID}/
    Files: TCPRIMED_v01r01-final_{ATCF_ID}_{SENSOR}_{PLATFORM}_{ORBIT}_{YYYYMMDDHHMMSS}.nc

Integration:
    from microwave_api import router as microwave_router
    app.include_router(microwave_router, prefix="/microwave")

Dependencies (all in deployment):
    fastapi, boto3, xarray, numpy, h5netcdf, matplotlib
"""

import base64
import io
import json
import logging
import os
import re
import threading
import time
from collections import defaultdict
from datetime import datetime as _dt, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

logger = logging.getLogger("microwave_api")

router = APIRouter()

# ---------------------------------------------------------------------------
# TC-PRIMED S3 configuration
# ---------------------------------------------------------------------------
TCPRIMED_BUCKET = "noaa-nesdis-tcprimed-pds"
TCPRIMED_PREFIX = "v01r01/final"

# Time window (hours) for matching overpasses to TDR analyses
OVERPASS_WINDOW_HOURS = 6

# Sensors that carry channels we need for 89 GHz PCT and 37 GHz
# Each entry: sensor_name -> dict of channel group info
SENSOR_INFO = {
    "GMI":   {"full_name": "GPM Microwave Imager",          "has_89": True, "has_37": True},
    "SSMIS": {"full_name": "Special Sensor MW Imager/Sounder", "has_89": True, "has_37": True},
    "AMSR2": {"full_name": "Advanced MW Scanning Radiometer 2", "has_89": True, "has_37": True},
    "SSMI":  {"full_name": "Special Sensor MW/Imager",       "has_89": True, "has_37": True},
    "TMI":   {"full_name": "TRMM Microwave Imager",          "has_89": True, "has_37": True},
    "ATMS":  {"full_name": "Advanced Technology MW Sounder",  "has_89": True, "has_37": False},
    "MHS":   {"full_name": "Microwave Humidity Sounder",      "has_89": True, "has_37": False},
}

# TC-PRIMED file naming pattern
# TCPRIMED_v01r01-final_{ATCF_ID}_{SENSOR}_{PLATFORM}_{ORBIT}_{YYYYMMDDHHMMSS}.nc
_FILE_PATTERN = re.compile(
    r"TCPRIMED_v01r01-final_"
    r"(?P<atcf_id>[A-Z]{2}\d+)_"
    r"(?P<sensor>[A-Z0-9]+)_"
    r"(?P<platform>[A-Za-z0-9_-]+)_"
    r"(?P<orbit>\d+)_"
    r"(?P<timestamp>\d{14})\.nc$"
)

# Basin code mapping: IBTrACS basin -> TC-PRIMED S3 directory name
# TC-PRIMED uses 2-letter ATCF basin codes as directory names
BASIN_MAP = {
    "NA": "AL",   # North Atlantic
    "EP": "EP",   # Eastern North Pacific
    "WP": "WP",   # Western North Pacific
    "NI": "IO",   # North Indian Ocean
    "SI": "SH",   # South Indian Ocean  -> Southern Hemisphere
    "SP": "SH",   # South Pacific       -> Southern Hemisphere
    "SA": "AL",   # South Atlantic (rare)
}

# ---------------------------------------------------------------------------
# Global index: built at startup, maps TC-RADAR cases to TC-PRIMED overpasses
# ---------------------------------------------------------------------------
_index_lock = threading.Lock()
_case_overpass_index: Dict[int, List[dict]] = {}   # case_index -> [overpass_info, ...]
_storm_overpass_index: Dict[str, List[dict]] = {}  # atcf_id -> [overpass_info, ...]
_atcf_map: Dict[Tuple[str, int], str] = {}         # (storm_name, year) -> atcf_id
_index_ready = threading.Event()
_index_error: Optional[str] = None


def _get_boto3_client():
    """Create an anonymous S3 client for the public TC-PRIMED bucket."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client("s3", config=Config(signature_version=UNSIGNED),
                        region_name="us-east-1")


def _parse_tcprimed_filename(filename: str) -> Optional[dict]:
    """Extract metadata from a TC-PRIMED filename."""
    basename = filename.rsplit("/", 1)[-1] if "/" in filename else filename
    m = _FILE_PATTERN.match(basename)
    if not m:
        return None
    ts_str = m.group("timestamp")
    try:
        dt = _dt.strptime(ts_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return {
        "atcf_id":   m.group("atcf_id"),
        "sensor":    m.group("sensor"),
        "platform":  m.group("platform"),
        "orbit":     m.group("orbit"),
        "datetime":  dt,
        "filename":  basename,
    }


def _build_atcf_map(ibtracs_path: str) -> Dict[Tuple[str, int], str]:
    """
    Build a mapping from (storm_name_upper, year) -> atcf_id
    using the IBTrACS storms JSON that's already part of this project.
    """
    mapping: Dict[Tuple[str, int], str] = {}
    try:
        with open(ibtracs_path, "r") as f:
            data = json.load(f)
        for storm in data.get("storms", []):
            atcf = storm.get("atcf_id")
            name = storm.get("name", "").upper()
            year = storm.get("year")
            if atcf and name and year:
                key = (name, year)
                # If multiple storms share name+year, keep the one with
                # the basin matching the ATCF prefix (AL, EP, etc.)
                if key not in mapping:
                    mapping[key] = atcf
        logger.info("ATCF map built: %d storm-year pairs", len(mapping))
    except Exception as e:
        logger.error("Failed to load IBTrACS for ATCF mapping: %s", e)
    return mapping


def _parse_case_datetime(case: dict) -> Optional[_dt]:
    """Parse a TC-RADAR case datetime string into a timezone-aware datetime."""
    dt_str = case.get("datetime", "")
    # Format: "2003-09-03 17:20 UTC"
    try:
        return _dt.strptime(dt_str, "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    # Fallback: reconstruct from fields
    try:
        return _dt(
            case["year"], case["month"], case["day"],
            case.get("hour", 0), case.get("minute", 0),
            tzinfo=timezone.utc,
        )
    except (KeyError, ValueError):
        return None


def _extract_annual_number(atcf_id: str) -> str:
    """
    Extract the 2-digit annual storm number from an ATCF ID.
    E.g. 'AL062018' -> '06', 'EP122020' -> '12', 'WP032019' -> '03'.
    """
    # ATCF format: BB##YYYY  (2-char basin + 2-digit number + 4-digit year)
    return atcf_id[2:4]


def _compute_tcprimed_season(atcf_id: str, year: int) -> int:
    """
    Compute the TC-PRIMED season directory year.

    For the Northern Hemisphere: season = calendar year.
    For the Southern Hemisphere: the season starts July 1, so the TC-PRIMED
    season is the calendar year + 1 (e.g., a storm in Dec 2019 is season 2020).

    Since IBTrACS 'year' already captures genesis year and TC-PRIMED follows
    the same convention as IBTrACS for SH storm years, we use the ATCF year
    directly — it already encodes the correct season.
    """
    # The 4-digit year in the ATCF ID IS the season year
    try:
        return int(atcf_id[4:8])
    except (ValueError, IndexError):
        return year


def _list_tcprimed_files_for_storm(s3_client, atcf_id: str, season: int) -> List[str]:
    """
    List all TC-PRIMED NetCDF files for a given ATCF ID.
    Returns list of S3 keys.

    TC-PRIMED S3 path structure:
        v01r01/final/{season}/{basin}/{annual_number}/
    where annual_number is the 2-digit storm number (e.g. '06'),
    NOT the full ATCF ID.
    """
    basin_code = atcf_id[:2]  # e.g. "AL", "EP", "WP"
    annual_num = _extract_annual_number(atcf_id)  # e.g. "06"
    tcprimed_season = _compute_tcprimed_season(atcf_id, season)

    prefix = f"{TCPRIMED_PREFIX}/{tcprimed_season}/{basin_code}/{annual_num}/"
    keys = []
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=TCPRIMED_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".nc"):
                    keys.append(key)
    except Exception as e:
        logger.warning("S3 list failed for %s/%s: %s", prefix, atcf_id, e)

    logger.debug("Listed %d files for %s at %s", len(keys), atcf_id, prefix)
    return keys


def _build_index_thread(metadata_path: str, ibtracs_path: str):
    """
    Background thread: builds the overpass index at startup.

    Steps:
    1. Load TC-RADAR metadata → extract unique (storm_name, year) pairs
    2. Load IBTrACS → map each pair to an ATCF ID
    3. For each unique ATCF ID, list TC-PRIMED files from S3
    4. For each TC-RADAR case, find overpasses within ±OVERPASS_WINDOW_HOURS
    """
    global _case_overpass_index, _storm_overpass_index, _atcf_map, _index_error

    try:
        logger.info("Microwave index build starting...")
        t0 = time.time()

        # Step 1: Load TC-RADAR metadata
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        cases = meta.get("cases", [])
        logger.info("Loaded %d TC-RADAR cases", len(cases))

        # Step 2: Build ATCF mapping
        atcf_map = _build_atcf_map(ibtracs_path)

        # Step 3: Group cases by (storm_name, year) -> list of cases
        storm_cases: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
        for case in cases:
            key = (case["storm_name"].upper(), case["year"])
            storm_cases[key] = storm_cases.get(key, [])
            storm_cases[key].append(case)

        # Step 4: For each unique ATCF ID, fetch overpass list from S3
        s3 = _get_boto3_client()
        unique_atcf: Dict[str, Tuple[str, int]] = {}  # atcf_id -> (name, year)
        for (name, year), case_list in storm_cases.items():
            atcf = atcf_map.get((name, year))
            if atcf and atcf not in unique_atcf:
                unique_atcf[atcf] = (name, year)

        logger.info("Querying TC-PRIMED for %d unique storms...", len(unique_atcf))

        storm_overpasses: Dict[str, List[dict]] = {}
        n_files_total = 0

        for atcf_id, (name, year) in unique_atcf.items():
            s3_keys = _list_tcprimed_files_for_storm(s3, atcf_id, year)
            overpasses = []
            for key in s3_keys:
                info = _parse_tcprimed_filename(key)
                if info is None:
                    continue
                # Verify this file belongs to our storm (directory may contain
                # files from other storms if annual numbers overlap across basins)
                if info["atcf_id"] != atcf_id:
                    continue
                # Only keep sensors that have useful channels
                sensor = info["sensor"]
                if sensor not in SENSOR_INFO:
                    continue
                info["s3_key"] = key
                info["has_89"] = SENSOR_INFO[sensor]["has_89"]
                info["has_37"] = SENSOR_INFO[sensor]["has_37"]
                info["sensor_full"] = SENSOR_INFO[sensor]["full_name"]
                overpasses.append(info)
            storm_overpasses[atcf_id] = overpasses
            n_files_total += len(overpasses)

        logger.info("Found %d total overpass files across %d storms",
                     n_files_total, len(storm_overpasses))

        # Step 5: For each TC-RADAR case, find temporally matched overpasses
        case_index: Dict[int, List[dict]] = {}
        window = timedelta(hours=OVERPASS_WINDOW_HOURS)

        for (name, year), case_list in storm_cases.items():
            atcf = atcf_map.get((name, year))
            if not atcf or atcf not in storm_overpasses:
                continue
            all_overpasses = storm_overpasses[atcf]
            for case in case_list:
                case_dt = _parse_case_datetime(case)
                if case_dt is None:
                    continue
                matched = []
                for op in all_overpasses:
                    offset = (op["datetime"] - case_dt).total_seconds()
                    if abs(offset) <= window.total_seconds():
                        matched.append({
                            "s3_key":    op["s3_key"],
                            "sensor":    op["sensor"],
                            "platform":  op["platform"],
                            "orbit":     op["orbit"],
                            "datetime":  op["datetime"].strftime("%Y-%m-%d %H:%M UTC"),
                            "offset_minutes": round(offset / 60.0, 1),
                            "has_89":    op["has_89"],
                            "has_37":    op["has_37"],
                            "sensor_full": op["sensor_full"],
                            "filename":  op["filename"],
                        })
                # Sort by absolute offset (closest first)
                matched.sort(key=lambda x: abs(x["offset_minutes"]))
                if matched:
                    case_index[case["case_index"]] = matched

        # Build storm-level index (serialisable)
        storm_idx: Dict[str, List[dict]] = {}
        for atcf_id, ops in storm_overpasses.items():
            storm_idx[atcf_id] = [
                {
                    "s3_key":    op["s3_key"],
                    "sensor":    op["sensor"],
                    "platform":  op["platform"],
                    "orbit":     op["orbit"],
                    "datetime":  op["datetime"].strftime("%Y-%m-%d %H:%M UTC"),
                    "has_89":    op["has_89"],
                    "has_37":    op["has_37"],
                    "sensor_full": op["sensor_full"],
                    "filename":  op["filename"],
                }
                for op in ops
            ]
            storm_idx[atcf_id].sort(key=lambda x: x["datetime"])

        # Commit to globals
        with _index_lock:
            _case_overpass_index.update(case_index)
            _storm_overpass_index.update(storm_idx)
            _atcf_map.update(atcf_map)

        elapsed = time.time() - t0
        logger.info(
            "Microwave index ready: %d cases with overpasses, "
            "%d storms indexed, %.1fs",
            len(case_index), len(storm_idx), elapsed,
        )

    except Exception as e:
        logger.exception("Microwave index build failed: %s", e)
        _index_error = str(e)
    finally:
        _index_ready.set()


def start_index_build(metadata_path: str = None, ibtracs_path: str = None):
    """
    Kick off the background index build. Called from the main app at startup.
    Paths default to the files alongside this module.
    """
    base = Path(__file__).parent
    if metadata_path is None:
        metadata_path = str(base / "tc_radar_metadata.json")
    if ibtracs_path is None:
        ibtracs_path = str(base / "ibtracs_storms.json")

    thread = threading.Thread(
        target=_build_index_thread,
        args=(metadata_path, ibtracs_path),
        daemon=True,
        name="microwave-index-builder",
    )
    thread.start()
    logger.info("Microwave index build thread started")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/status")
async def microwave_status():
    """Check whether the overpass index has finished building."""
    ready = _index_ready.is_set()
    with _index_lock:
        n_cases = len(_case_overpass_index)
        n_storms = len(_storm_overpass_index)
    return JSONResponse({
        "ready": ready,
        "error": _index_error,
        "cases_with_overpasses": n_cases,
        "storms_indexed": n_storms,
    })


@router.get("/overpasses")
async def get_overpasses(
    case_index: int = Query(..., description="TC-RADAR case index"),
):
    """
    Return all TC-PRIMED microwave overpasses within ±6 hours of a TC-RADAR case.
    Each overpass includes sensor, platform, timestamp, offset from TDR analysis,
    and available products (89 GHz PCT, 37 GHz).
    """
    if not _index_ready.is_set():
        raise HTTPException(503, "Microwave overpass index is still building. Try again shortly.")

    with _index_lock:
        overpasses = _case_overpass_index.get(case_index, [])

    return JSONResponse({
        "case_index": case_index,
        "overpasses": overpasses,
        "count": len(overpasses),
        "window_hours": OVERPASS_WINDOW_HOURS,
    })


@router.get("/storm_overpasses")
async def get_storm_overpasses(
    atcf_id: str = Query(None, description="ATCF storm ID, e.g. AL062018"),
    storm_name: str = Query(None, description="Storm name (alternative to atcf_id)"),
    year: int = Query(None, description="Storm year (required if using storm_name)"),
):
    """
    Return ALL TC-PRIMED overpasses for a full storm lifecycle.
    Supports both global archive (query by ATCF ID) and archive mode
    (query by storm_name + year, which gets mapped to ATCF ID internally).
    """
    if not _index_ready.is_set():
        raise HTTPException(503, "Microwave overpass index is still building. Try again shortly.")

    # Resolve ATCF ID
    resolved_atcf = atcf_id
    if not resolved_atcf and storm_name and year:
        with _index_lock:
            resolved_atcf = _atcf_map.get((storm_name.upper(), year))
        if not resolved_atcf:
            return JSONResponse({
                "atcf_id": None,
                "storm_name": storm_name,
                "year": year,
                "overpasses": [],
                "count": 0,
                "message": "No ATCF ID found for this storm",
            })

    if not resolved_atcf:
        raise HTTPException(400, "Provide either atcf_id or storm_name+year")

    resolved_atcf = resolved_atcf.upper()
    with _index_lock:
        overpasses = _storm_overpass_index.get(resolved_atcf, [])

    return JSONResponse({
        "atcf_id": resolved_atcf,
        "overpasses": overpasses,
        "count": len(overpasses),
    })


@router.get("/data")
async def get_microwave_data(
    s3_key: str = Query(..., description="Full S3 key of the TC-PRIMED NetCDF file"),
    product: str = Query("89pct", description="Product: '89pct' or '37h'"),
    center_lat: float = Query(None, description="Storm center latitude (for regridding)"),
    center_lon: float = Query(None, description="Storm center longitude (for regridding)"),
    radius_km: float = Query(500.0, description="Radius in km for the output domain"),
):
    """
    Fetch a TC-PRIMED NetCDF file from S3, compute the requested product,
    and return storm-centered gridded data suitable for Leaflet/Plotly overlay.

    Products:
        89pct — 89 GHz Polarization-Corrected Temperature
                PCT = 1.818 * TB_V − 0.818 * TB_H
        37h   — 37 GHz H-pol brightness temperature

    Returns:
        JSON with lat/lon grids, data values, metadata for frontend rendering.
    """
    import xarray as xr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if product not in ("89pct", "37h"):
        raise HTTPException(400, "product must be '89pct' or '37h'")

    # Validate S3 key
    if not s3_key.startswith(TCPRIMED_PREFIX) or not s3_key.endswith(".nc"):
        raise HTTPException(400, "Invalid S3 key")

    # Parse filename to know the sensor
    info = _parse_tcprimed_filename(s3_key)
    if info is None:
        raise HTTPException(400, "Cannot parse TC-PRIMED filename from key")

    sensor = info["sensor"]
    if sensor not in SENSOR_INFO:
        raise HTTPException(400, f"Unsupported sensor: {sensor}")

    if product == "37h" and not SENSOR_INFO[sensor]["has_37"]:
        raise HTTPException(400, f"Sensor {sensor} does not have 37 GHz channels")

    try:
        # Open the NetCDF from S3
        import fsspec
        s3_url = f"s3://{TCPRIMED_BUCKET}/{s3_key}"
        fs = fsspec.filesystem("s3", anon=True)

        with fs.open(s3_url, "rb") as f:
            ds = xr.open_dataset(f, engine="h5netcdf", group="")

        # TC-PRIMED files have a hierarchical structure with groups.
        # The swath data is typically in sensor-specific groups.
        # We need to find the right group for brightness temperatures.
        #
        # Common group structure:
        #   / (root) — storm metadata (lat, lon, time, etc.)
        #   /{SENSOR}/brightness_temperature — TB channels
        #   /interpolation/ — storm-relative grids
        #
        # Let's try the interpolation group first (pre-gridded to storm center)
        # then fall back to swath data.

        # Close root and try interpolation group
        ds.close()

        data_dict = None

        # Try the storm-centered interpolated grid first
        try:
            with fs.open(s3_url, "rb") as f:
                ds_interp = xr.open_dataset(
                    f, engine="h5netcdf",
                    group=f"/{sensor}/interpolation"
                )

            if product == "89pct":
                data_dict = _compute_89pct_interpolated(ds_interp, sensor)
            else:
                data_dict = _compute_37h_interpolated(ds_interp, sensor)

            ds_interp.close()

        except Exception as e_interp:
            logger.warning("Interpolated grid not found for %s: %s, trying swath",
                          sensor, e_interp)

            # Fall back to swath-level brightness temperatures
            try:
                with fs.open(s3_url, "rb") as f:
                    ds_bt = xr.open_dataset(
                        f, engine="h5netcdf",
                        group=f"/{sensor}/brightness_temperature"
                    )
                with fs.open(s3_url, "rb") as f:
                    ds_geo = xr.open_dataset(
                        f, engine="h5netcdf",
                        group=f"/{sensor}/geolocation"
                    )

                if product == "89pct":
                    data_dict = _compute_89pct_swath(ds_bt, ds_geo, sensor)
                else:
                    data_dict = _compute_37h_swath(ds_bt, ds_geo, sensor)

                ds_bt.close()
                ds_geo.close()

            except Exception as e_swath:
                logger.error("Failed to read swath data: %s", e_swath)
                raise HTTPException(500, f"Could not extract {product} data: {e_swath}")

        if data_dict is None:
            raise HTTPException(500, "No data could be extracted")

        # Generate a PNG image for efficient overlay
        png_b64 = _render_product_image(data_dict, product)

        return JSONResponse({
            "product": product,
            "sensor": sensor,
            "platform": info["platform"],
            "datetime": info["datetime"].strftime("%Y-%m-%d %H:%M UTC"),
            "image_b64": png_b64,
            "bounds": data_dict.get("bounds"),  # [[south, west], [north, east]]
            "center_lat": data_dict.get("center_lat"),
            "center_lon": data_dict.get("center_lon"),
            "grid": {
                "nx": data_dict.get("nx"),
                "ny": data_dict.get("ny"),
                "dx_km": data_dict.get("dx_km"),
            },
            "stats": data_dict.get("stats", {}),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error fetching microwave data: %s", e)
        raise HTTPException(500, f"Error processing microwave data: {e}")


# ---------------------------------------------------------------------------
# Product computation helpers
# ---------------------------------------------------------------------------

# TC-PRIMED channel naming conventions (approximate — varies by sensor)
# 89 GHz channels:  '89V', '89H' or '89.0V', '89.0H' or 'S1' (ATMS ch16~89GHz)
# 37 GHz channels:  '37V', '37H' or '36.5V', '36.5H' or '36V', '36H'

_89GHZ_PATTERNS = {
    "GMI":   {"v": "89V", "h": "89H"},
    "SSMIS": {"v": "91V", "h": "91H"},      # SSMIS has 91 GHz (close proxy for 89)
    "AMSR2": {"v": "89V_A", "h": "89H_A"},  # AMSR2 has A/B scan; use A
    "SSMI":  {"v": "85V", "h": "85H"},       # SSM/I has 85 GHz
    "TMI":   {"v": "85V", "h": "85H"},       # TMI has 85 GHz
    "ATMS":  {"v": "88V", "h": None},        # ATMS ch16 ~88 GHz, V-pol only
    "MHS":   {"v": "89V", "h": None},        # MHS ch1 ~89 GHz, V-pol only
}

_37GHZ_PATTERNS = {
    "GMI":   {"v": "37V", "h": "37H"},
    "SSMIS": {"v": "37V", "h": "37H"},
    "AMSR2": {"v": "37V", "h": "37H"},
    "SSMI":  {"v": "37V", "h": "37H"},
    "TMI":   {"v": "37V", "h": "37H"},
}


def _find_channel_var(ds, sensor: str, freq_ghz: int, pol: str) -> Optional[str]:
    """
    Search the dataset variables for one that matches the desired
    frequency and polarization. TC-PRIMED naming can vary, so we
    try several patterns.
    """
    patterns_dict = _89GHZ_PATTERNS if freq_ghz >= 85 else _37GHZ_PATTERNS
    sensor_info = patterns_dict.get(sensor, {})
    expected = sensor_info.get(pol.lower())

    if expected is None:
        return None

    # Direct match
    for var_name in ds.data_vars:
        vn_upper = var_name.upper().replace(".", "").replace("_", "")
        ex_upper = expected.upper().replace(".", "").replace("_", "")
        if ex_upper in vn_upper:
            return var_name

    # Fuzzy search — look for frequency number + polarization
    freq_str = str(freq_ghz)
    for var_name in ds.data_vars:
        vn = var_name.upper()
        if freq_str in vn and pol.upper() in vn:
            return var_name

    # Last resort — check for 'brightness_temperature' with channel dim
    if "brightness_temperature" in ds.data_vars:
        return "brightness_temperature"

    return None


def _compute_89pct_interpolated(ds, sensor: str) -> dict:
    """
    Compute 89 GHz PCT from the interpolated (storm-centered) grid.
    Returns dict with data array, lats, lons, bounds.
    """
    v_name = _find_channel_var(ds, sensor, 89, "V")
    h_name = _find_channel_var(ds, sensor, 89, "H")

    if v_name and h_name and v_name != h_name:
        tb_v = ds[v_name].values.astype(np.float32)
        tb_h = ds[h_name].values.astype(np.float32)
        pct = 1.818 * tb_v - 0.818 * tb_h
    elif v_name:
        # H-pol not available (ATMS, MHS) — just use V-pol as proxy
        pct = ds[v_name].values.astype(np.float32)
    else:
        raise ValueError(f"Cannot find 89 GHz channels for sensor {sensor}")

    # Get coordinate arrays
    result = _extract_grid_info(ds, pct)
    result["product_label"] = "89 GHz PCT (K)"
    result["stats"] = {
        "min": float(np.nanmin(pct)),
        "max": float(np.nanmax(pct)),
        "mean": float(np.nanmean(pct)),
    }
    return result


def _compute_37h_interpolated(ds, sensor: str) -> dict:
    """
    Compute 37 GHz H-pol from the interpolated grid.
    """
    h_name = _find_channel_var(ds, sensor, 37, "H")
    if not h_name:
        raise ValueError(f"Cannot find 37 GHz H-pol for sensor {sensor}")

    tb_h = ds[h_name].values.astype(np.float32)

    result = _extract_grid_info(ds, tb_h)
    result["product_label"] = "37 GHz H-pol TB (K)"
    result["stats"] = {
        "min": float(np.nanmin(tb_h)),
        "max": float(np.nanmax(tb_h)),
        "mean": float(np.nanmean(tb_h)),
    }
    return result


def _compute_89pct_swath(ds_bt, ds_geo, sensor: str) -> dict:
    """
    Compute 89 GHz PCT from swath-level data and regrid to a regular grid.
    """
    v_name = _find_channel_var(ds_bt, sensor, 89, "V")
    h_name = _find_channel_var(ds_bt, sensor, 89, "H")

    if v_name and h_name and v_name != h_name:
        tb_v = ds_bt[v_name].values.astype(np.float32)
        tb_h = ds_bt[h_name].values.astype(np.float32)
        data = 1.818 * tb_v - 0.818 * tb_h
    elif v_name:
        data = ds_bt[v_name].values.astype(np.float32)
    else:
        raise ValueError(f"Cannot find 89 GHz channels for sensor {sensor}")

    lats, lons = _get_swath_geolocation(ds_geo)
    gridded = _regrid_swath(data, lats, lons)
    gridded["product_label"] = "89 GHz PCT (K)"
    gridded["stats"] = {
        "min": float(np.nanmin(data[np.isfinite(data)])) if np.any(np.isfinite(data)) else None,
        "max": float(np.nanmax(data[np.isfinite(data)])) if np.any(np.isfinite(data)) else None,
    }
    return gridded


def _compute_37h_swath(ds_bt, ds_geo, sensor: str) -> dict:
    """
    Compute 37 GHz H-pol from swath data and regrid.
    """
    h_name = _find_channel_var(ds_bt, sensor, 37, "H")
    if not h_name:
        raise ValueError(f"Cannot find 37 GHz H-pol for sensor {sensor}")

    data = ds_bt[h_name].values.astype(np.float32)
    lats, lons = _get_swath_geolocation(ds_geo)
    gridded = _regrid_swath(data, lats, lons)
    gridded["product_label"] = "37 GHz H-pol TB (K)"
    gridded["stats"] = {
        "min": float(np.nanmin(data[np.isfinite(data)])) if np.any(np.isfinite(data)) else None,
        "max": float(np.nanmax(data[np.isfinite(data)])) if np.any(np.isfinite(data)) else None,
    }
    return gridded


def _get_swath_geolocation(ds_geo) -> Tuple[np.ndarray, np.ndarray]:
    """Extract lat/lon arrays from the geolocation group."""
    lat_candidates = ["latitude", "lat", "Latitude"]
    lon_candidates = ["longitude", "lon", "Longitude"]
    lats = lons = None
    for name in lat_candidates:
        if name in ds_geo.data_vars or name in ds_geo.coords:
            lats = ds_geo[name].values
            break
    for name in lon_candidates:
        if name in ds_geo.data_vars or name in ds_geo.coords:
            lons = ds_geo[name].values
            break
    if lats is None or lons is None:
        raise ValueError("Cannot find lat/lon in geolocation group")
    return lats, lons


def _extract_grid_info(ds, data: np.ndarray) -> dict:
    """
    Extract grid coordinate info from an interpolated (storm-centered) dataset.
    TC-PRIMED interpolated grids typically use x_distance, y_distance (km) coords,
    plus a storm center lat/lon attribute.
    """
    # Try to get storm-relative coordinates
    x_km = y_km = None
    for xname in ["x_distance", "x", "across_track"]:
        if xname in ds.coords or xname in ds.dims:
            x_km = ds[xname].values
            break
    for yname in ["y_distance", "y", "along_track"]:
        if yname in ds.coords or yname in ds.dims:
            y_km = ds[yname].values
            break

    # Get storm center from attributes
    center_lat = ds.attrs.get("storm_latitude", ds.attrs.get("center_latitude", 0.0))
    center_lon = ds.attrs.get("storm_longitude", ds.attrs.get("center_longitude", 0.0))

    if x_km is not None and y_km is not None:
        # Convert km offsets to lat/lon
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(np.radians(center_lat))
        lat_arr = center_lat + y_km / km_per_deg_lat
        lon_arr = center_lon + x_km / km_per_deg_lon
    else:
        # Try direct lat/lon coords
        for lname in ["latitude", "lat"]:
            if lname in ds.coords:
                lat_arr = ds[lname].values
                break
        else:
            lat_arr = np.arange(data.shape[0])
        for lname in ["longitude", "lon"]:
            if lname in ds.coords:
                lon_arr = ds[lname].values
                break
        else:
            lon_arr = np.arange(data.shape[1] if data.ndim > 1 else data.shape[0])

    south = float(np.min(lat_arr))
    north = float(np.max(lat_arr))
    west = float(np.min(lon_arr))
    east = float(np.max(lon_arr))

    return {
        "data": data,
        "center_lat": float(center_lat),
        "center_lon": float(center_lon),
        "bounds": [[south, west], [north, east]],
        "nx": data.shape[1] if data.ndim > 1 else data.shape[0],
        "ny": data.shape[0],
        "dx_km": float(np.abs(x_km[1] - x_km[0])) if x_km is not None and len(x_km) > 1 else None,
    }


def _regrid_swath(data: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                  grid_res_deg: float = 0.02) -> dict:
    """
    Regrid irregular swath data onto a regular lat/lon grid using
    nearest-neighbor binning. Fast and sufficient for image overlay.
    """
    from scipy.interpolate import griddata

    # Flatten
    mask = np.isfinite(data.ravel()) & np.isfinite(lats.ravel()) & np.isfinite(lons.ravel())
    flat_data = data.ravel()[mask]
    flat_lat = lats.ravel()[mask]
    flat_lon = lons.ravel()[mask]

    if len(flat_data) < 10:
        raise ValueError("Insufficient valid data points for regridding")

    # Define regular grid
    lat_min, lat_max = float(flat_lat.min()), float(flat_lat.max())
    lon_min, lon_max = float(flat_lon.min()), float(flat_lon.max())

    # Limit grid size to prevent memory issues
    max_grid_dim = 500
    n_lat = min(int((lat_max - lat_min) / grid_res_deg) + 1, max_grid_dim)
    n_lon = min(int((lon_max - lon_min) / grid_res_deg) + 1, max_grid_dim)

    grid_lat = np.linspace(lat_min, lat_max, n_lat)
    grid_lon = np.linspace(lon_min, lon_max, n_lon)
    glon, glat = np.meshgrid(grid_lon, grid_lat)

    # Regrid
    gridded = griddata(
        (flat_lon, flat_lat), flat_data,
        (glon, glat),
        method="nearest",
    )

    center_lat = (lat_min + lat_max) / 2.0
    center_lon = (lon_min + lon_max) / 2.0

    return {
        "data": gridded,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
        "nx": n_lon,
        "ny": n_lat,
        "dx_km": grid_res_deg * 111.0,
    }


# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------

def _render_product_image(data_dict: dict, product: str) -> str:
    """
    Render the gridded product data as a transparent PNG (base64-encoded)
    suitable for Leaflet image overlay.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    data = data_dict["data"]
    if data is None:
        return ""

    # Ensure 2D
    if data.ndim == 1:
        return ""

    ny, nx = data.shape

    # Set up colormaps and value ranges
    if product == "89pct":
        # 89 GHz PCT: low values = deep convection (ice scattering)
        # Typical range: ~80 K (intense convection) to ~280 K (warm ocean)
        vmin, vmax = 80, 280
        # Use reversed colormap: blue/purple for cold (convective), warm for clear
        cmap = plt.cm.get_cmap("RdYlBu")  # already warm-to-cool
    else:
        # 37 GHz H-pol: warm values = warm rain, low values = ice/land
        vmin, vmax = 130, 280
        cmap = plt.cm.get_cmap("RdYlBu_r")

    # Create figure with transparent background
    dpi = 100
    fig_w = nx / dpi
    fig_h = ny / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()

    # Mask invalid values and set transparency
    masked = np.ma.masked_invalid(data)
    cmap_copy = cmap.copy()
    cmap_copy.set_bad(alpha=0.0)

    ax.imshow(
        masked,
        origin="lower",
        cmap=cmap_copy,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True, bbox_inches="tight",
                pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------------------------------------------------------------------------
# Colorbar endpoint (for frontend legend)
# ---------------------------------------------------------------------------

@router.get("/colorbar")
async def get_colorbar(
    product: str = Query("89pct", description="Product: '89pct' or '37h'"),
):
    """
    Return a standalone colorbar image (PNG, base64) for the given product,
    suitable for rendering as a map legend.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if product == "89pct":
        vmin, vmax = 80, 280
        cmap = plt.cm.get_cmap("RdYlBu")
        label = "89 GHz PCT (K)"
    elif product == "37h":
        vmin, vmax = 130, 280
        cmap = plt.cm.get_cmap("RdYlBu_r")
        label = "37 GHz H-pol TB (K)"
    else:
        raise HTTPException(400, "product must be '89pct' or '37h'")

    fig, ax = plt.subplots(figsize=(4, 0.4))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")

    return JSONResponse({"image_b64": b64, "product": product, "label": label})


# ---------------------------------------------------------------------------
# Realtime support: match a real-time TDR case to TC-PRIMED overpasses
# ---------------------------------------------------------------------------

@router.get("/realtime_overpasses")
async def get_realtime_overpasses(
    storm_name: str = Query(..., description="Storm name from real-time TDR"),
    year: int = Query(..., description="Year"),
    analysis_time: str = Query(..., description="TDR analysis time, ISO format"),
):
    """
    For real-time mode: given a storm name, year, and TDR analysis time,
    find matching TC-PRIMED overpasses. This works by looking up the ATCF ID
    from the IBTrACS mapping and then filtering the storm's overpasses
    by the analysis time window.
    """
    if not _index_ready.is_set():
        raise HTTPException(503, "Microwave overpass index is still building.")

    # Parse analysis time
    try:
        analysis_dt = _dt.fromisoformat(analysis_time.replace("Z", "+00:00"))
        if analysis_dt.tzinfo is None:
            analysis_dt = analysis_dt.replace(tzinfo=timezone.utc)
    except ValueError:
        raise HTTPException(400, "Invalid analysis_time format. Use ISO 8601.")

    # Look up ATCF ID
    with _index_lock:
        atcf = _atcf_map.get((storm_name.upper(), year))

    if not atcf:
        return JSONResponse({
            "storm_name": storm_name,
            "year": year,
            "overpasses": [],
            "count": 0,
            "message": "No ATCF ID found for this storm",
        })

    # Get all storm overpasses and filter by time window
    with _index_lock:
        all_ops = _storm_overpass_index.get(atcf, [])

    window = timedelta(hours=OVERPASS_WINDOW_HOURS)
    matched = []
    for op in all_ops:
        op_dt = _dt.strptime(op["datetime"], "%Y-%m-%d %H:%M UTC").replace(
            tzinfo=timezone.utc
        )
        offset = (op_dt - analysis_dt).total_seconds()
        if abs(offset) <= window.total_seconds():
            entry = dict(op)
            entry["offset_minutes"] = round(offset / 60.0, 1)
            matched.append(entry)

    matched.sort(key=lambda x: abs(x["offset_minutes"]))

    return JSONResponse({
        "storm_name": storm_name,
        "year": year,
        "atcf_id": atcf,
        "analysis_time": analysis_time,
        "overpasses": matched,
        "count": len(matched),
        "window_hours": OVERPASS_WINDOW_HOURS,
    })
