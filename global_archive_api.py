"""
global_archive_api.py — FastAPI router for Global TC Archive (HURSAT-B1 IR imagery)

Include in tc_radar_api.py:
    from global_archive_api import router as global_router
    app.include_router(global_router, prefix="/global")

Endpoints:
    GET /global/hursat/meta?sid={SID}    — HURSAT frame list for a storm
    GET /global/hursat/frame?sid={SID}&frame_idx={N}  — Rendered IR frame as base64 PNG
    GET /global/health                   — Cache status

HURSAT-B1 data on NCEI is available via two path patterns:
  1. /data/hurricane-satellite-hursat-b1/archive/v06/{YEAR}/{SID}.nc  (combined file)
  2. /pub/data/satellite/hursat/{YEAR}/{SID}/B1-IR-BD/*.nc  (per-frame files)
This module tries both patterns with fallback.
"""

import base64
import gc
import io
import logging
import os
import re
import tempfile
from collections import OrderedDict
from datetime import datetime
from functools import lru_cache

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image

logger = logging.getLogger("global_archive")

router = APIRouter(tags=["global_archive"])

# ── Configuration ────────────────────────────────────────────

# Multiple NCEI URL patterns to try (in order of preference)
HURSAT_URLS = [
    # Pattern 1: v06 archive — combined NetCDF per storm (directory with .nc files)
    "https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/{year}/{sid}/",
    # Pattern 2: pub archive — per-frame NetCDFs in B1-IR-BD subfolder
    "https://www.ncei.noaa.gov/pub/data/satellite/hursat/{year}/{sid}/B1-IR-BD/",
    # Pattern 3: pub archive — sometimes B1-IR (no BD suffix)
    "https://www.ncei.noaa.gov/pub/data/satellite/hursat/{year}/{sid}/B1-IR/",
]

HURSAT_START_YEAR = 1978
HURSAT_END_YEAR = 2015

# ── Caches ───────────────────────────────────────────────────

# LRU cache for open xarray datasets (keyed by SID)
_ds_cache: OrderedDict = OrderedDict()
_DS_CACHE_MAX = 5  # Max storms in memory (~5–25 MB each)

# LRU cache for rendered PNG frames
_frame_cache: OrderedDict = OrderedDict()
_FRAME_CACHE_MAX = 200  # ~200 × 250KB ≈ 50 MB max

# LRU cache for HURSAT metadata (frame lists)
_meta_cache: OrderedDict = OrderedDict()
_META_CACHE_MAX = 500

# Cache for per-frame file URLs (keyed by SID)
_frame_urls_cache: OrderedDict = OrderedDict()
_FRAME_URLS_CACHE_MAX = 200


# ── IR Colormap (NOAA-style enhanced) ────────────────────────

IR_COLORMAP = [
    [0.0,    (8,   8,   8)],
    [0.15,   (40,  40,  40)],
    [0.35,   (90,  90,  90)],
    [0.45,   (20,  90,  200)],
    [0.55,   (0,   180, 255)],
    [0.65,   (0,   255, 180)],
    [0.72,   (255, 255, 0)],
    [0.80,   (255, 140, 0)],
    [0.88,   (255, 40,  40)],
    [0.94,   (200, 0,   200)],
    [1.0,    (255, 255, 255)],
]


@lru_cache(maxsize=1)
def _build_ir_lut():
    """Build 256-entry RGBA lookup table from IR_COLORMAP."""
    lut = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256):
        frac = i / 255.0
        for j in range(len(IR_COLORMAP) - 1):
            f0, c0 = IR_COLORMAP[j]
            f1, c1 = IR_COLORMAP[j + 1]
            if f0 <= frac <= f1:
                t = (frac - f0) / (f1 - f0) if f1 != f0 else 0
                r = int(c0[0] + t * (c1[0] - c0[0]))
                g = int(c0[1] + t * (c1[1] - c0[1]))
                b = int(c0[2] + t * (c1[2] - c0[2]))
                lut[i] = [r, g, b, 255]
                break
    return lut


_IR_LUT = _build_ir_lut()


def _render_ir_png(frame_2d, vmin=170.0, vmax=310.0):
    """Render a 2D brightness temperature array to a base64 PNG."""
    arr = np.asarray(frame_2d, dtype=np.float32)

    # Cold clouds (low Tb) → high index → bright colors
    frac = 1.0 - (arr - vmin) / (vmax - vmin)
    frac = np.clip(frac, 0.0, 1.0)
    indices = (frac * 255).astype(np.uint8)

    # Apply LUT
    rgba = _IR_LUT[indices]  # shape (H, W, 4)

    # Set NaN / invalid pixels to transparent
    mask = ~np.isfinite(arr) | (arr <= 0)
    rgba[mask] = [0, 0, 0, 0]

    # Flip vertically (NetCDF convention: first row = southernmost)
    rgba = rgba[::-1]

    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ── HURSAT Data Access ───────────────────────────────────────

def _parse_sid_year(sid: str) -> int:
    """Extract year from IBTrACS SID (e.g., '2005236N23285' → 2005)."""
    match = re.match(r"^(\d{4})", sid)
    if match:
        return int(match.group(1))
    return 0


def _discover_hursat_files(sid: str, year: int):
    """
    Discover HURSAT NetCDF files for a storm by trying multiple NCEI paths.
    Returns (dir_url, nc_files) tuple, or (None, []) if not found.
    """
    import requests

    headers = {
        "User-Agent": "TC-RADAR-API/1.0 (NOAA/HRD research; https://michaelfischerwx.github.io/TC-RADAR/)"
    }

    for url_pattern in HURSAT_URLS:
        dir_url = url_pattern.format(year=year, sid=sid)
        try:
            logger.info(f"HURSAT: trying {dir_url}")
            resp = requests.get(dir_url, timeout=15, headers=headers)
            logger.info(f"HURSAT: {dir_url} → HTTP {resp.status_code} ({len(resp.text)} bytes)")
            if resp.status_code != 200:
                continue

            # Parse HTML directory listing for .nc files
            nc_files = re.findall(r'href="([^"]*\.nc[^"]*)"', resp.text)
            # Filter out parent directory links, keep only filenames
            nc_files = [f for f in nc_files if '/' not in f and f.endswith('.nc')]

            if nc_files:
                # Sort by filename to get chronological order
                nc_files.sort()
                logger.info(f"HURSAT: found {len(nc_files)} .nc files at {dir_url}")
                return dir_url, nc_files

            logger.info(f"HURSAT: no .nc files parsed from {dir_url}")

        except Exception as e:
            logger.info(f"HURSAT: error checking {dir_url}: {e}")
            continue

    return None, []


def _get_hursat_combined_dataset(sid: str):
    """Fetch and cache a combined HURSAT NetCDF dataset (one file = all frames)."""
    if sid in _ds_cache:
        _ds_cache.move_to_end(sid)
        return _ds_cache[sid]

    import xarray as xr
    import requests

    year = _parse_sid_year(sid)
    if year < HURSAT_START_YEAR or year > HURSAT_END_YEAR:
        return None

    dir_url, nc_files = _discover_hursat_files(sid, year)
    if not dir_url or not nc_files:
        return None

    # If there's just one .nc file, it's likely a combined dataset
    # If there are many, they're per-frame files — store URLs for per-frame access
    if len(nc_files) > 1:
        # Per-frame mode — cache the file list and return None for combined ds
        frame_urls = [dir_url + f for f in nc_files]
        _frame_urls_cache[sid] = frame_urls
        if len(_frame_urls_cache) > _FRAME_URLS_CACHE_MAX:
            _frame_urls_cache.popitem(last=False)
        return "PER_FRAME"  # sentinel value

    # Single combined file
    nc_url = dir_url + nc_files[0]
    logger.info(f"Fetching combined HURSAT: {nc_url}")

    headers = {
        "User-Agent": "TC-RADAR-API/1.0 (NOAA/HRD research; https://michaelfischerwx.github.io/TC-RADAR/)"
    }

    try:
        resp = requests.get(nc_url, timeout=60, headers=headers)
        resp.raise_for_status()

        tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
        tmp.write(resp.content)
        tmp.close()

        ds = xr.open_dataset(tmp.name, engine="netcdf4")

        # Cache
        _ds_cache[sid] = ds
        if len(_ds_cache) > _DS_CACHE_MAX:
            old_sid, old_ds = _ds_cache.popitem(last=False)
            old_ds.close()
            gc.collect()

        return ds

    except Exception as e:
        logger.error(f"Failed to fetch combined HURSAT for {sid}: {e}")
        return None


def _parse_datetime_from_filename(filename: str):
    """
    Extract datetime from HURSAT filename.
    Common patterns:
        HURSAT_b1_v06_2005236N23285_d20050823_s181500.nc
        HURSAT-B1-2005236N23285-200508231815.nc
    """
    # Try pattern: d{YYYYMMDD}_s{HHMMSS}
    m = re.search(r'd(\d{8})_s(\d{6})', filename)
    if m:
        return f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}T{m.group(2)[:2]}:{m.group(2)[2:4]}:{m.group(2)[4:6]}"

    # Try pattern: {SID}-{YYYYMMDDHHMM}
    m = re.search(r'-(\d{12})\.nc', filename)
    if m:
        d = m.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}T{d[8:10]}:{d[10:12]}:00"

    # Try pattern: {YYYYMMDD}{HHMM} somewhere in the name
    m = re.search(r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})', filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}:{m.group(5)}:00"

    return filename


def _load_single_frame_nc(nc_url: str):
    """Download and open a single per-frame NetCDF file, return the IR 2D array."""
    import xarray as xr
    import requests

    headers = {
        "User-Agent": "TC-RADAR-API/1.0 (NOAA/HRD research; https://michaelfischerwx.github.io/TC-RADAR/)"
    }

    try:
        resp = requests.get(nc_url, timeout=30, headers=headers)
        resp.raise_for_status()

        tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
        tmp.write(resp.content)
        tmp.close()

        ds = xr.open_dataset(tmp.name, engine="netcdf4")

        # Find the IR variable
        var_name = None
        for candidate in ["irwin_cdr", "irwin", "irwin_2", "Tb", "IRWIN"]:
            if candidate in ds:
                var_name = candidate
                break

        if var_name is None:
            # Try to find any variable that looks like brightness temp
            for v in ds.data_vars:
                if ds[v].ndim >= 2:
                    var_name = v
                    break

        if var_name is None:
            ds.close()
            os.unlink(tmp.name)
            return None

        # Get the 2D frame (may have time dim of size 1)
        data = ds[var_name]
        if "time" in data.dims:
            frame = data.isel(time=0).values
        else:
            frame = data.values

        ds.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

        return frame

    except Exception as e:
        logger.warning(f"Failed to load frame {nc_url}: {e}")
        return None


# ── Endpoints ────────────────────────────────────────────────

@router.get("/hursat/meta")
def hursat_meta(sid: str = Query(..., description="IBTrACS storm ID")):
    """Return HURSAT frame metadata for a storm."""
    # Check cache
    if sid in _meta_cache:
        _meta_cache.move_to_end(sid)
        return JSONResponse(
            _meta_cache[sid],
            headers={"Cache-Control": "public, max-age=86400, immutable"},
        )

    year = _parse_sid_year(sid)
    if year < HURSAT_START_YEAR or year > HURSAT_END_YEAR:
        result = {
            "sid": sid,
            "available": False,
            "reason": f"HURSAT-B1 coverage is {HURSAT_START_YEAR}–{HURSAT_END_YEAR}",
        }
        return JSONResponse(result)

    # Try to discover files
    ds_or_sentinel = _get_hursat_combined_dataset(sid)

    if ds_or_sentinel == "PER_FRAME":
        # Per-frame mode — build metadata from filenames
        frame_urls = _frame_urls_cache.get(sid, [])
        if not frame_urls:
            result = {"sid": sid, "available": False, "reason": "Data not found on NCEI"}
            return JSONResponse(result)

        frames = []
        for i, url in enumerate(frame_urls):
            filename = url.rsplit("/", 1)[-1]
            dt = _parse_datetime_from_filename(filename)
            frames.append({"index": i, "datetime": dt})

        result = {
            "sid": sid,
            "available": True,
            "n_frames": len(frames),
            "frames": frames,
            "mode": "per_frame",
        }

    elif ds_or_sentinel is not None:
        # Combined dataset mode
        ds = ds_or_sentinel
        try:
            times = ds["time"].values if "time" in ds.dims else []
            n_frames = len(times)

            frames = []
            for i, t in enumerate(times):
                dt = str(np.datetime_as_string(t, unit="s")) if hasattr(t, "astype") else str(t)
                frames.append({"index": i, "datetime": dt})

            result = {
                "sid": sid,
                "available": True,
                "n_frames": n_frames,
                "frames": frames,
                "mode": "combined",
            }
        except Exception as e:
            logger.error(f"Error reading HURSAT metadata for {sid}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        result = {"sid": sid, "available": False, "reason": "Data not found on NCEI"}
        return JSONResponse(result)

    # Cache result
    _meta_cache[sid] = result
    if len(_meta_cache) > _META_CACHE_MAX:
        _meta_cache.popitem(last=False)

    return JSONResponse(
        result,
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )


@router.get("/hursat/frame")
def hursat_frame(
    sid: str = Query(..., description="IBTrACS storm ID"),
    frame_idx: int = Query(..., ge=0, description="Frame index (0-based)"),
):
    """Return a rendered IR frame as base64 PNG."""
    cache_key = (sid, frame_idx)

    # Check frame cache
    if cache_key in _frame_cache:
        _frame_cache.move_to_end(cache_key)
        return JSONResponse(
            _frame_cache[cache_key],
            headers={
                "Cache-Control": "public, max-age=86400, immutable",
                "X-Cache": "HIT",
            },
        )

    # Determine if per-frame or combined mode
    frame_urls = _frame_urls_cache.get(sid)

    if frame_urls:
        # Per-frame mode — download individual file
        if frame_idx >= len(frame_urls):
            raise HTTPException(
                status_code=404,
                detail=f"Frame index {frame_idx} out of range (0-{len(frame_urls)-1})",
            )

        nc_url = frame_urls[frame_idx]
        frame_2d = _load_single_frame_nc(nc_url)
        if frame_2d is None:
            raise HTTPException(status_code=500, detail="Failed to load frame data")

        filename = nc_url.rsplit("/", 1)[-1]
        frame_dt = _parse_datetime_from_filename(filename)

    else:
        # Combined dataset mode
        ds = _get_hursat_combined_dataset(sid)
        if ds is None or ds == "PER_FRAME":
            raise HTTPException(status_code=404, detail="HURSAT data not found")

        try:
            # Find IR variable
            var_name = "irwin_cdr"
            if var_name not in ds:
                for alt in ["irwin", "irwin_2", "Tb", "IRWIN"]:
                    if alt in ds:
                        var_name = alt
                        break
                else:
                    # Try any 3D variable
                    for v in ds.data_vars:
                        if ds[v].ndim >= 3:
                            var_name = v
                            break
                    else:
                        raise HTTPException(status_code=404, detail="No IR variable found")

            n_times = ds.dims.get("time", 0)
            if frame_idx >= n_times:
                raise HTTPException(
                    status_code=404,
                    detail=f"Frame index {frame_idx} out of range (0-{n_times-1})",
                )

            frame_2d = ds[var_name].isel(time=frame_idx).values

            frame_dt = ""
            if "time" in ds:
                t = ds["time"].values[frame_idx]
                frame_dt = str(np.datetime_as_string(t, unit="s"))

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error extracting frame {frame_idx} for {sid}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Render PNG
    png = _render_ir_png(frame_2d, vmin=170.0, vmax=310.0)

    result = {
        "sid": sid,
        "frame_idx": frame_idx,
        "datetime": frame_dt,
        "frame": png,
    }

    # Cache rendered frame
    _frame_cache[cache_key] = result
    if len(_frame_cache) > _FRAME_CACHE_MAX:
        _frame_cache.popitem(last=False)
        gc.collect()

    return JSONResponse(
        result,
        headers={
            "Cache-Control": "public, max-age=86400, immutable",
            "X-Cache": "MISS",
        },
    )


@router.get("/health")
def global_health():
    """Health check for global archive endpoints."""
    return {
        "status": "ok",
        "ds_cache_size": len(_ds_cache),
        "frame_cache_size": len(_frame_cache),
        "meta_cache_size": len(_meta_cache),
        "frame_urls_cache_size": len(_frame_urls_cache),
    }


@router.get("/hursat/debug")
def hursat_debug(sid: str = Query("1992230N11325", description="IBTrACS storm ID to test")):
    """Debug endpoint: test NCEI connectivity for a storm."""
    import requests

    year = _parse_sid_year(sid)
    headers = {
        "User-Agent": "TC-RADAR-API/1.0 (NOAA/HRD research; https://michaelfischerwx.github.io/TC-RADAR/)"
    }

    results = []
    for url_pattern in HURSAT_URLS:
        dir_url = url_pattern.format(year=year, sid=sid)
        try:
            resp = requests.get(dir_url, timeout=15, headers=headers)
            nc_files = re.findall(r'href="([^"]*\.nc[^"]*)"', resp.text) if resp.status_code == 200 else []
            nc_files = [f for f in nc_files if '/' not in f and f.endswith('.nc')]
            results.append({
                "url": dir_url,
                "status": resp.status_code,
                "content_length": len(resp.text),
                "nc_files_found": len(nc_files),
                "nc_files": nc_files[:5],  # first 5 for brevity
                "snippet": resp.text[:300] if resp.status_code == 200 else "",
            })
        except Exception as e:
            results.append({
                "url": dir_url,
                "error": str(e),
            })

    return {"sid": sid, "year": year, "url_checks": results}
