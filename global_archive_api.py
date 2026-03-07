"""
global_archive_api.py — FastAPI router for Global TC Archive (HURSAT-B1 IR imagery)

Include in tc_radar_api.py:
    from global_archive_api import router as global_router
    app.include_router(global_router, prefix="/global")

Endpoints:
    GET /global/hursat/meta?sid={SID}    — HURSAT frame list for a storm
    GET /global/hursat/frame?sid={SID}&frame_idx={N}  — Rendered IR frame as base64 PNG
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

HURSAT_BASE_URL = (
    "https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06"
)
HURSAT_START_YEAR = 1978
HURSAT_END_YEAR = 2015

# ── Caches ───────────────────────────────────────────────────

# LRU cache for open xarray datasets (keyed by SID)
_ds_cache: OrderedDict = OrderedDict()
_DS_CACHE_MAX = 5  # Max storms in memory (~5–25 MB each)

# LRU cache for rendered PNG frames
_frame_cache: OrderedDict = OrderedDict()
_FRAME_CACHE_MAX = 200  # ~200 × 250KB ≈ 50 MB max

# LRU cache for HURSAT metadata
_meta_cache: OrderedDict = OrderedDict()
_META_CACHE_MAX = 500


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
        # Find surrounding breakpoints
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


def _get_hursat_dataset(sid: str):
    """Fetch and cache HURSAT NetCDF dataset for a storm."""
    if sid in _ds_cache:
        _ds_cache.move_to_end(sid)
        return _ds_cache[sid]

    import xarray as xr
    import requests

    year = _parse_sid_year(sid)
    if year < HURSAT_START_YEAR or year > HURSAT_END_YEAR:
        return None

    # HURSAT-B1 v06 stores one NetCDF per storm
    # Directory: /v06/{YEAR}/{SID}/
    # File naming varies — we need to list the directory first
    dir_url = f"{HURSAT_BASE_URL}/{year}/{sid}/"

    try:
        # List files in the directory
        resp = requests.get(dir_url, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"HURSAT directory not found: {dir_url} (HTTP {resp.status_code})")
            return None

        # Parse HTML directory listing for .nc files
        nc_files = re.findall(r'href="([^"]+\.nc)"', resp.text)
        if not nc_files:
            logger.warning(f"No NetCDF files found in {dir_url}")
            return None

        # Usually there's one file per storm — take the first/largest
        nc_url = dir_url + nc_files[0]
        logger.info(f"Fetching HURSAT: {nc_url}")

        # Download to temp file (HURSAT files are small: 1-10 MB)
        resp = requests.get(nc_url, timeout=60)
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
        logger.error(f"Failed to fetch HURSAT for {sid}: {e}")
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

    ds = _get_hursat_dataset(sid)
    if ds is None:
        result = {"sid": sid, "available": False, "reason": "Data not found on NCEI"}
        return JSONResponse(result)

    # Extract frame info
    try:
        # HURSAT-B1 uses 'time' dimension
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
        }

        # Cache
        _meta_cache[sid] = result
        if len(_meta_cache) > _META_CACHE_MAX:
            _meta_cache.popitem(last=False)

        return JSONResponse(
            result,
            headers={"Cache-Control": "public, max-age=86400, immutable"},
        )

    except Exception as e:
        logger.error(f"Error reading HURSAT metadata for {sid}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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

    ds = _get_hursat_dataset(sid)
    if ds is None:
        raise HTTPException(status_code=404, detail="HURSAT data not found")

    try:
        # Primary IR variable: irwin_cdr (IR window, nadir observation)
        var_name = "irwin_cdr"
        if var_name not in ds:
            # Fallback to other IR variables
            for alt in ["irwin", "irwin_2", "Tb"]:
                if alt in ds:
                    var_name = alt
                    break
            else:
                raise HTTPException(status_code=404, detail="No IR variable found")

        n_times = ds.dims.get("time", 0)
        if frame_idx >= n_times:
            raise HTTPException(
                status_code=404,
                detail=f"Frame index {frame_idx} out of range (0-{n_times-1})",
            )

        # Extract 2D frame
        frame = ds[var_name].isel(time=frame_idx).values

        # Get datetime for this frame
        frame_dt = ""
        if "time" in ds:
            t = ds["time"].values[frame_idx]
            frame_dt = str(np.datetime_as_string(t, unit="s"))

        # Render PNG
        png = _render_ir_png(frame, vmin=170.0, vmax=310.0)

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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rendering HURSAT frame {frame_idx} for {sid}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def global_health():
    """Health check for global archive endpoints."""
    return {
        "status": "ok",
        "ds_cache_size": len(_ds_cache),
        "frame_cache_size": len(_frame_cache),
        "meta_cache_size": len(_meta_cache),
    }
