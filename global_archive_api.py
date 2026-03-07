"""
global_archive_api.py — FastAPI router for Global TC Archive (HURSAT-B1 IR imagery)

Include in tc_radar_api.py:
    from global_archive_api import router as global_router
    app.include_router(global_router, prefix="/global")

Endpoints:
    GET /global/hursat/meta?sid={SID}    — HURSAT frame list for a storm
    GET /global/hursat/frame?sid={SID}&frame_idx={N}  — Rendered IR frame as base64 PNG
    GET /global/health                   — Cache status
    GET /global/hursat/debug?sid={SID}   — Debug NCEI connectivity

HURSAT-B1 v06 data on NCEI is stored as tar.gz archives:
    https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/{YEAR}/
    HURSAT_b1_v06_{SID}_{NAME}_c{DATE}.tar.gz
Each tar.gz contains multiple per-frame NetCDF files.
"""

import base64
import gc
import io
import logging
import os
import re
import tarfile
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

HURSAT_V06_BASE = (
    "https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06"
)

HURSAT_START_YEAR = 1978
HURSAT_END_YEAR = 2015

_HTTP_HEADERS = {
    "User-Agent": "TC-RADAR-API/1.0 (NOAA/HRD research; https://michaelfischerwx.github.io/TC-RADAR/)"
}

# ── Caches ───────────────────────────────────────────────────

# Cache for extracted NetCDF file paths from tar.gz (keyed by SID)
# Value: list of (datetime_str, tmp_nc_path) tuples, sorted chronologically
_extracted_cache: OrderedDict = OrderedDict()
_EXTRACTED_CACHE_MAX = 5  # Max storms extracted at once

# LRU cache for rendered PNG frames
_frame_cache: OrderedDict = OrderedDict()
_FRAME_CACHE_MAX = 200  # ~200 × 250KB ≈ 50 MB max

# LRU cache for HURSAT metadata (frame lists)
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


def _find_tarball_url(sid: str, year: int) -> str | None:
    """
    Find the tar.gz URL for a storm on NCEI.
    Scans the year directory listing for a file containing the SID.
    """
    import requests

    year_url = f"{HURSAT_V06_BASE}/{year}/"

    try:
        logger.info(f"HURSAT: listing {year_url}")
        resp = requests.get(year_url, timeout=20, headers=_HTTP_HEADERS)
        if resp.status_code != 200:
            logger.info(f"HURSAT: year listing returned HTTP {resp.status_code}")
            return None

        # Find tar.gz files matching this SID
        # Pattern: HURSAT_b1_v06_{SID}_{NAME}_c{DATE}.tar.gz
        pattern = rf'href="(HURSAT_b1_v06_{re.escape(sid)}_[^"]+\.tar\.gz)"'
        matches = re.findall(pattern, resp.text)

        if matches:
            tarball_url = year_url + matches[0]
            logger.info(f"HURSAT: found tarball {matches[0]}")
            return tarball_url

        logger.info(f"HURSAT: no tarball found for SID {sid} in {year_url}")
        return None

    except Exception as e:
        logger.error(f"HURSAT: error listing {year_url}: {e}")
        return None


def _extract_tarball(sid: str, tarball_url: str) -> list:
    """
    Download a HURSAT tar.gz, extract NetCDF files to /tmp, return sorted list
    of (datetime_str, nc_path) tuples.
    """
    import requests

    try:
        logger.info(f"HURSAT: downloading {tarball_url}")
        resp = requests.get(tarball_url, timeout=120, headers=_HTTP_HEADERS,
                            stream=True)
        resp.raise_for_status()

        # Write to temp file
        tmp_tar = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
        for chunk in resp.iter_content(chunk_size=1024 * 64):
            tmp_tar.write(chunk)
        tmp_tar.close()
        tar_size = os.path.getsize(tmp_tar.name)
        logger.info(f"HURSAT: downloaded {tar_size / 1024 / 1024:.1f} MB")

        # Extract .nc files to a temp directory
        extract_dir = tempfile.mkdtemp(prefix=f"hursat_{sid}_")

        with tarfile.open(tmp_tar.name, "r:gz") as tar:
            # Security: only extract .nc files, no path traversal
            nc_members = [
                m for m in tar.getmembers()
                if m.name.endswith(".nc") and not m.name.startswith("/")
                and ".." not in m.name
            ]
            logger.info(f"HURSAT: tar contains {len(nc_members)} .nc files")

            frames = []
            for member in nc_members:
                # Extract to flat directory
                member.name = os.path.basename(member.name)
                tar.extract(member, path=extract_dir)
                nc_path = os.path.join(extract_dir, member.name)

                # Parse datetime from filename
                dt_str = _parse_datetime_from_filename(member.name)
                frames.append((dt_str, nc_path))

        # Clean up tar file
        os.unlink(tmp_tar.name)

        # Sort by datetime
        frames.sort(key=lambda x: x[0])

        # Cache extracted paths
        _evict_extracted_cache()
        _extracted_cache[sid] = frames
        if len(_extracted_cache) > _EXTRACTED_CACHE_MAX:
            _evict_extracted_cache()

        return frames

    except Exception as e:
        logger.error(f"HURSAT: failed to extract {tarball_url}: {e}")
        return []


def _evict_extracted_cache():
    """Evict oldest entry from extracted cache and clean up tmp files."""
    if len(_extracted_cache) >= _EXTRACTED_CACHE_MAX:
        old_sid, old_frames = _extracted_cache.popitem(last=False)
        for _, nc_path in old_frames:
            try:
                os.unlink(nc_path)
            except OSError:
                pass
        # Try to remove the directory
        if old_frames:
            try:
                os.rmdir(os.path.dirname(old_frames[0][1]))
            except OSError:
                pass
        gc.collect()
        logger.info(f"HURSAT: evicted cache for {old_sid}")


def _get_extracted_frames(sid: str) -> list | None:
    """Get extracted frames for a storm, downloading if needed."""
    if sid in _extracted_cache:
        _extracted_cache.move_to_end(sid)
        return _extracted_cache[sid]

    year = _parse_sid_year(sid)
    if year < HURSAT_START_YEAR or year > HURSAT_END_YEAR:
        return None

    tarball_url = _find_tarball_url(sid, year)
    if not tarball_url:
        return None

    frames = _extract_tarball(sid, tarball_url)
    return frames if frames else None


def _parse_datetime_from_filename(filename: str) -> str:
    """
    Extract datetime from HURSAT filename.
    Common patterns:
        HURSAT_b1_v06_2005236N23285_d20050823_s181500.nc
    """
    # Try pattern: d{YYYYMMDD}_s{HHMMSS}
    m = re.search(r'd(\d{8})_s(\d{6})', filename)
    if m:
        d, t = m.group(1), m.group(2)
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}"

    # Try pattern: {SID}-{YYYYMMDDHHMM}
    m = re.search(r'-(\d{12})\.nc', filename)
    if m:
        d = m.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}T{d[8:10]}:{d[10:12]}:00"

    # Fallback: any 8+ digit sequence that looks like a date
    m = re.search(r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})', filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}:{m.group(5)}:00"

    return filename


def _load_frame_from_nc(nc_path: str):
    """Open a local NetCDF file and return the IR 2D array."""
    import xarray as xr

    try:
        ds = xr.open_dataset(nc_path, engine="netcdf4")

        # Find the IR variable
        var_name = None
        for candidate in ["irwin_cdr", "irwin", "irwin_2", "Tb", "IRWIN"]:
            if candidate in ds:
                var_name = candidate
                break

        if var_name is None:
            for v in ds.data_vars:
                if ds[v].ndim >= 2:
                    var_name = v
                    break

        if var_name is None:
            ds.close()
            return None

        data = ds[var_name]
        if "time" in data.dims:
            frame = data.isel(time=0).values
        else:
            frame = data.values

        ds.close()
        return frame

    except Exception as e:
        logger.warning(f"Failed to load {nc_path}: {e}")
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

    frames = _get_extracted_frames(sid)
    if not frames:
        result = {"sid": sid, "available": False, "reason": "Data not found on NCEI"}
        return JSONResponse(result)

    frame_list = []
    for i, (dt_str, _) in enumerate(frames):
        frame_list.append({"index": i, "datetime": dt_str})

    result = {
        "sid": sid,
        "available": True,
        "n_frames": len(frames),
        "frames": frame_list,
    }

    # Cache metadata
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

    frames = _get_extracted_frames(sid)
    if not frames:
        raise HTTPException(status_code=404, detail="HURSAT data not found")

    if frame_idx >= len(frames):
        raise HTTPException(
            status_code=404,
            detail=f"Frame index {frame_idx} out of range (0-{len(frames)-1})",
        )

    dt_str, nc_path = frames[frame_idx]

    frame_2d = _load_frame_from_nc(nc_path)
    if frame_2d is None:
        raise HTTPException(status_code=500, detail="Failed to read frame data")

    # Render PNG
    png = _render_ir_png(frame_2d, vmin=170.0, vmax=310.0)

    result = {
        "sid": sid,
        "frame_idx": frame_idx,
        "datetime": dt_str,
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
        "extracted_cache_size": len(_extracted_cache),
        "frame_cache_size": len(_frame_cache),
        "meta_cache_size": len(_meta_cache),
        "extracted_storms": list(_extracted_cache.keys()),
    }


@router.get("/hursat/inspect")
def hursat_inspect(sid: str = Query("1992230N11325", description="IBTrACS storm ID")):
    """Inspect the first extracted NetCDF file to see variables and structure."""
    import xarray as xr

    frames = _get_extracted_frames(sid)
    if not frames:
        return {"error": "No frames extracted", "sid": sid}

    dt_str, nc_path = frames[0]
    result = {"sid": sid, "n_frames": len(frames), "first_file": nc_path, "datetime": dt_str}

    try:
        ds = xr.open_dataset(nc_path, engine="netcdf4")
        result["variables"] = {}
        for v in ds.data_vars:
            var = ds[v]
            result["variables"][v] = {
                "dims": list(var.dims),
                "shape": list(var.shape),
                "dtype": str(var.dtype),
            }
            # Add min/max for numeric vars
            try:
                vals = var.values
                if vals.size > 0 and vals.dtype.kind == 'f':
                    finite = vals[np.isfinite(vals)]
                    if len(finite) > 0:
                        result["variables"][v]["min"] = float(np.min(finite))
                        result["variables"][v]["max"] = float(np.max(finite))
            except Exception:
                pass
        result["dims"] = {k: v for k, v in ds.dims.items()}
        result["coords"] = list(ds.coords)
        ds.close()
    except Exception as e:
        result["error"] = str(e)

    return result


@router.get("/hursat/debug")
def hursat_debug(sid: str = Query("1992230N11325", description="IBTrACS storm ID to test")):
    """Debug endpoint: test NCEI connectivity for a storm."""
    import requests

    year = _parse_sid_year(sid)

    # Check year directory
    year_url = f"{HURSAT_V06_BASE}/{year}/"
    year_result = {}
    try:
        resp = requests.get(year_url, timeout=20, headers=_HTTP_HEADERS)
        year_result["url"] = year_url
        year_result["status"] = resp.status_code

        if resp.status_code == 200:
            # Find matching tarball
            pattern = rf'href="(HURSAT_b1_v06_{re.escape(sid)}_[^"]+\.tar\.gz)"'
            matches = re.findall(pattern, resp.text)
            year_result["matching_tarballs"] = matches

            # Also show sample tarballs for context
            all_tarballs = re.findall(r'href="(HURSAT_b1_v06_[^"]+\.tar\.gz)"', resp.text)
            year_result["total_tarballs"] = len(all_tarballs)
            year_result["sample_tarballs"] = all_tarballs[:5]
    except Exception as e:
        year_result["error"] = str(e)

    return {"sid": sid, "year": year, "year_directory": year_result}
