"""
global_archive_api.py — FastAPI router for Global TC Archive (IR satellite imagery)

Include in tc_radar_api.py:
    from global_archive_api import router as global_router
    app.include_router(global_router, prefix="/global")

Endpoints:
    GET /global/hursat/meta?sid={SID}    — HURSAT frame list for a storm
    GET /global/hursat/frame?sid={SID}&frame_idx={N}  — Rendered IR frame as base64 PNG
    GET /global/ir/meta?sid={SID}&track=...  — Unified IR meta (auto-selects source)
    GET /global/ir/frame?sid={SID}&frame_idx={N}&lat=...&lon=...  — Unified IR frame
    GET /global/health                   — Cache status
    GET /global/hursat/debug?sid={SID}   — Debug NCEI connectivity

Data sources:
    HURSAT-B1 v06 (1978–2015): NCEI tar.gz archives, storm-centered 8km 3-hourly
    MergIR (2000–present): NASA GES DISC, global 4km half-hourly, requires Earthdata token
    Priority: MergIR preferred for 2000+, HURSAT fallback for 1978–1999
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

# MergIR (NASA GES DISC GPM_MERGIR) — via OPeNDAP for server-side subsetting
MERGIR_OPENDAP = "https://disc2.gesdisc.eosdis.nasa.gov/opendap/MERGED_IR/GPM_MERGIR.1"
MERGIR_START_YEAR = 1998  # Extended back from Feb 2000 in June 2025

# Earthdata credentials for MergIR access (set via env vars on Render)
# Option 1: Bearer token (EARTHDATA_TOKEN) — preferred
# Option 2: Username/password (EARTHDATA_USER + EARTHDATA_PASS)
# Either option requires a ~/.netrc entry for urs.earthdata.nasa.gov
EARTHDATA_TOKEN = os.environ.get("EARTHDATA_TOKEN", "")
EARTHDATA_USER = os.environ.get("EARTHDATA_USER", "")
EARTHDATA_PASS = os.environ.get("EARTHDATA_PASS", "")


def _setup_earthdata_netrc():
    """
    Create ~/.netrc and ~/.dodsrc for NASA GES DISC OPeNDAP access.

    OPeNDAP (used by xarray) requires ~/.netrc with Earthdata credentials.
    Also creates ~/.dodsrc to tell the DAP client to follow redirects and
    check cookies from urs.earthdata.nasa.gov.
    """
    home = os.path.expanduser("~")
    netrc_path = os.path.join(home, ".netrc")
    dodsrc_path = os.path.join(home, ".dodsrc")

    # Skip if no credentials configured
    if not EARTHDATA_USER and not EARTHDATA_TOKEN:
        print("[global_archive] Earthdata: no credentials configured (MergIR disabled)")
        return

    # Write ~/.netrc
    if EARTHDATA_USER and EARTHDATA_PASS:
        netrc_entry = (
            f"machine urs.earthdata.nasa.gov\n"
            f"    login {EARTHDATA_USER}\n"
            f"    password {EARTHDATA_PASS}\n"
        )
    elif EARTHDATA_TOKEN:
        # Use token as password with 'token' as username
        netrc_entry = (
            f"machine urs.earthdata.nasa.gov\n"
            f"    login token\n"
            f"    password {EARTHDATA_TOKEN}\n"
        )
    else:
        return

    try:
        with open(netrc_path, "w") as f:
            f.write(netrc_entry)
        os.chmod(netrc_path, 0o600)
        print(f"[global_archive] Earthdata: wrote {netrc_path}")
    except Exception as e:
        print(f"[global_archive] Earthdata: FAILED to write {netrc_path}: {e}")

    # Write ~/.dodsrc for OPeNDAP cookie/redirect handling
    dodsrc_content = (
        "HTTP.COOKIEJAR=~/.urs_cookies\n"
        "HTTP.NETRC=~/.netrc\n"
    )
    try:
        with open(dodsrc_path, "w") as f:
            f.write(dodsrc_content)
        print(f"[global_archive] Earthdata: wrote {dodsrc_path}")
    except Exception as e:
        print(f"[global_archive] Earthdata: FAILED to write {dodsrc_path}: {e}")


# Run netrc setup at import time (before any OPeNDAP requests)
print(f"[global_archive] EARTHDATA_USER={'set' if EARTHDATA_USER else 'empty'}, "
      f"EARTHDATA_PASS={'set' if EARTHDATA_PASS else 'empty'}, "
      f"EARTHDATA_TOKEN={'set' if EARTHDATA_TOKEN else 'empty'}")
_setup_earthdata_netrc()

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

    # NOTE: Vertical orientation is handled in _load_frame_from_nc()
    # which checks actual latitude order and ensures north-at-top.
    # No flip needed here.

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


def _extract_tarball(sid: str, tarball_url: str, storm_lon: float = 0.0) -> list:
    """
    Download a HURSAT tar.gz, extract NetCDF files to /tmp, return sorted list
    of (datetime_str, nc_path, satellite) tuples.
    Deduplicates frames at the same time from different satellites,
    preferring the satellite with the best viewing angle for storm_lon.
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

            raw_frames = []
            for member in nc_members:
                # Extract to flat directory
                member.name = os.path.basename(member.name)
                tar.extract(member, path=extract_dir)
                nc_path = os.path.join(extract_dir, member.name)

                # Parse datetime from filename
                dt_str = _parse_datetime_from_filename(member.name)
                raw_frames.append((dt_str, nc_path))

        # Clean up tar file
        os.unlink(tmp_tar.name)

        # Sort by datetime
        raw_frames.sort(key=lambda x: x[0])

        # Deduplicate frames with same datetime from different satellites
        frames = _deduplicate_frames(raw_frames, storm_lon=storm_lon)
        logger.info(
            f"HURSAT: {len(raw_frames)} raw frames → {len(frames)} after dedup"
        )

        # Clean up skipped duplicate files
        kept_paths = {f[1] for f in frames}
        for _, nc_path in raw_frames:
            if nc_path not in kept_paths:
                try:
                    os.unlink(nc_path)
                except OSError:
                    pass

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
        for frame_tuple in old_frames:
            try:
                os.unlink(frame_tuple[1])  # nc_path is index 1
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


def _get_extracted_frames(sid: str, storm_lon: float = 0.0) -> list | None:
    """Get extracted frames for a storm, downloading if needed.
    storm_lon is used for satellite viewing angle selection during dedup."""
    if sid in _extracted_cache:
        _extracted_cache.move_to_end(sid)
        return _extracted_cache[sid]

    year = _parse_sid_year(sid)
    if year < HURSAT_START_YEAR or year > HURSAT_END_YEAR:
        return None

    tarball_url = _find_tarball_url(sid, year)
    if not tarball_url:
        return None

    frames = _extract_tarball(sid, tarball_url, storm_lon=storm_lon)
    return frames if frames else None


def _parse_datetime_from_filename(filename: str) -> str:
    """
    Extract datetime from HURSAT filename.
    Actual v06 format:
        {SID}.{NAME}.{YYYY}.{MM}.{DD}.{HHMM}.{...}.hursat-b1.v06.nc
        e.g. 1992230N11325.ANDREW.1992.08.16.1800.43.MET-4.022.hursat-b1.v06.nc
    """
    # Primary pattern: {SID}.{NAME}.{YYYY}.{MM}.{DD}.{HHMM}
    m = re.search(r'\.\d{4}\.(\d{2})\.(\d{2})\.(\d{4})\.', filename)
    if m:
        # Extract year from the 4-digit sequence before the MM.DD.HHMM
        ym = re.search(r'\.(\d{4})\.(\d{2})\.(\d{2})\.(\d{4})\.', filename)
        if ym:
            yyyy, mm, dd, hhmm = ym.group(1), ym.group(2), ym.group(3), ym.group(4)
            return f"{yyyy}-{mm}-{dd}T{hhmm[:2]}:{hhmm[2:]}:00"

    # Fallback pattern: d{YYYYMMDD}_s{HHMMSS}
    m = re.search(r'd(\d{8})_s(\d{6})', filename)
    if m:
        d, t = m.group(1), m.group(2)
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}"

    # Fallback: any 8+ digit sequence that looks like a date
    m = re.search(r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})', filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}:{m.group(5)}:00"

    return filename


def _parse_satellite_from_filename(filename: str) -> str:
    """
    Extract satellite name from HURSAT filename.
    Format: {SID}.{NAME}.{YYYY}.{MM}.{DD}.{HHMM}.{num}.{SATELLITE}.{num}.hursat-b1.v06.nc
    e.g. 1992230N11325.ANDREW.1992.08.22.0300.59.GOE-7.053.hursat-b1.v06.nc → "GOE-7"
    """
    # Match satellite field: letter(s) + optional dash + number, between two dot-number segments
    m = re.search(r'\.\d{4}\.\d{2}\.\d{2}\.\d{4}\.\d+\.([A-Z][\w-]+)\.\d+\.hursat', filename)
    if m:
        return m.group(1)
    return ""


# Known geostationary satellite sub-point longitudes (degrees East).
# Used to pick the satellite with the best viewing angle for a given storm.
_SAT_SUBLON = {
    # GOES-East series (~75°W = -75°)
    "GOE-8": -75, "GOE-12": -75, "GOE-13": -75, "GOE-14": -75, "GOE-16": -75,
    # GOES-West series (~137°W = -137°)
    "GOE-7": -112,  # GOES-7 drifted; was ~112°W in early 1990s
    "GOE-9": -135, "GOE-10": -135, "GOE-11": -135, "GOE-15": -135, "GOE-17": -137,
    # Meteosat (0° or 63°E for Indian Ocean)
    "MET-2": 0, "MET-3": 0, "MET-4": 0, "MET-5": 0, "MET-7": 0,
    "MET-8": 0, "MET-9": 0, "MET-10": 0, "MET-11": 0,
    # Meteosat Indian Ocean (IODC at ~63°E)
    "MET-6": 63, "MET-I7": 57,
    # GMS / MTSAT / Himawari (~140°E)
    "GMS-1": 140, "GMS-2": 140, "GMS-3": 140, "GMS-4": 140, "GMS-5": 140,
    "MTS-1": 140, "MTS-2": 140,
    "HIM-8": 140, "HIM-9": 140,
}

# Fallback sub-point by satellite prefix (for unrecognized specific IDs)
_SAT_PREFIX_SUBLON = {
    "GOE": -75,   # Default GOES → East position
    "GMS": 140,
    "MTS": 140,
    "HIM": 140,
    "MET": 0,
}


def _get_sat_sublon(sat_name: str) -> float | None:
    """Get approximate sub-satellite-point longitude for a satellite."""
    if sat_name in _SAT_SUBLON:
        return _SAT_SUBLON[sat_name]
    prefix = sat_name[:3].upper() if sat_name else ""
    return _SAT_PREFIX_SUBLON.get(prefix)


def _viewing_angle_score(sat_name: str, storm_lon: float) -> float:
    """
    Return a score representing how far the satellite is from the storm.
    Lower = better viewing angle (satellite is closer to being overhead).

    Uses great-circle-like longitude difference on the equator as proxy.
    Falls back to a large penalty if satellite position is unknown.
    """
    sublon = _get_sat_sublon(sat_name)
    if sublon is None:
        return 999.0
    # Longitude difference (handle wrap-around)
    diff = abs(storm_lon - sublon)
    if diff > 180:
        diff = 360 - diff
    return diff


def _deduplicate_frames(frames: list, storm_lon: float = 0.0) -> list:
    """
    Deduplicate HURSAT frames that share the same datetime but come from
    different satellites (e.g., GOE-7 and MET-4 both at 0300 UTC).

    Keeps the satellite with the best viewing angle for the storm's longitude
    (smallest longitude difference between satellite sub-point and storm center).
    Also stores satellite name in the tuple for metadata.

    Input: list of (datetime_str, nc_path) tuples
    Output: list of (datetime_str, nc_path, satellite) tuples, deduplicated
    """
    from collections import defaultdict

    # Group by datetime
    by_time = defaultdict(list)
    for dt_str, nc_path in frames:
        sat = _parse_satellite_from_filename(os.path.basename(nc_path))
        by_time[dt_str].append((dt_str, nc_path, sat))

    deduped = []
    for dt_str in sorted(by_time.keys()):
        candidates = by_time[dt_str]
        if len(candidates) == 1:
            deduped.append(candidates[0])
        else:
            # Pick the satellite closest to the storm (best viewing angle)
            candidates.sort(key=lambda x: _viewing_angle_score(x[2], storm_lon))
            chosen = candidates[0]
            skipped_info = [
                f"{c[2]}({_viewing_angle_score(c[2], storm_lon):.0f}°)"
                for c in candidates[1:]
            ]
            logger.info(
                f"HURSAT dedup: {dt_str} — kept {chosen[2]} "
                f"(Δlon={_viewing_angle_score(chosen[2], storm_lon):.0f}°), "
                f"skipped {skipped_info}"
            )
            deduped.append(chosen)

    return deduped


def _load_frame_from_nc(nc_path: str):
    """
    Open a local NetCDF file and return (frame_2d, bounds_dict).

    bounds_dict has keys: south, north, west, east (in degrees).
    HURSAT-B1 v06 files have 'latitude' and 'longitude' coordinate variables
    on a 301×301 storm-centered grid (~8km resolution ≈ ~12° box).
    """
    import xarray as xr

    try:
        ds = xr.open_dataset(nc_path, engine="h5netcdf")

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
            return None, None

        data = ds[var_name]
        if "htime" in data.dims:
            frame = data.isel(htime=0).values
        elif "time" in data.dims:
            frame = data.isel(time=0).values
        else:
            frame = data.values

        # Extract geographic bounds and determine latitude orientation
        bounds = None
        lat_var = None
        lon_var = None
        lat_increasing = True  # default assumption

        # HURSAT-B1 v06 uses 'latitude' and 'longitude' (1D or 2D)
        for lat_cand in ["latitude", "lat", "Latitude"]:
            if lat_cand in ds.coords or lat_cand in ds:
                lat_var = lat_cand
                break
        for lon_cand in ["longitude", "lon", "Longitude"]:
            if lon_cand in ds.coords or lon_cand in ds:
                lon_var = lon_cand
                break

        if lat_var and lon_var:
            try:
                lats = ds[lat_var].values
                lons = ds[lon_var].values

                logger.info(
                    f"HURSAT frame: {os.path.basename(nc_path)}, "
                    f"lat shape={lats.shape}, lon shape={lons.shape}, "
                    f"frame shape={frame.shape}"
                )

                # Determine latitude order (1D array or first column of 2D)
                if lats.ndim == 1:
                    lat_1d = lats
                elif lats.ndim == 2:
                    lat_1d = lats[:, lats.shape[1] // 2]  # center column
                else:
                    lat_1d = lats.ravel()

                finite_lats = lat_1d[np.isfinite(lat_1d)]
                if len(finite_lats) >= 2:
                    lat_increasing = finite_lats[-1] > finite_lats[0]
                    logger.info(
                        f"  lat[0]={finite_lats[0]:.2f}, "
                        f"lat[-1]={finite_lats[-1]:.2f}, "
                        f"increasing={lat_increasing}"
                    )

                # Handle NaN values
                lats_valid = lats[np.isfinite(lats)]
                lons_valid = lons[np.isfinite(lons)]
                if len(lats_valid) > 0 and len(lons_valid) > 0:
                    bounds = {
                        "south": float(np.min(lats_valid)),
                        "north": float(np.max(lats_valid)),
                        "west": float(np.min(lons_valid)),
                        "east": float(np.max(lons_valid)),
                    }
                    logger.info(
                        f"  bounds: S={bounds['south']:.2f} N={bounds['north']:.2f} "
                        f"W={bounds['west']:.2f} E={bounds['east']:.2f}"
                    )

                    # Log frame data stats for sanity check
                    valid_frame = frame[np.isfinite(frame)]
                    if len(valid_frame) > 0:
                        logger.info(
                            f"  Tb stats: min={np.min(valid_frame):.1f}, "
                            f"max={np.max(valid_frame):.1f}, "
                            f"mean={np.mean(valid_frame):.1f}, "
                            f"NaN%={100*(1 - len(valid_frame)/frame.size):.0f}%"
                        )
            except Exception as e:
                logger.warning(f"Could not extract bounds from {nc_path}: {e}")

        # Ensure frame is oriented with north at top (row 0 = north)
        # Leaflet imageOverlay expects top-left = NW corner
        if lat_increasing:
            # First row = south → flip so first row = north
            frame = frame[::-1]
            logger.info("  → Flipped frame (lat was increasing)")
        else:
            logger.info("  → No flip needed (lat already decreasing/north-at-top)")

        ds.close()
        return frame, bounds

    except Exception as e:
        logger.warning(f"Failed to load {nc_path}: {e}")
        return None, None


# ── Endpoints ────────────────────────────────────────────────

@router.get("/hursat/meta")
def hursat_meta(
    sid: str = Query(..., description="IBTrACS storm ID"),
    storm_lon: float = Query(0.0, description="Storm longitude for satellite selection"),
):
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

    frames = _get_extracted_frames(sid, storm_lon=storm_lon)
    if not frames:
        result = {"sid": sid, "available": False, "reason": "Data not found on NCEI"}
        return JSONResponse(result)

    frame_list = []
    for i, frame_tuple in enumerate(frames):
        entry = {"index": i, "datetime": frame_tuple[0]}
        if len(frame_tuple) > 2 and frame_tuple[2]:
            entry["satellite"] = frame_tuple[2]
        frame_list.append(entry)

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

    # storm_lon=0.0 is fine here — frames should already be cached from meta call
    frames = _get_extracted_frames(sid, storm_lon=0.0)
    if not frames:
        raise HTTPException(status_code=404, detail="HURSAT data not found")

    if frame_idx >= len(frames):
        raise HTTPException(
            status_code=404,
            detail=f"Frame index {frame_idx} out of range (0-{len(frames)-1})",
        )

    frame_tuple = frames[frame_idx]
    dt_str, nc_path = frame_tuple[0], frame_tuple[1]
    satellite = frame_tuple[2] if len(frame_tuple) > 2 else ""
    nc_filename = os.path.basename(nc_path)

    logger.info(
        f"HURSAT frame request: idx={frame_idx}, dt={dt_str}, "
        f"sat={satellite}, file={nc_filename}"
    )

    frame_2d, bounds = _load_frame_from_nc(nc_path)
    if frame_2d is None:
        raise HTTPException(status_code=500, detail="Failed to read frame data")

    # Render PNG
    png = _render_ir_png(frame_2d, vmin=170.0, vmax=310.0)

    result = {
        "sid": sid,
        "frame_idx": frame_idx,
        "datetime": dt_str,
        "frame": png,
        "nc_file": nc_filename,
    }
    if satellite:
        result["satellite"] = satellite
    if bounds:
        result["bounds"] = bounds

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


# ══════════════════════════════════════════════════════════════
#  MergIR Data Access (2000–present)
# ══════════════════════════════════════════════════════════════

# MergIR caches
_mergir_meta_cache: OrderedDict = OrderedDict()
_MERGIR_META_CACHE_MAX = 100

# Box size for storm-centered subset (degrees from center)
MERGIR_HALF_DOMAIN = 5.0  # 5° each direction = 10°×10° box


def _mergir_opendap_file_url(dt: datetime) -> str:
    """
    Build OPeNDAP URL for the MergIR file covering a given datetime.
    Each file has two half-hourly grids and is keyed by the truncated hour.

    Pattern: {BASE}/{YYYY}/{DOY:03d}/merg_{YYYYMMDDHH}_4km-pixel.nc4
    """
    jday = dt.timetuple().tm_yday
    time_str = dt.strftime("%Y%m%d%H")
    return (
        f"{MERGIR_OPENDAP}/{dt.year}/{jday:03d}/"
        f"merg_{time_str}_4km-pixel.nc4"
    )


def _build_mergir_frame_list(track_points: list) -> list:
    """
    Build a list of MergIR frames from track points.

    track_points: list of {"t": "YYYY-MM-DDTHH:MM", "la": lat, "lo": lon}
    Returns: list of {"datetime": str, "lat": float, "lon": float}

    We sample at 3-hourly intervals (synoptic hours) to keep frame counts
    manageable while providing good temporal coverage.
    """
    frames = []
    seen_times = set()

    for pt in track_points:
        if not pt.get("t") or not pt.get("la") or not pt.get("lo"):
            continue

        try:
            dt = datetime.fromisoformat(pt["t"].replace("Z", "+00:00").split("+")[0])
        except (ValueError, AttributeError):
            continue

        # Round to nearest 3-hour interval for consistent sampling
        hour_3 = (dt.hour // 3) * 3
        dt_rounded = dt.replace(hour=hour_3, minute=0, second=0, microsecond=0)
        time_key = dt_rounded.strftime("%Y%m%d%H")

        if time_key in seen_times:
            continue
        seen_times.add(time_key)

        frames.append({
            "datetime": dt_rounded.strftime("%Y-%m-%dT%H:%M:00"),
            "lat": float(pt["la"]),
            "lon": float(pt["lo"]),
        })

    frames.sort(key=lambda f: f["datetime"])
    return frames


def _find_nearest_half_hour(target_dt, ir_times):
    """
    Find the MergIR time index closest to target_dt.
    Each file has two half-hourly grids.
    Returns (time_index, time_diff_minutes).
    """
    import pandas as pd

    best_idx = 0
    best_diff = float("inf")
    for i, t in enumerate(ir_times):
        ts = pd.Timestamp(t)
        diff_min = abs((target_dt - ts).total_seconds()) / 60.0
        if diff_min < best_diff:
            best_diff = diff_min
            best_idx = i
    return best_idx, best_diff


# Earthdata session (reused across requests for cookie persistence)
_earthdata_session = None


def _get_earthdata_session():
    """Get or create a requests session with Earthdata credentials."""
    global _earthdata_session
    if _earthdata_session is not None:
        return _earthdata_session

    import requests

    session = requests.Session()

    if EARTHDATA_USER and EARTHDATA_PASS:
        session.auth = (EARTHDATA_USER, EARTHDATA_PASS)
    elif EARTHDATA_TOKEN:
        session.headers.update({"Authorization": f"Bearer {EARTHDATA_TOKEN}"})

    _earthdata_session = session
    return session


def _mergir_direct_file_url(dt: datetime) -> str:
    """
    Build direct download URL for the MergIR file covering a given datetime.
    Pattern: {BASE_DATA}/{YYYY}/{DOY:03d}/merg_{YYYYMMDDHH}_4km-pixel.nc4
    """
    jday = dt.timetuple().tm_yday
    time_str = dt.strftime("%Y%m%d%H")
    return (
        f"https://disc2.gesdisc.eosdis.nasa.gov/data/MERGED_IR/GPM_MERGIR.1"
        f"/{dt.year}/{jday:03d}/merg_{time_str}_4km-pixel.nc4"
    )


def _mergir_subset_url(dt: datetime, center_lat: float, center_lon: float) -> str:
    """
    Build OPeNDAP subset URL that returns only the geographic subset we need.
    Appends .nc4 suffix + constraint expression to request server-side subsetting.
    This downloads ~1-2 MB instead of ~130 MB for the full file.

    GES DISC OPeNDAP supports: {opendap_url}.nc4?Tb[time][lat_start:lat_end][lon_start:lon_end]
    MergIR grid: lat -60 to 60 (0.03636° spacing, ~3300 pts), lon -180 to 180 (~9896 pts)
    """
    jday = dt.timetuple().tm_yday
    time_str = dt.strftime("%Y%m%d%H")
    base = (
        f"{MERGIR_OPENDAP}/{dt.year}/{jday:03d}/"
        f"merg_{time_str}_4km-pixel.nc4"
    )

    # Convert lat/lon bounds to grid indices
    # MergIR: lat from -60 to 60 in ~3301 steps, lon from -180 to 180 in ~9896 steps
    # Add 1° margin to avoid edge effects; actual subsetting uses .sel() later
    margin = 1.0
    lat_min = center_lat - MERGIR_HALF_DOMAIN - margin
    lat_max = center_lat + MERGIR_HALF_DOMAIN + margin
    lon_min = center_lon - MERGIR_HALF_DOMAIN - margin
    lon_max = center_lon + MERGIR_HALF_DOMAIN + margin

    # Clamp to grid bounds
    lat_min = max(lat_min, -60.0)
    lat_max = min(lat_max, 60.0)
    lon_min = max(lon_min, -180.0)
    lon_max = min(lon_max, 180.0)

    # Grid spacing: ~0.036364° for lat, ~0.036378° for lon
    lat_idx_min = int((lat_min + 60.0) / 0.036364)
    lat_idx_max = min(int((lat_max + 60.0) / 0.036364), 3300)
    lon_idx_min = int((lon_min + 180.0) / 0.036378)
    lon_idx_max = min(int((lon_max + 180.0) / 0.036378), 9895)

    # Build constraint expression for server-side subsetting
    # Request both time steps (usually 2 per file), subset lat/lon
    constraint = (
        f"Tb[0:1][{lat_idx_min}:{lat_idx_max}][{lon_idx_min}:{lon_idx_max}],"
        f"lat[{lat_idx_min}:{lat_idx_max}],"
        f"lon[{lon_idx_min}:{lon_idx_max}],"
        f"time[0:1]"
    )

    return f"{base}.nc4?{constraint}"


def _load_mergir_subset(target_dt: datetime, center_lat: float, center_lon: float):
    """
    Fetch a single MergIR Tb snapshot, cropped to a
    MERGIR_HALF_DOMAIN degree box around (center_lat, center_lon).

    Uses requests with Earthdata session auth to download the file,
    then opens locally with xarray for subsetting.
    Downloads to temp file, extracts subset, deletes temp file.

    Returns (2D numpy array of Tb, bounds_dict) or (None, None) on failure.
    bounds_dict has keys: south, north, west, east — extracted from actual data.
    """
    import xarray as xr
    from datetime import timedelta

    session = _get_earthdata_session()
    if not session:
        logger.warning("MergIR: no Earthdata session available")
        return None, None

    file_dt = target_dt.replace(minute=0, second=0, microsecond=0)

    for attempt_dt in [file_dt, file_dt + timedelta(hours=1)]:
        # Try full file download (more reliable than OPeNDAP subset)
        full_url = _mergir_direct_file_url(attempt_dt)
        # OPeNDAP subset as fallback
        subset_url = _mergir_subset_url(attempt_dt, center_lat, center_lon)

        for url_label, url in [("subset", subset_url), ("full", full_url)]:
            tmp = None
            try:
                logger.info(f"MergIR: downloading {url_label} from {url[:120]}...")
                resp = session.get(url, timeout=90, allow_redirects=True)

                if resp.status_code in (401, 403):
                    logger.warning(
                        f"MergIR: auth failed ({resp.status_code}), "
                        f"resetting session"
                    )
                    global _earthdata_session
                    _earthdata_session = None
                    session = _get_earthdata_session()
                    continue

                if resp.status_code != 200:
                    logger.info(f"MergIR: HTTP {resp.status_code} for {url_label}")
                    continue

                # Write to temp file
                tmp = tempfile.NamedTemporaryFile(suffix=".nc4", delete=False)
                tmp.write(resp.content)
                tmp.close()
                file_size_mb = len(resp.content) / 1024 / 1024
                logger.info(f"MergIR: downloaded {url_label} {file_size_mb:.1f} MB")

            except Exception as e:
                logger.warning(f"MergIR: {url_label} download failed: {e}")
                continue

            try:
                ds = xr.open_dataset(tmp.name, engine="h5netcdf")

                ir_times = ds["time"].values
                tidx, tdiff = _find_nearest_half_hour(target_dt, ir_times)

                if tdiff > 20.0:
                    ds.close()
                    os.unlink(tmp.name)
                    continue

                # Always do spatial subsetting from the dataset to get
                # correct lat/lon bounds (even for OPeNDAP subsets)
                lat_min = center_lat - MERGIR_HALF_DOMAIN
                lat_max = center_lat + MERGIR_HALF_DOMAIN
                lon_min = center_lon - MERGIR_HALF_DOMAIN
                lon_max = center_lon + MERGIR_HALF_DOMAIN

                da = ds["Tb"].isel(time=tidx)

                # Subset spatially using coordinate selection
                da_sub = da.sel(
                    lat=slice(lat_min, lat_max),
                    lon=slice(lon_min, lon_max),
                )

                tb = da_sub.values
                actual_lats = da_sub.coords["lat"].values
                actual_lons = da_sub.coords["lon"].values

                ds.close()
                os.unlink(tmp.name)
                tmp = None

                if tb is not None and tb.size > 0 and len(actual_lats) > 1 and len(actual_lons) > 1:
                    # Extract actual bounds from coordinate arrays
                    actual_bounds = {
                        "south": float(np.min(actual_lats)),
                        "north": float(np.max(actual_lats)),
                        "west": float(np.min(actual_lons)),
                        "east": float(np.max(actual_lons)),
                    }
                    logger.info(
                        f"MergIR: got {tb.shape} subset for "
                        f"({center_lat:.1f}, {center_lon:.1f}), "
                        f"actual bounds: S={actual_bounds['south']:.2f} "
                        f"N={actual_bounds['north']:.2f} "
                        f"W={actual_bounds['west']:.2f} "
                        f"E={actual_bounds['east']:.2f}"
                    )
                    # MergIR lat is ascending (south→north), flip so
                    # row 0 = north (as Leaflet imageOverlay expects)
                    return tb[::-1], actual_bounds

            except Exception as e:
                logger.warning(f"MergIR: {url_label} parse failed: {e}")
                if tmp:
                    try:
                        os.unlink(tmp.name)
                    except OSError:
                        pass
                continue

    logger.warning(f"MergIR: no data found for {target_dt}")
    return None, None


# ── Unified IR Endpoints ──────────────────────────────────────

@router.get("/ir/meta")
def ir_meta(
    sid: str = Query(..., description="IBTrACS storm ID"),
    track: str = Query("", description="JSON-encoded track points array"),
    storm_lon: float = Query(0.0, description="Storm longitude for satellite selection"),
):
    """
    Return IR frame metadata for a storm, auto-selecting HURSAT or MergIR.

    Priority: MergIR for 2000+, HURSAT for 1978-1999.
    For MergIR, track data is needed to know storm positions.
    """
    import json as json_mod

    cache_key = f"ir_{sid}"
    if cache_key in _mergir_meta_cache:
        _mergir_meta_cache.move_to_end(cache_key)
        return JSONResponse(
            _mergir_meta_cache[cache_key],
            headers={"Cache-Control": "public, max-age=3600"},
        )

    year = _parse_sid_year(sid)

    # Determine source
    source = None
    _earthdata_configured = bool(EARTHDATA_TOKEN or EARTHDATA_USER)
    if year >= MERGIR_START_YEAR and _earthdata_configured:
        source = "mergir"
    elif HURSAT_START_YEAR <= year <= HURSAT_END_YEAR:
        source = "hursat"
    elif year >= MERGIR_START_YEAR and not _earthdata_configured:
        # MergIR available but no credentials configured
        result = {
            "sid": sid,
            "available": False,
            "source": "mergir",
            "reason": "MergIR requires Earthdata credentials (not configured)",
        }
        return JSONResponse(result)
    else:
        result = {
            "sid": sid,
            "available": False,
            "reason": f"No IR data available for year {year}",
        }
        return JSONResponse(result)

    # Handle HURSAT path
    if source == "hursat":
        frames = _get_extracted_frames(sid, storm_lon=storm_lon)
        if not frames:
            result = {
                "sid": sid, "available": False, "source": "hursat",
                "reason": "HURSAT data not found on NCEI",
            }
            return JSONResponse(result)

        frame_list = [
            {
                "index": i, "datetime": ft[0],
                **({"satellite": ft[2]} if len(ft) > 2 and ft[2] else {}),
            }
            for i, ft in enumerate(frames)
        ]
        result = {
            "sid": sid, "available": True, "source": "hursat",
            "n_frames": len(frames), "frames": frame_list,
        }

    # Handle MergIR path
    elif source == "mergir":
        # Parse track points
        track_points = []
        if track:
            try:
                track_points = json_mod.loads(track)
            except (json_mod.JSONDecodeError, TypeError):
                pass

        if not track_points:
            # Fall back to HURSAT if available
            if HURSAT_START_YEAR <= year <= HURSAT_END_YEAR:
                frames = _get_extracted_frames(sid, storm_lon=storm_lon)
                if frames:
                    frame_list = [
                        {
                            "index": i, "datetime": ft[0],
                            **({"satellite": ft[2]} if len(ft) > 2 and ft[2] else {}),
                        }
                        for i, ft in enumerate(frames)
                    ]
                    result = {
                        "sid": sid, "available": True, "source": "hursat",
                        "n_frames": len(frames), "frames": frame_list,
                    }
                    _mergir_meta_cache[cache_key] = result
                    if len(_mergir_meta_cache) > _MERGIR_META_CACHE_MAX:
                        _mergir_meta_cache.popitem(last=False)
                    return JSONResponse(
                        result,
                        headers={"Cache-Control": "public, max-age=3600"},
                    )

            result = {
                "sid": sid, "available": False, "source": "mergir",
                "reason": "Track data required for MergIR (pass track parameter)",
            }
            return JSONResponse(result)

        # Build MergIR frame list from track
        mergir_frames = _build_mergir_frame_list(track_points)
        if not mergir_frames:
            result = {
                "sid": sid, "available": False, "source": "mergir",
                "reason": "No valid track times for MergIR",
            }
            return JSONResponse(result)

        frame_list = [
            {
                "index": i,
                "datetime": f["datetime"],
                "lat": f["lat"],
                "lon": f["lon"],
            }
            for i, f in enumerate(mergir_frames)
        ]
        result = {
            "sid": sid, "available": True, "source": "mergir",
            "n_frames": len(mergir_frames), "frames": frame_list,
        }

    # Cache and return
    _mergir_meta_cache[cache_key] = result
    if len(_mergir_meta_cache) > _MERGIR_META_CACHE_MAX:
        _mergir_meta_cache.popitem(last=False)

    return JSONResponse(
        result,
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/ir/frame")
def ir_frame(
    sid: str = Query(..., description="IBTrACS storm ID"),
    frame_idx: int = Query(..., ge=0, description="Frame index (0-based)"),
    lat: float = Query(None, description="Storm center latitude (for MergIR)"),
    lon: float = Query(None, description="Storm center longitude (for MergIR)"),
):
    """
    Return a rendered IR frame as base64 PNG.
    Auto-selects HURSAT vs MergIR based on cached metadata.
    """
    cache_key = (f"ir_{sid}", frame_idx)

    # Check frame cache
    if cache_key in _frame_cache:
        _frame_cache.move_to_end(cache_key)
        return JSONResponse(
            _frame_cache[cache_key],
            headers={"Cache-Control": "public, max-age=86400", "X-Cache": "HIT"},
        )

    # Check what source was determined for this storm
    meta_key = f"ir_{sid}"
    meta = _mergir_meta_cache.get(meta_key)
    source = meta.get("source", "hursat") if meta else "hursat"

    year = _parse_sid_year(sid)

    if source == "mergir" and meta and meta.get("available"):
        # MergIR path: need lat/lon for the specific frame
        frames = meta.get("frames", [])
        if frame_idx >= len(frames):
            raise HTTPException(status_code=404, detail=f"Frame {frame_idx} out of range")

        frame_info = frames[frame_idx]
        frame_lat = lat or frame_info.get("lat")
        frame_lon = lon or frame_info.get("lon")
        frame_dt = datetime.fromisoformat(frame_info["datetime"])

        if frame_lat is None or frame_lon is None:
            raise HTTPException(status_code=400, detail="lat/lon required for MergIR")

        frame_2d, mergir_bounds = _load_mergir_subset(frame_dt, frame_lat, frame_lon)

        if frame_2d is None:
            raise HTTPException(
                status_code=502,
                detail="Failed to retrieve MergIR data (check Earthdata token)",
            )

        png = _render_ir_png(frame_2d, vmin=170.0, vmax=310.0)
        # Use actual bounds from data (not assumed center ± half_domain)
        bounds = mergir_bounds or {
            "south": frame_lat - MERGIR_HALF_DOMAIN,
            "north": frame_lat + MERGIR_HALF_DOMAIN,
            "west": frame_lon - MERGIR_HALF_DOMAIN,
            "east": frame_lon + MERGIR_HALF_DOMAIN,
        }
        result = {
            "sid": sid, "frame_idx": frame_idx,
            "datetime": frame_info["datetime"], "source": "mergir",
            "frame": png,
            "bounds": bounds,
        }

    else:
        # HURSAT path (default)
        # Use lon if available (from MergIR params), default 0.0 for cache hit
        frames = _get_extracted_frames(sid, storm_lon=lon or 0.0)
        if not frames:
            raise HTTPException(status_code=404, detail="IR data not found")

        if frame_idx >= len(frames):
            raise HTTPException(
                status_code=404,
                detail=f"Frame {frame_idx} out of range (0-{len(frames)-1})",
            )

        frame_tuple = frames[frame_idx]
        dt_str, nc_path = frame_tuple[0], frame_tuple[1]
        satellite = frame_tuple[2] if len(frame_tuple) > 2 else ""

        frame_2d, bounds = _load_frame_from_nc(nc_path)
        if frame_2d is None:
            raise HTTPException(status_code=500, detail="Failed to read frame data")

        png = _render_ir_png(frame_2d, vmin=170.0, vmax=310.0)
        result = {
            "sid": sid, "frame_idx": frame_idx,
            "datetime": dt_str, "source": "hursat",
            "frame": png,
        }
        if satellite:
            result["satellite"] = satellite
        if bounds:
            result["bounds"] = bounds

    # Cache rendered frame
    _frame_cache[cache_key] = result
    if len(_frame_cache) > _FRAME_CACHE_MAX:
        _frame_cache.popitem(last=False)
        gc.collect()

    return JSONResponse(
        result,
        headers={"Cache-Control": "public, max-age=86400", "X-Cache": "MISS"},
    )


@router.get("/ir/batch")
def ir_batch(
    sid: str = Query(..., description="IBTrACS storm ID"),
    indices: str = Query(..., description="Comma-separated frame indices, e.g. '0,1,2,3,4'"),
):
    """
    Fetch multiple IR frames in one request using concurrent workers.
    Returns a dict mapping frame index → frame data (or null on failure).
    Server-side concurrency is capped at 3 workers to limit RAM usage.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    MAX_BATCH = 10      # Max frames per batch request
    MAX_WORKERS = 3     # Concurrent OPeNDAP/file reads

    # Parse indices
    try:
        idx_list = [int(x.strip()) for x in indices.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid indices format")

    if len(idx_list) > MAX_BATCH:
        idx_list = idx_list[:MAX_BATCH]

    # Look up cached meta to determine source
    # The unified /ir/meta endpoint caches with "ir_{sid}" prefix
    meta = _mergir_meta_cache.get(f"ir_{sid}")
    if not meta:
        meta = _mergir_meta_cache.get(sid)
    if not meta:
        meta = _meta_cache.get(sid)
    source = meta.get("source", "hursat") if meta else "hursat"

    def fetch_single_frame(frame_idx: int):
        """Fetch one frame, return (idx, result_dict) or (idx, None)."""
        cache_key = (sid, frame_idx)

        # Check frame cache first
        if cache_key in _frame_cache:
            _frame_cache.move_to_end(cache_key)
            return (frame_idx, _frame_cache[cache_key])

        try:
            if source == "mergir" and meta and meta.get("available"):
                frames_meta = meta.get("frames", [])
                if frame_idx >= len(frames_meta):
                    return (frame_idx, None)

                frame_info = frames_meta[frame_idx]
                frame_dt = datetime.strptime(
                    frame_info["datetime"], "%Y-%m-%dT%H:%M:%S"
                )
                frame_lat = frame_info.get("lat")
                frame_lon = frame_info.get("lon")

                if frame_lat is None or frame_lon is None:
                    return (frame_idx, None)

                frame_2d, mergir_bounds = _load_mergir_subset(
                    frame_dt, frame_lat, frame_lon
                )
                if frame_2d is None:
                    return (frame_idx, None)

                png = _render_ir_png(frame_2d, vmin=170.0, vmax=310.0)
                bounds = mergir_bounds or {
                    "south": frame_lat - MERGIR_HALF_DOMAIN,
                    "north": frame_lat + MERGIR_HALF_DOMAIN,
                    "west": frame_lon - MERGIR_HALF_DOMAIN,
                    "east": frame_lon + MERGIR_HALF_DOMAIN,
                }
                result = {
                    "sid": sid, "frame_idx": frame_idx,
                    "datetime": frame_info["datetime"], "source": "mergir",
                    "frame": png,
                    "bounds": bounds,
                }
            else:
                # HURSAT path
                frames = _get_extracted_frames(sid, storm_lon=0.0)
                if not frames or frame_idx >= len(frames):
                    return (frame_idx, None)

                frame_tuple = frames[frame_idx]
                dt_str, nc_path = frame_tuple[0], frame_tuple[1]
                satellite = frame_tuple[2] if len(frame_tuple) > 2 else ""

                frame_2d, bounds = _load_frame_from_nc(nc_path)
                if frame_2d is None:
                    return (frame_idx, None)

                png = _render_ir_png(frame_2d, vmin=170.0, vmax=310.0)
                result = {
                    "sid": sid, "frame_idx": frame_idx,
                    "datetime": dt_str, "source": "hursat",
                    "frame": png,
                }
                if satellite:
                    result["satellite"] = satellite
                if bounds:
                    result["bounds"] = bounds

            # Cache it
            _frame_cache[cache_key] = result
            if len(_frame_cache) > _FRAME_CACHE_MAX:
                _frame_cache.popitem(last=False)

            return (frame_idx, result)

        except Exception as e:
            logger.warning(f"Batch frame {frame_idx} failed: {e}")
            return (frame_idx, None)

    # Fetch frames concurrently
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_single_frame, idx): idx for idx in idx_list}
        for future in as_completed(futures):
            idx, data = future.result()
            results[str(idx)] = data

    return JSONResponse(
        {"sid": sid, "source": source, "frames": results},
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/health")
def global_health():
    """Health check for global archive endpoints."""
    return {
        "status": "ok",
        "hursat_cache_size": len(_extracted_cache),
        "frame_cache_size": len(_frame_cache),
        "hursat_meta_cache_size": len(_meta_cache),
        "ir_meta_cache_size": len(_mergir_meta_cache),
        "extracted_storms": list(_extracted_cache.keys()),
        "earthdata_configured": bool(EARTHDATA_TOKEN or EARTHDATA_USER),
        "earthdata_method": "token" if EARTHDATA_TOKEN else ("user/pass" if EARTHDATA_USER else "none"),
        "mergir_available": bool(EARTHDATA_TOKEN or EARTHDATA_USER),
    }


@router.get("/hursat/inspect")
def hursat_inspect(sid: str = Query("1992230N11325", description="IBTrACS storm ID")):
    """Inspect the first extracted NetCDF file to see variables and structure."""
    import xarray as xr

    frames = _get_extracted_frames(sid, storm_lon=0.0)
    if not frames:
        return {"error": "No frames extracted", "sid": sid}

    frame_tuple = frames[0]
    dt_str, nc_path = frame_tuple[0], frame_tuple[1]
    satellite = frame_tuple[2] if len(frame_tuple) > 2 else ""
    result = {"sid": sid, "n_frames": len(frames), "first_file": nc_path, "datetime": dt_str}
    if satellite:
        result["satellite"] = satellite

    try:
        ds = xr.open_dataset(nc_path, engine="h5netcdf")
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
