"""
realtime_tdr_api.py — Real-Time TDR Endpoints
===============================================
Add-on module for tc_radar_api.py that provides endpoints for browsing
and visualizing real-time Tail Doppler Radar (TDR) analyses from
seb.omao.noaa.gov/pub/flight/radar/.

Also provides GOES IR satellite imagery endpoints for real-time
storm context, sourcing ABI Band 13 (10.3 µm clean IR window) data
from NOAA's public AWS S3 buckets with xarray byte-range subsetting.

How to integrate:
    In tc_radar_api.py, add near the bottom:
        from realtime_tdr_api import router as realtime_router
        app.include_router(realtime_router, prefix="/realtime")

Or simply paste the router endpoints into tc_radar_api.py directly.

Dependencies (all already in tc_radar_api.py):
    fastapi, xarray, numpy, matplotlib, requests (add if not present)
Additional for GOES IR: s3fs, pyproj, Pillow
"""

import base64
import gc
import gzip
import io
import re
import time
from collections import OrderedDict
from datetime import datetime as _dt, timedelta, timezone
from html.parser import HTMLParser
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response

try:
    import requests as _requests
except ImportError:
    import urllib.request as _urllib
    _requests = None

# Optional GOES IR dependencies — LAZY loaded to save ~80 MB RAM at startup.
# The actual imports happen inside _get_s3fs_module() and _get_pyproj_module()
# only when an IR endpoint is called.
_s3fs_mod = None      # lazy: import s3fs
_pyproj_mod = None    # lazy: import pyproj


def _get_s3fs_module():
    """Lazy-import s3fs on first use."""
    global _s3fs_mod
    if _s3fs_mod is None:
        try:
            import s3fs
            _s3fs_mod = s3fs
        except ImportError:
            return None
    return _s3fs_mod


def _get_pyproj_module():
    """Lazy-import pyproj on first use."""
    global _pyproj_mod
    if _pyproj_mod is None:
        try:
            import pyproj
            _pyproj_mod = pyproj
        except ImportError:
            return None
    return _pyproj_mod

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEB_BASE = "https://seb.omao.noaa.gov/pub/flight/radar"

# Variable mapping: real-time TDR → display metadata
# key: variable name in the NetCDF file
# value: (display_name, colormap, units, vmin, vmax)
RT_VARIABLES = {
    "TANGENTIAL_WIND": ("Tangential Wind",   "jet",        "m/s",  -10,  80),
    "RADIAL_WIND":     ("Radial Wind",        "RdBu_r",    "m/s",  -30,  30),
    "W":               ("Vertical Velocity",  "RdBu_r",    "m/s",   -5,   5),
    "REFLECTIVITY":    ("Reflectivity",       "Spectral_r", "dBZ", -10,  65),
    "WIND_SPEED":      ("Wind Speed",         "inferno",   "m/s",    0,  80),
    "VORT":            ("Relative Vorticity", "RdBu_r",    "s⁻¹", -5e-3, 5e-3),
    "U":               ("Eastward Wind (U)",  "RdBu_r",    "m/s",  -40,  40),
    "V":               ("Northward Wind (V)", "RdBu_r",    "m/s",  -40,  40),
}

# Derived variable: earth-relative wind speed from U and V
RT_DERIVED = {
    "EARTH_REL_WSPD": {
        "display_name": "Earth-Rel. Wind Speed",
        "components": ("U", "V"),
        "cmap": "jet",
        "units": "m/s",
        "vmin": 0,
        "vmax": 80,
    }
}

DEFAULT_RT_VARIABLE = "TANGENTIAL_WIND"

# ---------------------------------------------------------------------------
# GOES IR Satellite Configuration
# ---------------------------------------------------------------------------
# Satellite buckets (public, no auth needed)
#   GOES-East: GOES-16 (pre Apr 2025) / GOES-19 (post Apr 2025)
#   GOES-West: GOES-18
GOES_BUCKETS = {
    "east_16": "noaa-goes16",
    "east_19": "noaa-goes19",
    "west":    "noaa-goes18",
}
# Sub-satellite longitude for geostationary projection
GOES_LON_0 = {"east": -75.2, "west": -137.2}
GOES_SAT_HEIGHT = 35786023.0        # metres above Earth centre
GOES_TRANSITION_DT = _dt(2025, 4, 4, 15, 0, 0, tzinfo=timezone.utc)  # GOES-19 operational

IR_PRODUCT = "ABI-L2-CMIPF"         # full-disk Cloud & Moisture Imagery
IR_BAND = 13                         # 10.3 µm clean longwave IR window
IR_VARIABLE = "CMI"                  # variable name inside CMI single-band file
IR_LOOKBACK_H = 4                    # hours of lookback
IR_INTERVAL_MIN = 30                 # minutes between animation frames (was 15)
IR_N_FRAMES = int(IR_LOOKBACK_H * 60 / IR_INTERVAL_MIN) + 1  # 9 (t=0 … t−4h)
IR_BOX_DEG = 8.0                     # geographic crop box (degrees, was 10)
IR_VMIN = 190.0                      # brightness temperature colour limits (K)
IR_VMAX = 310.0

# Enhanced IR colormap LUT (cold → bright/colourful, warm → dark grey)
# Same stops as the archive MergIR LUT in tc_radar_api.py
_IR_STOPS = [
    (0.00,   8,   8,   8),
    (0.15,  40,  40,  40),
    (0.30,  90,  90,  90),
    (0.40, 140, 140, 140),
    (0.50, 200, 200, 200),
    (0.55,   0, 180, 255),
    (0.60,   0, 100, 255),
    (0.65,   0, 255,   0),
    (0.70, 255, 255,   0),
    (0.75, 255, 180,   0),
    (0.80, 255,  80,   0),
    (0.85, 255,   0,   0),
    (0.90, 180,   0, 180),
    (0.95, 255, 180, 255),
    (1.00, 255, 255, 255),
]


def _build_ir_lut() -> np.ndarray:
    """Build a 256-entry uint8 RGBA LUT for IR brightness temperatures."""
    lut = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256):
        frac = i / 255.0
        lo, hi = _IR_STOPS[0], _IR_STOPS[-1]
        for s in range(len(_IR_STOPS) - 1):
            if _IR_STOPS[s][0] <= frac <= _IR_STOPS[s + 1][0]:
                lo, hi = _IR_STOPS[s], _IR_STOPS[s + 1]
                break
        t = 0.0 if hi[0] == lo[0] else (frac - lo[0]) / (hi[0] - lo[0])
        lut[i, 0] = int(lo[1] + t * (hi[1] - lo[1]) + 0.5)
        lut[i, 1] = int(lo[2] + t * (hi[2] - lo[2]) + 0.5)
        lut[i, 2] = int(lo[3] + t * (hi[3] - lo[3]) + 0.5)
        lut[i, 3] = 220  # semi-transparent alpha
    return lut


_IR_LUT = _build_ir_lut()

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
_rt_ds_cache = OrderedDict()       # file_url → (xr.Dataset, timestamp)
_rt_dir_cache = OrderedDict()      # dir_url  → (link_list, timestamp)
_RT_DS_CACHE_MAX = 3               # was 8 — keep fewer datasets in RAM
_RT_DIR_CACHE_TTL = 300            # 5 minutes for directory listings

# GOES IR frame cache: (file_url, frame_index) → rendered result dict
# Kept small because browsers cache via Cache-Control headers.
_rt_ir_cache = OrderedDict()
_RT_IR_CACHE_MAX = 5               # minimal — browser handles long-term caching

# Shared S3 filesystem (lazy-initialised)
_goes_fs = None

router = APIRouter(tags=["realtime"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_bytes(url: str, timeout: int = 60) -> bytes:
    """Fetch raw bytes from a URL."""
    if _requests:
        resp = _requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    else:
        req = _urllib.Request(url)
        with _urllib.urlopen(req, timeout=timeout) as resp:
            return resp.read()


def _fetch_text(url: str, timeout: int = 30) -> str:
    """Fetch text content from a URL."""
    if _requests:
        resp = _requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    else:
        req = _urllib.Request(url)
        with _urllib.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")


class _LinkParser(HTMLParser):
    """Extract href links from an Apache directory listing."""
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value and not value.startswith("?") and not value.startswith("/"):
                    self.links.append(value)


def _parse_directory(url: str) -> list[str]:
    """Fetch an Apache directory listing and return the link names."""
    now = time.time()
    if url in _rt_dir_cache:
        links, ts = _rt_dir_cache[url]
        if now - ts < _RT_DIR_CACHE_TTL:
            _rt_dir_cache.move_to_end(url)
            return links

    html = _fetch_text(url)
    parser = _LinkParser()
    parser.feed(html)
    links = parser.links
    _rt_dir_cache[url] = (links, now)
    if len(_rt_dir_cache) > 50:
        _rt_dir_cache.popitem(last=False)
    return links


def _open_rt_dataset(file_url: str) -> xr.Dataset:
    """Download, decompress (if .gz), and open a real-time TDR NetCDF file."""
    if file_url in _rt_ds_cache:
        ds, _ = _rt_ds_cache[file_url]
        _rt_ds_cache.move_to_end(file_url)
        return ds

    raw = _fetch_bytes(file_url, timeout=120)

    # Decompress if gzipped — check the actual bytes, not just the URL,
    # because the HTTP server may transparently decompress via Content-Encoding.
    # Gzip magic number is b'\x1f\x8b'; HDF5/NetCDF4 starts with b'\x89HDF' or b'CDF'.
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)

    # Detect file format from magic bytes and choose the right xarray engine.
    # netCDF3 classic starts with b'CDF'; HDF5/netCDF4 starts with b'\x89HDF'.
    if raw[:3] == b'CDF':
        engine = "scipy"
    else:
        engine = "h5netcdf"

    ds = xr.open_dataset(io.BytesIO(raw), engine=engine)
    _rt_ds_cache[file_url] = (ds, time.time())
    if len(_rt_ds_cache) > _RT_DS_CACHE_MAX:
        _rt_ds_cache.popitem(last=False)
    return ds


def _cmap_to_plotly(cmap_name: str, n_steps: int = 64) -> list:
    """Convert a matplotlib colormap name to a Plotly-compatible colorscale."""
    cmap = plt.get_cmap(cmap_name)
    return [
        [round(i / (n_steps - 1), 4),
         f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"]
        for i, c in enumerate(cmap(np.linspace(0, 1, n_steps)))
    ]


def _clean_2d(data):
    """Convert 2D numpy array to JSON-safe nested list (NaN → None)."""
    return [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in data]


def _get_xy_coords(ds):
    """Extract x/y coordinate arrays in km (storm-centered)."""
    x = ds["x"].values.astype(float)
    y = ds["y"].values.astype(float)

    # The file stores grid indices; convert to km using resolution and origin offset.
    # If coordinates are already in km (e.g. values range beyond 0–250), use them directly.
    if x.max() <= 250 and x.min() >= 0:
        res_ew = float(ds.attrs.get("EAST_WEST_RESOLUTION_KM", 2.0))
        res_ns = float(ds.attrs.get("NORTH_SOUTH_RESOLUTION_KM", 2.0))
        origin_ew = float(ds.attrs.get("ORIGIN_EAST_WEST_DISTANCE_FROM_LOWER_LEFT_CORNER_KM", 250.0))
        origin_ns = float(ds.attrs.get("ORIGIN_NORTH_SOUTH_DISTANCE_FROM_LOWER_LEFT_CORNER_KM", 250.0))
        x_km = (x * res_ew) - origin_ew
        y_km = (y * res_ns) - origin_ns
    else:
        x_km = x
        y_km = y
    return x_km, y_km


def _get_level_axis(ds):
    """Return the vertical level array in km."""
    return ds["level"].values.astype(float)


def _extract_2d(ds, variable: str, level_km: float):
    """Extract a 2D (x, y) slice at a given height level, handling dim order."""
    levels = _get_level_axis(ds)
    z_idx = int(np.argmin(np.abs(levels - level_km)))
    actual_level = float(levels[z_idx])

    if variable in RT_DERIVED:
        info = RT_DERIVED[variable]
        u_name, v_name = info["components"]
        u = ds[u_name].isel(time=0, level=z_idx).values
        v = ds[v_name].isel(time=0, level=z_idx).values
        data = np.sqrt(u**2 + v**2)
    else:
        data = ds[variable].isel(time=0, level=z_idx).values

    # Ensure shape is (y, x) for Plotly heatmap (row = y, col = x)
    # File dims are (x, y, level, time), so after isel we have (x, y).
    # Transpose so rows = y-axis, cols = x-axis.
    if data.shape[0] == ds.sizes.get("x", 0) and data.shape[1] == ds.sizes.get("y", 0):
        data = data.T

    return data, actual_level


def _extract_3d(ds, variable: str, max_height_km: float = 18.0):
    """Extract the full 3D volume (level, y, x)."""
    levels = _get_level_axis(ds)
    h_mask = levels <= max_height_km + 0.01

    if variable in RT_DERIVED:
        info = RT_DERIVED[variable]
        u_name, v_name = info["components"]
        u = ds[u_name].isel(time=0).values   # (x, y, level)
        v = ds[v_name].isel(time=0).values
        vol = np.sqrt(u**2 + v**2)
    else:
        vol = ds[variable].isel(time=0).values  # (x, y, level)

    # Reorder from (x, y, level) → (level, y, x) for consistency
    vol = np.transpose(vol, (2, 1, 0))  # (level, y, x)

    # Trim to max height
    n_h = int(h_mask.sum())
    vol = vol[:n_h, :, :]
    return vol, levels[h_mask]


def _build_case_meta(ds) -> dict:
    """Build metadata dict from global attributes."""
    attrs = ds.attrs
    # Construct analysis center time
    hour = int(attrs.get("ANALYSIS CENTER HOUR", 0))
    minute = int(attrs.get("ANALYSIS CENTER MINUTE", 0))
    second = int(attrs.get("ANALYSIS CENTER SECOND", 0))
    year = int(attrs.get("FLIGHT YEAR", 0))
    month = int(attrs.get("FLIGHT MONTH", 0))
    day = int(attrs.get("FLIGHT DAY", 0))

    datetime_str = ""
    if year > 0:
        # Handle hour >= 24 (common in met data — e.g. hour 25 = 01Z next day)
        if hour >= 24:
            base_date = _dt(year, month, day, tzinfo=timezone.utc)
            adjusted = base_date + timedelta(hours=hour, minutes=minute, seconds=second)
            datetime_str = adjusted.strftime("%Y-%m-%d %H:%M:%SZ")
        else:
            datetime_str = f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}Z"

    return {
        "storm_name": str(attrs.get("STORM NAME", attrs.get("STMNAME", "Unknown"))).strip(),
        "mission_id": str(attrs.get("FLTNAME", "")).strip(),
        "datetime": datetime_str,
        "latitude": float(attrs.get("ORIGIN_LATITUDE", 0)),
        "longitude": float(attrs.get("ORIGIN_LONGITUDE", 0)),
        "radar": str(attrs.get("RADAR", "TAIL")).strip(),
        "experiment": str(attrs.get("EXPERIMENT", "")).strip(),
        "resolution_km": float(attrs.get("EAST_WEST_RESOLUTION_KM", 2.0)),
        "storm_motion_north_ms": float(attrs.get("NORTHWARD STORM MOTION (METERS PER SECOND)", -999)),
        "storm_motion_east_ms": float(attrs.get("EASTWARD STORM MOTION (METERS PER SECOND)", -999)),
        "analysis_level": str(attrs.get("ANALYSIS LEVEL (1--REAL-TIME,2--RESEARCH QUALITY)", "")).strip(),
        "melting_height_km": float(attrs.get("HEIGHT OF CENTER OF MELTING BAND (KM)", -999)),
        "vmax_kt": None,  # Not available in real-time files
    }


# ---------------------------------------------------------------------------
# GOES IR Helpers
# ---------------------------------------------------------------------------

def _get_goes_fs():
    """Return a shared s3fs filesystem for public NOAA GOES buckets."""
    global _goes_fs
    if _goes_fs is None:
        s3fs = _get_s3fs_module()
        if s3fs is None:
            return None
        _goes_fs = s3fs.S3FileSystem(anon=True)
    return _goes_fs


def _select_goes_sat(longitude: float, analysis_dt: _dt) -> tuple[str, str]:
    """
    Select GOES satellite based on storm longitude and analysis date.

    Returns (bucket_name, sat_key) where sat_key is 'east' or 'west'.
    GOES-East: lon > −115°  (GOES-16 before Apr 2025, GOES-19 after)
    GOES-West: lon ≤ −115°  (GOES-18)
    """
    if longitude > -115:
        sat_key = "east"
        # GOES-19 became operational GOES-East on 2025-04-04 15:00 UTC
        if analysis_dt.replace(tzinfo=timezone.utc) >= GOES_TRANSITION_DT:
            bucket = GOES_BUCKETS["east_19"]
        else:
            bucket = GOES_BUCKETS["east_16"]
    else:
        sat_key = "west"
        bucket = GOES_BUCKETS["west"]
    return bucket, sat_key


def _find_goes_file(bucket: str, target_dt: _dt, tolerance_min: int = 15) -> Optional[str]:
    """
    Find the GOES ABI Band 13 full-disk file closest to target_dt.

    Searches the S3 directory for the target hour, parses start-time from
    each filename, and returns the full S3 key of the best match (or None).
    """
    fs = _get_goes_fs()
    if fs is None:
        return None

    jday = target_dt.timetuple().tm_yday
    prefix = f"{bucket}/{IR_PRODUCT}/{target_dt.year}/{jday:03d}/{target_dt.hour:02d}/"

    try:
        files = fs.ls(prefix, detail=False)
    except Exception:
        return None

    # Filter to Band 13 files only
    band_tag = f"C{IR_BAND:02d}"
    candidates = [f for f in files if band_tag in f.split("/")[-1]]
    if not candidates:
        return None

    # Parse start timestamp from filename:
    #   GOES-16: OR_ABI-L2-CMIPF-M6C13-G16-s20231501200432_e…_c….nc
    #   GOES-19: OR_ABI-L2-CMIPF-M6C13_G19_s20253011300205_e…_c….nc
    best_file = None
    best_delta = timedelta(minutes=tolerance_min + 1)
    ts_re = re.compile(r"[-_]s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})")

    for fpath in candidates:
        fname = fpath.split("/")[-1]
        m = ts_re.search(fname)
        if not m:
            continue
        try:
            yr, jd, hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
            file_dt = _dt(yr, 1, 1, hh, mm, ss, tzinfo=timezone.utc) + timedelta(days=jd - 1)
            delta = abs(file_dt - target_dt.replace(tzinfo=timezone.utc))
            if delta < best_delta:
                best_delta = delta
                best_file = fpath
        except Exception:
            continue

    if best_delta > timedelta(minutes=tolerance_min):
        return None
    return best_file


def _latlon_to_goes_xy(lat: float, lon: float, sat_key: str) -> tuple[float, float]:
    """
    Convert geographic (lat, lon) to GOES fixed-grid (x, y) in radians.

    Uses the geostationary projection with sweep='x' (GOES-R convention).
    Returns (x_rad, y_rad).
    """
    pyproj = _get_pyproj_module()
    if pyproj is None:
        raise RuntimeError("pyproj is required for GOES IR subsetting")
    lon_0 = GOES_LON_0[sat_key]
    proj = pyproj.Proj(proj="geos", h=GOES_SAT_HEIGHT, lon_0=lon_0, sweep="x")
    x_m, y_m = proj(lon, lat)
    # Convert metres → scanning-angle radians (divide by satellite height)
    return x_m / GOES_SAT_HEIGHT, y_m / GOES_SAT_HEIGHT


def _open_goes_subset(s3_key: str, center_lat: float, center_lon: float,
                      sat_key: str) -> np.ndarray:
    """
    Open a GOES CMI file from S3 and return a geographically-subsetted
    2D brightness-temperature array (y, x) in Kelvin.

    Uses xarray + s3fs byte-range reads so only the subset chunks are
    downloaded (~2-4 MB instead of ~25 MB).
    """
    fs = _get_goes_fs()
    if fs is None:
        raise RuntimeError("s3fs not available")

    half = IR_BOX_DEG / 2.0
    x_min, y_min = _latlon_to_goes_xy(center_lat - half, center_lon - half, sat_key)
    x_max, y_max = _latlon_to_goes_xy(center_lat + half, center_lon + half, sat_key)
    # Ensure ascending order for x, descending for y (GOES convention)
    x_lo, x_hi = min(x_min, x_max), max(x_min, x_max)
    y_lo, y_hi = min(y_min, y_max), max(y_min, y_max)

    fobj = fs.open(f"s3://{s3_key}", "rb")
    try:
        ds = xr.open_dataset(fobj, engine="h5netcdf")

        # Subset — GOES y-axis is descending, x is ascending
        ds_sub = ds.sel(x=slice(x_lo, x_hi), y=slice(y_hi, y_lo))

        # Extract brightness temperature
        if IR_VARIABLE in ds_sub:
            tb = ds_sub[IR_VARIABLE].values.astype(np.float32)
        else:
            # Some files use CMI_C13 instead of CMI
            alt_var = f"CMI_C{IR_BAND:02d}"
            if alt_var in ds_sub:
                tb = ds_sub[alt_var].values.astype(np.float32)
            else:
                raise ValueError(f"Neither {IR_VARIABLE} nor {alt_var} found in dataset")
    finally:
        ds.close()
        fobj.close()
        del ds, ds_sub  # noqa: F821
        gc.collect()
    return tb


def _render_ir_png(frame_2d: np.ndarray) -> Optional[str]:
    """
    Render a 2D Tb array to a base64-encoded PNG data-URL.
    Uses the enhanced IR colormap LUT (cold → bright colours).
    Returns None if all data is NaN.
    """
    from PIL import Image

    arr = np.asarray(frame_2d, dtype=np.float32)
    if not np.any(np.isfinite(arr)):
        return None

    # Normalise: cold clouds (low Tb) → high index → bright colours
    frac = 1.0 - (arr - IR_VMIN) / (IR_VMAX - IR_VMIN)
    frac = np.clip(frac, 0.0, 1.0)
    indices = (frac * 255).astype(np.uint8)

    # Apply LUT
    rgba = _IR_LUT[indices]  # (H, W, 4)

    # Set NaN / invalid pixels to transparent
    mask = ~np.isfinite(arr) | (arr <= 0)
    rgba[mask] = [0, 0, 0, 0]

    # NOTE: No vertical flip here.  GOES y-axis is descending (north at
    # row 0), which matches standard image orientation (top = north) and
    # Leaflet's L.imageOverlay expectation.  For Plotly underlays the
    # client swaps the y-axis bounds instead.

    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _build_frame_times(analysis_dt: _dt) -> list[_dt]:
    """
    Build list of target GOES scan times for the IR animation.
    Returns IR_N_FRAMES datetimes from t=0 (most recent) to t−4h.
    """
    base = analysis_dt.replace(tzinfo=timezone.utc) if analysis_dt.tzinfo is None else analysis_dt
    return [base - timedelta(minutes=i * IR_INTERVAL_MIN) for i in range(IR_N_FRAMES)]


def _parse_tdr_datetime(meta: dict) -> _dt:
    """Parse the analysis datetime from TDR case_meta dict."""
    dt_str = meta.get("datetime", "")
    if not dt_str:
        raise ValueError("No datetime in TDR metadata")
    # Format: "YYYY-MM-DD HH:MM:SSZ"
    dt_str = dt_str.rstrip("Z").strip()
    return _dt.fromisoformat(dt_str).replace(tzinfo=timezone.utc)


def _get_ir_bounds_km(center_lat: float) -> dict:
    """
    Return the IR box extents in km (for Plotly underlay positioning).
    Box is IR_BOX_DEG × IR_BOX_DEG centred on the storm.
    """
    half = IR_BOX_DEG / 2.0
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(center_lat))
    return {
        "y_min_km": -half * km_per_deg_lat,
        "y_max_km":  half * km_per_deg_lat,
        "x_min_km": -half * km_per_deg_lon,
        "x_max_km":  half * km_per_deg_lon,
    }


def _get_variable_info(variable: str) -> dict:
    """Get display metadata for a variable key."""
    if variable in RT_VARIABLES:
        display_name, cmap, units, vmin, vmax = RT_VARIABLES[variable]
    elif variable in RT_DERIVED:
        info = RT_DERIVED[variable]
        display_name = info["display_name"]
        cmap = info["cmap"]
        units = info["units"]
        vmin = info["vmin"]
        vmax = info["vmax"]
    else:
        raise HTTPException(status_code=400, detail=f"Unknown variable '{variable}'.")

    return {
        "key": variable,
        "display_name": display_name,
        "units": units,
        "vmin": vmin,
        "vmax": vmax,
        "colorscale": _cmap_to_plotly(cmap),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/missions")
def list_missions():
    """
    List available mission folders from the SEB radar archive.
    Returns folder names that look like missions (e.g. '20251028H1/').
    """
    try:
        links = _parse_directory(SEB_BASE + "/")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not reach SEB server: {e}")

    # Filter to directories that look like mission IDs (digits + letter + digit pattern)
    missions = []
    for link in links:
        name = link.rstrip("/")
        if re.match(r"^\d{6,8}[A-Za-z]\d", name):
            missions.append(name)

    # Sort reverse chronological
    missions.sort(reverse=True)
    return JSONResponse({"missions": missions})


@router.get("/files")
def list_files(
    mission: str = Query(..., description="Mission folder name, e.g. '20251028H1'"),
):
    """
    List available xy.nc(.gz) analysis files within a mission folder.
    Returns file entries with name and URL.
    """
    url = f"{SEB_BASE}/{mission}/"
    try:
        links = _parse_directory(url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not list files for {mission}: {e}")

    # Filter to xy analysis files
    files = []
    for link in links:
        if "_xy.nc" in link:
            # Parse analysis time from filename like 251028H1_1349_xy.nc.gz
            m = re.match(r".*?_(\d{4})_xy\.nc", link)
            time_label = m.group(1) if m else ""
            files.append({
                "filename": link,
                "url": f"{SEB_BASE}/{mission}/{link}",
                "time_label": time_label,
            })

    files.sort(key=lambda f: f["filename"])
    return JSONResponse({"mission": mission, "files": files})


@router.get("/variables")
def list_rt_variables():
    """List available variables for real-time TDR files."""
    result = []
    for key, (display_name, _, units, _, _) in RT_VARIABLES.items():
        result.append({"key": key, "display_name": display_name, "units": units})
    for key, info in RT_DERIVED.items():
        result.append({"key": key, "display_name": info["display_name"], "units": info["units"]})
    return result


@router.get("/data")
def get_rt_data(
    file_url:   str   = Query(...,                           description="Full URL to the xy.nc(.gz) file"),
    variable:   str   = Query(DEFAULT_RT_VARIABLE,           description="Variable name"),
    level_km:   float = Query(2.0,            ge=0.0, le=18, description="Altitude in km"),
    overlay:    str   = Query("",                            description="Optional overlay variable"),
):
    """Return a 2D plan-view data slice as JSON for client-side Plotly rendering."""
    if variable not in RT_VARIABLES and variable not in RT_DERIVED:
        raise HTTPException(status_code=400, detail=f"Unknown variable '{variable}'.")
    if overlay and overlay not in RT_VARIABLES and overlay not in RT_DERIVED:
        raise HTTPException(status_code=400, detail=f"Unknown overlay variable '{overlay}'.")

    try:
        ds = _open_rt_dataset(file_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open file: {e}")

    x_km, y_km = _get_xy_coords(ds)
    data, actual_level = _extract_2d(ds, variable, level_km)
    var_info = _get_variable_info(variable)
    case_meta = _build_case_meta(ds)

    result = {
        "data": _clean_2d(data),
        "x": np.round(x_km, 2).tolist(),
        "y": np.round(y_km, 2).tolist(),
        "actual_level_km": actual_level,
        "variable": var_info,
        "case_meta": case_meta,
    }

    # Optional overlay
    if overlay:
        ov_data, _ = _extract_2d(ds, overlay, level_km)
        ov_info = _get_variable_info(overlay)
        result["overlay"] = {
            "data": _clean_2d(ov_data),
            "display_name": ov_info["display_name"],
            "units": ov_info["units"],
            "vmin": ov_info["vmin"],
            "vmax": ov_info["vmax"],
        }

    return JSONResponse(result)


@router.get("/cross_section")
def get_rt_cross_section(
    file_url:   str   = Query(...,                           description="Full URL to the xy.nc(.gz) file"),
    variable:   str   = Query(DEFAULT_RT_VARIABLE,           description="Variable name"),
    x0:         float = Query(...,                           description="Start X (km)"),
    y0:         float = Query(...,                           description="Start Y (km)"),
    x1:         float = Query(...,                           description="End X (km)"),
    y1:         float = Query(...,                           description="End Y (km)"),
    n_points:   int   = Query(150,            ge=10, le=500, description="Sample points along line"),
    overlay:    str   = Query("",                            description="Optional overlay variable"),
):
    """Return a vertical cross-section along a user-defined line."""
    if variable not in RT_VARIABLES and variable not in RT_DERIVED:
        raise HTTPException(status_code=400, detail=f"Unknown variable '{variable}'.")
    if overlay and overlay not in RT_VARIABLES and overlay not in RT_DERIVED:
        raise HTTPException(status_code=400, detail=f"Unknown overlay variable '{overlay}'.")

    try:
        ds = _open_rt_dataset(file_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open file: {e}")

    x_km, y_km = _get_xy_coords(ds)
    levels = _get_level_axis(ds)
    n_heights = len(levels)

    # Sample points along the line
    xs = np.linspace(x0, x1, n_points)
    ys = np.linspace(y0, y1, n_points)

    # Find nearest grid indices
    xi_idx = np.array([int(np.argmin(np.abs(x_km - xp))) for xp in xs])
    yi_idx = np.array([int(np.argmin(np.abs(y_km - yp))) for yp in ys])

    # Distance along line
    distance = np.sqrt((xs - x0)**2 + (ys - y0)**2)

    def _extract_cs(var_key):
        vol, _ = _extract_3d(ds, var_key, max_height_km=18.0)  # (level, y, x)
        cs = np.full((n_heights, n_points), np.nan)
        for h in range(n_heights):
            for p in range(n_points):
                cs[h, p] = vol[h, yi_idx[p], xi_idx[p]]
        return cs

    cs_data = _extract_cs(variable)
    var_info = _get_variable_info(variable)
    case_meta = _build_case_meta(ds)

    result = {
        "cross_section": _clean_2d(cs_data),
        "distance_km": np.round(distance, 2).tolist(),
        "height_km": np.round(levels, 2).tolist(),
        "variable": var_info,
        "case_meta": case_meta,
        "endpoints": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
    }

    if overlay:
        ov_cs = _extract_cs(overlay)
        ov_info = _get_variable_info(overlay)
        result["overlay"] = {
            "cross_section": _clean_2d(ov_cs),
            "display_name": ov_info["display_name"],
            "units": ov_info["units"],
            "vmin": ov_info["vmin"],
            "vmax": ov_info["vmax"],
        }

    return JSONResponse(result)


@router.get("/volume")
def get_rt_volume(
    file_url:      str   = Query(...,                              description="Full URL to the xy.nc(.gz) file"),
    variable:      str   = Query(DEFAULT_RT_VARIABLE,              description="Variable name"),
    stride:        int   = Query(2,                ge=1, le=5,     description="Spatial subsampling stride"),
    max_height_km: float = Query(15.0,             ge=1, le=18,   description="Maximum height (km)"),
):
    """Return the full 3D volume for Plotly isosurface rendering (compact mode)."""
    if variable not in RT_VARIABLES and variable not in RT_DERIVED:
        raise HTTPException(status_code=400, detail=f"Unknown variable '{variable}'.")

    try:
        ds = _open_rt_dataset(file_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open file: {e}")

    x_km, y_km = _get_xy_coords(ds)
    vol, height_sub = _extract_3d(ds, variable, max_height_km)

    # Subsample spatially
    x_sub = x_km[::stride]
    y_sub = y_km[::stride]
    vol_sub = vol[:, ::stride, ::stride]

    nz, ny, nx = vol_sub.shape
    v_flat = vol_sub.ravel()

    # Data range (excluding NaN)
    valid = v_flat[np.isfinite(v_flat)]
    data_min = float(np.nanmin(valid)) if len(valid) > 0 else 0
    data_max = float(np.nanmax(valid)) if len(valid) > 0 else 1

    # Replace NaN with sentinel
    SENTINEL = -9999.0
    v_flat = np.where(np.isfinite(v_flat), np.round(v_flat, 3), SENTINEL)

    var_info = _get_variable_info(variable)
    var_info["data_min"] = round(data_min, 3)
    var_info["data_max"] = round(data_max, 3)

    case_meta = _build_case_meta(ds)

    return JSONResponse({
        "value": v_flat.tolist(),
        "sentinel": SENTINEL,
        "grid_shape": [nz, ny, nx],
        "x_axis": np.round(x_sub, 2).tolist(),
        "y_axis": np.round(y_sub, 2).tolist(),
        "z_axis": np.round(height_sub, 2).tolist(),
        "variable": var_info,
        "case_meta": case_meta,
    })


# ---------------------------------------------------------------------------
# Diagnostic endpoint — remove once IR is confirmed working
# ---------------------------------------------------------------------------

@router.get("/ir_debug")
def debug_ir(
    file_url: str = Query(..., description="URL to the TDR xy.nc(.gz) file"),
):
    """Diagnostic: show what the IR pipeline sees without rendering frames."""
    info = {"steps": []}

    # Step 1: Check dependencies
    s3fs_ok = _get_s3fs_module() is not None
    pyproj_ok = _get_pyproj_module() is not None
    info["s3fs_available"] = s3fs_ok
    info["pyproj_available"] = pyproj_ok
    if not s3fs_ok or not pyproj_ok:
        info["error"] = "Missing dependencies"
        return JSONResponse(info)
    info["steps"].append("dependencies OK")

    # Step 2: Open TDR file
    try:
        ds = _open_rt_dataset(file_url)
        meta = _build_case_meta(ds)
        info["meta"] = meta
        info["steps"].append("TDR file opened")
    except Exception as e:
        info["error"] = f"TDR open failed: {e}"
        return JSONResponse(info)

    # Step 3: Parse datetime and select satellite
    try:
        analysis_dt = _parse_tdr_datetime(meta)
        info["analysis_dt"] = analysis_dt.isoformat()
    except Exception as e:
        info["error"] = f"datetime parse failed: {e}"
        return JSONResponse(info)

    bucket, sat_key = _select_goes_sat(meta["longitude"], analysis_dt)
    info["bucket"] = bucket
    info["sat_key"] = sat_key
    info["steps"].append(f"satellite: {bucket} ({sat_key})")

    # Step 4: Build frame times and check t=0
    frame_times = _build_frame_times(analysis_dt)
    t0 = frame_times[0]
    jday = t0.timetuple().tm_yday
    prefix = f"{bucket}/{IR_PRODUCT}/{t0.year}/{jday:03d}/{t0.hour:02d}/"
    info["s3_prefix"] = prefix
    info["t0_target"] = t0.isoformat()

    # Step 5: List S3 directory
    fs = _get_goes_fs()
    try:
        files = fs.ls(prefix, detail=False)
        info["s3_file_count"] = len(files)
        # Show first 5 filenames
        info["s3_files_sample"] = [f.split("/")[-1] for f in files[:5]]
        info["steps"].append(f"S3 ls returned {len(files)} files")
    except Exception as e:
        info["s3_ls_error"] = str(e)
        info["steps"].append(f"S3 ls FAILED: {e}")
        return JSONResponse(info)

    # Step 6: Filter to Band 13
    band_tag = f"C{IR_BAND:02d}"
    candidates = [f for f in files if band_tag in f.split("/")[-1]]
    info["band13_count"] = len(candidates)
    if candidates:
        info["band13_sample"] = [f.split("/")[-1] for f in candidates[:3]]

    # Step 7: Find best match
    try:
        best = _find_goes_file(bucket, t0)
        info["best_file"] = best.split("/")[-1] if best else None
        info["steps"].append(f"best match: {best}")
    except Exception as e:
        info["find_error"] = str(e)

    return JSONResponse(info)


# ---------------------------------------------------------------------------
# HTTP Caching Helpers
# ---------------------------------------------------------------------------

# Historical GOES IR frames are immutable — once rendered they never change.
# We set aggressive Cache-Control headers so the browser (and any CDN) caches
# them, eliminating repeat requests and reducing server RAM usage.
_IR_CACHE_MAX_AGE = 86400          # 24 hours for individual frames
_IR_META_CACHE_MAX_AGE = 300       # 5 minutes for the /ir metadata+frame0 response


def _cached_json_response(data: dict, max_age: int = _IR_CACHE_MAX_AGE) -> Response:
    """Return a JSONResponse with Cache-Control headers for browser caching."""
    import ujson
    body = ujson.dumps(data)
    return Response(
        content=body,
        media_type="application/json",
        headers={
            "Cache-Control": f"public, max-age={max_age}, immutable",
        },
    )


# ---------------------------------------------------------------------------
# GOES IR Satellite Imagery Endpoints
# ---------------------------------------------------------------------------

@router.get("/ir")
def get_realtime_ir(
    file_url: str = Query(..., description="URL to the TDR xy.nc(.gz) file"),
):
    """
    Return GOES IR metadata and the t=0 frame (most recent) for instant display.

    The client then calls /ir_frame for each additional frame index to
    progressively build the animation (same two-phase pattern as the
    archive IR system).
    """
    # Lazy-load IR dependencies on first call (saves ~80 MB at startup)
    missing = []
    if _get_s3fs_module() is None:
        missing.append("s3fs")
    if _get_pyproj_module() is None:
        missing.append("pyproj")
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"GOES IR not available — missing packages: {', '.join(missing)}",
        )

    # Open TDR file (likely already cached) and extract metadata
    try:
        ds = _open_rt_dataset(file_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open TDR file: {e}")

    meta = _build_case_meta(ds)
    center_lat = meta["latitude"]
    center_lon = meta["longitude"]

    try:
        analysis_dt = _parse_tdr_datetime(meta)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Select GOES satellite
    bucket, sat_key = _select_goes_sat(center_lon, analysis_dt)
    sat_label = f"GOES-East ({bucket.split('-')[-1].upper()})" if sat_key == "east" else "GOES-West (G18)"

    # Build frame time list
    frame_times = _build_frame_times(analysis_dt)
    lag_minutes = [i * IR_INTERVAL_MIN for i in range(len(frame_times))]

    # Fetch t=0 frame for instant display
    frame0_png = None
    frame0_dt_iso = None
    try:
        t0_key = _find_goes_file(bucket, frame_times[0])
        if t0_key:
            tb = _open_goes_subset(t0_key, center_lat, center_lon, sat_key)
            frame0_png = _render_ir_png(tb)
            del tb  # free raw array immediately
            frame0_dt_iso = frame_times[0].isoformat()
            # Cache it
            _rt_ir_cache[(file_url, 0)] = {
                "frame_index": 0,
                "datetime_iso": frame0_dt_iso,
                "frame": frame0_png,
            }
            if len(_rt_ir_cache) > _RT_IR_CACHE_MAX:
                _rt_ir_cache.popitem(last=False)
    except Exception as e:
        # Non-fatal: t=0 frame unavailable, client will show placeholder
        frame0_png = None

    # Build frame datetime list for the client
    frame_datetimes = []
    for ft in frame_times:
        frame_datetimes.append(ft.strftime("%Y-%m-%d %H:%M UTC"))

    # IR box bounds in km (for Plotly underlay positioning)
    bounds_km = _get_ir_bounds_km(center_lat)

    # Lat/lon bounds (for Leaflet overlay positioning)
    half_deg = IR_BOX_DEG / 2.0
    bounds_deg = {
        "lat_min": center_lat - half_deg,
        "lat_max": center_lat + half_deg,
        "lon_min": center_lon - half_deg,
        "lon_max": center_lon + half_deg,
    }

    return _cached_json_response({
        "center_lat": center_lat,
        "center_lon": center_lon,
        "satellite": sat_key,
        "satellite_label": sat_label,
        "bucket": bucket,
        "n_frames": len(frame_times),
        "lag_minutes": lag_minutes,
        "frame_datetimes": frame_datetimes,
        "frame0": frame0_png,
        "bounds_km": bounds_km,
        "bounds_deg": bounds_deg,
        "units": "K",
    }, max_age=_IR_META_CACHE_MAX_AGE)


@router.get("/ir_frame")
def get_realtime_ir_frame(
    file_url:    str = Query(..., description="URL to the TDR xy.nc(.gz) file"),
    frame_index: int = Query(..., ge=0, description="Frame index (0 = most recent)"),
):
    """
    Return a single server-rendered IR PNG frame.
    Called progressively by the client to build up the animation.
    """
    if _get_s3fs_module() is None or _get_pyproj_module() is None:
        raise HTTPException(status_code=503, detail="GOES IR not available")

    # Check server-side cache first (small — browser is primary cache)
    cache_key = (file_url, frame_index)
    if cache_key in _rt_ir_cache:
        _rt_ir_cache.move_to_end(cache_key)
        return _cached_json_response(_rt_ir_cache[cache_key])

    # Open TDR file for metadata
    try:
        ds = _open_rt_dataset(file_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open TDR file: {e}")

    meta = _build_case_meta(ds)
    center_lat = meta["latitude"]
    center_lon = meta["longitude"]

    try:
        analysis_dt = _parse_tdr_datetime(meta)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    frame_times = _build_frame_times(analysis_dt)
    if frame_index >= len(frame_times):
        raise HTTPException(status_code=400, detail=f"frame_index {frame_index} out of range (max {len(frame_times)-1})")

    bucket, sat_key = _select_goes_sat(center_lon, analysis_dt)
    target_dt = frame_times[frame_index]

    # Fetch and render
    png = None
    try:
        s3_key = _find_goes_file(bucket, target_dt)
        if s3_key:
            tb = _open_goes_subset(s3_key, center_lat, center_lon, sat_key)
            png = _render_ir_png(tb)
            del tb
    except Exception:
        png = None
    finally:
        gc.collect()

    result = {
        "frame_index": frame_index,
        "datetime_iso": target_dt.strftime("%Y-%m-%d %H:%M UTC"),
        "frame": png,
    }

    # Light server-side cache (browser is primary via Cache-Control)
    _rt_ir_cache[cache_key] = result
    if len(_rt_ir_cache) > _RT_IR_CACHE_MAX:
        _rt_ir_cache.popitem(last=False)

    return _cached_json_response(result)


# ---------------------------------------------------------------------------
# Azimuthal Mean
# ---------------------------------------------------------------------------

def _compute_azimuthal_mean(vol, x_coords, y_coords, height_vals,
                            max_radius, dr, coverage_min):
    """
    Compute azimuthal mean from a 3D Cartesian volume (level, y, x).

    Returns:
        az_mean:  2D array (n_heights × n_rbins) — NaN where coverage < threshold
        coverage: 2D array (n_heights × n_rbins)
        r_bins:   1D array of radius bin centres (km)
    """
    xx, yy = np.meshgrid(x_coords, y_coords)
    rr = np.sqrt(xx**2 + yy**2)

    r_edges = np.arange(0, max_radius + dr, dr)
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2.0
    n_rbins = len(r_centers)
    n_heights = len(height_vals)

    bin_idx = np.digitize(rr, r_edges) - 1  # (ny, nx), values 0..n_rbins-1

    az_mean  = np.full((n_heights, n_rbins), np.nan)
    coverage = np.full((n_heights, n_rbins), 0.0)

    for h in range(n_heights):
        slab = vol[h, :, :]  # (y, x)
        valid = ~np.isnan(slab)
        for r in range(n_rbins):
            mask = (bin_idx == r)
            n_total = np.count_nonzero(mask)
            if n_total == 0:
                continue
            in_bin = mask & valid
            n_valid = np.count_nonzero(in_bin)
            frac = n_valid / n_total
            coverage[h, r] = frac
            if frac >= coverage_min:
                az_mean[h, r] = float(np.nanmean(slab[in_bin]))

    return az_mean, coverage, r_centers


@router.get("/azimuthal_mean")
def get_rt_azimuthal_mean(
    file_url:      str   = Query(..., description="URL of the TDR netCDF file"),
    variable:      str   = Query(DEFAULT_RT_VARIABLE, description="Variable key"),
    max_radius_km: float = Query(200.0, ge=10, le=500, description="Max radius (km)"),
    dr_km:         float = Query(2.0, ge=0.5, le=20, description="Radial bin width (km)"),
    coverage_min:  float = Query(0.5, ge=0.0, le=1.0, description="Min data coverage fraction"),
    overlay:       str   = Query("", description="Optional overlay variable key"),
):
    """Return azimuthal-mean radius-height cross-section for a real-time TDR file."""
    if variable not in RT_VARIABLES and variable not in RT_DERIVED:
        raise HTTPException(status_code=400, detail=f"Unknown variable '{variable}'.")
    if overlay and overlay not in RT_VARIABLES and overlay not in RT_DERIVED:
        raise HTTPException(status_code=400, detail=f"Unknown overlay variable '{overlay}'.")

    ds = _open_rt_dataset(file_url)
    x_coords, y_coords = _get_xy_coords(ds)
    vol, heights = _extract_3d(ds, variable)

    az_mean, cov, r_centers = _compute_azimuthal_mean(
        vol, x_coords, y_coords, heights,
        max_radius_km, dr_km, coverage_min
    )

    var_info = _get_variable_info(variable)
    meta = _build_case_meta(ds)

    result = {
        "azimuthal_mean": _clean_2d(az_mean),
        "coverage": _clean_2d(cov),
        "radius_km": [round(float(r), 2) for r in r_centers],
        "height_km": [round(float(h), 2) for h in heights],
        "coverage_min": coverage_min,
        "variable": var_info,
        "case_meta": meta,
    }

    # Optional overlay
    if overlay:
        try:
            ov_vol, ov_heights = _extract_3d(ds, overlay)
            ov_az, _, _ = _compute_azimuthal_mean(
                ov_vol, x_coords, y_coords, ov_heights,
                max_radius_km, dr_km, coverage_min
            )
            ov_info = _get_variable_info(overlay)
            result["overlay"] = {
                "azimuthal_mean": _clean_2d(ov_az),
                "key": overlay,
                "display_name": ov_info["display_name"],
                "units": ov_info["units"],
                "vmin": ov_info["vmin"],
                "vmax": ov_info["vmax"],
            }
        except Exception:
            pass

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Dropsonde Observations
# ---------------------------------------------------------------------------

SEB_SONDE_BASE = "https://seb.omao.noaa.gov/pub/flight/ASPEN_Data"
SONDE_TIME_WINDOW_MIN = 45  # ±45 minutes from TDR analysis center time

# Cache: file_url → (response_dict, timestamp)
_rt_sonde_cache = OrderedDict()
_RT_SONDE_CACHE_MAX = 3
_RT_SONDE_CACHE_TTL = 300  # 5 minutes


def _extract_mission_id(file_url: str) -> Optional[str]:
    """
    Extract mission ID from a TDR file URL.

    URLs look like: .../radar/20251028H1/251028H1_1349_xy.nc.gz
    The mission folder name follows the /radar/ segment.
    """
    m = re.search(r"/radar/(\d{6,8}[A-Za-z]\d[^/]*)/", file_url)
    if m:
        return m.group(1)
    # Fallback: try to parse from filename prefix (e.g. 251028H1_1349_xy.nc)
    fname = file_url.rstrip("/").split("/")[-1]
    m2 = re.match(r"(\d{6,8}[A-Za-z]\d)", fname)
    if m2:
        short_id = m2.group(1)
        if len(short_id) >= 7 and short_id[0:2].isdigit():
            yr2 = short_id[:2]
            century = "20" if int(yr2) < 70 else "19"
            return century + short_id
    return None


def _parse_sonde_launch_time(filename: str) -> Optional[_dt]:
    """
    Parse launch datetime from a dropsonde CSV filename.

    Filename format: D20251028_135043_PQC.csv
    """
    m = re.match(r"D(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", filename)
    if not m:
        return None
    try:
        hr = int(m.group(4))
        mi = int(m.group(5))
        sc = int(m.group(6))
        if hr >= 24:
            base = _dt(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                        tzinfo=timezone.utc)
            return base + timedelta(hours=hr, minutes=mi, seconds=sc)
        return _dt(
            int(m.group(1)), int(m.group(2)), int(m.group(3)),
            hr, mi, sc, tzinfo=timezone.utc,
        )
    except ValueError:
        return None


def _parse_dropsonde_csv(csv_text: str) -> Optional[dict]:
    """
    Parse an ASPEN PQC dropsonde CSV file.

    Returns a dict with:
        meta: dict of header metadata
        profile: dict of arrays (time_s, pres, temp, rh, wspd, wdir,
                                 lat, lon, alt, gps_alt, uwnd, vwnd)
    or None if parsing fails.
    """
    lines = csv_text.splitlines()
    meta = {}
    fields_line = None
    data_start = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Fields,"):
            fields_line = i
            data_start = i + 2  # skip Units line
            break
        parts = stripped.split(",", 1)
        if len(parts) == 2:
            key = parts[0].strip()
            val = parts[1].strip().strip('"')
            meta[key] = val

    if fields_line is None or data_start is None:
        return None

    # Parse column names from Fields line
    field_names = lines[fields_line].split(",")
    if field_names and field_names[0].strip().lower() == "fields":
        field_names = field_names[1:]
    field_names = [f.strip() for f in field_names]

    col_map = {}
    for idx, name in enumerate(field_names):
        col_map[name.lower()] = idx

    profile = {
        "time_s": [], "pres": [], "temp": [], "rh": [],
        "wspd": [], "wdir": [], "lat": [], "lon": [],
        "alt": [], "gps_alt": [], "uwnd": [], "vwnd": [],
        "ascent": [], "dewpoint": [],
    }

    aliases = {
        "time_s": ["time"],
        "pres": ["pressure"],
        "temp": ["temperature"],
        "rh": ["rh"],
        "wspd": ["speed"],
        "wdir": ["direction"],
        "lat": ["latitude"],
        "lon": ["longitude"],
        "alt": ["altitude"],
        "gps_alt": ["gps altitude", "gpsaltitude"],
        "uwnd": ["uwnd"],
        "vwnd": ["vwnd"],
        "ascent": ["ascent"],
        "dewpoint": ["dewpoint", "dp", "dew point", "dew pt", "td"],
    }

    var_cols = {}
    for var_key, names in aliases.items():
        for name in names:
            if name in col_map:
                var_cols[var_key] = col_map[name]
                break

    for i in range(data_start, len(lines)):
        stripped = lines[i].strip()
        if not stripped or not stripped.startswith("Data,"):
            continue
        parts = stripped.split(",")
        values = parts[1:]

        for var_key in profile:
            if var_key not in var_cols:
                profile[var_key].append(None)
                continue
            ci = var_cols[var_key]
            if ci < len(values) and values[ci].strip():
                try:
                    profile[var_key].append(float(values[ci].strip()))
                except (ValueError, TypeError):
                    profile[var_key].append(None)
            else:
                profile[var_key].append(None)

    if not profile["time_s"]:
        return None

    return {"meta": meta, "profile": profile}


def _latlon_to_storm_km(
    lat: float, lon: float, center_lat: float, center_lon: float
) -> tuple[float, float]:
    """Convert geographic lat/lon to storm-relative (x_km, y_km)."""
    x_km = (lon - center_lon) * 111.0 * np.cos(np.radians(center_lat))
    y_km = (lat - center_lat) * 111.0
    return float(x_km), float(y_km)


def _filter_valid_profile(profile: dict) -> dict:
    """Filter profile to rows with valid lat, lon, and alt (or gps_alt)."""
    n = len(profile["time_s"])
    mask = []
    for i in range(n):
        lat_ok = profile["lat"][i] is not None
        lon_ok = profile["lon"][i] is not None
        alt_ok = (profile["alt"][i] is not None) or (profile["gps_alt"][i] is not None)
        mask.append(lat_ok and lon_ok and alt_ok)

    filtered = {}
    for key in profile:
        filtered[key] = [profile[key][i] for i in range(n) if mask[i]]
    return filtered


def _build_sonde_response(
    parsed: dict,
    center_lat: float,
    center_lon: float,
    analysis_dt: _dt,
) -> Optional[dict]:
    """Build a single dropsonde entry for the API response."""
    meta = parsed["meta"]
    profile = _filter_valid_profile(parsed["profile"])

    if not profile["lat"]:
        return None

    x_km_arr = []
    y_km_arr = []
    alt_km_arr = []
    for i in range(len(profile["lat"])):
        x, y = _latlon_to_storm_km(
            profile["lat"][i], profile["lon"][i], center_lat, center_lon
        )
        x_km_arr.append(round(x, 3))
        y_km_arr.append(round(y, 3))
        alt_m = profile["gps_alt"][i] if profile["gps_alt"][i] is not None else profile["alt"][i]
        alt_km_arr.append(round(alt_m / 1000.0, 4) if alt_m is not None else None)

    launch = {
        "lat": profile["lat"][0],
        "lon": profile["lon"][0],
        "alt_m": (profile["gps_alt"][0] or profile["alt"][0]),
        "x_km": x_km_arr[0],
        "y_km": y_km_arr[0],
    }

    surface = {
        "lat": profile["lat"][-1],
        "lon": profile["lon"][-1],
        "alt_m": (profile["gps_alt"][-1] or profile["alt"][-1]),
        "x_km": x_km_arr[-1],
        "y_km": y_km_arr[-1],
    }

    # Launch time for time offset computation (handle hour >= 24)
    launch_dt = None
    try:
        yr = int(meta.get("Year", 0))
        mo = int(meta.get("Month", 0))
        dy = int(meta.get("Day", 0))
        hr = int(meta.get("Hour", 0))
        mi = int(meta.get("Minute", 0))
        sc = int(meta.get("Second", 0))
        if yr > 0:
            if hr >= 24:
                base = _dt(yr, mo, dy, tzinfo=timezone.utc)
                launch_dt = base + timedelta(hours=hr, minutes=mi, seconds=sc)
            else:
                launch_dt = _dt(yr, mo, dy, hr, mi, sc, tzinfo=timezone.utc)
    except (ValueError, TypeError):
        pass

    time_offset_min = None
    launch_time_str = ""
    if launch_dt:
        launch_time_str = launch_dt.strftime("%Y-%m-%d %H:%M:%SZ")
        delta = (launch_dt - analysis_dt).total_seconds() / 60.0
        time_offset_min = round(delta, 1)

    def _round_list(arr, decimals=3):
        return [round(v, decimals) if v is not None else None for v in arr]

    hit_sfc = meta.get("DropsondeHitSfc", meta.get("DropSondeHitSfc", "0"))

    return {
        "sonde_id": meta.get("SondeId", ""),
        "launch_time": launch_time_str,
        "time_offset_min": time_offset_min,
        "comments": meta.get("Comments", meta.get("Comment", "")),
        "flight": meta.get("Flight", ""),
        "platform": meta.get("PlatformId", ""),
        "project": meta.get("Project", ""),
        "hit_surface": str(hit_sfc).strip() == "1",
        "launch": launch,
        "surface": surface,
        "profile": {
            "time_s": _round_list(profile["time_s"], 1),
            "lat": _round_list(profile["lat"], 5),
            "lon": _round_list(profile["lon"], 5),
            "x_km": x_km_arr,
            "y_km": y_km_arr,
            "alt_km": alt_km_arr,
            "wspd": _round_list(profile["wspd"], 2),
            "wdir": _round_list(profile["wdir"], 1),
            "temp": _round_list(profile["temp"], 2),
            "pres": _round_list(profile["pres"], 2),
            "rh": _round_list(profile["rh"], 1),
            "dewpoint": _round_list(profile.get("dewpoint", []), 2),
            "uwnd": _round_list(profile["uwnd"], 2),
            "vwnd": _round_list(profile["vwnd"], 2),
        },
    }


@router.get("/dropsondes")
def get_dropsondes(
    file_url: str = Query(..., description="URL to the TDR xy.nc(.gz) file"),
):
    """
    Return dropsonde profiles within ±45 min of the TDR analysis time.

    Searches the matching ASPEN_Data mission folder on the SEB server,
    parses PQC CSV files, converts to storm-relative coordinates, and
    returns full-resolution profiles for visualization.
    """
    now = time.time()
    if file_url in _rt_sonde_cache:
        cached, ts = _rt_sonde_cache[file_url]
        if now - ts < _RT_SONDE_CACHE_TTL:
            _rt_sonde_cache.move_to_end(file_url)
            return JSONResponse(cached)

    try:
        ds = _open_rt_dataset(file_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open TDR file: {e}")

    case_meta = _build_case_meta(ds)
    center_lat = case_meta["latitude"]
    center_lon = case_meta["longitude"]

    try:
        analysis_dt = _parse_tdr_datetime(case_meta)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Extract mission ID from file URL
    mission_id = _extract_mission_id(file_url)
    if not mission_id:
        flt = case_meta.get("mission_id", "")
        m = re.match(r"(\d{6,8}[A-Za-z]\d)", flt)
        if m:
            mission_id = m.group(1)

    if not mission_id:
        raise HTTPException(
            status_code=400,
            detail="Could not determine mission ID from file URL",
        )

    # List dropsonde CSVs for this mission
    sonde_dir_url = f"{SEB_SONDE_BASE}/{mission_id}/"
    try:
        links = _parse_directory(sonde_dir_url)
    except Exception:
        result = {
            "dropsondes": [],
            "analysis_time": analysis_dt.strftime("%Y-%m-%d %H:%M:%SZ"),
            "center_lat": center_lat,
            "center_lon": center_lon,
            "time_window_min": SONDE_TIME_WINDOW_MIN,
            "n_sondes": 0,
            "message": f"No dropsonde directory found for mission {mission_id}",
        }
        _rt_sonde_cache[file_url] = (result, now)
        return JSONResponse(result)

    # Filter to PQC CSV files within time window
    csv_candidates = []
    for link in links:
        if not link.endswith("_PQC.csv"):
            continue
        launch_dt = _parse_sonde_launch_time(link)
        if launch_dt is None:
            continue
        delta_min = abs((launch_dt - analysis_dt).total_seconds()) / 60.0
        if delta_min <= SONDE_TIME_WINDOW_MIN:
            csv_candidates.append((link, launch_dt, delta_min))

    csv_candidates.sort(key=lambda x: x[2])

    # Fetch and parse each CSV in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_and_parse(item):
        filename, _ldt, _dmin = item
        csv_url = f"{SEB_SONDE_BASE}/{mission_id}/{filename}"
        try:
            csv_text = _fetch_text(csv_url, timeout=30)
            parsed = _parse_dropsonde_csv(csv_text)
            if parsed is None:
                return None
            return _build_sonde_response(parsed, center_lat, center_lon, analysis_dt)
        except Exception:
            return None

    dropsondes = []
    if csv_candidates:
        max_workers = min(len(csv_candidates), 6)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_and_parse, item): item for item in csv_candidates}
            for future in as_completed(futures):
                sonde_entry = future.result()
                if sonde_entry is not None:
                    dropsondes.append(sonde_entry)

    dropsondes.sort(
        key=lambda s: abs(s["time_offset_min"]) if s["time_offset_min"] is not None else 999
    )

    result = {
        "dropsondes": dropsondes,
        "analysis_time": analysis_dt.strftime("%Y-%m-%d %H:%M:%SZ"),
        "center_lat": center_lat,
        "center_lon": center_lon,
        "time_window_min": SONDE_TIME_WINDOW_MIN,
        "n_sondes": len(dropsondes),
    }

    _rt_sonde_cache[file_url] = (result, now)
    if len(_rt_sonde_cache) > _RT_SONDE_CACHE_MAX:
        _rt_sonde_cache.popitem(last=False)

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Flight-Level (In Situ) Observations — IWG1 + MELISSA from acdata
# ---------------------------------------------------------------------------

SEB_ACDATA_BASE = "https://seb.omao.noaa.gov/pub/acdata"
FL_TIME_WINDOW_MIN = 45  # +/-45 minutes from TDR analysis centre time

# NOAA P-3 IWG1 field indices (0-based, after splitting by comma).
# The P-3 includes a WGS-84 altitude field at position 5 that shifts all
# subsequent fields by +1 compared to the base IWG1 standard.
# Spec: https://archive.eol.ucar.edu/raf/Software/IWG1_Def.html
_IWG1 = {
    "timestamp":    1,   # ISO-8601 UTC (yyyymmddThhmmss)
    "lat":          2,   # degrees
    "lon":          3,   # degrees
    "gps_alt":      4,   # GPS MSL altitude (m)
    # 5: WGS-84 altitude (m) — often empty on P-3
    "press_alt":    6,   # pressure altitude (feet)
    "radar_alt":    7,   # radar altitude (feet)
    "ground_spd":   8,   # ground speed (m/s)
    "true_airspd":  9,   # true airspeed (m/s)
    "ias":         10,   # indicated airspeed (knots)
    "mach":        11,   # Mach number
    "vert_vel":    12,   # vertical velocity (m/s)
    "heading":     13,   # true heading (deg)
    "track":       14,   # track angle (deg)
    "drift":       15,   # drift angle (deg)
    "pitch":       16,   # pitch (deg)
    "roll":        17,   # roll (deg)
    "sideslip":    18,   # side slip (deg)
    "aoa":         19,   # angle of attack (deg)
    "temp":        20,   # ambient/static temperature (deg C)
    "dewpoint":    21,   # dew point (deg C)
    "total_temp":  22,   # total temperature (deg C)
    "static_pres": 23,   # static pressure (hPa)
    "dyn_pres":    24,   # dynamic pressure (hPa)
    "cabin_pres":  25,   # cabin pressure (hPa)
    "wind_spd":    26,   # environmental wind speed (m/s)
    "wind_dir":    27,   # environmental wind direction (deg)
    "vert_wind":   28,   # vertical wind speed (m/s)
    "flight_id":   33,   # e.g. 20251028H1 (P-3 extension)
}

# MELISSA fields — only the SFMR surface wind estimates (positions 2-6)
# are reliably identified; the remaining 90+ MELISSA fields vary by
# aircraft configuration and are not yet mapped.
_MELISSA_SFMR_INDICES = [2, 3, 4, 5]  # 4 SFMR sfc wind estimates (m/s)
_MELISSA_EXTRAP_SFC = 6               # extrapolated sfc wind (m/s)

# Cache
_rt_fl_cache = OrderedDict()
_RT_FL_CACHE_MAX = 3
_RT_FL_CACHE_TTL = 300  # 5 minutes


def _parse_iwg1_timestamp(ts_str: str) -> Optional[_dt]:
    """Parse IWG1 timestamp like '20251028T101345' into datetime."""
    ts = ts_str.strip()
    if not ts or len(ts) < 15:
        return None
    try:
        return _dt.strptime(ts[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _safe_float(val: str) -> Optional[float]:
    """Safely convert a string to float, returning None on failure."""
    v = val.strip() if val else ""
    if not v:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _iwg1_field(parts: list, idx: int) -> Optional[float]:
    """Extract a float from IWG1 parts at the given index, or None."""
    if idx < len(parts):
        return _safe_float(parts[idx])
    return None


def _parse_acdata_serial(text: str, analysis_dt: _dt, time_window_min: float) -> list[dict]:
    """
    Parse a _serial.dat file containing IWG1 + MELISSA records.

    On the NOAA P-3, IWG1 and MELISSA are on the SAME line, separated
    by a space (e.g. "...2313A MELISSA,AL132025,...").  This parser
    splits on " MELISSA," to separate them.

    Returns a list of 1-Hz observation dicts within +/-time_window_min
    of analysis_dt.
    """
    lines = text.splitlines()
    observations = []

    for line in lines:
        line = line.strip()
        if not line.startswith("IWG1,"):
            continue

        # Split IWG1 and MELISSA (they share the same line on P-3)
        iwg1_text = line
        melissa_text = None
        mel_split = line.split(" MELISSA,", 1)
        if len(mel_split) == 2:
            iwg1_text = mel_split[0]
            melissa_text = "MELISSA," + mel_split[1]

        parts = iwg1_text.split(",")
        ts = _parse_iwg1_timestamp(parts[_IWG1["timestamp"]] if len(parts) > 1 else "")
        if ts is None:
            continue

        # Time-window filter
        delta_sec = (ts - analysis_dt).total_seconds()
        if abs(delta_sec) > time_window_min * 60:
            continue

        lat = _iwg1_field(parts, _IWG1["lat"])
        lon = _iwg1_field(parts, _IWG1["lon"])
        if lat is None or lon is None:
            continue

        obs = {
            "time": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "time_offset_s": round(delta_sec, 1),
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "gps_alt_m":      _iwg1_field(parts, _IWG1["gps_alt"]),
            "press_alt_ft":   _iwg1_field(parts, _IWG1["press_alt"]),
            "ground_spd_ms":  _iwg1_field(parts, _IWG1["ground_spd"]),
            "true_airspd_ms": _iwg1_field(parts, _IWG1["true_airspd"]),
            "vert_vel_ms":    _iwg1_field(parts, _IWG1["vert_vel"]),
            "heading":        _iwg1_field(parts, _IWG1["heading"]),
            "track":          _iwg1_field(parts, _IWG1["track"]),
            "static_pres_hpa": _iwg1_field(parts, _IWG1["static_pres"]),
            "temp_c":         _iwg1_field(parts, _IWG1["temp"]),
            "dewpoint_c":     _iwg1_field(parts, _IWG1["dewpoint"]),
            "fl_wspd_ms":     _iwg1_field(parts, _IWG1["wind_spd"]),
            "fl_wdir_deg":    _iwg1_field(parts, _IWG1["wind_dir"]),
            # MELISSA fields
            "sfmr_wspd_ms": None,
            "extrapolated_sfc_wspd_ms": None,
        }

        # Round values
        for key in obs:
            if isinstance(obs[key], float) and key not in ("lat", "lon", "time_offset_s"):
                obs[key] = round(obs[key], 2)

        # Parse MELISSA line (same line, after the space)
        if melissa_text:
            mparts = melissa_text.split(",")
            # SFMR surface wind: take the maximum positive of the 4 estimates
            best_sfmr = None
            for si in _MELISSA_SFMR_INDICES:
                if si < len(mparts):
                    v = _safe_float(mparts[si])
                    if v is not None and v >= 0:
                        if best_sfmr is None or v > best_sfmr:
                            best_sfmr = v
            if best_sfmr is not None:
                obs["sfmr_wspd_ms"] = round(best_sfmr, 2)

            if _MELISSA_EXTRAP_SFC < len(mparts):
                v = _safe_float(mparts[_MELISSA_EXTRAP_SFC])
                if v is not None:
                    obs["extrapolated_sfc_wspd_ms"] = round(v, 2)

        observations.append(obs)

    return observations


def _average_fl_window(
    observations: list[dict], interval_s: float = 10.0
) -> list[dict]:
    """
    Compute interval_s-second averages of flight-level observations.

    Groups consecutive 1-Hz obs into non-overlapping windows and averages
    all numeric fields.  Lat/lon/time use the window centre observation.
    """
    if not observations or interval_s <= 1:
        return observations

    # Numeric keys to average
    _AVG_KEYS = [
        "gps_alt_m", "press_alt_ft", "ground_spd_ms", "true_airspd_ms",
        "vert_vel_ms", "static_pres_hpa", "temp_c", "dewpoint_c",
        "fl_wspd_ms", "fl_wdir_deg", "sfmr_wspd_ms",
        "extrapolated_sfc_wspd_ms",
    ]

    result = []
    n = len(observations)
    i = 0
    while i < n:
        t0 = observations[i]["time_offset_s"]
        window = []
        while i < n and observations[i]["time_offset_s"] - t0 < interval_s:
            window.append(observations[i])
            i += 1

        # Use the centre observation for identity fields
        mid_idx = len(window) // 2
        mid = window[mid_idx]
        averaged = {
            "time": mid["time"],
            "time_offset_s": mid["time_offset_s"],
            "lat": mid["lat"],
            "lon": mid["lon"],
        }

        # Average numeric fields (skip None values)
        for key in _AVG_KEYS:
            vals = [o[key] for o in window if o.get(key) is not None]
            if vals:
                # For wind direction, use circular mean
                if key == "fl_wdir_deg":
                    rad = [v * np.pi / 180.0 for v in vals]
                    mean_sin = sum(np.sin(r) for r in rad) / len(rad)
                    mean_cos = sum(np.cos(r) for r in rad) / len(rad)
                    avg = (np.arctan2(mean_sin, mean_cos) * 180.0 / np.pi) % 360.0
                else:
                    avg = sum(vals) / len(vals)
                averaged[key] = round(avg, 2)
            else:
                averaged[key] = None

        # Carry forward heading/track from centre obs
        averaged["heading"] = mid.get("heading")
        averaged["track"] = mid.get("track")

        result.append(averaged)

    return result


def _find_acdata_serial_file(mission_id: str, year: int) -> Optional[str]:
    """
    Locate the _serial.dat file for a given mission in the acdata tree.

    Directory structure: /pub/acdata/{year}/MET/{mission_id}/{mission_id}_serial.dat
    """
    base_url = f"{SEB_ACDATA_BASE}/{year}/MET/{mission_id}/"
    try:
        links = _parse_directory(base_url)
    except Exception:
        return None

    for link in links:
        if link.endswith("_serial.dat"):
            return f"{base_url}{link}"
    return None


@router.get("/flightlevel")
def get_flight_level(
    file_url: str = Query(..., description="URL to the TDR xy.nc(.gz) file"),
    avg_interval_s: float = Query(10.0, ge=1, le=60,
                                  description="Averaging interval in seconds"),
):
    """
    Return flight-level (in situ) observations from the IWG1/MELISSA serial
    data stream for the aircraft mission matching the given TDR analysis file.

    Parses 1-Hz IWG1 + MELISSA from seb.omao.noaa.gov/pub/acdata/, filters
    to +/-45 min of the TDR analysis time, and returns 10-second averaged
    observations with storm-relative coordinates.
    """
    now = time.time()
    cache_key = f"{file_url}__avg{avg_interval_s}"
    if cache_key in _rt_fl_cache:
        cached, ts = _rt_fl_cache[cache_key]
        if now - ts < _RT_FL_CACHE_TTL:
            _rt_fl_cache.move_to_end(cache_key)
            return JSONResponse(cached)

    # Open the TDR file to extract mission metadata
    try:
        ds = _open_rt_dataset(file_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open TDR file: {e}")

    case_meta = _build_case_meta(ds)
    center_lat = case_meta["latitude"]
    center_lon = case_meta["longitude"]

    try:
        analysis_dt = _parse_tdr_datetime(case_meta)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Extract mission ID
    mission_id = _extract_mission_id(file_url)
    if not mission_id:
        flt = case_meta.get("mission_id", "")
        m = re.match(r"(\d{6,8}[A-Za-z]\d)", flt)
        if m:
            mission_id = m.group(1)
    if not mission_id:
        raise HTTPException(status_code=400, detail="Could not determine mission ID")

    year = analysis_dt.year

    # Find the serial data file
    serial_url = _find_acdata_serial_file(mission_id, year)
    if serial_url is None:
        result = {
            "observations": [],
            "analysis_time": analysis_dt.strftime("%Y-%m-%d %H:%M:%SZ"),
            "center_lat": center_lat,
            "center_lon": center_lon,
            "mission_id": mission_id,
            "time_window_min": FL_TIME_WINDOW_MIN,
            "n_obs": 0,
            "message": f"No flight-level serial data found for mission {mission_id}",
        }
        _rt_fl_cache[cache_key] = (result, now)
        return JSONResponse(result)

    # Fetch and parse the serial data
    try:
        serial_text = _fetch_text(serial_url, timeout=60)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Could not fetch serial data from {serial_url}: {e}",
        )

    raw_obs = _parse_acdata_serial(serial_text, analysis_dt, FL_TIME_WINDOW_MIN)

    # Compute 10-second (or user-specified) averages
    averaged_obs = _average_fl_window(raw_obs, interval_s=avg_interval_s)

    # Add storm-relative coordinates
    for obs in averaged_obs:
        x_km, y_km = _latlon_to_storm_km(
            obs["lat"], obs["lon"], center_lat, center_lon
        )
        obs["x_km"] = round(x_km, 3)
        obs["y_km"] = round(y_km, 3)

    # Compute summary statistics from the RAW 1-Hz data for accuracy
    fl_wspds = [o["fl_wspd_ms"] for o in raw_obs if o["fl_wspd_ms"] is not None]
    sfmr_wspds = [o["sfmr_wspd_ms"] for o in raw_obs if o["sfmr_wspd_ms"] is not None]
    static_pres = [o["static_pres_hpa"] for o in raw_obs
                   if o["static_pres_hpa"] is not None and 200 < o["static_pres_hpa"] < 1100]
    temps = [o["temp_c"] for o in raw_obs if o["temp_c"] is not None]

    summary = {
        "max_fl_wspd_ms": round(max(fl_wspds), 2) if fl_wspds else None,
        "max_sfmr_wspd_ms": round(max(sfmr_wspds), 2) if sfmr_wspds else None,
        "min_slp_hpa": None,  # SLP not yet reliably parsed from MELISSA
        "max_temp_c": round(max(temps), 2) if temps else None,
        "min_temp_c": round(min(temps), 2) if temps else None,
        "min_static_pres_hpa": round(min(static_pres), 2) if static_pres else None,
        "total_obs_1hz": len(raw_obs),
        "avg_interval_s": avg_interval_s,
        "mean_alt_m": round(
            sum(o["gps_alt_m"] for o in raw_obs if o["gps_alt_m"] is not None)
            / max(1, sum(1 for o in raw_obs if o["gps_alt_m"] is not None)),
            0,
        ) if any(o["gps_alt_m"] is not None for o in raw_obs) else None,
    }

    result = {
        "observations": averaged_obs,
        "analysis_time": analysis_dt.strftime("%Y-%m-%d %H:%M:%SZ"),
        "center_lat": center_lat,
        "center_lon": center_lon,
        "mission_id": mission_id,
        "serial_url": serial_url,
        "time_window_min": FL_TIME_WINDOW_MIN,
        "n_obs": len(averaged_obs),
        "n_obs_total": len(raw_obs),
        "summary": summary,
    }

    _rt_fl_cache[cache_key] = (result, now)
    if len(_rt_fl_cache) > _RT_FL_CACHE_MAX:
        _rt_fl_cache.popitem(last=False)

    return JSONResponse(result)
