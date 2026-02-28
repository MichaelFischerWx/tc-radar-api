"""
realtime_tdr_api.py — Real-Time TDR Endpoints
===============================================
Add-on module for tc_radar_api.py that provides endpoints for browsing
and visualizing real-time Tail Doppler Radar (TDR) analyses from
seb.omao.noaa.gov/pub/flight/radar/.

How to integrate:
    In tc_radar_api.py, add near the bottom:
        from realtime_tdr_api import router as realtime_router
        app.include_router(realtime_router, prefix="/realtime")

Or simply paste the router endpoints into tc_radar_api.py directly.

Dependencies (all already in tc_radar_api.py):
    fastapi, xarray, numpy, matplotlib, requests (add if not present)
"""

import gzip
import io
import re
import time
from collections import OrderedDict
from html.parser import HTMLParser
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

try:
    import requests as _requests
except ImportError:
    import urllib.request as _urllib
    _requests = None

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
# Caching
# ---------------------------------------------------------------------------
_rt_ds_cache = OrderedDict()       # file_url → (xr.Dataset, timestamp)
_rt_dir_cache = OrderedDict()      # dir_url  → (link_list, timestamp)
_RT_DS_CACHE_MAX = 8
_RT_DIR_CACHE_TTL = 300            # 5 minutes for directory listings

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

    # Decompress if gzipped
    if file_url.endswith(".gz"):
        raw = gzip.decompress(raw)

    ds = xr.open_dataset(io.BytesIO(raw), engine="h5netcdf")
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
