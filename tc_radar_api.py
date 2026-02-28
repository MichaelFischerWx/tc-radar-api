"""
tc_radar_api.py  (S3/Zarr edition)
====================================
FastAPI backend for on-demand TC-RADAR plot generation.

Data backend: Zarr stores on S3 — case-level chunks mean each plot
request is a handful of small S3 GETs, typically completing in <1 s
(vs 5–15 s for HTTP range requests against AOML).

Deploy on Render (free tier is fine), co-located in the same AWS region
as your S3 bucket (us-east-1 recommended) for zero-latency data access.

Environment variables
---------------------
    TC_RADAR_S3_BUCKET   S3 bucket name  (required for S3 mode)
    TC_RADAR_S3_PREFIX   prefix in bucket (default: tc-radar)
    AWS_ACCESS_KEY_ID    \\ standard AWS creds — or use an IAM role
    AWS_SECRET_ACCESS_KEY /
    AWS_DEFAULT_REGION   (default: us-east-1)
    METADATA_PATH        path to tc_radar_metadata.json (default: ./tc_radar_metadata.json)

If TC_RADAR_S3_BUCKET is not set the API falls back to the original
AOML HTTP range-request mode automatically.

Local dev
---------
    pip install fastapi uvicorn h5netcdf h5py fsspec xarray zarr s3fs matplotlib numpy Pillow
    TC_RADAR_S3_BUCKET=your-bucket uvicorn tc_radar_api:app --reload --port 8000
"""

import base64
import gc
import io
import json
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import fsspec
from scipy import ndimage as _ndimage
from scipy.interpolate import RegularGridInterpolator
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import Response, JSONResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
S3_BUCKET  = os.environ.get("TC_RADAR_S3_BUCKET", "")
S3_PREFIX  = os.environ.get("TC_RADAR_S3_PREFIX", "tc-radar")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
USE_S3     = bool(S3_BUCKET)

AOML_BASE = "https://www.aoml.noaa.gov/ftp/pub/hrd/data/radar/level3"

# S3 Zarr paths  (set after conversion script runs)
S3_PATHS = {
    ("swath", "early"):  f"s3://{S3_BUCKET}/{S3_PREFIX}/swath_early",
    ("swath", "recent"): f"s3://{S3_BUCKET}/{S3_PREFIX}/swath_recent",
    ("merge", "early"):  f"s3://{S3_BUCKET}/{S3_PREFIX}/merge_early",
    ("merge", "recent"): f"s3://{S3_BUCKET}/{S3_PREFIX}/merge_recent",
}

# AOML fallback (original NetCDF via HTTP range requests)
AOML_FILES = {
    ("swath", "early"):  f"{AOML_BASE}/tc_radar_v3m_1997_2019_xy_rel_swath_ships.nc",
    ("swath", "recent"): f"{AOML_BASE}/tc_radar_v3m_2020_2024_xy_rel_swath_ships.nc",
    ("merge", "early"):  f"{AOML_BASE}/tc_radar_v3m_1997_2019_xy_rel_merge_ships.nc",
    ("merge", "recent"): f"{AOML_BASE}/tc_radar_v3m_2020_2024_xy_rel_merge_ships.nc",
}


# IR satellite imagery (MergIR) Zarr store
IR_S3_PATH = f"s3://{S3_BUCKET}/{S3_PREFIX}/mergir" if S3_BUCKET else ""

# IR colormap: NOAA-style enhanced IR (warm=dark, cold=bright/colorful)
IR_COLORMAP = [
    [0.0,    "rgb(8,8,8)"],
    [0.15,   "rgb(40,40,40)"],
    [0.30,   "rgb(90,90,90)"],
    [0.40,   "rgb(140,140,140)"],
    [0.50,   "rgb(200,200,200)"],
    [0.55,   "rgb(0,180,255)"],
    [0.60,   "rgb(0,100,255)"],
    [0.65,   "rgb(0,255,0)"],
    [0.70,   "rgb(255,255,0)"],
    [0.75,   "rgb(255,180,0)"],
    [0.80,   "rgb(255,80,0)"],
    [0.85,   "rgb(255,0,0)"],
    [0.90,   "rgb(180,0,180)"],
    [0.95,   "rgb(255,180,255)"],
    [1.0,    "rgb(255,255,255)"],
]

# ERA5 environmental diagnostics Zarr store
ERA5_S3_PATH = f"s3://{S3_BUCKET}/{S3_PREFIX}/era5" if S3_BUCKET else ""

ERA5_FIELD_CONFIG = {
    "shear_mag": {
        "display_name": "Deep-Layer Shear (200\u2013850 hPa)",
        "units": "m/s", "vmin": 0, "vmax": 30,
        "colorscale": [
            [0.0, "rgb(255,255,204)"], [0.15, "rgb(255,237,160)"],
            [0.30, "rgb(254,217,118)"], [0.45, "rgb(254,178,76)"],
            [0.60, "rgb(253,141,60)"], [0.75, "rgb(240,59,32)"],
            [0.90, "rgb(189,0,38)"], [1.0, "rgb(128,0,38)"],
        ],
        "has_vectors": True,
        "vector_u": "shear_u", "vector_v": "shear_v",
    },
    "rh_mid": {
        "display_name": "Mid-Level RH (500\u2013700 hPa)",
        "units": "%", "vmin": 0, "vmax": 100,
        "colorscale": [
            [0.0, "rgb(140,81,10)"], [0.15, "rgb(191,129,45)"],
            [0.30, "rgb(223,194,125)"], [0.45, "rgb(245,245,220)"],
            [0.55, "rgb(199,234,229)"], [0.70, "rgb(128,205,193)"],
            [0.85, "rgb(53,151,143)"], [1.0, "rgb(1,102,94)"],
        ],
        "has_vectors": False,
    },
    "div200": {
        "display_name": "200 hPa Divergence",
        "units": "s\u207b\u00b9", "vmin": -3e-5, "vmax": 3e-5,
        "colorscale": [
            [0.0, "rgb(178,24,43)"], [0.15, "rgb(214,96,77)"],
            [0.30, "rgb(244,165,130)"], [0.45, "rgb(253,219,199)"],
            [0.55, "rgb(209,229,240)"], [0.70, "rgb(146,197,222)"],
            [0.85, "rgb(67,147,195)"], [1.0, "rgb(33,102,172)"],
        ],
        "has_vectors": True,
        "vector_u": "u200", "vector_v": "v200",
    },
    "sst": {
        "display_name": "Sea Surface Temperature",
        "units": "°C", "vmin": 18, "vmax": 32,
        "colorscale": [
            [0.0, "rgb(49,54,149)"], [0.15, "rgb(69,117,180)"],
            [0.30, "rgb(116,173,209)"], [0.45, "rgb(171,217,233)"],
            [0.55, "rgb(253,174,97)"], [0.70, "rgb(244,109,67)"],
            [0.85, "rgb(215,48,39)"], [1.0, "rgb(165,0,38)"],
        ],
        "has_vectors": False,
    },
    "entropy_def": {
        "display_name": "Entropy Deficit (χₘ)",
        "units": "", "vmin": 0, "vmax": 2.0,
        "colorscale": [
            [0.0, "rgb(255,247,236)"], [0.15, "rgb(254,232,200)"],
            [0.30, "rgb(253,212,158)"], [0.45, "rgb(253,187,132)"],
            [0.60, "rgb(227,145,86)"], [0.75, "rgb(189,109,53)"],
            [0.90, "rgb(140,81,10)"], [1.0, "rgb(84,48,5)"],
        ],
        "has_vectors": False,
    },
}

CASE_COUNTS = {
    ("swath", "early"):  710,
    ("swath", "recent"): 800,
    ("merge", "early"):  310,
    ("merge", "recent"): 126,
}

# ---------------------------------------------------------------------------
# Variable config
# ---------------------------------------------------------------------------
VARIABLES = {
    "recentered_tangential_wind":           ("Tangential Wind (WCM)",          "recentered_tangential_wind",           "jet",       "m/s",  -10,   80),
    "recentered_radial_wind":               ("Radial Wind (WCM)",               "recentered_radial_wind",               "RdBu_r",    "m/s",  -30,   30),
    "recentered_upward_air_velocity":       ("Vertical Velocity (WCM)",         "recentered_upward_air_velocity",       "RdBu_r",    "m/s",   -5,    5),
    "recentered_reflectivity":              ("Reflectivity (WCM)",              "recentered_reflectivity",              "Spectral_r","dBZ",  -10,   65),
    "recentered_wind_speed":                ("Wind Speed (WCM)",                "recentered_wind_speed",                "inferno",   "m/s",    0,   80),
    "recentered_earth_relative_wind_speed": ("Earth-Rel. Wind Speed (WCM)",    "recentered_earth_relative_wind_speed", "jet",       "m/s",    0,   80),
    "recentered_relative_vorticity":        ("Relative Vorticity (WCM)",        "recentered_relative_vorticity",        "RdBu_r",    "s⁻¹",-5e-3, 5e-3),
    "recentered_divergence":                ("Divergence (WCM)",                "recentered_divergence",                "RdBu_r",    "s⁻¹",-5e-3, 5e-3),
    "total_recentered_tangential_wind":     ("Tangential Wind (tilt-relative)", "total_recentered_tangential_wind",     "jet",       "m/s",  -10,   80),
    "total_recentered_radial_wind":         ("Radial Wind (tilt-relative)",     "total_recentered_radial_wind",         "RdBu_r",    "m/s",  -30,   30),
    "total_recentered_upward_air_velocity": ("Vertical Velocity (tilt-rel.)",   "total_recentered_upward_air_velocity", "RdBu_r",    "m/s",   -5,    5),
    "total_recentered_reflectivity":        ("Reflectivity (tilt-relative)",    "total_recentered_reflectivity",        "Spectral_r","dBZ",  -10,   65),
    "total_recentered_wind_speed":          ("Wind Speed (tilt-relative)",      "total_recentered_wind_speed",          "inferno",   "m/s",    0,   80),
    "total_recentered_earth_relative_wind_speed": ("Earth-Rel. Wind Speed (tilt-rel.)", "total_recentered_earth_relative_wind_speed", "jet", "m/s", 0, 80),
    "swath_tangential_wind":                ("Tangential Wind (original)",      "swath_tangential_wind",                "jet",       "m/s",  -10,   80),
    "swath_radial_wind":                    ("Radial Wind (original)",          "swath_radial_wind",                    "RdBu_r",    "m/s",  -30,   30),
    "swath_reflectivity":                   ("Reflectivity (original)",         "swath_reflectivity",                   "Spectral_r","dBZ",  -10,   65),
    "swath_wind_speed":                     ("Wind Speed (original)",           "swath_wind_speed",                     "inferno",   "m/s",    0,   80),
    "swath_earth_relative_wind_speed":      ("Earth-Rel. Wind Speed (original)","swath_earth_relative_wind_speed",      "jet",       "m/s",    0,   80),
    # Merged (flight-averaged) domain
    "merged_tangential_wind":               ("Tangential Wind (merged)",        "merged_tangential_wind",               "jet",       "m/s",  -10,   80),
    "merged_radial_wind":                   ("Radial Wind (merged)",            "merged_radial_wind",                   "RdBu_r",    "m/s",  -30,   30),
    "merged_reflectivity":                  ("Reflectivity (merged)",           "merged_reflectivity",                  "Spectral_r","dBZ",  -10,   65),
    "merged_wind_speed":                    ("Wind Speed (merged)",             "merged_wind_speed",                    "inferno",   "m/s",    0,   80),
    "merged_upward_air_velocity":           ("Vertical Velocity (merged)",      "merged_upward_air_velocity",           "RdBu_r",    "m/s",   -5,    5),
    "merged_relative_vorticity":            ("Relative Vorticity (merged)",     "merged_relative_vorticity",            "RdBu_r",    "s⁻¹",-5e-3, 5e-3),
    "merged_divergence":                    ("Divergence (merged)",             "merged_divergence",                    "RdBu_r",    "s⁻¹",-5e-3, 5e-3),
}

# Derived variables: computed as sqrt(u² + v²) from component pairs
# key → (u_varname, v_varname)
DERIVED_VARIABLES = {
    "recentered_earth_relative_wind_speed":       ("recentered_earth_relative_eastward_wind",       "recentered_earth_relative_northward_wind"),
    "total_recentered_earth_relative_wind_speed":  ("total_recentered_earth_relative_eastward_wind",  "total_recentered_earth_relative_northward_wind"),
    "swath_earth_relative_wind_speed":             ("swath_earth_relative_eastward_wind",             "swath_earth_relative_northward_wind"),
}

DEFAULT_VARIABLE = "recentered_tangential_wind"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="TC-RADAR API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://michaelfischerwx.github.io", "http://localhost:8000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# HTTP cache headers — scientific data is immutable, cache aggressively
# ---------------------------------------------------------------------------

class CacheHeaderMiddleware(BaseHTTPMiddleware):
    """
    Add Cache-Control headers based on endpoint data mutability.

    Immutable endpoints (radar data, IR, ERA5) get long TTLs — the data
    will never change for a given case_index + variable + level.
    Semi-stable endpoints (composites) get shorter TTLs since new cases
    could be added.  Metadata/health get brief caching.
    """
    IMMUTABLE_PREFIXES = (
        '/data', '/ir', '/ir_frame', '/era5',
        '/azimuthal_mean', '/quadrant_mean',
        '/cross_section', '/plot', '/volume',
    )
    SEMI_STABLE_PREFIXES = ('/composite',)
    SHORT_CACHE_PATHS = ('/health', '/metadata', '/variables', '/levels')

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        path = request.url.path

        # Only cache successful responses
        if response.status_code == 200:
            if any(path.startswith(p) for p in self.IMMUTABLE_PREFIXES):
                response.headers['Cache-Control'] = 'public, max-age=86400, immutable'
            elif any(path.startswith(p) for p in self.SEMI_STABLE_PREFIXES):
                response.headers['Cache-Control'] = 'public, max-age=3600'
            elif path in self.SHORT_CACHE_PATHS:
                response.headers['Cache-Control'] = 'public, max-age=300'

        return response


app.add_middleware(CacheHeaderMiddleware)


class CompositeGCMiddleware(BaseHTTPMiddleware):
    """
    Force garbage collection after composite endpoints to reclaim numpy arrays
    and prevent OOM on the 512 MB Render free tier.
    """
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.startswith('/composite'):
            gc.collect()
        return response

app.add_middleware(CompositeGCMiddleware)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def get_dataset(data_type: str, era: str) -> xr.Dataset:
    """
    Open a TC-RADAR dataset.

    S3 mode  : xr.open_zarr() — case-chunked, <1 s per plot after warm open
    AOML mode: xr.open_dataset() via fsspec HTTP — ~5-15 s per plot
    """
    if USE_S3:
        import s3fs
        path = S3_PATHS[(data_type, era)]
        fs   = s3fs.S3FileSystem(anon=False, client_kwargs={"region_name": AWS_REGION})
        store = s3fs.S3Map(root=path, s3=fs, check=False)
        ds = xr.open_zarr(store, consolidated=True)
        print(f"Opened Zarr from S3: {path}")
    else:
        url = AOML_FILES[(data_type, era)]
        of  = fsspec.open(url, "rb")
        ds  = xr.open_dataset(of.open(), engine="h5netcdf")
        print(f"Opened NetCDF from AOML: {url}")
    return ds



@lru_cache(maxsize=1)
def get_ir_dataset():
    """Open the MergIR Zarr store from S3."""
    if not USE_S3 or not IR_S3_PATH:
        return None
    import s3fs
    fs = s3fs.S3FileSystem(anon=False, client_kwargs={"region_name": AWS_REGION})
    store = s3fs.S3Map(root=IR_S3_PATH, s3=fs, check=False)
    import zarr
    return zarr.open(store, mode='r')


@lru_cache(maxsize=1)
def get_era5_dataset():
    """Open the ERA5 Zarr store from S3."""
    if not USE_S3 or not ERA5_S3_PATH:
        return None
    import s3fs
    fs = s3fs.S3FileSystem(anon=False, client_kwargs={"region_name": AWS_REGION})
    store = s3fs.S3Map(root=ERA5_S3_PATH, s3=fs, check=False)
    import zarr
    return zarr.open(store, mode='r')


def resolve_case(case_index: int, data_type: str) -> tuple[xr.Dataset, int]:
    early_count = CASE_COUNTS[(data_type, "early")]
    if case_index < early_count:
        return get_dataset(data_type, "early"), case_index
    else:
        return get_dataset(data_type, "recent"), case_index - early_count


# ---------------------------------------------------------------------------
# Plot rendering
# ---------------------------------------------------------------------------

def render_planview(
    ds: xr.Dataset,
    local_idx: int,
    variable_key: str,
    level_km: float,
    case_meta: dict,
) -> bytes:
    display_name, varname, cmap, units, vmin, vmax = VARIABLES[variable_key]

    # Derived variable: compute sqrt(u² + v²) from component pair
    if variable_key in DERIVED_VARIABLES:
        u_name, v_name = DERIVED_VARIABLES[variable_key]
        if u_name not in ds or v_name not in ds:
            raise ValueError(f"Components '{u_name}' / '{v_name}' not in dataset.")
        ref_varname = u_name  # use u component for dim detection
        height_vals = ds["height"].values
        z_idx = int(np.argmin(np.abs(height_vals - level_km)))
        actual_level = float(height_vals[z_idx])
        u = ds[u_name].isel(num_cases=local_idx, height=z_idx).values
        v = ds[v_name].isel(num_cases=local_idx, height=z_idx).values
        data = np.sqrt(u**2 + v**2)
    else:
        if varname not in ds:
            available = [k for k, v in VARIABLES.items() if v[1] in ds]
            raise ValueError(f"'{varname}' not in dataset. Available: {available}")
        ref_varname = varname
        da = ds[varname].isel(num_cases=local_idx)
        height_vals = ds["height"].values
        z_idx = int(np.argmin(np.abs(height_vals - level_km)))
        actual_level = float(height_vals[z_idx])
        da = da.isel(height=z_idx)
        data = da.values

    # Determine which spatial grid this variable is on:
    #   - recentered / total_recentered vars: (northward_distance, eastward_distance) — 201×201
    #   - original swath vars:                (latitude, longitude)                   — 200×200
    var_dims = set(ds[ref_varname].dims)
    if "eastward_distance" in var_dims and "northward_distance" in var_dims:
        x = ds["eastward_distance"].values
        y = ds["northward_distance"].values
    elif "latitude" in var_dims and "longitude" in var_dims:
        # Original grid: construct storm-centered distance axes (2 km spacing)
        nx = ds.sizes["longitude"]
        ny = ds.sizes["latitude"]
        x = np.linspace(-(nx - 1), (nx - 1), nx)   # e.g. -199 to +199 km
        y = np.linspace(-(ny - 1), (ny - 1), ny)
    else:
        # Fallback: use first two remaining dims as index arrays
        remaining = [d for d in da.dims if d != "num_cases" and d != "height"]
        x = np.arange(data.shape[-1])
        y = np.arange(data.shape[-2])

    fig, ax = plt.subplots(figsize=(7, 6.5), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    im = ax.pcolormesh(x, y, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")

    rmw = case_meta.get("rmw_km")
    if rmw is not None and not np.isnan(float(rmw)):
        rmw = float(rmw)
        theta = np.linspace(0, 2 * np.pi, 360)
        ax.plot(rmw * np.cos(theta), rmw * np.sin(theta),
                "w--", lw=1.5, alpha=0.85, label=f"RMW = {rmw:.0f} km")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.3, labelcolor="white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(units, color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    storm   = case_meta.get("storm_name", "")
    dt      = case_meta.get("datetime", "")
    vmax_kt = case_meta.get("vmax_kt", "")
    vmax_str = f"  |  Vmax = {float(vmax_kt):.0f} kt" if vmax_kt != "" else ""
    ax.set_title(
        f"{storm}  |  {dt}{vmax_str}\n{display_name} @ {actual_level:.1f} km",
        color="white", fontsize=10, pad=8,
    )
    ax.set_xlabel("Eastward distance (km)", color="white", fontsize=9)
    ax.set_ylabel("Northward distance (km)", color="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.set_aspect("equal")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Metadata + plot cache
# ---------------------------------------------------------------------------
METADATA_PATH = Path(os.environ.get("METADATA_PATH", "./tc_radar_metadata.json"))
MERGE_METADATA_PATH = Path(os.environ.get("MERGE_METADATA_PATH", "./tc_radar_metadata_merge.json"))
_metadata_cache: dict[int, dict] = {}
_merge_metadata_cache: dict[int, dict] = {}
_plot_cache: OrderedDict = OrderedDict()
_PLOT_CACHE_MAX = 150  # ~150 plots × ~150 KB ≈ 22 MB max

_data_cache: OrderedDict = OrderedDict()
_DATA_CACHE_MAX = 100  # ~100 entries × ~100 KB ≈ 10 MB max


@app.on_event("startup")
def startup():
    # Load metadata
    global _metadata_cache, _merge_metadata_cache
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            data = json.load(f)
        _metadata_cache = {c["case_index"]: c for c in data.get("cases", [])}
        print(f"Loaded {len(_metadata_cache)} cases from {METADATA_PATH}")
    else:
        print(f"Warning: {METADATA_PATH} not found")

    if MERGE_METADATA_PATH.exists():
        with open(MERGE_METADATA_PATH) as f:
            data = json.load(f)
        _merge_metadata_cache = {c["case_index"]: c for c in data.get("cases", [])}
        print(f"Loaded {len(_merge_metadata_cache)} merge cases from {MERGE_METADATA_PATH}")
    else:
        print(f"Warning: {MERGE_METADATA_PATH} not found — merge composites will not filter correctly")

    backend = f"S3 Zarr (s3://{S3_BUCKET}/{S3_PREFIX})" if USE_S3 else "AOML HTTP (fallback)"
    print(f"Data backend: {backend}")

    # Pre-warm datasets in background threads
    # Also enrich metadata with SHIPS shear values once datasets are loaded
    def prewarm_and_enrich(data_type, era):
        try:
            ds = get_dataset(data_type, era)
            print(f"Pre-warmed {data_type}/{era}")
            _enrich_metadata_with_ships(ds, data_type, era)
        except Exception as e:
            print(f"Pre-warm failed {data_type}/{era}: {e}")

    for dt in ("swath", "merge"):
        for era in ("early", "recent"):
            threading.Thread(target=prewarm_and_enrich, args=(dt, era), daemon=True).start()


def _enrich_metadata_with_ships(ds, data_type, era):
    """
    Enrich metadata cache with SHIPS shear values (SDDC, SHDC) read
    from the Zarr store.  Runs once per dataset at startup so that
    composite filtering never needs to open the Zarr just to check shear.
    """
    cache = _merge_metadata_cache if data_type == "merge" else _metadata_cache
    early_count = CASE_COUNTS[(data_type, "early")]
    offset = 0 if era == "early" else early_count
    n_cases = ds.sizes.get("num_cases", 0)
    enriched = 0
    for local_idx in range(n_cases):
        case_index = local_idx + offset
        if case_index not in cache:
            continue
        sddc = _get_ships_value(ds, local_idx, "sddc_ships")
        shdc = _get_ships_value(ds, local_idx, "shdc_ships")
        if sddc is not None:
            cache[case_index]["sddc"] = sddc
        if shdc is not None:
            cache[case_index]["shdc"] = shdc
        enriched += 1
    print(f"Enriched {enriched} cases with SHIPS shear data ({data_type}/{era})")


def _filter_cases_for_composite(
    min_intensity: float, max_intensity: float,
    min_vmax_change: float, max_vmax_change: float,
    min_tilt: float, max_tilt: float,
    min_year: int, max_year: int,
    min_shear_mag: float, max_shear_mag: float,
    min_shear_dir: float, max_shear_dir: float,
    data_type: str = "swath",
) -> list[int]:
    """Return list of case_index values that pass all composite filters."""
    cache = _merge_metadata_cache if data_type == "merge" else _metadata_cache
    matching = []
    for idx, meta in cache.items():
        vmax = meta.get("vmax_kt")
        if vmax is None:
            continue
        if vmax < min_intensity or vmax > max_intensity:
            continue

        vc = meta.get("24-h_vmax_change_kt")
        if min_vmax_change > -100 or max_vmax_change < 85:
            if vc is None:
                continue
            if vc < min_vmax_change or vc > max_vmax_change:
                continue

        tilt = meta.get("tilt_magnitude_km")
        if min_tilt > 0 or max_tilt < 200:
            if tilt is None:
                continue
            if tilt < min_tilt or tilt > max_tilt:
                continue

        year = meta.get("year")
        if year is not None and (year < min_year or year > max_year):
            continue

        # Shear magnitude: swath uses "shdc" (enriched), merge uses "shear_magnitude_kt" (from JSON)
        shear_mag = meta.get("shdc") or meta.get("shear_magnitude_kt")
        if min_shear_mag > 0 or max_shear_mag < 100:
            if shear_mag is None:
                continue
            if shear_mag < min_shear_mag or shear_mag > max_shear_mag:
                continue

        sddc = meta.get("sddc")
        if min_shear_dir > 0 or max_shear_dir < 360:
            if sddc is None:
                continue
            # Handle wraparound: if min > max, it's a range crossing 360°
            if min_shear_dir <= max_shear_dir:
                if sddc < min_shear_dir or sddc > max_shear_dir:
                    continue
            else:
                # e.g. min=315, max=45 means "NW through NE"
                if sddc < min_shear_dir and sddc > max_shear_dir:
                    continue

        matching.append(idx)
    return matching


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "s3_zarr" if USE_S3 else "aoml_http",
        "bucket": S3_BUCKET if USE_S3 else None,
    }


# /debug endpoint removed for production — exposed internal dataset structure.
# To re-enable for development, uncomment the block below and protect with auth.
# @app.get("/debug")
# def debug():
#     """Debug endpoint — shows what xarray sees in the Zarr store."""
#     try:
#         ds = get_dataset("swath", "early")
#         return {"dims": dict(ds.sizes), "data_vars": list(ds.data_vars), "coords": list(ds.coords)}
#     except Exception as e:
#         return {"error": str(e)}


@app.get("/health")
def health_check():
    """Lightweight health/readiness check for monitoring cold-start status."""
    return {"status": "ok", "version": "2.0"}


@app.get("/variables")
def list_variables():
    return [{"key": k, "display_name": v[0], "units": v[3]} for k, v in VARIABLES.items()]


@app.get("/levels")
def list_levels():
    return {"levels_km": [round(i * 0.5, 1) for i in range(37)]}


@app.get("/metadata")
def get_metadata(
    case_index: int = Query(..., ge=0),
    data_type:  str = Query("swath", description="'swath' or 'merge'"),
):
    cache = _merge_metadata_cache if data_type == "merge" else _metadata_cache
    if case_index not in cache:
        raise HTTPException(status_code=404, detail=f"case_index {case_index} not found")
    return JSONResponse(cache[case_index])


# ---------------------------------------------------------------------------
# Helper: matplotlib colormap → Plotly colorscale
# ---------------------------------------------------------------------------
def _cmap_to_plotly(cmap_name: str, n_steps: int = 64) -> list:
    """Convert a matplotlib colormap name to a Plotly-compatible colorscale list."""
    cmap = plt.get_cmap(cmap_name)
    return [
        [round(i / (n_steps - 1), 4),
         f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"]
        for i, c in enumerate(cmap(np.linspace(0, 1, n_steps)))
    ]


def _extract_2d_slice(ds, local_idx, variable_key, z_idx):
    """Extract a 2D (y, x) data slice for a variable at a given height index."""
    _, varname, _, _, _, _ = VARIABLES[variable_key]
    if variable_key in DERIVED_VARIABLES:
        u_name, v_name = DERIVED_VARIABLES[variable_key]
        u = ds[u_name].isel(num_cases=local_idx, height=z_idx).values
        v = ds[v_name].isel(num_cases=local_idx, height=z_idx).values
        return np.sqrt(u**2 + v**2), u_name
    else:
        da = ds[varname].isel(num_cases=local_idx, height=z_idx)
        return da.values, varname


def _clean_2d(data):
    """Convert 2D numpy array to JSON-safe nested list (NaN → None)."""
    return [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in data]


@app.get("/data")
def get_data(
    case_index: int   = Query(...,              ge=0,          description="0-based case index"),
    variable:   str   = Query(DEFAULT_VARIABLE,                description="Variable key — see /variables"),
    level_km:   float = Query(2.0,              ge=0.0, le=18, description="Altitude in km"),
    data_type:  str   = Query("swath",                         description="'swath' or 'merge'"),
    overlay:    str   = Query("",                              description="Optional overlay variable key"),
):
    """Return the raw 2D data slice as JSON for client-side Plotly rendering."""
    if variable not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"Unknown variable '{variable}'. See /variables.")
    if data_type not in ("swath", "merge"):
        raise HTTPException(status_code=400, detail="data_type must be 'swath' or 'merge'")
    if overlay and overlay not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"Unknown overlay variable '{overlay}'. See /variables.")

    # Serve from cache if available (instant — no S3 read or computation)
    cache_key = (case_index, variable, round(level_km, 1), data_type, overlay)
    if cache_key in _data_cache:
        _data_cache.move_to_end(cache_key)
        return JSONResponse(_data_cache[cache_key], headers={"X-Cache": "HIT"})

    try:
        ds, local_idx = resolve_case(case_index, data_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open dataset: {e}")

    display_name, varname, cmap, units, vmin, vmax = VARIABLES[variable]
    height_vals = ds["height"].values
    z_idx = int(np.argmin(np.abs(height_vals - level_km)))
    actual_level = float(height_vals[z_idx])

    # Primary variable
    data, ref_varname = _extract_2d_slice(ds, local_idx, variable, z_idx)

    # Determine spatial grid
    var_dims = set(ds[ref_varname].dims)
    if "eastward_distance" in var_dims and "northward_distance" in var_dims:
        x = ds["eastward_distance"].values.tolist()
        y = ds["northward_distance"].values.tolist()
    elif "latitude" in var_dims and "longitude" in var_dims:
        nx = ds.sizes["longitude"]
        ny = ds.sizes["latitude"]
        x = np.linspace(-(nx - 1), (nx - 1), nx).tolist()
        y = np.linspace(-(ny - 1), (ny - 1), ny).tolist()
    else:
        x = list(range(data.shape[-1]))
        y = list(range(data.shape[-2]))

    case_meta = _build_case_meta(case_index, ds, local_idx, data_type)

    result = {
        "data": _clean_2d(data),
        "x": x,
        "y": y,
        "actual_level_km": actual_level,
        "variable": {
            "key": variable,
            "display_name": display_name,
            "units": units,
            "vmin": vmin,
            "vmax": vmax,
            "colorscale": _cmap_to_plotly(cmap),
        },
        "case_meta": case_meta,
    }

    # Optional overlay variable
    if overlay:
        ov_display, ov_varname, _, ov_units, ov_vmin, ov_vmax = VARIABLES[overlay]
        try:
            ov_data, _ = _extract_2d_slice(ds, local_idx, overlay, z_idx)
            result["overlay"] = {
                "data": _clean_2d(ov_data),
                "key": overlay,
                "display_name": ov_display,
                "units": ov_units,
                "vmin": ov_vmin,
                "vmax": ov_vmax,
            }
        except Exception:
            pass  # silently skip overlay if variable unavailable

    # Store in cache
    _data_cache[cache_key] = result
    if len(_data_cache) > _DATA_CACHE_MAX:
        _data_cache.popitem(last=False)  # evict oldest entry

    return JSONResponse(result, headers={"X-Cache": "MISS"})


def _extract_cross_section(ds, local_idx, variable_key, x_coords, y_coords, xi_idx, yi_idx, h_axis, n_heights, n_points):
    """Extract a vertical cross-section for a variable along sample indices."""
    _, varname, _, _, _, _ = VARIABLES[variable_key]
    if variable_key in DERIVED_VARIABLES:
        u_name, v_name = DERIVED_VARIABLES[variable_key]
        u_3d = ds[u_name].isel(num_cases=local_idx).values
        v_3d = ds[v_name].isel(num_cases=local_idx).values
        vol = np.sqrt(u_3d**2 + v_3d**2)
    else:
        vol = ds[varname].isel(num_cases=local_idx).values

    cs = np.full((n_heights, n_points), np.nan)
    for h in range(n_heights):
        for p in range(n_points):
            if h_axis == 0:
                cs[h, p] = vol[h, yi_idx[p], xi_idx[p]]
            elif h_axis == 2:
                cs[h, p] = vol[yi_idx[p], xi_idx[p], h]
            else:
                cs[h, p] = vol[yi_idx[p], h, xi_idx[p]]
    return cs


@app.get("/cross_section")
def cross_section(
    case_index: int   = Query(...,              ge=0,          description="0-based case index"),
    variable:   str   = Query(DEFAULT_VARIABLE,                description="Variable key — see /variables"),
    data_type:  str   = Query("swath",                         description="'swath' or 'merge'"),
    x0:         float = Query(...,                             description="Start X (km)"),
    y0:         float = Query(...,                             description="Start Y (km)"),
    x1:         float = Query(...,                             description="End X (km)"),
    y1:         float = Query(...,                             description="End Y (km)"),
    n_points:   int   = Query(150,              ge=10, le=500, description="Sample points along line"),
    overlay:    str   = Query("",                              description="Optional overlay variable key"),
):
    """Return a vertical cross-section along a user-defined line."""
    if variable not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"Unknown variable '{variable}'. See /variables.")
    if data_type not in ("swath", "merge"):
        raise HTTPException(status_code=400, detail="data_type must be 'swath' or 'merge'")
    if overlay and overlay not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"Unknown overlay variable '{overlay}'.")

    try:
        ds, local_idx = resolve_case(case_index, data_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open dataset: {e}")

    display_name, varname, cmap, units, vmin, vmax = VARIABLES[variable]
    height_vals = ds["height"].values

    # Determine ref variable for grid detection
    if variable in DERIVED_VARIABLES:
        ref_varname = DERIVED_VARIABLES[variable][0]
    else:
        if varname not in ds:
            raise HTTPException(status_code=400, detail=f"'{varname}' not in dataset.")
        ref_varname = varname

    # Determine spatial grid
    var_dims = set(ds[ref_varname].dims)
    if "eastward_distance" in var_dims and "northward_distance" in var_dims:
        x_coords = ds["eastward_distance"].values
        y_coords = ds["northward_distance"].values
    elif "latitude" in var_dims and "longitude" in var_dims:
        nx = ds.sizes["longitude"]
        ny = ds.sizes["latitude"]
        x_coords = np.linspace(-(nx - 1), (nx - 1), nx)
        y_coords = np.linspace(-(ny - 1), (ny - 1), ny)
    else:
        raise HTTPException(status_code=500, detail="Cannot determine spatial grid")

    # Determine height axis position
    dim_list = [d for d in ds[ref_varname].dims if d != "num_cases"]
    if "height" not in dim_list:
        raise HTTPException(status_code=500, detail="Cannot determine height axis")
    h_axis = dim_list.index("height")

    # Sample points along the line
    t = np.linspace(0, 1, n_points)
    xs = x0 + t * (x1 - x0)
    ys = y0 + t * (y1 - y0)
    dist = np.sqrt((xs - x0)**2 + (ys - y0)**2)

    xi_idx = np.array([int(np.argmin(np.abs(x_coords - xp))) for xp in xs])
    yi_idx = np.array([int(np.argmin(np.abs(y_coords - yp))) for yp in ys])

    n_heights = len(height_vals)

    # Primary cross-section
    cs = _extract_cross_section(ds, local_idx, variable, x_coords, y_coords, xi_idx, yi_idx, h_axis, n_heights, n_points)

    case_meta = _build_case_meta(case_index, ds, local_idx, data_type)

    result = {
        "cross_section": _clean_2d(cs),
        "distance_km": [round(float(d), 2) for d in dist],
        "height_km": [round(float(h), 2) for h in height_vals],
        "endpoints": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "variable": {
            "key": variable,
            "display_name": display_name,
            "units": units,
            "vmin": vmin,
            "vmax": vmax,
            "colorscale": _cmap_to_plotly(cmap),
        },
        "case_meta": case_meta,
    }

    # Optional overlay cross-section
    if overlay:
        ov_display, _, _, ov_units, ov_vmin, ov_vmax = VARIABLES[overlay]
        try:
            ov_cs = _extract_cross_section(ds, local_idx, overlay, x_coords, y_coords, xi_idx, yi_idx, h_axis, n_heights, n_points)
            result["overlay"] = {
                "cross_section": _clean_2d(ov_cs),
                "key": overlay,
                "display_name": ov_display,
                "units": ov_units,
                "vmin": ov_vmin,
                "vmax": ov_vmax,
            }
        except Exception:
            pass

    return JSONResponse(result)


@app.get("/volume")
def get_volume(
    case_index: int   = Query(...,              ge=0,          description="0-based case index"),
    variable:   str   = Query(DEFAULT_VARIABLE,                description="Variable key — see /variables"),
    data_type:  str   = Query("swath",                         description="'swath' or 'merge'"),
    stride:     int   = Query(2,                ge=1, le=5,    description="Spatial subsampling stride (2 = half res)"),
    max_height_km: float = Query(15.0,          ge=1, le=18,   description="Maximum height to include (km)"),
    compact:    bool  = Query(False,                            description="If true, send 1D axis vectors instead of flattened meshgrid"),
):
    """
    Return the full 3D volume as flattened arrays for Plotly isosurface rendering.

    The grid is subsampled spatially by `stride` to reduce transfer size.
    NaN values are replaced with a sentinel (-9999) so the grid stays regular;
    the client should set isomin above this sentinel.

    compact=true mode sends 1D axis vectors (x_axis, y_axis, z_axis) instead
    of the full flattened meshgrid (x, y, z), reducing payload ~4× before gzip.
    The client reconstructs the meshgrid from axes + grid_shape.
    """
    if variable not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"Unknown variable '{variable}'. See /variables.")
    if data_type not in ("swath", "merge"):
        raise HTTPException(status_code=400, detail="data_type must be 'swath' or 'merge'")

    try:
        ds, local_idx = resolve_case(case_index, data_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open dataset: {e}")

    display_name, varname, cmap, units, vmin, vmax = VARIABLES[variable]
    height_vals = ds["height"].values

    # Cap height
    h_mask = height_vals <= max_height_km + 0.01
    height_sub = height_vals[h_mask]

    # Extract full 3D volume
    vol, ref_varname = _extract_3d_volume(ds, local_idx, variable)

    # Determine spatial grid and height axis position
    dim_list = [d for d in ds[ref_varname].dims if d != "num_cases"]
    if "height" not in dim_list:
        raise HTTPException(status_code=500, detail="Cannot determine height axis")
    h_axis = dim_list.index("height")

    var_dims = set(ds[ref_varname].dims)
    if "eastward_distance" in var_dims and "northward_distance" in var_dims:
        x_full = ds["eastward_distance"].values
        y_full = ds["northward_distance"].values
    elif "latitude" in var_dims and "longitude" in var_dims:
        nx = ds.sizes["longitude"]
        ny = ds.sizes["latitude"]
        x_full = np.linspace(-(nx - 1), (nx - 1), nx)
        y_full = np.linspace(-(ny - 1), (ny - 1), ny)
    else:
        x_full = np.arange(vol.shape[-1], dtype=float)
        y_full = np.arange(vol.shape[-2], dtype=float)

    # Subsample spatial dimensions
    x_sub = x_full[::stride]
    y_sub = y_full[::stride]
    n_h = int(h_mask.sum())

    # Slice the volume: need to handle varying axis orders
    if h_axis == 0:
        vol_sub = vol[:n_h, ::stride, ::stride]
    elif h_axis == 2:
        vol_sub = vol[::stride, ::stride, :n_h]
    else:
        vol_sub = vol[::stride, :n_h, ::stride]

    # Reorder to (height, y, x) if needed
    if h_axis == 1:
        vol_sub = np.transpose(vol_sub, (1, 0, 2))  # (y, h, x) -> (h, y, x)
    elif h_axis == 2:
        vol_sub = np.transpose(vol_sub, (2, 0, 1))  # (y, x, h) -> (h, y, x)

    nz, ny, nx = vol_sub.shape
    v_flat = vol_sub.ravel()

    # Compute actual data range (excluding NaN)
    valid = v_flat[np.isfinite(v_flat)]
    data_min = float(np.nanmin(valid)) if len(valid) > 0 else vmin
    data_max = float(np.nanmax(valid)) if len(valid) > 0 else vmax

    # Replace NaN with sentinel for regular-grid isosurface
    SENTINEL = -9999.0
    v_flat = np.where(np.isfinite(v_flat), np.round(v_flat, 3), SENTINEL)

    case_meta = _build_case_meta(case_index, ds, local_idx, data_type)

    result = {
        "value": v_flat.tolist(),
        "sentinel": SENTINEL,
        "grid_shape": [nz, ny, nx],
        "variable": {
            "key": variable,
            "display_name": display_name,
            "units": units,
            "vmin": vmin,
            "vmax": vmax,
            "data_min": round(data_min, 3),
            "data_max": round(data_max, 3),
            "colorscale": _cmap_to_plotly(cmap),
        },
        "case_meta": case_meta,
    }

    if compact:
        # Compact mode: send 1D axis vectors (~232 values total)
        # Client reconstructs meshgrid from axes + grid_shape
        result["x_axis"] = np.round(x_sub, 2).tolist()
        result["y_axis"] = np.round(y_sub, 2).tolist()
        result["z_axis"] = np.round(height_sub, 2).tolist()
    else:
        # Legacy mode: send full flattened meshgrid (~918K values)
        Z, Y, X = np.meshgrid(height_sub, y_sub, x_sub, indexing='ij')
        result["x"] = np.round(X.ravel(), 2).tolist()
        result["y"] = np.round(Y.ravel(), 2).tolist()
        result["z"] = np.round(Z.ravel(), 2).tolist()

    return JSONResponse(result)


def _extract_3d_volume(ds, local_idx, variable_key):
    """Extract the full 3D volume (height × y × x) for a variable."""
    _, varname, _, _, _, _ = VARIABLES[variable_key]
    if variable_key in DERIVED_VARIABLES:
        u_name, v_name = DERIVED_VARIABLES[variable_key]
        u_3d = ds[u_name].isel(num_cases=local_idx).values
        v_3d = ds[v_name].isel(num_cases=local_idx).values
        return np.sqrt(u_3d**2 + v_3d**2), u_name
    else:
        vol = ds[varname].isel(num_cases=local_idx).values
        return vol, varname


def _get_ships_value(ds, local_idx, varname):
    """
    Look up a SHIPS variable at t=0 h.

    SHIPS lag-hour axis has 17 entries: -48, -42, -36, …, 0, …, +42, +48 h.
    t=0 is index 8 (0-based).
    Returns float or None if unavailable / missing (9999).
    """
    SHIPS_T0_IDX = 8
    if varname not in ds:
        return None
    try:
        raw = ds[varname].isel(num_cases=local_idx).values
        val = float(raw[SHIPS_T0_IDX]) if raw.ndim >= 1 else float(raw)
    except Exception:
        return None
    if val == 9999 or np.isnan(val):
        return None
    return round(val, 1)


def _get_sddc(ds, local_idx):
    """SDDC: deep-layer shear heading (deg, met convention)."""
    return _get_ships_value(ds, local_idx, "sddc_ships")


def _get_shdc(ds, local_idx):
    """SHDC: deep-layer shear magnitude (kt)."""
    return _get_ships_value(ds, local_idx, "shdc_ships")




# ---------------------------------------------------------------------------
# IR satellite imagery – server-side PNG rendering
# ---------------------------------------------------------------------------

# Build a 256-entry RGBA lookup table matching the JS IR colormap.
# The colormap maps normalized (1 - (Tb-vmin)/(vmax-vmin)) to RGB.
def _build_ir_lut():
    """Create a 256-entry uint8 RGBA LUT for IR brightness temperatures."""
    stops = [
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
    lut = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256):
        frac = i / 255.0
        # Find bounding stops
        lo, hi = stops[0], stops[-1]
        for s in range(len(stops) - 1):
            if frac >= stops[s][0] and frac <= stops[s + 1][0]:
                lo, hi = stops[s], stops[s + 1]
                break
        t = 0.0 if hi[0] == lo[0] else (frac - lo[0]) / (hi[0] - lo[0])
        lut[i, 0] = int(lo[1] + t * (hi[1] - lo[1]) + 0.5)
        lut[i, 1] = int(lo[2] + t * (hi[2] - lo[2]) + 0.5)
        lut[i, 2] = int(lo[3] + t * (hi[3] - lo[3]) + 0.5)
        lut[i, 3] = 220  # alpha
    return lut

_IR_LUT = _build_ir_lut()


def _render_ir_png(frame_2d, vmin=190.0, vmax=310.0):
    """
    Render a 2D Tb array to a base64-encoded PNG string.
    Returns a data-URL ready for use as an image src.
    """
    from PIL import Image

    arr = np.asarray(frame_2d, dtype=np.float32)
    # Normalize: cold clouds (low Tb) → high index → bright colors
    frac = 1.0 - (arr - vmin) / (vmax - vmin)
    frac = np.clip(frac, 0.0, 1.0)
    indices = (frac * 255).astype(np.uint8)

    # Apply LUT
    rgba = _IR_LUT[indices]  # shape (H, W, 4)

    # Set NaN / invalid pixels to transparent
    mask = ~np.isfinite(arr) | (arr <= 0)
    rgba[mask] = [0, 0, 0, 0]

    # Flip vertically (lat ascending → image top-to-bottom)
    rgba = rgba[::-1]

    # Encode as PNG (level=1: 4× faster, ~same size as optimize)
    img = Image.fromarray(rgba, 'RGBA')
    buf = io.BytesIO()
    img.save(buf, format='PNG', compress_level=1)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/png;base64,{b64}"


@app.get("/ir")
def get_ir(case_index: int = Query(..., ge=0)):
    """
    Return IR metadata and the t=0 frame (most recent) for instant display.
    The client then calls /ir_frame for each additional lag index to progressively
    """
    ir_store = get_ir_dataset()
    if ir_store is None:
        raise HTTPException(status_code=503, detail="IR data not available (S3 not configured)")

    # Check status
    try:
        status = int(ir_store['status'][case_index])
    except (IndexError, KeyError):
        raise HTTPException(status_code=404, detail=f"Case {case_index} not found in IR store")

    if status != 1:
        raise HTTPException(status_code=404, detail=f"No IR data for case {case_index}")

    # Read metadata
    import numpy as np
    center_lat = float(ir_store['center_lat'][case_index])
    center_lon = float(ir_store['center_lon'][case_index])
    lat_offsets = ir_store['lat_offsets'][:].tolist()
    lon_offsets = ir_store['lon_offsets'][:].tolist()
    lag_hours = ir_store['lag_hours'][:].tolist()

    # Render ONLY the t=0 frame (index 0) for instant display
    tb_frame0 = ir_store['Tb'][case_index, 0]  # shape: (n_lat, n_lon)
    if np.all(np.isnan(tb_frame0)):
        frame0_png = None
    else:
        frame0_png = _render_ir_png(tb_frame0, vmin=190.0, vmax=310.0)

    # Read IR datetimes
    ir_dt_raw = ir_store['ir_datetime'][case_index]  # epoch minutes
    ir_datetimes = []
    for val in ir_dt_raw:
        val = int(val)
        if val > 0:
            dt = datetime.utcfromtimestamp(val * 60)
            ir_datetimes.append(dt.strftime('%Y-%m-%d %H:%M UTC'))
        else:
            ir_datetimes.append(None)

    return {
        "case_index": case_index,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "lat_offsets": lat_offsets,
        "lon_offsets": lon_offsets,
        "lag_hours": lag_hours,
        "ir_datetimes": ir_datetimes,
        "n_frames": len(lag_hours),
        "frame0": frame0_png,
        "units": "K",
    }


@app.get("/ir_frame")
def get_ir_frame(case_index: int = Query(..., ge=0), lag_index: int = Query(..., ge=0)):
    """
    Return a single server-rendered IR PNG frame.
    Called progressively by the client to build up the animation.
    """
    ir_store = get_ir_dataset()
    if ir_store is None:
        raise HTTPException(status_code=503, detail="IR data not available (S3 not configured)")

    try:
        status = int(ir_store['status'][case_index])
    except (IndexError, KeyError):
        raise HTTPException(status_code=404, detail=f"Case {case_index} not found in IR store")

    if status != 1:
        raise HTTPException(status_code=404, detail=f"No IR data for case {case_index}")

    import numpy as np
    n_lags = ir_store['Tb'].shape[1]
    if lag_index >= n_lags:
        raise HTTPException(status_code=404, detail=f"lag_index {lag_index} out of range (max {n_lags-1})")

    frame = ir_store['Tb'][case_index, lag_index]
    if np.all(np.isnan(frame)):
        png = None
    else:
        png = _render_ir_png(frame, vmin=190.0, vmax=310.0)

    return {
        "case_index": case_index,
        "lag_index": lag_index,
        "frame": png,
    }



# ---------------------------------------------------------------------------
# ERA5 environmental diagnostics endpoint
# ---------------------------------------------------------------------------
@app.get("/era5")
def get_era5(
    case_index: int = Query(..., ge=0),
    field: str = Query("shear_mag"),
    include_profiles: bool = Query(True),
    radius_km: float = Query(0, ge=0, le=1200, description="Crop radius in km (0=full domain)"),
):
    """
    Return ERA5 environmental diagnostics for a TC-RADAR case.

    Parameters
    ----------
    case_index : int
    field : str - 2D field to return (shear_mag, rh_mid, div200)
    include_profiles : bool - include vertical profiles & hodograph data
    radius_km : float - crop to this radius from TC center (0=full 20° domain)
    """
    era5 = get_era5_dataset()
    if era5 is None:
        raise HTTPException(status_code=503, detail="ERA5 data not available")

    try:
        status = int(era5['status'][case_index])
    except (IndexError, KeyError):
        raise HTTPException(status_code=404, detail=f"Case {case_index} not in ERA5 store")
    if status != 1:
        raise HTTPException(status_code=404, detail=f"No ERA5 data for case {case_index}")

    if field not in ERA5_FIELD_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown field: {field}. Valid: {list(ERA5_FIELD_CONFIG.keys())}")

    cfg = ERA5_FIELD_CONFIG[field]

    # Read full 2D field and coordinate offsets
    data_2d = era5[field][case_index]  # (81, 81)
    lat_offsets = era5['lat_offsets'][:]
    lon_offsets = era5['lon_offsets'][:]

    # Crop to radius_km if specified
    if radius_km > 0:
        center_lat = float(era5['center_lat'][case_index])
        crop_deg = radius_km / 111.0  # approximate degrees
        lat_mask = np.abs(lat_offsets) <= crop_deg
        lon_mask = np.abs(lon_offsets) <= crop_deg
        data_2d = data_2d[np.ix_(lat_mask, lon_mask)]
        lat_offsets = lat_offsets[lat_mask]
        lon_offsets = lon_offsets[lon_mask]

    field_list = []
    for row in data_2d:
        field_list.append([
            None if (np.isnan(v)) else round(float(v), 4)
            for v in row
        ])

    result = {
        "case_index": case_index,
        "field": field,
        "field_config": cfg,
        "data": field_list,
        "center_lat": float(era5['center_lat'][case_index]),
        "center_lon": float(era5['center_lon'][case_index]),
        "lat_offsets": lat_offsets.tolist(),
        "lon_offsets": lon_offsets.tolist(),
    }

    # Add vector components if field has them (subsampled for readability)
    if cfg.get("has_vectors"):
        u_full = era5[cfg["vector_u"]][case_index]
        v_full = era5[cfg["vector_v"]][case_index]
        if radius_km > 0:
            u_full = u_full[np.ix_(lat_mask, lon_mask)]
            v_full = v_full[np.ix_(lat_mask, lon_mask)]
        # Stride: aim for ~10-15 arrows per axis
        n_pts = len(lat_offsets)
        stride = max(1, n_pts // 12)
        u_sub = u_full[::stride, ::stride]
        v_sub = v_full[::stride, ::stride]
        result["vectors"] = {
            "u": [[round(float(v), 2) if not np.isnan(v) else None for v in row] for row in u_sub],
            "v": [[round(float(v), 2) if not np.isnan(v) else None for v in row] for row in v_sub],
            "stride": stride,
        }

    # Scalar diagnostics
    # Scalar diagnostics — include all available fields
    result["scalars"] = {}
    for sname in ['shear_mag_env', 'shear_dir_env', 'rh_mid_env', 'div200_env',
                   'sst_env', 'chi_m', 'v_pi', 'vent_index']:
        try:
            val = float(era5[sname][case_index])
            if np.isnan(val):
                result["scalars"][sname] = None
            elif 'div' in sname:
                result["scalars"][sname] = round(val, 4)
            elif sname == 'vent_index':
                result["scalars"][sname] = round(val, 3)
            elif sname == 'chi_m':
                result["scalars"][sname] = round(val, 2)
            else:
                result["scalars"][sname] = round(val, 1)
        except Exception:
            result["scalars"][sname] = None

    # Vertical profiles (200-600 km annulus mean) with derived thermodynamics
    if include_profiles:
        try:
            plev = era5['plev'][:].astype(float)           # hPa
            t_arr = era5['t_profile'][case_index].astype(float)   # K
            q_arr = era5['q_profile'][case_index].astype(float)   # g/kg from Zarr

            # Convert q from g/kg to kg/kg if stored as g/kg
            q_kgkg = q_arr / 1000.0 if np.nanmax(q_arr) > 0.5 else q_arr

            # Potential temperature: θ = T * (1000/p)^(Rd/Cp)
            Rd_Cp = 287.04 / 1005.7
            theta = t_arr * (1000.0 / plev) ** Rd_Cp

            # Equivalent potential temperature (Bolton 1980):
            # θe = θ_d * exp(Lv * r_s / (Cp_d * T_LCL))
            # Simplified: θe ≈ T * (1000/p)^0.2854*(1-0.28*q) * exp((3036/T_LCL - 1.78)*q*(1+0.448*q))
            # Using the more standard approximation:
            # θe = θ * exp(Lv * q / (Cpd * T))
            Lv = 2.501e6  # J/kg
            Cpd = 1005.7  # J/kg/K
            theta_e = theta * np.exp(Lv * q_kgkg / (Cpd * t_arr))

            result["profiles"] = {
                "plev": plev.tolist(),
                "u": [round(float(v), 2) if not np.isnan(v) else None for v in era5['u_profile'][case_index]],
                "v": [round(float(v), 2) if not np.isnan(v) else None for v in era5['v_profile'][case_index]],
                "rh": [round(float(v), 1) if not np.isnan(v) else None for v in era5['rh_profile'][case_index]],
                "t": [round(float(v), 2) if not np.isnan(v) else None for v in t_arr],
                "theta": [round(float(v), 1) if not np.isnan(v) else None for v in theta],
                "theta_e": [round(float(v), 1) if not np.isnan(v) else None for v in theta_e],
                "q": [round(float(v), 6) if not np.isnan(v) else None for v in q_arr],
            }
        except Exception:
            result["profiles"] = None

    # Report whether 3D fields are available (so frontend can enable/disable radius slider)
    _3d_t_names = ['t_3d', 'temperature_3d', 't3d', 'temperature', 'temp_3d']
    _3d_q_names = ['q_3d', 'specific_humidity_3d', 'q3d', 'specific_humidity', 'shum_3d']
    result["has_3d"] = any(k in era5 for k in _3d_t_names) and any(k in era5 for k in _3d_q_names)

    return result


# ---------------------------------------------------------------------------
# ERA5 scalar recomputation at user-specified annuli
# ---------------------------------------------------------------------------
@app.get("/era5_scalars")
def get_era5_scalars(
    case_index: int = Query(..., ge=0),
    inner_km: float = Query(200, ge=0, le=1000, description="Inner annulus radius (km)"),
    outer_km: float = Query(800, ge=50, le=1200, description="Outer annulus radius (km)"),
):
    """
    Recompute area-mean scalar diagnostics from the gridded ERA5 2D fields
    at a user-specified annulus.

    This allows users to explore sensitivity of environmental diagnostics
    to the averaging domain without requiring Zarr-level precomputation.
    """
    era5 = get_era5_dataset()
    if era5 is None:
        raise HTTPException(status_code=503, detail="ERA5 data not available")

    try:
        status = int(era5['status'][case_index])
    except (IndexError, KeyError):
        raise HTTPException(status_code=404, detail=f"Case {case_index} not in ERA5 store")
    if status != 1:
        raise HTTPException(status_code=404, detail=f"No ERA5 data for case {case_index}")

    center_lat = float(era5['center_lat'][case_index])
    lat_offsets = era5['lat_offsets'][:]
    lon_offsets = era5['lon_offsets'][:]
    cos_lat = np.cos(np.deg2rad(center_lat))

    # Build distance grid (km)
    dlon_grid, dlat_grid = np.meshgrid(lon_offsets, lat_offsets)
    dist_km = np.sqrt((dlat_grid * 111.0)**2 + (dlon_grid * 111.0 * cos_lat)**2)
    mask = (dist_km >= inner_km) & (dist_km <= outer_km)

    if mask.sum() < 4:
        raise HTTPException(status_code=400,
                            detail=f"Annulus {inner_km}-{outer_km} km contains too few grid points")

    result = {
        "case_index": case_index,
        "inner_km": inner_km,
        "outer_km": outer_km,
        "n_points": int(mask.sum()),
    }

    # Recompute shear from vector components
    try:
        shear_u = era5['shear_u'][case_index]
        shear_v = era5['shear_v'][case_index]
        su = float(np.nanmean(shear_u[mask]))
        sv = float(np.nanmean(shear_v[mask]))
        result["shear_mag_env"] = round(float(np.sqrt(su**2 + sv**2)), 1)
        shear_dir_math = np.degrees(np.arctan2(sv, su))
        result["shear_dir_env"] = round(float((270 - shear_dir_math) % 360), 0)
    except Exception:
        result["shear_mag_env"] = None
        result["shear_dir_env"] = None

    # Recompute RH mid
    try:
        rh_mid = era5['rh_mid'][case_index]
        result["rh_mid_env"] = round(float(np.nanmean(rh_mid[mask])), 0)
    except Exception:
        result["rh_mid_env"] = None

    # Recompute divergence
    try:
        div200 = era5['div200'][case_index]
        result["div200_env"] = round(float(np.nanmean(div200[mask])), 4)
    except Exception:
        result["div200_env"] = None

    # Recompute SST
    try:
        sst = era5['sst'][case_index]
        result["sst_env"] = round(float(np.nanmean(sst[mask])), 1)
    except Exception:
        result["sst_env"] = None

    # PI, chi_m, and vent_index are precomputed and not recomputable
    # from 2D grids alone (require vertical profiles), so return stored values
    for sname in ['chi_m', 'v_pi', 'vent_index']:
        try:
            val = float(era5[sname][case_index])
            result[sname] = round(val, 3) if not np.isnan(val) else None
        except Exception:
            result[sname] = None

    return result


# ---------------------------------------------------------------------------
# ERA5 sounding at user-specified averaging radius (for Skew-T)
# ---------------------------------------------------------------------------
@app.get("/era5_sounding")
def get_era5_sounding(
    case_index: int = Query(..., ge=0),
    radius_km: float = Query(200, ge=50, le=800, description="Averaging radius (km) for sounding"),
):
    """
    Return ERA5 vertical profiles for Skew-T rendering.

    If 3D ERA5 fields are available in the Zarr store, profiles are recomputed
    as disc-averages within the specified radius.  Otherwise falls back to the
    precomputed 200-600 km annulus profiles.

    Parameters
    ----------
    case_index : int
    radius_km  : float  - disc radius for azimuthal averaging (km)
    """
    era5 = get_era5_dataset()
    if era5 is None:
        raise HTTPException(status_code=503, detail="ERA5 data not available")

    try:
        status = int(era5['status'][case_index])
    except (IndexError, KeyError):
        raise HTTPException(status_code=404, detail=f"Case {case_index} not in ERA5 store")
    if status != 1:
        raise HTTPException(status_code=404, detail=f"No ERA5 data for case {case_index}")

    result = {"case_index": case_index, "radius_km": radius_km}

    # Check if full 3D T/q fields are available for custom radius
    # Support alternate variable names common in ERA5 stores
    _3d_t_names = ['t_3d', 'temperature_3d', 't3d', 'temperature', 'temp_3d']
    _3d_q_names = ['q_3d', 'specific_humidity_3d', 'q3d', 'specific_humidity', 'shum_3d']
    _3d_u_names = ['u_3d', 'u_wind_3d', 'u3d', 'u_component_of_wind']
    _3d_v_names = ['v_3d', 'v_wind_3d', 'v3d', 'v_component_of_wind']

    t_3d_key = next((k for k in _3d_t_names if k in era5), None)
    q_3d_key = next((k for k in _3d_q_names if k in era5), None)
    u_3d_key = next((k for k in _3d_u_names if k in era5), None)
    v_3d_key = next((k for k in _3d_v_names if k in era5), None)

    has_3d = t_3d_key is not None and q_3d_key is not None
    result["has_3d"] = has_3d

    if has_3d:
        try:
            center_lat = float(era5['center_lat'][case_index])
            lat_offsets = era5['lat_offsets'][:]
            lon_offsets = era5['lon_offsets'][:]
            cos_lat = np.cos(np.deg2rad(center_lat))
            dlon_grid, dlat_grid = np.meshgrid(lon_offsets, lat_offsets)
            dist_km = np.sqrt((dlat_grid * 111.0)**2 + (dlon_grid * 111.0 * cos_lat)**2)
            mask_2d = dist_km <= radius_km

            if mask_2d.sum() < 2:
                raise ValueError("Too few points")

            plev = era5['plev'][:].astype(float)
            t_3d = era5[t_3d_key][case_index]   # (nlev, nlat, nlon)
            q_3d = era5[q_3d_key][case_index]

            t_prof = np.array([float(np.nanmean(t_3d[k][mask_2d])) for k in range(len(plev))])
            q_prof = np.array([float(np.nanmean(q_3d[k][mask_2d])) for k in range(len(plev))])
            q_kgkg = q_prof / 1000.0 if np.nanmax(q_prof) > 0.5 else q_prof

            Rd_Cp = 287.04 / 1005.7
            theta = t_prof * (1000.0 / plev) ** Rd_Cp
            Lv, Cpd = 2.501e6, 1005.7
            theta_e = theta * np.exp(Lv * q_kgkg / (Cpd * t_prof))

            # RH: use rh_3d directly if available, else compute from q and T
            rh_3d_key = next((k for k in ['rh_3d', 'relative_humidity_3d'] if k in era5), None)
            if rh_3d_key:
                rh_3d_arr = era5[rh_3d_key][case_index]
                rh_prof = np.array([float(np.nanmean(rh_3d_arr[k][mask_2d])) for k in range(len(plev))])
                rh_prof = np.clip(rh_prof, 0, 100)
            else:
                es = 6.112 * np.exp(17.67 * (t_prof - 273.15) / (t_prof - 273.15 + 243.5))
                ws = 0.622 * es / (plev - es)
                rh_prof = np.clip(q_kgkg / ws * 100.0, 0, 100)

            profiles = {
                "plev": plev.tolist(),
                "t": [round(float(v), 2) if not np.isnan(v) else None for v in t_prof],
                "q": [round(float(v), 6) if not np.isnan(v) else None for v in q_prof],
                "rh": [round(float(v), 1) if not np.isnan(v) else None for v in rh_prof],
                "theta": [round(float(v), 1) if not np.isnan(v) else None for v in theta],
                "theta_e": [round(float(v), 1) if not np.isnan(v) else None for v in theta_e],
            }

            # Also average u/v 3D winds if available
            has_uv_3d = u_3d_key is not None and v_3d_key is not None
            if has_uv_3d:
                u_3d = era5[u_3d_key][case_index]
                v_3d = era5[v_3d_key][case_index]
                u_prof = np.array([float(np.nanmean(u_3d[k][mask_2d])) for k in range(len(plev))])
                v_prof = np.array([float(np.nanmean(v_3d[k][mask_2d])) for k in range(len(plev))])
                profiles["u"] = [round(float(v), 2) if not np.isnan(v) else None for v in u_prof]
                profiles["v"] = [round(float(v), 2) if not np.isnan(v) else None for v in v_prof]
            else:
                # Fall back to precomputed u/v profiles
                try:
                    profiles["u"] = [round(float(v), 2) if not np.isnan(v) else None for v in era5['u_profile'][case_index]]
                    profiles["v"] = [round(float(v), 2) if not np.isnan(v) else None for v in era5['v_profile'][case_index]]
                except Exception:
                    pass

            result["source"] = "recomputed"
            result["profiles"] = profiles
            return result
        except Exception:
            pass  # Fall through to precomputed profiles

    # Fallback: return precomputed annular-mean profiles
    try:
        plev = era5['plev'][:].astype(float)
        t_arr = era5['t_profile'][case_index].astype(float)
        q_arr = era5['q_profile'][case_index].astype(float)
        q_kgkg = q_arr / 1000.0 if np.nanmax(q_arr) > 0.5 else q_arr

        Rd_Cp = 287.04 / 1005.7
        theta = t_arr * (1000.0 / plev) ** Rd_Cp
        Lv, Cpd = 2.501e6, 1005.7
        theta_e = theta * np.exp(Lv * q_kgkg / (Cpd * t_arr))

        result["source"] = "precomputed"
        result["precomputed_domain"] = "200-600 km annulus"
        result["profiles"] = {
            "plev": plev.tolist(),
            "t": [round(float(v), 2) if not np.isnan(v) else None for v in t_arr],
            "q": [round(float(v), 6) if not np.isnan(v) else None for v in q_arr],
            "rh": [round(float(v), 1) if not np.isnan(v) else None for v in era5['rh_profile'][case_index]],
            "theta": [round(float(v), 1) if not np.isnan(v) else None for v in theta],
            "theta_e": [round(float(v), 1) if not np.isnan(v) else None for v in theta_e],
        }
        # Include precomputed u/v profiles for wind barbs
        try:
            result["profiles"]["u"] = [round(float(v), 2) if not np.isnan(v) else None for v in era5['u_profile'][case_index]]
            result["profiles"]["v"] = [round(float(v), 2) if not np.isnan(v) else None for v in era5['v_profile'][case_index]]
        except Exception:
            pass
    except Exception:
        result["profiles"] = None

    return result


def _build_case_meta(case_index, ds=None, local_idx=None, data_type="swath"):
    """Build case_meta dict with SDDC included."""
    cache = _merge_metadata_cache if data_type == "merge" else _metadata_cache
    case_meta = cache.get(case_index, {"case_index": case_index})
    meta = {
        "storm_name": case_meta.get("storm_name", ""),
        "datetime": case_meta.get("datetime", ""),
        "vmax_kt": case_meta.get("vmax_kt"),
        "rmw_km": case_meta.get("rmw_km"),
        "mission_id": case_meta.get("mission_id", ""),
    }
    if ds is not None and local_idx is not None:
        sddc = _get_sddc(ds, local_idx)
        shdc = _get_shdc(ds, local_idx)
        meta["sddc"] = sddc if sddc is not None else 9999
        meta["shdc"] = shdc if shdc is not None else 9999
    return meta


def _build_case_list(processed_indices: list[int], data_type: str = "swath") -> list[dict]:
    """Build a compact list of case metadata for composite reproducibility."""
    cache = _merge_metadata_cache if data_type == "merge" else _metadata_cache
    case_list = []
    for ci in processed_indices:
        meta = cache.get(ci, {})
        case_list.append({
            "case_index": ci,
            "storm_name": meta.get("storm_name", ""),
            "datetime": meta.get("datetime", ""),
            "vmax_kt": meta.get("vmax_kt"),
        })
    return case_list


def _compute_azimuthal_mean(vol, x_coords, y_coords, height_vals, h_axis,
                            max_radius, dr, coverage_min, rmw=None):
    """
    Compute azimuthal mean from a 3D Cartesian volume.

    If rmw is provided, radii are normalised by RMW (output bins in R/RMW).
    Otherwise bins are in km.

    Returns:
        az_mean: 2D array (n_heights × n_rbins) — NaN where coverage < threshold
        coverage: 2D array (n_heights × n_rbins) — fraction of valid data
        r_bins: 1D array of radius bin centres (km or R/RMW)
    """
    # Build 2D radius grid from coordinate arrays
    xx, yy = np.meshgrid(x_coords, y_coords)
    rr = np.sqrt(xx**2 + yy**2)

    # If RMW-normalising, convert radius grid to R/RMW
    if rmw is not None and rmw > 0:
        rr = rr / rmw

    # Define radius bins
    r_edges = np.arange(0, max_radius + dr, dr)
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2.0
    n_rbins = len(r_centers)
    n_heights = len(height_vals)

    # Pre-compute bin membership for each (y, x) grid point
    bin_idx = np.digitize(rr, r_edges) - 1  # shape: (ny, nx), values 0..n_rbins-1

    az_mean  = np.full((n_heights, n_rbins), np.nan)
    coverage = np.full((n_heights, n_rbins), 0.0)

    for h in range(n_heights):
        # Extract 2D slice at this height
        if h_axis == 0:
            slab = vol[h, :, :]
        elif h_axis == 2:
            slab = vol[:, :, h]
        else:
            slab = vol[:, h, :]

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


@app.get("/azimuthal_mean")
def azimuthal_mean(
    case_index:    int   = Query(...,   ge=0,            description="0-based case index"),
    variable:      str   = Query(DEFAULT_VARIABLE,       description="Variable key — see /variables"),
    data_type:     str   = Query("swath",                description="'swath' or 'merge'"),
    max_radius_km: float = Query(200.0, ge=10, le=500,   description="Maximum radius in km"),
    dr_km:         float = Query(2.0,   ge=0.5, le=20,   description="Radial bin width in km"),
    coverage_min:  float = Query(0.5,   ge=0.0, le=1.0,  description="Min fraction of valid data per bin"),
    overlay:       str   = Query("",                      description="Optional overlay variable key"),
):
    """Return azimuthal-mean radius-height cross-section as JSON."""
    if variable not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"Unknown variable '{variable}'. See /variables.")
    if data_type not in ("swath", "merge"):
        raise HTTPException(status_code=400, detail="data_type must be 'swath' or 'merge'")
    if overlay and overlay not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"Unknown overlay variable '{overlay}'.")

    try:
        ds, local_idx = resolve_case(case_index, data_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open dataset: {e}")

    display_name, varname, cmap, units, vmin, vmax = VARIABLES[variable]
    height_vals = ds["height"].values

    # Get ref variable for grid detection
    if variable in DERIVED_VARIABLES:
        ref_varname = DERIVED_VARIABLES[variable][0]
    else:
        if varname not in ds:
            raise HTTPException(status_code=400, detail=f"'{varname}' not in dataset.")
        ref_varname = varname

    # Determine spatial grid
    var_dims = set(ds[ref_varname].dims)
    if "eastward_distance" in var_dims and "northward_distance" in var_dims:
        x_coords = ds["eastward_distance"].values
        y_coords = ds["northward_distance"].values
    elif "latitude" in var_dims and "longitude" in var_dims:
        nx = ds.sizes["longitude"]
        ny = ds.sizes["latitude"]
        x_coords = np.linspace(-(nx - 1), (nx - 1), nx)
        y_coords = np.linspace(-(ny - 1), (ny - 1), ny)
    else:
        raise HTTPException(status_code=500, detail="Cannot determine spatial grid")

    # Determine height axis position
    dim_list = [d for d in ds[ref_varname].dims if d != "num_cases"]
    if "height" not in dim_list:
        raise HTTPException(status_code=500, detail="Cannot determine height axis")
    h_axis = dim_list.index("height")

    # Extract 3D volume and compute azimuthal mean
    vol, _ = _extract_3d_volume(ds, local_idx, variable)
    az_mean, coverage, r_centers = _compute_azimuthal_mean(
        vol, x_coords, y_coords, height_vals, h_axis,
        max_radius_km, dr_km, coverage_min
    )

    case_meta = _build_case_meta(case_index, ds, local_idx, data_type)

    result = {
        "azimuthal_mean": _clean_2d(az_mean),
        "coverage": _clean_2d(coverage),
        "radius_km": [round(float(r), 2) for r in r_centers],
        "height_km": [round(float(h), 2) for h in height_vals],
        "coverage_min": coverage_min,
        "variable": {
            "key": variable,
            "display_name": display_name,
            "units": units,
            "vmin": vmin,
            "vmax": vmax,
            "colorscale": _cmap_to_plotly(cmap),
        },
        "case_meta": case_meta,
    }

    # Optional overlay azimuthal mean
    if overlay:
        ov_display, _, ov_cmap, ov_units, ov_vmin, ov_vmax = VARIABLES[overlay]
        try:
            ov_vol, _ = _extract_3d_volume(ds, local_idx, overlay)
            ov_az, _, _ = _compute_azimuthal_mean(
                ov_vol, x_coords, y_coords, height_vals, h_axis,
                max_radius_km, dr_km, coverage_min
            )
            result["overlay"] = {
                "azimuthal_mean": _clean_2d(ov_az),
                "key": overlay,
                "display_name": ov_display,
                "units": ov_units,
                "vmin": ov_vmin,
                "vmax": ov_vmax,
            }
        except Exception:
            pass

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Shear-Relative Quadrant Means
# ---------------------------------------------------------------------------
QUADRANT_DEFS = {
    # key: (start_deg, end_deg) in shear-relative met heading
    # θ_sr = (θ_met - SDDC) mod 360, where 0° = downshear, CW positive
    # "left" = counterclockwise from downshear (met convention)
    "DSR": (0,   90),    # downshear-right
    "USR": (90,  180),   # upshear-right
    "USL": (180, 270),   # upshear-left
    "DSL": (270, 360),   # downshear-left
}


def _compute_quadrant_means(vol, x_coords, y_coords, height_vals, h_axis,
                            sddc, max_radius, dr, coverage_min, rmw=None):
    """
    Compute shear-relative quadrant means from a 3D Cartesian volume.

    If rmw is provided, radii are normalised by RMW (output bins in R/RMW).

    Parameters
    ----------
    vol : 3D array — full volume with axes depending on h_axis
    sddc : float — deep-layer shear heading (met deg, 0=N, 90=E, CW)
    rmw : float or None — if provided, normalise radius by RMW

    Returns
    -------
    quad_means : dict[str, 2D array] — {DSL, DSR, USL, USR} each (n_heights × n_rbins)
    r_centers  : 1D array of radial bin centres (km or R/RMW)
    """
    # Build 2D radius and azimuth grids
    xx, yy = np.meshgrid(x_coords, y_coords)
    rr = np.sqrt(xx**2 + yy**2)

    # If RMW-normalising, convert radius grid to R/RMW
    if rmw is not None and rmw > 0:
        rr = rr / rmw

    # Math angle → meteorological heading
    azimuth_math_deg = np.degrees(np.arctan2(yy, xx))        # -180..180, CCW from +x
    azimuth_met = (90.0 - azimuth_math_deg) % 360.0          # met heading, CW from N

    # Shear-relative azimuth: 0° = downshear, 90° = right-of-shear (CW)
    shear_rel_az = (azimuth_met - sddc) % 360.0

    # Radial bins
    r_edges = np.arange(0, max_radius + dr, dr)
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2.0
    n_rbins = len(r_centers)
    n_heights = len(height_vals)

    bin_idx = np.digitize(rr, r_edges) - 1   # (ny, nx)

    # Pre-compute quadrant masks (ny, nx) for each quadrant
    q_masks = {}
    for qname, (az_start, az_end) in QUADRANT_DEFS.items():
        q_masks[qname] = (shear_rel_az >= az_start) & (shear_rel_az < az_end)

    quad_means = {q: np.full((n_heights, n_rbins), np.nan) for q in QUADRANT_DEFS}

    for h in range(n_heights):
        # Extract 2D slab at this height
        if h_axis == 0:
            slab = vol[h, :, :]
        elif h_axis == 2:
            slab = vol[:, :, h]
        else:
            slab = vol[:, h, :]

        valid = ~np.isnan(slab)

        for r in range(n_rbins):
            r_mask = (bin_idx == r)
            for qname, q_mask in q_masks.items():
                mask = r_mask & q_mask
                n_total = np.count_nonzero(mask)
                if n_total == 0:
                    continue
                in_bin = mask & valid
                n_valid = np.count_nonzero(in_bin)
                frac = n_valid / n_total
                if frac >= coverage_min:
                    quad_means[qname][h, r] = float(np.nanmean(slab[in_bin]))

    return quad_means, r_centers


@app.get("/quadrant_mean")
def quadrant_mean(
    case_index:    int   = Query(...,   ge=0,            description="0-based case index"),
    variable:      str   = Query(DEFAULT_VARIABLE,       description="Variable key — see /variables"),
    data_type:     str   = Query("swath",                description="'swath' or 'merge'"),
    max_radius_km: float = Query(200.0, ge=10, le=500,   description="Maximum radius in km"),
    dr_km:         float = Query(2.0,   ge=0.5, le=20,   description="Radial bin width in km"),
    coverage_min:  float = Query(0.5,   ge=0.0, le=1.0,  description="Min fraction of valid data per bin"),
    overlay:       str   = Query("",                      description="Optional overlay variable key"),
):
    """Return shear-relative quadrant-mean radius-height sections as JSON."""
    if variable not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"Unknown variable '{variable}'. See /variables.")
    if data_type not in ("swath", "merge"):
        raise HTTPException(status_code=400, detail="data_type must be 'swath' or 'merge'")
    if overlay and overlay not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"Unknown overlay variable '{overlay}'.")

    try:
        ds, local_idx = resolve_case(case_index, data_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open dataset: {e}")

    # Look up SDDC — required for this endpoint
    sddc = _get_sddc(ds, local_idx)
    if sddc is None:
        raise HTTPException(
            status_code=400,
            detail="Shear direction (SDDC) not available for this case — cannot compute shear-relative quadrants."
        )

    display_name, varname, cmap, units, vmin, vmax = VARIABLES[variable]
    height_vals = ds["height"].values

    # Ref variable for grid detection
    if variable in DERIVED_VARIABLES:
        ref_varname = DERIVED_VARIABLES[variable][0]
    else:
        if varname not in ds:
            raise HTTPException(status_code=400, detail=f"'{varname}' not in dataset.")
        ref_varname = varname

    # Determine spatial grid
    var_dims = set(ds[ref_varname].dims)
    if "eastward_distance" in var_dims and "northward_distance" in var_dims:
        x_coords = ds["eastward_distance"].values
        y_coords = ds["northward_distance"].values
    elif "latitude" in var_dims and "longitude" in var_dims:
        nx = ds.sizes["longitude"]
        ny = ds.sizes["latitude"]
        x_coords = np.linspace(-(nx - 1), (nx - 1), nx)
        y_coords = np.linspace(-(ny - 1), (ny - 1), ny)
    else:
        raise HTTPException(status_code=500, detail="Cannot determine spatial grid")

    # Determine height axis position
    dim_list = [d for d in ds[ref_varname].dims if d != "num_cases"]
    if "height" not in dim_list:
        raise HTTPException(status_code=500, detail="Cannot determine height axis")
    h_axis = dim_list.index("height")

    # Extract 3D volume and compute quadrant means
    vol, _ = _extract_3d_volume(ds, local_idx, variable)
    quad_means, r_centers = _compute_quadrant_means(
        vol, x_coords, y_coords, height_vals, h_axis,
        sddc, max_radius_km, dr_km, coverage_min
    )

    case_meta = _build_case_meta(case_index, ds, local_idx, data_type)

    result = {
        "quadrant_means": {
            q: {"data": _clean_2d(quad_means[q])} for q in QUADRANT_DEFS
        },
        "radius_km": [round(float(r), 2) for r in r_centers],
        "height_km": [round(float(h), 2) for h in height_vals],
        "coverage_min": coverage_min,
        "variable": {
            "key": variable,
            "display_name": display_name,
            "units": units,
            "vmin": vmin,
            "vmax": vmax,
            "colorscale": _cmap_to_plotly(cmap),
        },
        "case_meta": case_meta,
    }

    # Optional overlay quadrant means
    if overlay:
        ov_display, _, ov_cmap, ov_units, ov_vmin, ov_vmax = VARIABLES[overlay]
        try:
            ov_vol, _ = _extract_3d_volume(ds, local_idx, overlay)
            ov_quads, _ = _compute_quadrant_means(
                ov_vol, x_coords, y_coords, height_vals, h_axis,
                sddc, max_radius_km, dr_km, coverage_min
            )
            result["overlay"] = {
                "quadrant_means": {
                    q: {"data": _clean_2d(ov_quads[q])} for q in QUADRANT_DEFS
                },
                "key": overlay,
                "display_name": ov_display,
                "units": ov_units,
                "vmin": ov_vmin,
                "vmax": ov_vmax,
            }
        except Exception:
            pass

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Composite endpoints
# ---------------------------------------------------------------------------
_COMPOSITE_MAX_CASES = 150  # safety cap — keep low for 512 MB free-tier servers

# Common query parameters for composite filters
def _composite_filter_params(
    min_intensity:  float = Query(0,    ge=0,   le=200,  description="Min Vmax (kt)"),
    max_intensity:  float = Query(200,  ge=0,   le=200,  description="Max Vmax (kt)"),
    min_vmax_change:float = Query(-100, ge=-100,le=85,   description="Min 24-h Vmax change (kt)"),
    max_vmax_change:float = Query(85,   ge=-100,le=85,   description="Max 24-h Vmax change (kt)"),
    min_tilt:       float = Query(0,    ge=0,   le=200,  description="Min tilt magnitude (km)"),
    max_tilt:       float = Query(200,  ge=0,   le=200,  description="Max tilt magnitude (km)"),
    min_year:       int   = Query(1997, ge=1997,le=2024,  description="Min year"),
    max_year:       int   = Query(2024, ge=1997,le=2024,  description="Max year"),
    min_shear_mag:  float = Query(0,    ge=0,   le=100,  description="Min shear magnitude (kt)"),
    max_shear_mag:  float = Query(100,  ge=0,   le=100,  description="Max shear magnitude (kt)"),
    min_shear_dir:  float = Query(0,    ge=0,   le=360,  description="Min shear direction (deg)"),
    max_shear_dir:  float = Query(360,  ge=0,   le=360,  description="Max shear direction (deg)"),
):
    return dict(
        min_intensity=min_intensity, max_intensity=max_intensity,
        min_vmax_change=min_vmax_change, max_vmax_change=max_vmax_change,
        min_tilt=min_tilt, max_tilt=max_tilt,
        min_year=min_year, max_year=max_year,
        min_shear_mag=min_shear_mag, max_shear_mag=max_shear_mag,
        min_shear_dir=min_shear_dir, max_shear_dir=max_shear_dir,
    )


def _resolve_grid_and_haxis(ds, ref_varname):
    """Determine spatial grid and height axis from a dataset + variable."""
    var_dims = set(ds[ref_varname].dims)
    if "eastward_distance" in var_dims and "northward_distance" in var_dims:
        x_coords = ds["eastward_distance"].values
        y_coords = ds["northward_distance"].values
    elif "latitude" in var_dims and "longitude" in var_dims:
        nx = ds.sizes["longitude"]
        ny = ds.sizes["latitude"]
        x_coords = np.linspace(-(nx - 1), (nx - 1), nx)
        y_coords = np.linspace(-(ny - 1), (ny - 1), ny)
    else:
        raise ValueError("Cannot determine spatial grid")
    dim_list = [d for d in ds[ref_varname].dims if d != "num_cases"]
    if "height" not in dim_list:
        raise ValueError("Cannot determine height axis")
    h_axis = dim_list.index("height")
    return x_coords, y_coords, h_axis


# ---------------------------------------------------------------------------
# Plan-view composite helpers
# ---------------------------------------------------------------------------

def _rotate_2d_grid(data, angle_deg):
    """
    Rotate a 2D array counter-clockwise by *angle_deg* degrees.

    Uses bilinear interpolation; out-of-bounds filled with NaN.
    Returns an array of the same shape (no reshape).
    """
    return _ndimage.rotate(data, angle_deg, order=1, cval=np.nan, reshape=False)


def _regrid_to_rmw_normalized(data_2d, x_phys, y_phys, rmw,
                               max_r_rmw=5.0, dr_rmw=0.1):
    """
    Re-grid a physical-coordinate 2D slice onto an RMW-normalised Cartesian
    grid spanning ±max_r_rmw with spacing dr_rmw.

    Parameters
    ----------
    data_2d : 2D ndarray (ny, nx)
    x_phys, y_phys : 1D arrays — physical distance coordinates (km)
    rmw : float — radius of maximum wind (km)
    max_r_rmw, dr_rmw : float — target grid extent & resolution in R/RMW

    Returns
    -------
    data_norm : 2D ndarray on the normalised grid
    x_norm, y_norm : 1D coordinate arrays in R/RMW
    """
    # Build target coordinate arrays (R/RMW)
    half_n = int(round(max_r_rmw / dr_rmw))
    x_norm = np.linspace(-max_r_rmw, max_r_rmw, 2 * half_n + 1)
    y_norm = np.linspace(-max_r_rmw, max_r_rmw, 2 * half_n + 1)

    # Target grid in physical space
    xx_norm, yy_norm = np.meshgrid(x_norm, y_norm)
    xx_phys_target = xx_norm * rmw
    yy_phys_target = yy_norm * rmw

    # Build interpolator on the original physical grid
    interp = RegularGridInterpolator(
        (y_phys, x_phys), data_2d,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    pts = np.column_stack([yy_phys_target.ravel(), xx_phys_target.ravel()])
    data_norm = interp(pts).reshape(len(y_norm), len(x_norm))
    return data_norm, x_norm, y_norm


# Number of parallel threads for composite S3 reads.
# 2 keeps concurrent memory usage manageable on the 512 MB Render free tier
# (~2 concurrent 3D volumes × ~10 MB ≈ 20 MB peak).
_COMPOSITE_WORKERS = 2

# Batch size for composite processing — process this many cases at a time,
# then GC between batches to keep peak memory bounded.
_COMPOSITE_BATCH_SIZE = 25


def _process_composites_batched(worker_fn, work_items, accumulate_fn):
    """
    Process composite cases in batches to bound peak memory on low-RAM servers.

    Parameters
    ----------
    worker_fn : callable
        Function to submit to thread pool. Receives unpacked args from each work item.
    work_items : list of tuples
        Each tuple is unpacked as args to worker_fn.
    accumulate_fn : callable(result)
        Called with each non-None worker result in the main thread.
        Should accumulate into the caller's arrays.

    Returns nothing — accumulation is done via side-effect through accumulate_fn.
    """
    total = len(work_items)
    for batch_start in range(0, total, _COMPOSITE_BATCH_SIZE):
        batch = work_items[batch_start:batch_start + _COMPOSITE_BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=_COMPOSITE_WORKERS) as pool:
            futures = {
                pool.submit(worker_fn, *item): item[0]
                for item in batch
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    accumulate_fn(result)
                del result
            del futures
        # Free thread pool + intermediate memory between batches
        gc.collect()


def _process_one_case_azimuthal(case_idx, rmw, variable, overlay, data_type,
                                 x_coords, y_coords, height_km, h_axis,
                                 max_r_rmw, dr_rmw, coverage_min):
    """
    Process a single case for composite azimuthal mean.
    Runs in a thread pool — all args passed explicitly, no shared mutable state.
    Returns (case_idx, az_mean, r_centers, ov_az_or_None) or None on failure.
    """
    try:
        ds, local_idx = resolve_case(case_idx, data_type)
        vol, _ = _extract_3d_volume(ds, local_idx, variable)
        az_mean, _, rc = _compute_azimuthal_mean(
            vol, x_coords, y_coords, height_km, h_axis,
            max_r_rmw, dr_rmw, coverage_min, rmw=rmw
        )

        ov_az = None
        if overlay:
            try:
                ov_vol, _ = _extract_3d_volume(ds, local_idx, overlay)
                ov_az, _, _ = _compute_azimuthal_mean(
                    ov_vol, x_coords, y_coords, height_km, h_axis,
                    max_r_rmw, dr_rmw, coverage_min, rmw=rmw
                )
            except Exception:
                pass

        return (case_idx, az_mean, rc, ov_az)
    except Exception as e:
        print(f"Composite az: skipping case {case_idx}: {e}")
        return None


def _process_one_case_quadrant(case_idx, sddc, rmw, variable, overlay, data_type,
                                x_coords, y_coords, height_km, h_axis,
                                max_r_rmw, dr_rmw, coverage_min):
    """
    Process a single case for composite quadrant mean.
    Runs in a thread pool — all args passed explicitly, no shared mutable state.
    Returns (case_idx, quad_means, r_centers, ov_quad_means_or_None) or None on failure.
    """
    try:
        ds, local_idx = resolve_case(case_idx, data_type)
        vol, _ = _extract_3d_volume(ds, local_idx, variable)
        quad_means, rc = _compute_quadrant_means(
            vol, x_coords, y_coords, height_km, h_axis,
            sddc, max_r_rmw, dr_rmw, coverage_min, rmw=rmw
        )

        ov_quads = None
        if overlay:
            try:
                ov_vol, _ = _extract_3d_volume(ds, local_idx, overlay)
                ov_quads, _ = _compute_quadrant_means(
                    ov_vol, x_coords, y_coords, height_km, h_axis,
                    sddc, max_r_rmw, dr_rmw, coverage_min, rmw=rmw
                )
            except Exception:
                pass

        return (case_idx, quad_means, rc, ov_quads)
    except Exception as e:
        print(f"Composite quad: skipping case {case_idx}: {e}")
        return None


def _process_one_case_plan_view(case_idx, rmw, sddc, variable, overlay,
                                 data_type, x_coords, y_coords, z_idx,
                                 normalize_rmw, max_r_rmw, dr_rmw,
                                 shear_relative):
    """
    Process a single case for plan-view composite.

    Returns (case_idx, plan_2d, x_grid, y_grid, ov_plan_2d) or None.
    """
    try:
        ds, local_idx = resolve_case(case_idx, data_type)
        data_2d, _ = _extract_2d_slice(ds, local_idx, variable, z_idx)

        # Shear-relative rotation: rotate so shear vector points right (+x)
        if shear_relative and sddc is not None:
            # ndimage.rotate rotates CCW by the given angle.
            # We need to map SDDC (met heading) to +x (math 0°).
            # Math angle of shear = 90 - SDDC.
            # To rotate shear to 0° math, rotate CCW by -(90 - SDDC) = SDDC - 90.
            # BUT ndimage.rotate(angle) rotates CCW by `angle`, so pass 90 - SDDC
            # to bring shear to the right.
            rotation_angle = 90.0 - float(sddc)
            data_2d = _rotate_2d_grid(data_2d, rotation_angle)

        # RMW normalisation
        if normalize_rmw and rmw is not None and rmw > 0:
            data_2d, x_grid, y_grid = _regrid_to_rmw_normalized(
                data_2d, x_coords, y_coords, rmw, max_r_rmw, dr_rmw,
            )
        else:
            x_grid, y_grid = x_coords, y_coords

        # Overlay
        ov_2d = None
        if overlay:
            try:
                ov_data, _ = _extract_2d_slice(ds, local_idx, overlay, z_idx)
                if shear_relative and sddc is not None:
                    ov_data = _rotate_2d_grid(ov_data, rotation_angle)
                if normalize_rmw and rmw is not None and rmw > 0:
                    ov_data, _, _ = _regrid_to_rmw_normalized(
                        ov_data, x_coords, y_coords, rmw, max_r_rmw, dr_rmw,
                    )
                ov_2d = ov_data
            except Exception:
                pass

        return (case_idx, data_2d, x_grid, y_grid, ov_2d)
    except Exception as e:
        print(f"Composite plan_view: skipping case {case_idx}: {e}")
        return None


@app.get("/composite/count")
def composite_count(
    data_type:      str   = Query("swath",                description="'swath' or 'merge'"),
    min_intensity:  float = Query(0,    ge=0,   le=200),
    max_intensity:  float = Query(200,  ge=0,   le=200),
    min_vmax_change:float = Query(-100, ge=-100,le=85),
    max_vmax_change:float = Query(85,   ge=-100,le=85),
    min_tilt:       float = Query(0,    ge=0,   le=200),
    max_tilt:       float = Query(200,  ge=0,   le=200),
    min_year:       int   = Query(1997, ge=1997,le=2024),
    max_year:       int   = Query(2024, ge=1997,le=2024),
    min_shear_mag:  float = Query(0,    ge=0,   le=100),
    max_shear_mag:  float = Query(100,  ge=0,   le=100),
    min_shear_dir:  float = Query(0,    ge=0,   le=360),
    max_shear_dir:  float = Query(360,  ge=0,   le=360),
):
    """Quick endpoint to get case count for given filter criteria (no data loading)."""
    cases = _filter_cases_for_composite(
        min_intensity, max_intensity, min_vmax_change, max_vmax_change,
        min_tilt, max_tilt, min_year, max_year,
        min_shear_mag, max_shear_mag, min_shear_dir, max_shear_dir,
        data_type=data_type,
    )
    return {
        "count": len(cases),
        "max_cases": _COMPOSITE_MAX_CASES,
        "capped": len(cases) > _COMPOSITE_MAX_CASES,
        "case_indices": cases[:20],
    }


@app.get("/composite/azimuthal_mean")
def composite_azimuthal_mean(
    variable:      str   = Query(DEFAULT_VARIABLE,       description="Variable key"),
    overlay:       str   = Query("",                     description="Optional overlay variable key"),
    data_type:     str   = Query("swath",                description="'swath' or 'merge'"),
    max_r_rmw:     float = Query(8.0,   ge=1, le=20,    description="Max radius in R/RMW"),
    dr_rmw:        float = Query(0.25,  ge=0.1, le=2,   description="Radial bin width in R/RMW"),
    coverage_min:  float = Query(0.25,  ge=0.0, le=1.0),
    min_intensity:  float = Query(0),    max_intensity:  float = Query(200),
    min_vmax_change:float = Query(-100), max_vmax_change:float = Query(85),
    min_tilt:       float = Query(0),    max_tilt:       float = Query(200),
    min_year:       int   = Query(1997), max_year:       int   = Query(2024),
    min_shear_mag:  float = Query(0),    max_shear_mag:  float = Query(100),
    min_shear_dir:  float = Query(0),    max_shear_dir:  float = Query(360),
):
    """Compute RMW-normalised composite azimuthal mean across matching cases."""
    if variable not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"Unknown variable '{variable}'.")
    if overlay and overlay not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"Unknown overlay variable '{overlay}'.")
    if data_type not in ("swath", "merge"):
        raise HTTPException(status_code=400, detail="data_type must be 'swath' or 'merge'")

    meta_cache = _merge_metadata_cache if data_type == "merge" else _metadata_cache
    matching = _filter_cases_for_composite(
        min_intensity, max_intensity, min_vmax_change, max_vmax_change,
        min_tilt, max_tilt, min_year, max_year,
        min_shear_mag, max_shear_mag, min_shear_dir, max_shear_dir,
        data_type=data_type,
    )
    if not matching:
        raise HTTPException(status_code=400, detail="No cases match the specified criteria.")
    if len(matching) > _COMPOSITE_MAX_CASES:
        matching = matching[:_COMPOSITE_MAX_CASES]

    # Filter to cases with valid RMW
    cases_with_rmw = []
    for ci in matching:
        rmw = meta_cache.get(ci, {}).get("rmw_km")
        if rmw is not None and not np.isnan(float(rmw)) and float(rmw) > 0:
            cases_with_rmw.append((ci, float(rmw)))
    if not cases_with_rmw:
        raise HTTPException(status_code=400, detail="No matching cases have valid RMW data.")

    display_name, varname, cmap, units, vmin, vmax = VARIABLES[variable]
    ref_varname = DERIVED_VARIABLES[variable][0] if variable in DERIVED_VARIABLES else varname

    # Pre-fetch grid info from the first valid case so all workers share it
    first_ds, _ = resolve_case(cases_with_rmw[0][0], data_type)
    height_km = first_ds["height"].values
    x_coords, y_coords, h_axis = _resolve_grid_and_haxis(first_ds, ref_varname)

    accum_sum = None
    accum_count = None
    ov_accum_sum = None
    ov_accum_count = None
    r_centers = None
    n_processed = 0
    processed_indices = []

    # Process cases in batches — keeps peak memory bounded on 512 MB servers.
    work_items = [
        (case_idx, rmw, variable, overlay, data_type,
         x_coords, y_coords, height_km, h_axis,
         max_r_rmw, dr_rmw, coverage_min)
        for case_idx, rmw in cases_with_rmw
    ]

    def _accum_az(result):
        nonlocal accum_sum, accum_count, ov_accum_sum, ov_accum_count
        nonlocal r_centers, n_processed
        case_idx, az_mean, rc, ov_az = result
        if accum_sum is None:
            r_centers = rc
            accum_sum = np.zeros_like(az_mean)
            accum_count = np.zeros_like(az_mean)
        valid = ~np.isnan(az_mean)
        accum_sum[valid] += az_mean[valid]
        accum_count[valid] += 1
        if ov_az is not None:
            if ov_accum_sum is None:
                ov_accum_sum = np.zeros_like(ov_az)
                ov_accum_count = np.zeros_like(ov_az)
            ov_valid = ~np.isnan(ov_az)
            ov_accum_sum[ov_valid] += ov_az[ov_valid]
            ov_accum_count[ov_valid] += 1
        n_processed += 1
        processed_indices.append(case_idx)

    _process_composites_batched(_process_one_case_azimuthal, work_items, _accum_az)

    if n_processed == 0:
        raise HTTPException(status_code=500, detail="Could not process any matching cases.")

    # Require each grid cell to have data from at least 33% of processed cases
    # (or 3, whichever is greater) to suppress noisy bins with sparse sampling
    min_cases = max(3, int(np.ceil(0.33 * n_processed)))
    composite = np.where(accum_count >= min_cases, accum_sum / accum_count, np.nan)

    result = {
        "azimuthal_mean": _clean_2d(composite),
        "radius_rrmw": [round(float(r), 3) for r in r_centers],
        "height_km": [round(float(h), 2) for h in height_km],
        "normalized": True,
        "coverage_min": coverage_min,
        "min_cases_per_bin": min_cases,
        "n_cases": n_processed,
        "n_matched": len(matching),
        "n_with_rmw": len(cases_with_rmw),
        "case_list": _build_case_list(processed_indices, data_type),
        "variable": {
            "key": variable,
            "display_name": display_name,
            "units": units,
            "vmin": vmin,
            "vmax": vmax,
            "colorscale": _cmap_to_plotly(cmap),
        },
        "filters": {
            "intensity": [min_intensity, max_intensity],
            "vmax_change": [min_vmax_change, max_vmax_change],
            "tilt": [min_tilt, max_tilt],
            "year": [min_year, max_year],
            "shear_mag": [min_shear_mag, max_shear_mag],
            "shear_dir": [min_shear_dir, max_shear_dir],
        },
    }

    # Add overlay if computed
    if overlay and ov_accum_sum is not None:
        ov_composite = np.where(ov_accum_count >= min_cases, ov_accum_sum / ov_accum_count, np.nan)
        ov_display, _, ov_cmap, ov_units, ov_vmin, ov_vmax = VARIABLES[overlay]
        clean_ov = _clean_2d(ov_composite)
        flat = [v for row in clean_ov for v in row if v is not None]
        result["overlay"] = {
            "display_name": ov_display,
            "key": overlay,
            "units": ov_units,
            "azimuthal_mean": clean_ov,
            "vmin": min(flat) if flat else ov_vmin,
            "vmax": max(flat) if flat else ov_vmax,
        }

    return JSONResponse(result)


@app.get("/composite/quadrant_mean")
def composite_quadrant_mean(
    variable:      str   = Query(DEFAULT_VARIABLE,       description="Variable key"),
    overlay:       str   = Query("",                     description="Optional overlay variable key"),
    data_type:     str   = Query("swath",                description="'swath' or 'merge'"),
    max_r_rmw:     float = Query(8.0,   ge=1, le=20,    description="Max radius in R/RMW"),
    dr_rmw:        float = Query(0.25,  ge=0.1, le=2,   description="Radial bin width in R/RMW"),
    coverage_min:  float = Query(0.25,  ge=0.0, le=1.0),
    min_intensity:  float = Query(0),    max_intensity:  float = Query(200),
    min_vmax_change:float = Query(-100), max_vmax_change:float = Query(85),
    min_tilt:       float = Query(0),    max_tilt:       float = Query(200),
    min_year:       int   = Query(1997), max_year:       int   = Query(2024),
    min_shear_mag:  float = Query(0),    max_shear_mag:  float = Query(100),
    min_shear_dir:  float = Query(0),    max_shear_dir:  float = Query(360),
):
    """Compute RMW-normalised composite shear-relative quadrant means."""
    if variable not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"Unknown variable '{variable}'.")
    if overlay and overlay not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"Unknown overlay variable '{overlay}'.")
    if data_type not in ("swath", "merge"):
        raise HTTPException(status_code=400, detail="data_type must be 'swath' or 'merge'")

    meta_cache = _merge_metadata_cache if data_type == "merge" else _metadata_cache
    matching = _filter_cases_for_composite(
        min_intensity, max_intensity, min_vmax_change, max_vmax_change,
        min_tilt, max_tilt, min_year, max_year,
        min_shear_mag, max_shear_mag, min_shear_dir, max_shear_dir,
        data_type=data_type,
    )
    if not matching:
        raise HTTPException(status_code=400, detail="No cases match the specified criteria.")
    if len(matching) > _COMPOSITE_MAX_CASES:
        matching = matching[:_COMPOSITE_MAX_CASES]

    display_name, varname, cmap, units, vmin, vmax = VARIABLES[variable]
    ref_varname = DERIVED_VARIABLES[variable][0] if variable in DERIVED_VARIABLES else varname

    # Cases need both valid SDDC and valid RMW
    valid_cases = []
    for ci in matching:
        meta = meta_cache.get(ci, {})
        sddc = meta.get("sddc")
        rmw = meta.get("rmw_km")
        if sddc is not None and rmw is not None and not np.isnan(float(rmw)) and float(rmw) > 0:
            valid_cases.append((ci, sddc, float(rmw)))
    if not valid_cases:
        raise HTTPException(status_code=400, detail="No matching cases have both shear direction and valid RMW.")

    # Pre-fetch grid info from the first valid case so all workers share it
    first_ds, _ = resolve_case(valid_cases[0][0], data_type)
    height_km = first_ds["height"].values
    x_coords, y_coords, h_axis = _resolve_grid_and_haxis(first_ds, ref_varname)

    accum_sum = None
    accum_count = None
    ov_accum_sum = None
    ov_accum_count = None
    r_centers = None
    n_processed = 0
    processed_indices = []

    # Process cases in batches — keeps peak memory bounded on 512 MB servers.
    work_items = [
        (case_idx, sddc, rmw, variable, overlay, data_type,
         x_coords, y_coords, height_km, h_axis,
         max_r_rmw, dr_rmw, coverage_min)
        for case_idx, sddc, rmw in valid_cases
    ]

    def _accum_quad(result):
        nonlocal accum_sum, accum_count, ov_accum_sum, ov_accum_count
        nonlocal r_centers, n_processed
        case_idx, quad_means, rc, ov_quads = result
        if accum_sum is None:
            r_centers = rc
            accum_sum = {q: np.zeros_like(quad_means[q]) for q in QUADRANT_DEFS}
            accum_count = {q: np.zeros_like(quad_means[q]) for q in QUADRANT_DEFS}
        for q in QUADRANT_DEFS:
            valid = ~np.isnan(quad_means[q])
            accum_sum[q][valid] += quad_means[q][valid]
            accum_count[q][valid] += 1
        if ov_quads is not None:
            if ov_accum_sum is None:
                ov_accum_sum = {q: np.zeros_like(ov_quads[q]) for q in QUADRANT_DEFS}
                ov_accum_count = {q: np.zeros_like(ov_quads[q]) for q in QUADRANT_DEFS}
            for q in QUADRANT_DEFS:
                ov_valid = ~np.isnan(ov_quads[q])
                ov_accum_sum[q][ov_valid] += ov_quads[q][ov_valid]
                ov_accum_count[q][ov_valid] += 1
        n_processed += 1
        processed_indices.append(case_idx)

    _process_composites_batched(_process_one_case_quadrant, work_items, _accum_quad)

    if n_processed == 0:
        raise HTTPException(status_code=500, detail="Could not process any matching cases.")

    # Require each grid cell to have data from at least 33% of processed cases
    # (or 3, whichever is greater) to suppress noisy bins with sparse sampling
    min_cases = max(3, int(np.ceil(0.33 * n_processed)))
    composite = {}
    for q in QUADRANT_DEFS:
        composite[q] = np.where(accum_count[q] >= min_cases, accum_sum[q] / accum_count[q], np.nan)

    result = {
        "quadrant_means": {q: {"data": _clean_2d(composite[q])} for q in QUADRANT_DEFS},
        "radius_rrmw": [round(float(r), 3) for r in r_centers],
        "height_km": [round(float(h), 2) for h in height_km],
        "normalized": True,
        "coverage_min": coverage_min,
        "min_cases_per_bin": min_cases,
        "n_cases": n_processed,
        "n_matched": len(matching),
        "n_with_shear_and_rmw": len(valid_cases),
        "case_list": _build_case_list(processed_indices, data_type),
        "variable": {
            "key": variable,
            "display_name": display_name,
            "units": units,
            "vmin": vmin,
            "vmax": vmax,
            "colorscale": _cmap_to_plotly(cmap),
        },
        "filters": {
            "intensity": [min_intensity, max_intensity],
            "vmax_change": [min_vmax_change, max_vmax_change],
            "tilt": [min_tilt, max_tilt],
            "year": [min_year, max_year],
            "shear_mag": [min_shear_mag, max_shear_mag],
            "shear_dir": [min_shear_dir, max_shear_dir],
        },
    }

    # Add overlay if computed
    if overlay and ov_accum_sum is not None:
        ov_display, _, ov_cmap, ov_units, ov_vmin, ov_vmax = VARIABLES[overlay]
        ov_composite = {}
        all_flat = []
        for q in QUADRANT_DEFS:
            ov_composite[q] = np.where(ov_accum_count[q] >= min_cases, ov_accum_sum[q] / ov_accum_count[q], np.nan)
            clean_q = _clean_2d(ov_composite[q])
            ov_composite[q] = clean_q
            all_flat.extend(v for row in clean_q for v in row if v is not None)
        result["overlay"] = {
            "display_name": ov_display,
            "key": overlay,
            "units": ov_units,
            "quadrant_means": {q: {"data": ov_composite[q]} for q in QUADRANT_DEFS},
            "vmin": min(all_flat) if all_flat else ov_vmin,
            "vmax": max(all_flat) if all_flat else ov_vmax,
        }

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Composite plan-view endpoint
# ---------------------------------------------------------------------------

@app.get("/composite/plan_view")
def composite_plan_view(
    variable:       str   = Query(DEFAULT_VARIABLE,        description="Variable key"),
    overlay:        str   = Query("",                      description="Optional overlay variable key"),
    data_type:      str   = Query("swath",                 description="'swath' or 'merge'"),
    level_km:       float = Query(2.0,  ge=0.5, le=18.0,  description="Height level (km)"),
    normalize_rmw:  bool  = Query(False,                   description="Normalise X/Y by RMW?"),
    max_r_rmw:      float = Query(5.0,  ge=1,   le=20,    description="Max extent in R/RMW"),
    dr_rmw:         float = Query(0.1,  ge=0.05, le=1,    description="Grid spacing in R/RMW"),
    shear_relative: bool  = Query(False,                   description="Rotate to shear-relative frame?"),
    coverage_min:   float = Query(0.25, ge=0.0,  le=1.0,  description="Min coverage fraction"),
    min_intensity:  float = Query(0),    max_intensity:  float = Query(200),
    min_vmax_change:float = Query(-100), max_vmax_change:float = Query(85),
    min_tilt:       float = Query(0),    max_tilt:       float = Query(200),
    min_year:       int   = Query(1997), max_year:       int   = Query(2024),
    min_shear_mag:  float = Query(0),    max_shear_mag:  float = Query(100),
    min_shear_dir:  float = Query(0),    max_shear_dir:  float = Query(360),
):
    """Compute a composite plan-view mean at a specified height level."""

    if variable not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"Unknown variable '{variable}'.")
    if overlay and overlay not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"Unknown overlay variable '{overlay}'.")
    if data_type not in ("swath", "merge"):
        raise HTTPException(status_code=400, detail="data_type must be 'swath' or 'merge'")

    meta_cache = _merge_metadata_cache if data_type == "merge" else _metadata_cache

    matching = _filter_cases_for_composite(
        min_intensity, max_intensity, min_vmax_change, max_vmax_change,
        min_tilt, max_tilt, min_year, max_year,
        min_shear_mag, max_shear_mag, min_shear_dir, max_shear_dir,
        data_type=data_type,
    )
    if not matching:
        raise HTTPException(status_code=400, detail="No cases match the specified criteria.")
    if len(matching) > _COMPOSITE_MAX_CASES:
        matching = matching[:_COMPOSITE_MAX_CASES]

    # ------------------------------------------------------------------
    # Collect per-case metadata (RMW, optionally SDDC)
    # ------------------------------------------------------------------
    cases_ready: list[tuple] = []   # (case_idx, rmw, sddc_or_None)
    for ci in matching:
        meta = meta_cache.get(ci, {})
        rmw = meta.get("rmw_km")
        if normalize_rmw:
            # RMW is mandatory when normalising
            if rmw is None or np.isnan(float(rmw)) or float(rmw) <= 0:
                continue
        else:
            rmw = rmw if (rmw is not None and not np.isnan(float(rmw)) and float(rmw) > 0) else None

        sddc = meta.get("sddc") if shear_relative else None
        if shear_relative and sddc is None:
            continue  # need SDDC for rotation

        cases_ready.append((ci, float(rmw) if rmw is not None else None, sddc))

    if not cases_ready:
        detail = "No matching cases have valid "
        parts = []
        if normalize_rmw:
            parts.append("RMW")
        if shear_relative:
            parts.append("shear direction")
        detail += " and ".join(parts) + " data."
        raise HTTPException(status_code=400, detail=detail)

    # ------------------------------------------------------------------
    # Pre-fetch grid info & locate height index
    # ------------------------------------------------------------------
    display_name, varname, cmap, units, vmin, vmax = VARIABLES[variable]
    ref_varname = DERIVED_VARIABLES[variable][0] if variable in DERIVED_VARIABLES else varname

    first_ds, _ = resolve_case(cases_ready[0][0], data_type)
    height_km = first_ds["height"].values
    z_idx = int(np.argmin(np.abs(height_km - level_km)))
    actual_level = float(height_km[z_idx])
    x_coords, y_coords, _ = _resolve_grid_and_haxis(first_ds, ref_varname)

    # ------------------------------------------------------------------
    # Parallel processing
    # ------------------------------------------------------------------
    accum_sum = None
    accum_count = None
    ov_accum_sum = None
    ov_accum_count = None
    n_processed = 0
    processed_indices: list[int] = []

    # Process cases in batches — keeps peak memory bounded on 512 MB servers.
    work_items = [
        (ci, rmw, sddc, variable, overlay, data_type,
         x_coords, y_coords, z_idx,
         normalize_rmw, max_r_rmw, dr_rmw, shear_relative)
        for ci, rmw, sddc in cases_ready
    ]

    def _accum_pv(result):
        nonlocal accum_sum, accum_count, ov_accum_sum, ov_accum_count, n_processed
        ci, plan_2d, x_grid, y_grid, ov_2d = result
        if accum_sum is None:
            accum_sum = np.zeros((len(y_grid), len(x_grid)))
            accum_count = np.zeros_like(accum_sum)
        valid = ~np.isnan(plan_2d)
        accum_sum[valid] += plan_2d[valid]
        accum_count[valid] += 1
        if ov_2d is not None:
            if ov_accum_sum is None:
                ov_accum_sum = np.zeros_like(ov_2d)
                ov_accum_count = np.zeros_like(ov_2d)
            ov_valid = ~np.isnan(ov_2d)
            ov_accum_sum[ov_valid] += ov_2d[ov_valid]
            ov_accum_count[ov_valid] += 1
        n_processed += 1
        processed_indices.append(ci)

    _process_composites_batched(_process_one_case_plan_view, work_items, _accum_pv)

    if n_processed == 0:
        raise HTTPException(status_code=500, detail="Could not process any matching cases.")

    # ------------------------------------------------------------------
    # Build composite — same 33 % minimum-cases rule as az-mean
    # ------------------------------------------------------------------
    min_cases = max(3, int(np.ceil(0.33 * n_processed)))
    composite = np.where(accum_count >= min_cases, accum_sum / accum_count, np.nan)

    # Determine output axis arrays & labels
    if normalize_rmw:
        half_n = int(round(max_r_rmw / dr_rmw))
        x_out = np.linspace(-max_r_rmw, max_r_rmw, 2 * half_n + 1)
        y_out = np.linspace(-max_r_rmw, max_r_rmw, 2 * half_n + 1)
        x_label = "X / RMW"
        y_label = "Y / RMW"
    else:
        x_out = x_coords
        y_out = y_coords
        x_label = "Eastward distance (km)"
        y_label = "Northward distance (km)"

    result = {
        "plan_view": _clean_2d(composite),
        "x_axis": [round(float(v), 3) for v in x_out],
        "y_axis": [round(float(v), 3) for v in y_out],
        "x_label": x_label,
        "y_label": y_label,
        "level_km": round(actual_level, 2),
        "normalize_rmw": normalize_rmw,
        "shear_relative": shear_relative,
        "coverage_min": coverage_min,
        "min_cases_per_bin": min_cases,
        "n_cases": n_processed,
        "n_matched": len(matching),
        "n_with_valid_meta": len(cases_ready),
        "case_list": _build_case_list(processed_indices, data_type),
        "variable": {
            "key": variable,
            "display_name": display_name,
            "units": units,
            "vmin": vmin,
            "vmax": vmax,
            "colorscale": _cmap_to_plotly(cmap),
        },
        "filters": {
            "intensity": [min_intensity, max_intensity],
            "vmax_change": [min_vmax_change, max_vmax_change],
            "tilt": [min_tilt, max_tilt],
            "year": [min_year, max_year],
            "shear_mag": [min_shear_mag, max_shear_mag],
            "shear_dir": [min_shear_dir, max_shear_dir],
        },
    }

    # Overlay
    if overlay and ov_accum_sum is not None:
        ov_composite = np.where(
            ov_accum_count >= min_cases, ov_accum_sum / ov_accum_count, np.nan
        )
        ov_display, _, _, ov_units, ov_vmin, ov_vmax = VARIABLES[overlay]
        clean_ov = _clean_2d(ov_composite)
        flat = [v for row in clean_ov for v in row if v is not None]
        result["overlay"] = {
            "display_name": ov_display,
            "key": overlay,
            "units": ov_units,
            "plan_view": clean_ov,
            "vmin": min(flat) if flat else ov_vmin,
            "vmax": max(flat) if flat else ov_vmax,
        }

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Environmental composite endpoints (ERA5)
# ---------------------------------------------------------------------------

# Common target grid for storm-relative km compositing
_ENV_COMP_GRID_KM = np.linspace(-1100, 1100, 81)  # ±1100 km at ~27 km spacing


def _process_one_case_era5_plan_view(case_idx, field, include_vectors, target_x_km, target_y_km,
                                     sddc=None):
    """
    Process a single case for ERA5 environmental plan-view composite.
    Regrids from degree-offset to a common km grid.
    If sddc is not None, rotate to shear-relative frame (shear → +x).
    Returns (case_idx, data_km, vec_u_km, vec_v_km) or None.
    """
    try:
        era5 = get_era5_dataset()
        if era5 is None:
            return None
        status = int(era5['status'][case_idx])
        if status != 1:
            return None

        data_deg = era5[field][case_idx]  # (81, 81) in degree offsets
        center_lat = float(era5['center_lat'][case_idx])
        lat_offsets = era5['lat_offsets'][:]
        lon_offsets = era5['lon_offsets'][:]

        # Convert degree offsets to km
        y_km_src = lat_offsets * 111.0
        x_km_src = lon_offsets * 111.0 * np.cos(np.deg2rad(center_lat))

        # Regrid onto common km grid
        interp = RegularGridInterpolator(
            (y_km_src, x_km_src), np.array(data_deg, dtype=np.float64),
            method='linear', bounds_error=False, fill_value=np.nan
        )
        yy, xx = np.meshgrid(target_y_km, target_x_km, indexing='ij')
        data_km = interp((yy, xx)).astype(np.float32)

        # Shear-relative rotation: rotate so shear vector points right (+x)
        # Same convention as TDR plan-view: rotation_angle = 90 - SDDC
        if sddc is not None:
            rotation_angle = 90.0 - float(sddc)
            data_km = _rotate_2d_grid(data_km, rotation_angle)

        vec_u_km = None
        vec_v_km = None
        if include_vectors:
            cfg = ERA5_FIELD_CONFIG.get(field, {})
            if cfg.get('has_vectors'):
                u_data = era5[cfg['vector_u']][case_idx]
                v_data = era5[cfg['vector_v']][case_idx]
                interp_u = RegularGridInterpolator(
                    (y_km_src, x_km_src), np.array(u_data, dtype=np.float64),
                    method='linear', bounds_error=False, fill_value=np.nan
                )
                interp_v = RegularGridInterpolator(
                    (y_km_src, x_km_src), np.array(v_data, dtype=np.float64),
                    method='linear', bounds_error=False, fill_value=np.nan
                )
                vec_u_km = interp_u((yy, xx)).astype(np.float32)
                vec_v_km = interp_v((yy, xx)).astype(np.float32)

                # Rotate vector field to shear-relative frame
                if sddc is not None:
                    vec_u_km = _rotate_2d_grid(vec_u_km, rotation_angle)
                    vec_v_km = _rotate_2d_grid(vec_v_km, rotation_angle)
                    # Also rotate the vector components themselves
                    # (rotating the grid moves pixels, but vectors still point
                    #  in their original directions — we must also rotate them)
                    theta_rad = np.deg2rad(rotation_angle)
                    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
                    u_rot =  cos_t * vec_u_km + sin_t * vec_v_km
                    v_rot = -sin_t * vec_u_km + cos_t * vec_v_km
                    vec_u_km, vec_v_km = u_rot, v_rot

        return (case_idx, data_km, vec_u_km, vec_v_km)
    except Exception as e:
        print(f"Composite era5_plan_view: skipping case {case_idx}: {e}")
        return None


@app.get("/composite/era5_plan_view")
def composite_era5_plan_view(
    filters: dict = Depends(_composite_filter_params),
    field: str = Query("shear_mag", description="ERA5 field name"),
    radius_km: float = Query(500, ge=100, le=1100, description="Crop radius (km)"),
    include_vectors: bool = Query(False, description="Include vector components"),
    shear_relative: bool = Query(False, description="Rotate to shear-relative frame?"),
    data_type: str = Query("swath", description="'swath' or 'merge'"),
):
    """Composite mean + std of an ERA5 2D field in storm-relative km coordinates."""
    era5 = get_era5_dataset()
    if era5 is None:
        raise HTTPException(status_code=503, detail="ERA5 data not available")
    if field not in ERA5_FIELD_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown field: {field}")

    matching = _filter_cases_for_composite(**filters, data_type=data_type)
    if not matching:
        raise HTTPException(status_code=404, detail="No matching cases")

    # Filter to cases with ERA5 data
    era5_status = era5['status'][:]
    matching = [ci for ci in matching if ci < len(era5_status) and int(era5_status[ci]) == 1]
    if not matching:
        raise HTTPException(status_code=404, detail="No matching cases with ERA5 data")

    # If shear_relative, look up SDDC per case from metadata and skip cases without it
    meta_cache = _merge_metadata_cache if data_type == "merge" else _metadata_cache
    cases_with_sddc: list[tuple] = []  # (case_idx, sddc_or_None)
    for ci in matching[:_COMPOSITE_MAX_CASES]:
        if shear_relative:
            meta = meta_cache.get(ci, {})
            sddc = meta.get("sddc")
            if sddc is None or sddc == 9999:
                continue  # need valid SDDC for rotation
            cases_with_sddc.append((ci, float(sddc)))
        else:
            cases_with_sddc.append((ci, None))

    if not cases_with_sddc:
        detail = "No matching cases with ERA5 data"
        if shear_relative:
            detail += " and valid shear direction (SDDC)"
        raise HTTPException(status_code=404, detail=detail)

    target_x_km = _ENV_COMP_GRID_KM.copy()
    target_y_km = _ENV_COMP_GRID_KM.copy()
    grid_shape = (len(target_y_km), len(target_x_km))

    accum_sum = np.zeros(grid_shape, dtype=np.float64)
    accum_sq = np.zeros(grid_shape, dtype=np.float64)
    accum_count = np.zeros(grid_shape, dtype=np.float64)
    vec_u_sum = np.zeros(grid_shape, dtype=np.float64) if include_vectors else None
    vec_v_sum = np.zeros(grid_shape, dtype=np.float64) if include_vectors else None
    vec_count = np.zeros(grid_shape, dtype=np.float64) if include_vectors else None
    processed = []

    # Process cases in batches — keeps peak memory bounded on 512 MB servers.
    work_items = [
        (ci, field, include_vectors, target_x_km, target_y_km, sddc)
        for ci, sddc in cases_with_sddc
    ]

    def _accum_era5(result):
        nonlocal vec_u_sum, vec_v_sum, vec_count
        ci, data_km, vu, vv = result
        valid = np.isfinite(data_km)
        accum_sum[valid] += data_km[valid]
        accum_sq[valid] += data_km[valid] ** 2
        accum_count[valid] += 1
        if include_vectors and vu is not None:
            vu_valid = np.isfinite(vu)
            vec_u_sum[vu_valid] += vu[vu_valid]
            vv_valid = np.isfinite(vv)
            vec_v_sum[vv_valid] += vv[vv_valid]
            vec_count[vu_valid] += 1
        processed.append(ci)

    _process_composites_batched(_process_one_case_era5_plan_view, work_items, _accum_era5)

    n_cases = len(processed)
    if n_cases == 0:
        raise HTTPException(status_code=404, detail="No valid ERA5 cases processed")

    min_cases = max(2, int(0.33 * n_cases))
    mean_2d = np.where(accum_count >= min_cases, accum_sum / accum_count, np.nan)
    variance = np.where(
        accum_count >= min_cases,
        accum_sq / accum_count - (accum_sum / accum_count) ** 2,
        np.nan
    )
    std_2d = np.sqrt(np.maximum(variance, 0))

    # Crop to radius_km
    yy, xx = np.meshgrid(target_y_km, target_x_km, indexing='ij')
    dist = np.sqrt(xx**2 + yy**2)
    crop_mask = dist > radius_km
    mean_2d[crop_mask] = np.nan
    std_2d[crop_mask] = np.nan

    cfg = ERA5_FIELD_CONFIG[field]
    response = {
        "field": field,
        "field_config": cfg,
        "mean": _clean_2d(mean_2d),
        "std": _clean_2d(std_2d),
        "x_km": [round(float(v), 1) for v in target_x_km],
        "y_km": [round(float(v), 1) for v in target_y_km],
        "n_cases": n_cases,
        "shear_relative": shear_relative,
        "case_list": _build_case_list(processed, data_type),
    }

    if include_vectors and vec_u_sum is not None:
        vec_mean_u = np.where(vec_count >= min_cases, vec_u_sum / vec_count, np.nan)
        vec_mean_v = np.where(vec_count >= min_cases, vec_v_sum / vec_count, np.nan)
        vec_mean_u[crop_mask] = np.nan
        vec_mean_v[crop_mask] = np.nan
        stride = max(1, len(target_x_km) // 12)
        response["vectors"] = {
            "u": _clean_2d(vec_mean_u[::stride, ::stride]),
            "v": _clean_2d(vec_mean_v[::stride, ::stride]),
            "stride": stride,
        }

    return JSONResponse(response)


@app.get("/composite/era5_profiles")
def composite_era5_profiles(
    filters: dict = Depends(_composite_filter_params),
    data_type: str = Query("swath", description="'swath' or 'merge'"),
):
    """
    Composite mean +/- std vertical profiles (T, Td, RH, u, v, theta, theta_e)
    from precomputed ERA5 200-600 km annulus profiles.
    """
    era5 = get_era5_dataset()
    if era5 is None:
        raise HTTPException(status_code=503, detail="ERA5 data not available")

    matching = _filter_cases_for_composite(**filters, data_type=data_type)
    if not matching:
        raise HTTPException(status_code=404, detail="No matching cases")

    era5_status = era5['status'][:]
    matching = [ci for ci in matching if ci < len(era5_status) and int(era5_status[ci]) == 1]
    if not matching:
        raise HTTPException(status_code=404, detail="No matching cases with ERA5 data")

    matching = matching[:_COMPOSITE_MAX_CASES]

    plev = era5['plev'][:].astype(float)
    n_lev = len(plev)
    n_cases = len(matching)

    t_all = np.full((n_cases, n_lev), np.nan, dtype=np.float64)
    q_all = np.full((n_cases, n_lev), np.nan, dtype=np.float64)
    rh_all = np.full((n_cases, n_lev), np.nan, dtype=np.float64)
    u_all = np.full((n_cases, n_lev), np.nan, dtype=np.float64)
    v_all = np.full((n_cases, n_lev), np.nan, dtype=np.float64)

    valid_count = 0
    for i, ci in enumerate(matching):
        try:
            t_all[i] = era5['t_profile'][ci].astype(float)
            q_all[i] = era5['q_profile'][ci].astype(float)
            rh_all[i] = era5['rh_profile'][ci].astype(float)
            u_all[i] = era5['u_profile'][ci].astype(float)
            v_all[i] = era5['v_profile'][ci].astype(float)
            valid_count += 1
        except Exception:
            pass

    if valid_count < 2:
        raise HTTPException(status_code=404, detail="Insufficient valid profiles")

    q_kgkg = q_all.copy()
    if np.nanmax(q_all) > 0.5:
        q_kgkg = q_all / 1000.0

    # Dewpoint from specific humidity and pressure (vectorized per level)
    td_all = np.full_like(t_all, np.nan)
    for lev in range(n_lev):
        p_pa = plev[lev] * 100.0
        q_lev = q_kgkg[:, lev]
        t_lev = t_all[:, lev]
        valid_mask = np.isfinite(q_lev) & (q_lev > 0) & np.isfinite(t_lev)
        if not valid_mask.any():
            continue
        e = q_lev[valid_mask] * p_pa / (0.622 + 0.378 * q_lev[valid_mask])
        e_sat = 610.94 * np.exp(17.625 * (t_lev[valid_mask] - 273.15) / (t_lev[valid_mask] - 273.15 + 243.04))
        e = np.minimum(e, e_sat)
        e = np.maximum(e, 1e-3)
        ln_e = np.log(e / 611.2)
        td_c = 243.5 * ln_e / (17.67 - ln_e)
        td_all[valid_mask, lev] = td_c + 273.15

    Rd_Cp = 287.04 / 1005.7
    theta_all = t_all * (1000.0 / plev[np.newaxis, :]) ** Rd_Cp
    Lv, Cpd = 2.501e6, 1005.7
    theta_e_all = theta_all * np.exp(Lv * q_kgkg / (Cpd * t_all))

    def _profile_stats(arr, decimals=2):
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        med = np.nanmedian(arr, axis=0)
        p25 = np.nanpercentile(arr, 25, axis=0)
        p75 = np.nanpercentile(arr, 75, axis=0)
        pmin = np.nanmin(arr, axis=0)
        pmax = np.nanmax(arr, axis=0)
        count = np.sum(np.isfinite(arr), axis=0)

        def _to_list(a, d=decimals):
            return [round(float(v), d) if np.isfinite(v) else None for v in a]

        return {
            "mean": _to_list(mean), "std": _to_list(std),
            "median": _to_list(med), "p25": _to_list(p25), "p75": _to_list(p75),
            "min": _to_list(pmin), "max": _to_list(pmax),
            "n_valid": [int(v) for v in count],
        }

    return JSONResponse({
        "plev": plev.tolist(),
        "n_cases": valid_count,
        "t": _profile_stats(t_all),
        "td": _profile_stats(td_all),
        "q": _profile_stats(q_all, decimals=4),
        "rh": _profile_stats(rh_all, decimals=1),
        "u": _profile_stats(u_all),
        "v": _profile_stats(v_all),
        "theta": _profile_stats(theta_all, decimals=1),
        "theta_e": _profile_stats(theta_e_all, decimals=1),
        "case_list": _build_case_list(matching, data_type),
    })


@app.get("/composite/era5_scalars")
def composite_era5_scalars(
    filters: dict = Depends(_composite_filter_params),
    data_type: str = Query("swath", description="'swath' or 'merge'"),
):
    """Composite statistics for all ERA5 scalar diagnostics."""
    era5 = get_era5_dataset()
    if era5 is None:
        raise HTTPException(status_code=503, detail="ERA5 data not available")

    matching = _filter_cases_for_composite(**filters, data_type=data_type)
    if not matching:
        raise HTTPException(status_code=404, detail="No matching cases")

    era5_status = era5['status'][:]
    matching = [ci for ci in matching if ci < len(era5_status) and int(era5_status[ci]) == 1]
    if not matching:
        raise HTTPException(status_code=404, detail="No matching cases with ERA5 data")

    matching = matching[:_COMPOSITE_MAX_CASES]
    indices = np.array(matching)

    scalar_names = [
        'shear_mag_env', 'shear_dir_env', 'rh_mid_env', 'div200_env',
        'sst_env', 'chi_m', 'v_pi', 'vent_index',
    ]
    scalar_meta = {
        'shear_mag_env': {"display_name": "Deep-Layer Shear", "units": "m/s", "decimals": 1},
        'shear_dir_env': {"display_name": "Shear Direction", "units": "\u00b0", "decimals": 0},
        'rh_mid_env': {"display_name": "Mid-Level RH", "units": "%", "decimals": 0},
        'div200_env': {"display_name": "200-hPa Divergence", "units": "\u00d710\u207b\u2075 s\u207b\u00b9", "decimals": 2, "scale": 1e5},
        'sst_env': {"display_name": "SST", "units": "\u00b0C", "decimals": 1},
        'chi_m': {"display_name": "Entropy Deficit (\u03c7\u2098)", "units": "", "decimals": 2},
        'v_pi': {"display_name": "Potential Intensity", "units": "m/s", "decimals": 1},
        'vent_index': {"display_name": "Ventilation Index", "units": "", "decimals": 3},
    }

    scalars = {}
    for sname in scalar_names:
        try:
            values = np.array([float(era5[sname][ci]) for ci in indices])
            valid = values[np.isfinite(values)]
            if len(valid) == 0:
                scalars[sname] = None
                continue

            meta = scalar_meta.get(sname, {"display_name": sname, "units": "", "decimals": 2})
            scale = meta.get("scale", 1.0)
            d = meta["decimals"]

            scalars[sname] = {
                "display_name": meta["display_name"],
                "units": meta["units"],
                "mean": round(float(np.mean(valid)) * scale, d),
                "std": round(float(np.std(valid)) * scale, d),
                "median": round(float(np.median(valid)) * scale, d),
                "p25": round(float(np.percentile(valid, 25)) * scale, d),
                "p75": round(float(np.percentile(valid, 75)) * scale, d),
                "min": round(float(np.min(valid)) * scale, d),
                "max": round(float(np.max(valid)) * scale, d),
                "n_valid": len(valid),
                "values": [round(float(v) * scale, d) for v in valid],
            }
        except Exception:
            scalars[sname] = None

    return JSONResponse({
        "n_cases": len(matching),
        "scalars": scalars,
        "case_list": _build_case_list(matching, data_type),
    })


@app.get("/plot")
def plot(
    case_index: int   = Query(...,              ge=0,          description="0-based case index"),
    variable:   str   = Query(DEFAULT_VARIABLE,                description="Variable key — see /variables"),
    level_km:   float = Query(2.0,              ge=0.0, le=18, description="Altitude in km"),
    data_type:  str   = Query("swath",                         description="'swath' or 'merge'"),
):
    if variable not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"Unknown variable '{variable}'. See /variables.")
    if data_type not in ("swath", "merge"):
        raise HTTPException(status_code=400, detail="data_type must be 'swath' or 'merge'")

    # Serve from cache if available (instant)
    cache_key = (case_index, variable, round(level_km, 1), data_type)
    if cache_key in _plot_cache:
        _plot_cache.move_to_end(cache_key)
        return Response(content=_plot_cache[cache_key], media_type="image/png",
                        headers={"X-Cache": "HIT"})

    try:
        ds, local_idx = resolve_case(case_index, data_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open dataset: {e}")

    cache = _merge_metadata_cache if data_type == "merge" else _metadata_cache
    case_meta = cache.get(case_index, {"case_index": case_index})

    try:
        png = render_planview(ds, local_idx, variable, level_km, case_meta)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Render error: {e}")

    _plot_cache[cache_key] = png
    if len(_plot_cache) > _PLOT_CACHE_MAX:
        _plot_cache.popitem(last=False)  # evict oldest entry
    return Response(content=png, media_type="image/png", headers={"X-Cache": "MISS"})

from realtime_tdr_api import router as realtime_router
app.include_router(realtime_router, prefix="/realtime")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
