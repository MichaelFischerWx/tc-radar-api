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
import io
import json
import os
import threading
from collections import OrderedDict
from functools import lru_cache
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import fsspec
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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
_PLOT_CACHE_MAX = 500  # ~500 plots × ~150 KB ≈ 75 MB max


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


@app.get("/debug")
def debug():
    """Temporary debug endpoint — shows what xarray sees in the Zarr store."""
    try:
        ds = get_dataset("swath", "early")
        return {
            "dims": dict(ds.sizes),
            "data_vars": list(ds.data_vars),
            "coords": list(ds.coords),
        }
    except Exception as e:
        return {"error": str(e)}


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

    return JSONResponse(result)


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
):
    """
    Return the full 3D volume as flattened arrays for Plotly isosurface rendering.

    The grid is subsampled spatially by `stride` to reduce transfer size.
    NaN values are replaced with a sentinel (-9999) so the grid stays regular;
    the client should set isomin above this sentinel.
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

    # Build flattened coordinate arrays via meshgrid
    Z, Y, X = np.meshgrid(height_sub, y_sub, x_sub, indexing='ij')
    x_flat = np.round(X.ravel(), 2)
    y_flat = np.round(Y.ravel(), 2)
    z_flat = np.round(Z.ravel(), 2)
    v_flat = vol_sub.ravel()

    # Compute actual data range (excluding NaN)
    valid = v_flat[np.isfinite(v_flat)]
    data_min = float(np.nanmin(valid)) if len(valid) > 0 else vmin
    data_max = float(np.nanmax(valid)) if len(valid) > 0 else vmax

    # Replace NaN with sentinel for regular-grid isosurface
    SENTINEL = -9999.0
    v_flat = np.where(np.isfinite(v_flat), np.round(v_flat, 3), SENTINEL)

    case_meta = _build_case_meta(case_index, ds, local_idx, data_type)

    return JSONResponse({
        "x": x_flat.tolist(),
        "y": y_flat.tolist(),
        "z": z_flat.tolist(),
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
    })


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
):
    """
    Return ERA5 environmental diagnostics for a TC-RADAR case.

    Parameters
    ----------
    case_index : int
    field : str - 2D field to return (shear_mag, rh_mid, div200)
    include_profiles : bool - include vertical profiles & hodograph data
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

    # Read 2D field
    data_2d = era5[field][case_index]  # (81, 81)
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
        "lat_offsets": era5['lat_offsets'][:].tolist(),
        "lon_offsets": era5['lon_offsets'][:].tolist(),
    }

    # Add vector components if field has them (subsampled every 4 pts = ~1°)
    if cfg.get("has_vectors"):
        stride = 4
        u_sub = era5[cfg["vector_u"]][case_index][::stride, ::stride]
        v_sub = era5[cfg["vector_v"]][case_index][::stride, ::stride]
        result["vectors"] = {
            "u": [[round(float(v), 2) if not np.isnan(v) else None for v in row] for row in u_sub],
            "v": [[round(float(v), 2) if not np.isnan(v) else None for v in row] for row in v_sub],
            "stride": stride,
        }

    # Scalar diagnostics
    result["scalars"] = {}
    for sname in ['shear_mag_env', 'shear_dir_env', 'rh_mid_env', 'div200_env']:
        try:
            val = float(era5[sname][case_index])
            result["scalars"][sname] = round(val, 4) if 'div' in sname else round(val, 1)
        except Exception:
            result["scalars"][sname] = None

    # Vertical profiles (200-600 km annulus mean)
    if include_profiles:
        try:
            result["profiles"] = {
                "plev": era5['plev'][:].tolist(),
                "u": [round(float(v), 2) if not np.isnan(v) else None for v in era5['u_profile'][case_index]],
                "v": [round(float(v), 2) if not np.isnan(v) else None for v in era5['v_profile'][case_index]],
                "rh": [round(float(v), 1) if not np.isnan(v) else None for v in era5['rh_profile'][case_index]],
                "t": [round(float(v), 2) if not np.isnan(v) else None for v in era5['t_profile'][case_index]],
            }
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
_COMPOSITE_MAX_CASES = 500  # safety cap

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
    return {"count": len(cases), "case_indices": cases[:20]}  # preview first 20


@app.get("/composite/azimuthal_mean")
def composite_azimuthal_mean(
    variable:      str   = Query(DEFAULT_VARIABLE,       description="Variable key"),
    overlay:       str   = Query("",                     description="Optional overlay variable key"),
    data_type:     str   = Query("swath",                description="'swath' or 'merge'"),
    max_r_rmw:     float = Query(8.0,   ge=1, le=20,    description="Max radius in R/RMW"),
    dr_rmw:        float = Query(0.25,  ge=0.1, le=2,   description="Radial bin width in R/RMW"),
    coverage_min:  float = Query(0.5,   ge=0.0, le=1.0),
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

    accum_sum = None
    accum_count = None
    ov_accum_sum = None
    ov_accum_count = None
    r_centers = None
    height_km = None
    n_processed = 0
    processed_indices = []

    for case_idx, rmw in cases_with_rmw:
        try:
            ds, local_idx = resolve_case(case_idx, data_type)
            if height_km is None:
                height_km = ds["height"].values
                x_coords, y_coords, h_axis = _resolve_grid_and_haxis(ds, ref_varname)
            vol, _ = _extract_3d_volume(ds, local_idx, variable)
            az_mean, _, rc = _compute_azimuthal_mean(
                vol, x_coords, y_coords, height_km, h_axis,
                max_r_rmw, dr_rmw, coverage_min, rmw=rmw
            )
            if accum_sum is None:
                r_centers = rc
                accum_sum = np.zeros_like(az_mean)
                accum_count = np.zeros_like(az_mean)
            valid = ~np.isnan(az_mean)
            accum_sum[valid] += az_mean[valid]
            accum_count[valid] += 1

            # Overlay accumulation
            if overlay:
                try:
                    ov_vol, _ = _extract_3d_volume(ds, local_idx, overlay)
                    ov_az, _, _ = _compute_azimuthal_mean(
                        ov_vol, x_coords, y_coords, height_km, h_axis,
                        max_r_rmw, dr_rmw, coverage_min, rmw=rmw
                    )
                    if ov_accum_sum is None:
                        ov_accum_sum = np.zeros_like(ov_az)
                        ov_accum_count = np.zeros_like(ov_az)
                    ov_valid = ~np.isnan(ov_az)
                    ov_accum_sum[ov_valid] += ov_az[ov_valid]
                    ov_accum_count[ov_valid] += 1
                except Exception:
                    pass  # skip overlay for this case

            n_processed += 1
            processed_indices.append(case_idx)
        except Exception as e:
            print(f"Composite: skipping case {case_idx}: {e}")
            continue

    if n_processed == 0:
        raise HTTPException(status_code=500, detail="Could not process any matching cases.")

    composite = np.where(accum_count > 0, accum_sum / accum_count, np.nan)

    result = {
        "azimuthal_mean": _clean_2d(composite),
        "radius_rrmw": [round(float(r), 3) for r in r_centers],
        "height_km": [round(float(h), 2) for h in height_km],
        "normalized": True,
        "coverage_min": coverage_min,
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
        ov_composite = np.where(ov_accum_count > 0, ov_accum_sum / ov_accum_count, np.nan)
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
    coverage_min:  float = Query(0.5,   ge=0.0, le=1.0),
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

    accum_sum = None
    accum_count = None
    ov_accum_sum = None
    ov_accum_count = None
    r_centers = None
    height_km = None
    n_processed = 0
    processed_indices = []

    for case_idx, sddc, rmw in valid_cases:
        try:
            ds, local_idx = resolve_case(case_idx, data_type)
            if height_km is None:
                height_km = ds["height"].values
                x_coords, y_coords, h_axis = _resolve_grid_and_haxis(ds, ref_varname)
            vol, _ = _extract_3d_volume(ds, local_idx, variable)
            quad_means, rc = _compute_quadrant_means(
                vol, x_coords, y_coords, height_km, h_axis,
                sddc, max_r_rmw, dr_rmw, coverage_min, rmw=rmw
            )
            if accum_sum is None:
                r_centers = rc
                accum_sum = {q: np.zeros_like(quad_means[q]) for q in QUADRANT_DEFS}
                accum_count = {q: np.zeros_like(quad_means[q]) for q in QUADRANT_DEFS}
            for q in QUADRANT_DEFS:
                valid = ~np.isnan(quad_means[q])
                accum_sum[q][valid] += quad_means[q][valid]
                accum_count[q][valid] += 1

            # Overlay accumulation
            if overlay:
                try:
                    ov_vol, _ = _extract_3d_volume(ds, local_idx, overlay)
                    ov_quad_means, _ = _compute_quadrant_means(
                        ov_vol, x_coords, y_coords, height_km, h_axis,
                        sddc, max_r_rmw, dr_rmw, coverage_min, rmw=rmw
                    )
                    if ov_accum_sum is None:
                        ov_accum_sum = {q: np.zeros_like(ov_quad_means[q]) for q in QUADRANT_DEFS}
                        ov_accum_count = {q: np.zeros_like(ov_quad_means[q]) for q in QUADRANT_DEFS}
                    for q in QUADRANT_DEFS:
                        ov_valid = ~np.isnan(ov_quad_means[q])
                        ov_accum_sum[q][ov_valid] += ov_quad_means[q][ov_valid]
                        ov_accum_count[q][ov_valid] += 1
                except Exception:
                    pass  # skip overlay for this case

            n_processed += 1
            processed_indices.append(case_idx)
        except Exception as e:
            print(f"Composite quad: skipping case {case_idx}: {e}")
            continue

    if n_processed == 0:
        raise HTTPException(status_code=500, detail="Could not process any matching cases.")

    composite = {}
    for q in QUADRANT_DEFS:
        composite[q] = np.where(accum_count[q] > 0, accum_sum[q] / accum_count[q], np.nan)

    result = {
        "quadrant_means": {q: {"data": _clean_2d(composite[q])} for q in QUADRANT_DEFS},
        "radius_rrmw": [round(float(r), 3) for r in r_centers],
        "height_km": [round(float(h), 2) for h in height_km],
        "normalized": True,
        "coverage_min": coverage_min,
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
            ov_composite[q] = np.where(ov_accum_count[q] > 0, ov_accum_sum[q] / ov_accum_count[q], np.nan)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
