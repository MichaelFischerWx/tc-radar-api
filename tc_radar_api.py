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
    pip install fastapi uvicorn h5netcdf h5py fsspec xarray zarr s3fs matplotlib numpy
    TC_RADAR_S3_BUCKET=your-bucket uvicorn tc_radar_api:app --reload --port 8000
"""

import io
import json
import os
import threading
from collections import OrderedDict
from functools import lru_cache
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
    "recentered_tangential_wind":           ("Tangential Wind (WCM)",          "recentered_tangential_wind",           "RdBu_r",    "m/s",  -10,   80),
    "recentered_radial_wind":               ("Radial Wind (WCM)",               "recentered_radial_wind",               "RdBu_r",    "m/s",  -30,   30),
    "recentered_upward_air_velocity":       ("Vertical Velocity (WCM)",         "recentered_upward_air_velocity",       "RdBu_r",    "m/s",   -5,    5),
    "recentered_reflectivity":              ("Reflectivity (WCM)",              "recentered_reflectivity",              "Spectral_r","dBZ",  -10,   65),
    "recentered_wind_speed":                ("Wind Speed (WCM)",                "recentered_wind_speed",                "inferno",   "m/s",    0,   80),
    "recentered_earth_relative_wind_speed": ("Earth-Rel. Wind Speed (WCM)",    "recentered_earth_relative_wind_speed", "inferno",   "m/s",    0,   80),
    "recentered_relative_vorticity":        ("Relative Vorticity (WCM)",        "recentered_relative_vorticity",        "RdBu_r",    "s⁻¹",-5e-3, 5e-3),
    "recentered_divergence":                ("Divergence (WCM)",                "recentered_divergence",                "RdBu_r",    "s⁻¹",-5e-3, 5e-3),
    "total_recentered_tangential_wind":     ("Tangential Wind (tilt-relative)", "total_recentered_tangential_wind",     "RdBu_r",    "m/s",  -10,   80),
    "total_recentered_radial_wind":         ("Radial Wind (tilt-relative)",     "total_recentered_radial_wind",         "RdBu_r",    "m/s",  -30,   30),
    "total_recentered_upward_air_velocity": ("Vertical Velocity (tilt-rel.)",   "total_recentered_upward_air_velocity", "RdBu_r",    "m/s",   -5,    5),
    "total_recentered_reflectivity":        ("Reflectivity (tilt-relative)",    "total_recentered_reflectivity",        "Spectral_r","dBZ",  -10,   65),
    "total_recentered_wind_speed":          ("Wind Speed (tilt-relative)",      "total_recentered_wind_speed",          "inferno",   "m/s",    0,   80),
    "total_recentered_earth_relative_wind_speed": ("Earth-Rel. Wind Speed (tilt-rel.)", "total_recentered_earth_relative_wind_speed", "inferno", "m/s", 0, 80),
    "swath_tangential_wind":                ("Tangential Wind (original)",      "swath_tangential_wind",                "RdBu_r",    "m/s",  -10,   80),
    "swath_radial_wind":                    ("Radial Wind (original)",          "swath_radial_wind",                    "RdBu_r",    "m/s",  -30,   30),
    "swath_reflectivity":                   ("Reflectivity (original)",         "swath_reflectivity",                   "Spectral_r","dBZ",  -10,   65),
    "swath_wind_speed":                     ("Wind Speed (original)",           "swath_wind_speed",                     "inferno",   "m/s",    0,   80),
    "swath_earth_relative_wind_speed":      ("Earth-Rel. Wind Speed (original)","swath_earth_relative_wind_speed",      "inferno",   "m/s",    0,   80),
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
_metadata_cache: dict[int, dict] = {}
_plot_cache: OrderedDict = OrderedDict()
_PLOT_CACHE_MAX = 500  # ~500 plots × ~150 KB ≈ 75 MB max


@app.on_event("startup")
def startup():
    # Load metadata
    global _metadata_cache
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            data = json.load(f)
        _metadata_cache = {c["case_index"]: c for c in data.get("cases", [])}
        print(f"Loaded {len(_metadata_cache)} cases from {METADATA_PATH}")
    else:
        print(f"Warning: {METADATA_PATH} not found")

    backend = f"S3 Zarr (s3://{S3_BUCKET}/{S3_PREFIX})" if USE_S3 else "AOML HTTP (fallback)"
    print(f"Data backend: {backend}")

    # Pre-warm swath datasets in background threads
    def prewarm(data_type, era):
        try:
            get_dataset(data_type, era)
            print(f"Pre-warmed {data_type}/{era}")
        except Exception as e:
            print(f"Pre-warm failed {data_type}/{era}: {e}")

    for era in ("early", "recent"):
        threading.Thread(target=prewarm, args=("swath", era), daemon=True).start()


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
def get_metadata(case_index: int = Query(..., ge=0)):
    if case_index not in _metadata_cache:
        raise HTTPException(status_code=404, detail=f"case_index {case_index} not found")
    return JSONResponse(_metadata_cache[case_index])


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

    case_meta = _metadata_cache.get(case_index, {"case_index": case_index})

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
        "case_meta": {
            "storm_name": case_meta.get("storm_name", ""),
            "datetime": case_meta.get("datetime", ""),
            "vmax_kt": case_meta.get("vmax_kt"),
            "rmw_km": case_meta.get("rmw_km"),
            "mission_id": case_meta.get("mission_id", ""),
        },
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

    case_meta = _metadata_cache.get(case_index, {"case_index": case_index})

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
        "case_meta": {
            "storm_name": case_meta.get("storm_name", ""),
            "datetime": case_meta.get("datetime", ""),
            "vmax_kt": case_meta.get("vmax_kt"),
        },
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

    case_meta = _metadata_cache.get(case_index, {"case_index": case_index})

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
