"""
tc_radar_api.py
===============
FastAPI backend for on-demand TC-RADAR plot generation.
Opens AOML NetCDF files lazily via fsspec HTTP range requests — no full
downloads, no kerchunk indexing required. Each plot request fetches only
the chunks needed for the requested case/variable/level (~1–5 MB).

Deploy on Render, Railway, or Fly.io (free tier is sufficient).

Local dev
---------
    pip install fastapi uvicorn h5netcdf h5py fsspec xarray matplotlib numpy ujson aiohttp
    uvicorn tc_radar_api:app --reload --port 8000

    # Test:
    # http://localhost:8000/plot?case_index=0&variable=recentered_tangential_wind&level_km=2.0
    # http://localhost:8000/health
    # http://localhost:8000/variables

API Endpoints
-------------
GET /plot
    ?case_index=<int>       Required. 0-based index (matches tc_radar_metadata.json)
    ?variable=<str>         Wind variable key (default: recentered_tangential_wind)
    ?level_km=<float>       Altitude in km (default: 2.0)
    ?data_type=<str>        'swath' or 'merge' (default: swath)
    Returns: PNG image

GET /variables
    Returns: JSON list of available variable keys and display names

GET /levels
    Returns: JSON list of available height levels in km

GET /metadata?case_index=<int>
    Returns: JSON metadata for a single case

GET /health
    Returns: {"status": "ok"}
"""

import io
import json
import os
from functools import lru_cache
from pathlib import Path

import fsspec
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

# ---------------------------------------------------------------------------
# AOML file URLs
# ---------------------------------------------------------------------------
AOML_BASE = "https://www.aoml.noaa.gov/ftp/pub/hrd/data/radar/level3"

AOML_FILES = {
    ("swath", "early"):  f"{AOML_BASE}/tc_radar_v3m_1997_2019_xy_rel_swath_ships.nc",
    ("swath", "recent"): f"{AOML_BASE}/tc_radar_v3m_2020_2024_xy_rel_swath_ships.nc",
    ("merge", "early"):  f"{AOML_BASE}/tc_radar_v3m_1997_2019_xy_rel_merge_ships.nc",
    ("merge", "recent"): f"{AOML_BASE}/tc_radar_v3m_2020_2024_xy_rel_merge_ships.nc",
}

# Cases per era file. Swath early confirmed = 710.
# Update merge counts after running the validation script on those files.
CASE_COUNTS = {
    ("swath", "early"):  710,
    ("swath", "recent"): 800,   # 1510 - 710
    ("merge", "early"):  310,   # update after checking merge file dims
    ("merge", "recent"): 126,   # update after checking merge file dims
}

# ---------------------------------------------------------------------------
# Variable config
# Three reference frameworks:
#   swath_*            : original, no recentering
#   recentered_*       : WCM recentered at 2 km (best for vortex-relative analysis)
#   total_recentered_* : tilt-relative, recentered at every level
#
# Format per key: (display_name, nc_varname, colormap, units, vmin, vmax)
# ---------------------------------------------------------------------------
VARIABLES = {
    # Recentered (WCM 2 km) — default framework
    "recentered_tangential_wind":        ("Tangential Wind (WCM)",         "recentered_tangential_wind",           "RdBu_r",    "m/s",  -10,   80),
    "recentered_radial_wind":            ("Radial Wind (WCM)",              "recentered_radial_wind",               "RdBu_r",    "m/s",  -30,   30),
    "recentered_upward_air_velocity":    ("Vertical Velocity (WCM)",        "recentered_upward_air_velocity",       "RdBu_r",    "m/s",   -5,    5),
    "recentered_reflectivity":           ("Reflectivity (WCM)",             "recentered_reflectivity",              "Spectral_r","dBZ",  -10,   65),
    "recentered_wind_speed":             ("Wind Speed (WCM)",               "recentered_wind_speed",                "inferno",   "m/s",    0,   80),
    "recentered_relative_vorticity":     ("Relative Vorticity (WCM)",       "recentered_relative_vorticity",        "RdBu_r",    "s⁻¹", -5e-3, 5e-3),
    "recentered_divergence":             ("Divergence (WCM)",               "recentered_divergence",                "RdBu_r",    "s⁻¹", -5e-3, 5e-3),
    # Tilt-relative
    "total_recentered_tangential_wind":  ("Tangential Wind (tilt-relative)","total_recentered_tangential_wind",     "RdBu_r",    "m/s",  -10,   80),
    "total_recentered_radial_wind":      ("Radial Wind (tilt-relative)",    "total_recentered_radial_wind",         "RdBu_r",    "m/s",  -30,   30),
    "total_recentered_upward_air_velocity":("Vertical Velocity (tilt-rel)", "total_recentered_upward_air_velocity", "RdBu_r",    "m/s",   -5,    5),
    "total_recentered_reflectivity":     ("Reflectivity (tilt-relative)",   "total_recentered_reflectivity",        "Spectral_r","dBZ",  -10,   65),
    "total_recentered_wind_speed":       ("Wind Speed (tilt-relative)",     "total_recentered_wind_speed",          "inferno",   "m/s",    0,   80),
    # Original swath
    "swath_tangential_wind":             ("Tangential Wind (original)",     "swath_tangential_wind",                "RdBu_r",    "m/s",  -10,   80),
    "swath_radial_wind":                 ("Radial Wind (original)",         "swath_radial_wind",                    "RdBu_r",    "m/s",  -30,   30),
    "swath_reflectivity":                ("Reflectivity (original)",        "swath_reflectivity",                   "Spectral_r","dBZ",  -10,   65),
    "swath_wind_speed":                  ("Wind Speed (original)",          "swath_wind_speed",                     "inferno",   "m/s",    0,   80),
}

DEFAULT_VARIABLE = "recentered_tangential_wind"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="TC-RADAR API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to https://michaelfischerwx.github.io in production
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Dataset loading — one open lazy dataset per file, cached across requests
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def get_dataset(data_type: str, era: str) -> xr.Dataset:
    """
    Open a TC-RADAR AOML file lazily via fsspec HTTP.
    Only HDF5 metadata is read on open; data chunks fetched on demand.
    Cached so repeated requests reuse the same open dataset.
    """
    url = AOML_FILES[(data_type, era)]
    of = fsspec.open(url, "rb")
    ds = xr.open_dataset(of.open(), engine="h5netcdf", chunks={})
    return ds


def resolve_case(case_index: int, data_type: str) -> tuple[xr.Dataset, int]:
    """
    Map a global case_index (0-based, matching tc_radar_metadata.json)
    to the correct era file and local index within that file.
    """
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
    """
    Render a storm-relative plan-view PNG for one TC-RADAR case.
    The .values call triggers HTTP range requests only for the
    selected variable/level/case chunk — typically a few MB.
    """
    display_name, varname, cmap, units, vmin, vmax = VARIABLES[variable_key]

    if varname not in ds:
        available = [k for k, v in VARIABLES.items() if v[1] in ds]
        raise ValueError(f"'{varname}' not in this dataset. Available: {available}")

    # Select case and height level
    da = ds[varname].isel(num_cases=local_idx)
    height_vals = ds["height"].values
    z_idx = int(np.argmin(np.abs(height_vals - level_km)))
    actual_level = float(height_vals[z_idx])
    da = da.isel(height=z_idx)

    # Trigger HTTP range request — fetches only this chunk
    data = da.values

    # Spatial axes (km, storm-relative)
    x = ds["eastward_distance"].values
    y = ds["northward_distance"].values

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(7, 6.5), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    im = ax.pcolormesh(x, y, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")

    # RMW ring
    rmw = case_meta.get("rmw_km")
    if rmw is not None and not np.isnan(float(rmw)):
        rmw = float(rmw)
        theta = np.linspace(0, 2 * np.pi, 360)
        ax.plot(rmw * np.cos(theta), rmw * np.sin(theta),
                "w--", lw=1.5, alpha=0.85, label=f"RMW = {rmw:.0f} km")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.3, labelcolor="white")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(units, color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Title
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
# Metadata cache — loaded once at startup from tc_radar_metadata.json
# ---------------------------------------------------------------------------
METADATA_PATH = Path(os.environ.get("METADATA_PATH", "./tc_radar_metadata.json"))
_metadata_cache: dict[int, dict] = {}


@app.on_event("startup")
def load_metadata():
    global _metadata_cache
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            data = json.load(f)
        _metadata_cache = {c["case_index"]: c for c in data.get("cases", [])}
        print(f"Loaded {len(_metadata_cache)} cases from {METADATA_PATH}")
    else:
        print(f"Warning: {METADATA_PATH} not found — plot titles will lack metadata")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/variables")
def list_variables():
    """Available variable keys and display names."""
    return [
        {"key": k, "display_name": v[0], "units": v[3]}
        for k, v in VARIABLES.items()
    ]


@app.get("/levels")
def list_levels():
    """Available height levels in km."""
    try:
        ds = get_dataset("swath", "early")
        return {"levels_km": [round(float(h), 2) for h in ds["height"].values]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/metadata")
def get_metadata(case_index: int = Query(..., ge=0)):
    if case_index not in _metadata_cache:
        raise HTTPException(status_code=404, detail=f"case_index {case_index} not found")
    return JSONResponse(_metadata_cache[case_index])


@app.get("/plot")
def plot(
    case_index: int   = Query(...,              ge=0,          description="0-based case index"),
    variable:   str   = Query(DEFAULT_VARIABLE,                description="Variable key — see /variables"),
    level_km:   float = Query(2.0,              ge=0.0, le=18, description="Altitude in km"),
    data_type:  str   = Query("swath",                         description="'swath' or 'merge'"),
):
    """
    Return a plan-view PNG for a single TC-RADAR case.
    Fetches only the required data chunk from AOML via HTTP range request.
    Typical response time: 2–8 s depending on AOML server latency.
    """
    if variable not in VARIABLES:
        raise HTTPException(status_code=400,
                            detail=f"Unknown variable '{variable}'. See /variables.")
    if data_type not in ("swath", "merge"):
        raise HTTPException(status_code=400, detail="data_type must be 'swath' or 'merge'")

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

    return Response(content=png, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
