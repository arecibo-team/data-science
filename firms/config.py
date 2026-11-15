"""Shared configuration: paths, FIRMS source catalogue, and the common schema."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
NORM_DIR = DATA_DIR / "normalized"
ANALYSIS_DIR = DATA_DIR / "analysis"
WEB_DATA_DIR = ROOT / "web" / "public" / "data"
INCIDENTS_DIR = DATA_DIR / "incidents"

# Independent ground truth. Nothing here is derived from FIRMS/MODIS/VIIRS: EFFIS burnt
# areas are MODIS/VIIRS-based and were removed for that reason.
EMS_ACTIVATIONS = "https://mapping.emergency.copernicus.eu/activations/api/activations/"
EMS_DASHBOARD = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/"
# Earliest date we harvest incidents for; matches the start of the FIRMS archive we use.
INCIDENTS_FROM = "2024-01-01"
# Rough Europe bounding box (lon_min, lat_min, lon_max, lat_max) used to reject
# non-European EMS activations.
EUROPE_BBOX = (-25.0, 34.0, 45.0, 72.0)

# Sentinel-2 burn scars, computed here from Microsoft Planetary Computer STAC (open data).
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
# Fire events: NASA detections linked within S2_LINK_KM and S2_LINK_DAYS (connected components).
S2_LINK_KM = 2.0
S2_LINK_DAYS = 2
S2_MIN_DETECTIONS = 10
# Events whose bounding box exceeds this span are too large for a single scene check.
S2_MAX_SPAN_KM = 50.0
# Chaining can link a whole season of persistent sources (industrial flares, volcanoes) into
# one event; events lasting longer than this are not a single fire and are skipped.
S2_MAX_DAYS = 21
S2_PAD_KM = 1.0
# Search windows: pre-fire scene before the first detection, post-fire scene after the last.
S2_PRE_DAYS = 30
S2_POST_DAYS = 45
S2_POST_OFFSET_DAYS = 2
S2_CLOUD_MAX = 40          # scene-level eo:cloud_cover upper bound, percent
S2_MAX_CANDIDATES = 4      # scenes loaded per side, lowest cloud cover first
S2_MIN_CLEAR = 0.5         # share of the event box that must be usable land in a scene
S2_RESOLUTION = 20         # metres; B8A, B12 and SCL are all native at 20 m
# Burn scar rules: dNBR threshold (~0.27 is the low-severity boundary used by USGS),
# minimum total scar area, and how close a scar must be to a NASA detection to count.
S2_DNBR = 0.27
S2_MIN_SCAR_HA = 5.0
S2_NEAR_KM = 1.0
# Matching of NASA detections to a Sentinel-2 scar: tight, with an exact date window.
S2_MATCH_BUFFER_KM = 0.3

# Matching of NASA detections to EMS AOIs: a detection is "confirmed" when it lies within
# this distance of an AOI and its date is within the AOI's dates +/- slack.
MATCH_BUFFER_KM = 1.0
MATCH_DATE_SLACK_DAYS = 2

API_BASE = "https://firms.modaps.eosdis.nasa.gov/api"
# FIRMS area API accepts 1..5 days per request (verified: 400 'Expects [1..5]' for 10).
MAX_DAY_RANGE = 5

# One "family" per satellite platform. SP = standard processing (archive),
# NRT = near-real-time (recent tail). sp=None when no SP product exists.
FAMILIES: dict[str, dict] = {
    "MODIS":      {"sensor": "MODIS", "sp": "MODIS_SP",         "nrt": "MODIS_NRT"},
    "VIIRS_SNPP": {"sensor": "VIIRS", "sp": "VIIRS_SNPP_SP",    "nrt": "VIIRS_SNPP_NRT"},
    "VIIRS_NOAA20": {"sensor": "VIIRS", "sp": "VIIRS_NOAA20_SP", "nrt": "VIIRS_NOAA20_NRT"},
    "VIIRS_NOAA21": {"sensor": "VIIRS", "sp": None,              "nrt": "VIIRS_NOAA21_NRT"},
}

# Map FIRMS source id -> (family, product)
SOURCE_INFO: dict[str, tuple[str, str]] = {}
for _fam, _spec in FAMILIES.items():
    if _spec["sp"]:
        SOURCE_INFO[_spec["sp"]] = (_fam, "SP")
    SOURCE_INFO[_spec["nrt"]] = (_fam, "NRT")

# FIRMS satellite codes -> platform names (MODIS uses names, VIIRS uses codes)
PLATFORM_NAMES = {"Terra": "Terra", "Aqua": "Aqua", "N": "SNPP", "N20": "NOAA-20", "N21": "NOAA-21"}

# Common schema columns, in output order.
SCHEMA = [
    "uid", "sensor", "platform", "product", "lat", "lon", "acq_time_utc", "daynight",
    "confidence_class", "confidence_raw", "confidence_pct", "frp_mw",
    "bt_mid_k", "bt_secondary_k", "scan_km", "track_km", "pixel_area_km2",
    "firms_version", "firms_type", "source",
]


def load_map_keys() -> list[str]:
    """Read FIRMS MAP_KEY(s) from the environment or .env. Comma-separated for several.

    Values are never printed or logged.
    """
    env = dotenv_values(ROOT / ".env")
    raw = (
        os.environ.get("FIRMS_MAP_KEYS")
        or os.environ.get("KEY")
        or env.get("FIRMS_MAP_KEYS")
        or env.get("KEY")
        or ""
    )
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        raise RuntimeError("No FIRMS MAP_KEY found. Put KEY=<key> in .env (comma-separate several).")
    return keys
