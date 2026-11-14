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

# Ground-truth incident sources (European fires only).
EFFIS_WFS = "https://maps.effis.emergency.copernicus.eu/effis"
EMS_ACTIVATIONS = "https://mapping.emergency.copernicus.eu/activations/api/activations/"
EMS_DASHBOARD = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/"
# Earliest date we harvest incidents for; matches the start of the FIRMS archive we use.
INCIDENTS_FROM = "2024-01-01"

# Country codes as used by EFFIS (EL = Greece, UK = United Kingdom) and EMS names.
# Turkey and Russia are excluded: only partly European and dominated by non-European fires.
EUROPE_COUNTRIES = {
    "AL", "AD", "AT", "BY", "BE", "BA", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE",
    "EL", "GR", "HU", "IS", "IE", "IT", "XK", "LV", "LI", "LT", "LU", "MT", "MD", "MC", "ME",
    "NL", "MK", "NO", "PL", "PT", "RO", "SM", "RS", "SK", "SI", "ES", "SE", "CH", "UA", "GB",
    "UK", "VA",
}
# Rough Europe bounding box (lon_min, lat_min, lon_max, lat_max) used to reject
# non-European EMS activations and stray EFFIS polygons.
EUROPE_BBOX = (-25.0, 34.0, 45.0, 72.0)

# Matching: a NASA detection is "confirmed" when it lies within this distance of an
# incident polygon and its acquisition date is within the incident's dates +/- slack.
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
