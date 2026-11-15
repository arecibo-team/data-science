"""Amber Index: a 0-100 wildfire protection-gap score per EU-27 country and year.

Shared by the notebook (training) and the API (serving), so both apply the same preprocessing.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MANUAL_DIR = DATA_DIR / "manual"
ARTIFACTS_DIR = ROOT / "artifacts"
SEED = 42

# EU-27: ISO2 (Eurostat uses EL for Greece; we normalise to GR), ISO3, English name
EU27 = {
    "AT": ("AUT", "Austria"), "BE": ("BEL", "Belgium"), "BG": ("BGR", "Bulgaria"), "HR": ("HRV", "Croatia"),
    "CY": ("CYP", "Cyprus"), "CZ": ("CZE", "Czechia"), "DK": ("DNK", "Denmark"), "EE": ("EST", "Estonia"),
    "FI": ("FIN", "Finland"), "FR": ("FRA", "France"), "DE": ("DEU", "Germany"), "GR": ("GRC", "Greece"),
    "HU": ("HUN", "Hungary"), "IE": ("IRL", "Ireland"), "IT": ("ITA", "Italy"), "LV": ("LVA", "Latvia"),
    "LT": ("LTU", "Lithuania"), "LU": ("LUX", "Luxembourg"), "MT": ("MLT", "Malta"), "NL": ("NLD", "Netherlands"),
    "PL": ("POL", "Poland"), "PT": ("PRT", "Portugal"), "RO": ("ROU", "Romania"), "SK": ("SVK", "Slovakia"),
    "SI": ("SVN", "Slovenia"), "ES": ("ESP", "Spain"), "SE": ("SWE", "Sweden"),
}
NEIGHBOURS = {
    "UA": ("UKR", "Ukraine"), "MD": ("MDA", "Moldova"), "BY": ("BLR", "Belarus"), "GB": ("GBR", "United Kingdom"),
    "NO": ("NOR", "Norway"), "CH": ("CHE", "Switzerland"), "RS": ("SRB", "Serbia"), "BA": ("BIH", "Bosnia and Herzegovina"),
    "ME": ("MNE", "Montenegro"), "AL": ("ALB", "Albania"), "XK": ("XKX", "Kosovo"), "TR": ("TUR", "Turkey"),
    "GE": ("GEO", "Georgia"), "AM": ("ARM", "Armenia"), "AZ": ("AZE", "Azerbaijan"),
}
FIRE_COUNTRIES = {**EU27, **NEIGHBOURS}
FIRE_NAMES = {iso2: name for iso2, (_, name) in FIRE_COUNTRIES.items()}
ISO3_TO_ISO2 = {iso3: iso2 for iso2, (iso3, _) in EU27.items()}
NAME_TO_ISO2 = {name: iso2 for iso2, (_, name) in EU27.items()}
NAMES = {iso2: name for iso2, (_, name) in EU27.items()}

SOURCES = {
    "eurostat_api": "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/",
    "spending": "gov_10a_exp?cofog99=GF0302&sector=S13&na_item=TE&unit=MIO_EUR",
    "population": "demo_pjan?sex=T&age=TOTAL",
    "area": "reg_area3?landuse=TOTAL&unit=KM2",
    "firefighters_2024": "https://ec.europa.eu/eurostat/documents/4187653/20614201/eu-lfs-extraction-firefighters-2024.xlsx/05e74593-6a14-807e-5429-c6732668aa78?t=1755599278722",
    "firefighters_news": [
        "https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20240823-1",
        "https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20230807-1",
    ],
    "burned_area": "https://ourworldindata.org/grapher/annual-area-burnt-by-wildfires-gwis.csv?useColumnShortNames=true",
    "overpass": ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
                 "https://overpass.private.coffee/api/interpreter"],
    "boundaries": "https://gisco-services.ec.europa.eu/distribution/v2/countries/geojson/CNTR_RG_20M_2020_4326.geojson",
    "hazards": "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH",
}
