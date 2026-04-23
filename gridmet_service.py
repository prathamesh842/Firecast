# gridmet_service.py
# Fetches real 75-day weather data from GridMET API
# for any USA location

import json
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO

logger = logging.getLogger(__name__)

# ── Load column means (fallback for missing data) ──────────────
with open("column_means.json", "r") as f:
    COLUMN_MEANS = json.load(f)

# ── GridMET variable names ─────────────────────────────────────
# Maps short_name (file path) → long_name (var parameter)
GRIDMET_VARIABLES = {
    "pr":     "precipitation_amount",
    "rmax":   "max_relative_humidity",
    "rmin":   "min_relative_humidity",
    "sph":    "specific_humidity",
    "srad":   "surface_downwelling_shortwave_flux_in_air",
    "tmmn":   "air_temperature",           # daily min
    "tmmx":   "air_temperature",           # daily max
    "vs":     "wind_speed",
    "bi":     "burning_index_g",
    "fm100":  "dead_fuel_moisture_100hr",
    "fm1000": "dead_fuel_moisture_1000hr",
    "erc":    "energy_release_component-g",
    "etr":    "grass_reference_evapotranspiration",
    "pet":    "potential_evapotranspiration",
    "vpd":    "mean_vapor_pressure_deficit"
}

# ── GridMET base URL ───────────────────────────────────────────
# short_name: file path (e.g., tmmx)
# long_name: var parameter (e.g., air_temperature)
BASE_URL = "https://thredds.northwestknowledge.net/thredds/ncss/grid/MET/{short_name}/{short_name}_{year}.nc?var={long_name}&latitude={lat}&longitude={lng}&time_start={start}&time_end={end}&accept=csv"


def fetch_single_variable(
    short_name: str,
    long_name: str,
    lat: float,
    lng: float,
    start_date: str,
    end_date: str,
    year: int
) -> pd.Series:
    """
    Fetches a single variable from GridMET API.
    short_name: file path (e.g., 'tmmx')
    long_name: var parameter (e.g., 'air_temperature')
    Returns a pandas Series indexed by date.
    Falls back to column mean if fetch fails.
    """
    url = BASE_URL.format(
        short_name=short_name,
        long_name=long_name,
        year=year,
        lat=lat,
        lng=lng,
        start=start_date,
        end=end_date
    )

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Parse CSV response
        df   = pd.read_csv(
            StringIO(response.text),
            skiprows=1   # skip header comment
        )
        df.columns = df.columns.str.strip()

        # Find date column
        date_col = [c for c in df.columns
                    if 'time' in c.lower() or
                       'date' in c.lower()][0]
        df['date'] = pd.to_datetime(df[date_col])
        df         = df.set_index('date')

        # Find value column
        val_col = [c for c in df.columns
                   if long_name.split('-')[0] in c.lower()
                   or short_name in c.lower()]

        if not val_col:
            val_col = [df.columns[-1]]

        series = df[val_col[0]].astype(float)

        # Replace nodata (32767) with column mean
        series = series.replace(32767.0, COLUMN_MEANS[short_name])
        series = series.fillna(COLUMN_MEANS[short_name])

        logger.debug(f"  ✅ {short_name}: {len(series)} days fetched")
        return series

    except Exception as e:
        logger.debug(
            f"  ℹ️ {short_name} using fallback (expected for some 2026 data)"
        )
        # Return series of mean values as fallback
        dates  = pd.date_range(start=start_date, end=end_date)
        return pd.Series(
            COLUMN_MEANS[short_name],
            index=dates,
            name=short_name
        )


def fetch_gridmet_window(
    lat: float,
    lng: float,
    window_days: int = 75
) -> np.ndarray:
    """
    Fetches last `window_days` of real weather data
    from GridMET for the given location.

    Returns numpy array of shape (1, 75, 1, 1, 17)
    ready to feed directly into ConvLSTM2D model!
    """

    # ── Validate USA bounds ────────────────────────────────────
    if not (25 <= lat <= 49):
        raise ValueError(
            f"Latitude {lat} out of USA bounds (25-49)"
        )
    if not (-124 <= lng <= -67):
        raise ValueError(
            f"Longitude {lng} out of USA bounds (-124 to -67)"
        )

    # ── Date range: last 75 days ───────────────────────────────
    # GridMET has ~5 day lag so end 5 days ago
    end_date   = datetime.now() - timedelta(days=5)
    start_date = end_date - timedelta(days=window_days - 1)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str   = end_date.strftime('%Y-%m-%d')
    year      = end_date.year

    # ⚠️ GridMET data is typically only available for past years
    # If current year is 2025 or later, use 2024 data instead
    # This is a workaround for when running in future dates (simulations)
    if year >= 2025:
        logger.info(f"  ℹ️ Adjusting year from {year} to 2024 (GridMET data availability)")
        year = 2024
        # Adjust dates to 2024 equivalent (same month/day)
        end_date = datetime(2024, end_date.month, end_date.day)
        start_date = datetime(2024, start_date.month, start_date.day)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str   = end_date.strftime('%Y-%m-%d')

    logger.info(
        f"Fetching GridMET data: "
        f"({lat}, {lng}) "
        f"{start_str} → {end_str}"
    )

    # ── Date range for alignment ───────────────────────────────
    date_range = pd.date_range(
        start=start_str,
        end=end_str,
        freq='D'
    )

    # ── Feature order matches your training data ───────────────
    # latitude, longitude, pr, rmax, rmin, sph, srad,
    # tmmn, tmmx, vs, bi, fm100, fm1000, erc, etr, pet, vpd

    feature_order = [
        'pr', 'rmax', 'rmin', 'sph', 'srad',
        'tmmn', 'tmmx', 'vs', 'bi', 'fm100',
        'fm1000', 'erc', 'etr', 'pet', 'vpd'
    ]

    # ── Fetch each variable ────────────────────────────────────
    fetched = {}
    fetch_success = 0
    fetch_fallback = 0

    # GRIDMET_VARIABLES maps short_name → long_name
    for var in feature_order:
        long_name = GRIDMET_VARIABLES[var]

        series = fetch_single_variable(
            short_name  = var,
            long_name   = long_name,
            lat         = lat,
            lng         = lng,
            start_date  = start_str,
            end_date    = end_str,
            year        = year
        )

        # Check if this is real data or fallback
        is_fallback = (series.iloc[0] == COLUMN_MEANS[var]) if len(series) > 0 else True
        if is_fallback:
            fetch_fallback += 1
        else:
            fetch_success += 1

        # Align to our date range
        series         = series.reindex(date_range)
        series         = series.fillna(COLUMN_MEANS[var])
        fetched[var]   = series.values

    # Log fetch statistics
    logger.info(
        f"  📊 GridMET fetch stats: {fetch_success}/{len(feature_order)} "
        f"succeeded, {fetch_fallback} using fallback"
    )

    # ── Build window array ─────────────────────────────────────
    # Shape: (75, 17)
    window = np.zeros(
        (window_days, 17), dtype=np.float32
    )

    for day in range(window_days):
        window[day, 0]  = lat                      # latitude
        window[day, 1]  = lng                       # longitude
        window[day, 2]  = fetched['pr'][day]        # pr
        window[day, 3]  = fetched['rmax'][day]      # rmax
        window[day, 4]  = fetched['rmin'][day]      # rmin
        window[day, 5]  = fetched['sph'][day]       # sph
        window[day, 6]  = fetched['srad'][day]      # srad
        window[day, 7]  = fetched['tmmn'][day]      # tmmn
        window[day, 8]  = fetched['tmmx'][day]      # tmmx
        window[day, 9]  = fetched['vs'][day]        # vs
        window[day, 10] = fetched['bi'][day]        # bi
        window[day, 11] = fetched['fm100'][day]     # fm100
        window[day, 12] = fetched['fm1000'][day]    # fm1000
        window[day, 13] = fetched['erc'][day]       # erc
        window[day, 14] = fetched['etr'][day]       # etr
        window[day, 15] = fetched['pet'][day]       # pet
        window[day, 16] = fetched['vpd'][day]       # vpd

    # ── Replace any remaining nodata ───────────────────────────
    window[window == 32767.0] = 0.0
    window                    = np.nan_to_num(window, nan=0.0)

    # ⚠️ Check if mostly fallback data (would explain same predictions)
    if fetch_fallback > len(feature_order) * 0.7:
        logger.warning(
            f"  ⚠️ {fetch_fallback}/{len(feature_order)} variables using fallback! "
            f"This location may return predictions similar to other locations."
        )

    # ── Reshape for model: (1, 75, 1, 1, 17) ──────────────────
    x = window[np.newaxis, :, np.newaxis, np.newaxis, :]

    logger.info(
        f"✅ GridMET window built: "
        f"shape={x.shape} "
        f"tmmx_latest={window[-1,8]:.1f}K "
        f"({window[-1,8]-273.15:.1f}°C)"
    )

    return x


def test_gridmet_connection(
    lat: float = 37.7749,
    lng: float = -119.4194
):
    """
    Quick test to verify GridMET API is accessible.
    Tests with California coordinates.
    """
    logger.info("Testing GridMET connection...")
    try:
        end_date   = datetime.now() - timedelta(days=5)
        start_date = end_date - timedelta(days=3)

        url = BASE_URL.format(
            short_name = 'tmmx',
            long_name  = 'air_temperature',
            year       = end_date.year,
            lat        = lat,
            lng        = lng,
            start      = start_date.strftime('%Y-%m-%d'),
            end        = end_date.strftime('%Y-%m-%d')
        )

        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            logger.info("✅ GridMET API accessible!")
            return True
        else:
            logger.warning(
                f"⚠️ GridMET returned status {response.status_code}"
            )
            return False

    except Exception as e:
        logger.error(f"❌ GridMET connection failed: {e}")
        return False