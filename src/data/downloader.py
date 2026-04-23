from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests


@dataclass
class StationSpec:
    name: str
    lat: float
    lon: float
    noaa_station: str


def _parse_noaa_numeric(value: str | float | int | None, scale: float = 1.0) -> float:
    if value is None or value == "" or str(value).strip() == "9999" or str(value).strip() == "99999":
        return np.nan
    try:
        txt = str(value).split(",")[0]
        if txt in {"+9999", "-9999", "+99999", "-99999", "99999"}:
            return np.nan
        return float(txt) / scale
    except Exception:
        return np.nan


def _download_noaa_global_hourly(station_id: str, start: str, end: str) -> pd.DataFrame:
    url = "https://www.ncei.noaa.gov/access/services/data/v1"
    params = {
        "dataset": "global-hourly",
        "stations": station_id,
        "startDate": start,
        "endDate": end,
        "format": "json",
        "includeAttributes": "false",
        "includeStationName": "true",
        "includeStationLocation": "1",
        "units": "metric",
    }
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    rows = r.json()
    if not isinstance(rows, list) or len(rows) == 0:
        raise RuntimeError(f"NOAA global-hourly empty for station={station_id}")

    df = pd.DataFrame(rows)
    # Key fields in NOAA global-hourly
    # DATE, TMP, DEW, SLP, VISIB, WDSP, MXSPD, PRCP
    out = pd.DataFrame()
    def col(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series([np.nan] * len(df))
    out["time"] = pd.to_datetime(df["DATE"], utc=True, errors="coerce")
    out["temperature"] = col("TMP").apply(lambda x: _parse_noaa_numeric(x, scale=10.0))
    out["dewpoint"] = col("DEW").apply(lambda x: _parse_noaa_numeric(x, scale=10.0))
    out["pressure"] = col("SLP").apply(lambda x: _parse_noaa_numeric(x, scale=10.0))
    out["visibility"] = col("VISIB").apply(lambda x: _parse_noaa_numeric(x, scale=1.0))
    out["wind_speed"] = col("WDSP").apply(lambda x: _parse_noaa_numeric(x, scale=10.0))
    out["wind_gust"] = col("MXSPD").apply(lambda x: _parse_noaa_numeric(x, scale=10.0))
    out["precipitation"] = col("PRCP").apply(lambda x: _parse_noaa_numeric(x, scale=10.0))
    out["wind_dir"] = col("WND").apply(lambda x: _parse_noaa_numeric(str(x).split(",")[0], scale=1.0))

    # Relative humidity from temp/dewpoint approximation
    t = out["temperature"]
    td = out["dewpoint"]
    es = np.exp((17.625 * t) / (243.04 + t))
    ed = np.exp((17.625 * td) / (243.04 + td))
    out["relative_humidity"] = 100.0 * ed / es

    out = out.dropna(subset=["time"]).copy()
    out = out.sort_values("time").groupby(pd.Grouper(key="time", freq="1H")).mean(numeric_only=True).reset_index()
    return out


def _fetch_openmeteo_hourly(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
        "soil_moisture_0_to_1cm",
        "precipitation",
        "visibility",
    ]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(hourly_vars),
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    payload = r.json()
    if "hourly" not in payload:
        raise RuntimeError(f"Open-Meteo response missing hourly: {payload}")

    hourly = payload["hourly"]
    out = pd.DataFrame({"time": pd.to_datetime(hourly["time"], utc=True)})
    for k, v in hourly.items():
        if k == "time":
            continue
        out[k] = v
    return out


def _fetch_openmeteo_elevation(lat: float, lon: float) -> float:
    url = "https://api.open-meteo.com/v1/elevation"
    params = {"latitude": lat, "longitude": lon}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    elevs = payload.get("elevation", [np.nan])
    return float(elevs[0]) if elevs else float("nan")


def download_all(config: Dict, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    start = config["region"]["start_date"]
    end = config["region"]["end_date"]
    specs = [StationSpec(**x) for x in config["region"]["stations"]]
    source_lat = float(config["region"]["source_proxy_point"]["lat"])
    source_lon = float(config["region"]["source_proxy_point"]["lon"])

    obs_frames: List[pd.DataFrame] = []
    bg_frames: List[pd.DataFrame] = []
    station_meta: List[Dict] = []
    static_rows: List[Dict] = []

    for spec in specs:
        obs = _download_noaa_global_hourly(spec.noaa_station, start, end)
        obs["station_id"] = spec.noaa_station
        obs["station_name"] = spec.name
        obs["lat"] = spec.lat
        obs["lon"] = spec.lon
        obs_frames.append(obs)

        bg = _fetch_openmeteo_hourly(spec.lat, spec.lon, start, end)
        bg = bg.rename(
            columns={
                "temperature_2m": "bg_temperature",
                "relative_humidity_2m": "bg_relative_humidity",
                "surface_pressure": "bg_surface_pressure",
                "wind_speed_10m": "bg_wind_speed",
                "wind_direction_10m": "bg_wind_dir",
                "wind_gusts_10m": "bg_wind_gust",
                "soil_moisture_0_to_1cm": "bg_soil_moisture",
                "precipitation": "bg_precipitation",
                "visibility": "bg_visibility",
            }
        )
        for c in [
            "bg_temperature",
            "bg_relative_humidity",
            "bg_surface_pressure",
            "bg_wind_speed",
            "bg_wind_dir",
            "bg_wind_gust",
            "bg_soil_moisture",
            "bg_precipitation",
            "bg_visibility",
        ]:
            if c not in bg.columns:
                bg[c] = np.nan
        bg["station_id"] = spec.noaa_station
        bg["station_name"] = spec.name
        bg["lat"] = spec.lat
        bg["lon"] = spec.lon
        bg_frames.append(bg)

        elevation = _fetch_openmeteo_elevation(spec.lat, spec.lon)
        dist_to_source = haversine_km(spec.lat, spec.lon, source_lat, source_lon)
        station_meta.append(
            {
                "station_id": spec.noaa_station,
                "station_name": spec.name,
                "lat": spec.lat,
                "lon": spec.lon,
                "elevation": elevation,
            }
        )
        static_rows.append(
            {
                "station_id": spec.noaa_station,
                "station_name": spec.name,
                "lat": spec.lat,
                "lon": spec.lon,
                "elevation": elevation,
                "distance_to_source_km": dist_to_source,
            }
        )

    obs_df = pd.concat(obs_frames, ignore_index=True).sort_values(["time", "station_id"])
    bg_df = pd.concat(bg_frames, ignore_index=True).sort_values(["time", "station_id"])
    static_df = pd.DataFrame(static_rows)
    station_meta_df = pd.DataFrame(station_meta)

    obs_df.to_parquet(output / "station_observations.parquet", index=False)
    bg_df.to_parquet(output / "background_openmeteo.parquet", index=False)
    static_df.to_csv(output / "static_features.csv", index=False)
    station_meta_df.to_csv(output / "station_meta.csv", index=False)

    with (output / "download_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "region": config["region"]["name"],
                "start_date": start,
                "end_date": end,
                "stations": station_meta,
                "source": ["NOAA NCEI Global Hourly", "Open-Meteo Archive API", "Open-Meteo Elevation API"],
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))
