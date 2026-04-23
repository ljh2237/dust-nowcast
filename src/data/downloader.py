from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


@dataclass
class StationSpec:
    name: str
    lat: float
    lon: float
    noaa_station: Optional[str] = None


def _safe_get_json(url: str, params: Dict, timeout: int = 60, retries: int = 3) -> Dict:
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(1.2 * (i + 1))
    if last_err is not None:
        raise last_err
    raise RuntimeError("Unknown request error")


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
    rows = _safe_get_json(url, params=params, timeout=90, retries=4)
    if not isinstance(rows, list) or len(rows) == 0:
        raise RuntimeError(f"NOAA global-hourly empty for station={station_id}")

    df = pd.DataFrame(rows)

    def col(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series([np.nan] * len(df))

    out = pd.DataFrame()
    out["time"] = pd.to_datetime(df["DATE"], utc=True, errors="coerce")
    out["temperature"] = col("TMP").apply(lambda x: _parse_noaa_numeric(x, scale=10.0))
    out["dewpoint"] = col("DEW").apply(lambda x: _parse_noaa_numeric(x, scale=10.0))
    out["pressure"] = col("SLP").apply(lambda x: _parse_noaa_numeric(x, scale=10.0))
    out["visibility"] = col("VISIB").apply(lambda x: _parse_noaa_numeric(x, scale=1.0))
    out["wind_speed"] = col("WDSP").apply(lambda x: _parse_noaa_numeric(x, scale=10.0))
    out["wind_gust"] = col("MXSPD").apply(lambda x: _parse_noaa_numeric(x, scale=10.0))
    out["precipitation"] = col("PRCP").apply(lambda x: _parse_noaa_numeric(x, scale=10.0))
    out["wind_dir"] = col("WND").apply(lambda x: _parse_noaa_numeric(str(x).split(",")[0], scale=1.0))

    t = out["temperature"]
    td = out["dewpoint"]
    es = np.exp((17.625 * t) / (243.04 + t))
    ed = np.exp((17.625 * td) / (243.04 + td))
    out["relative_humidity"] = np.clip(100.0 * ed / es, 0.0, 100.0)

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
    payload = _safe_get_json(url, params=params, timeout=90, retries=4)
    if "hourly" not in payload:
        raise RuntimeError(f"Open-Meteo response missing hourly: {payload}")

    hourly = payload["hourly"]
    out = pd.DataFrame({"time": pd.to_datetime(hourly["time"], utc=True)})
    for k, v in hourly.items():
        if k == "time":
            continue
        out[k] = v
    return out


def _openmeteo_as_observation(bg_df: pd.DataFrame) -> pd.DataFrame:
    obs = pd.DataFrame()
    obs["time"] = bg_df["time"]
    obs["wind_speed"] = bg_df["bg_wind_speed"]
    obs["wind_dir"] = bg_df["bg_wind_dir"]
    obs["temperature"] = bg_df["bg_temperature"]
    obs["relative_humidity"] = bg_df["bg_relative_humidity"]
    obs["pressure"] = bg_df["bg_surface_pressure"]
    obs["precipitation"] = bg_df["bg_precipitation"]
    obs["visibility"] = bg_df["bg_visibility"]
    obs["wind_gust"] = bg_df["bg_wind_gust"]
    return obs


def _fetch_openmeteo_elevation(lat: float, lon: float) -> float:
    url = "https://api.open-meteo.com/v1/elevation"
    params = {"latitude": lat, "longitude": lon}
    try:
        payload = _safe_get_json(url, params=params, timeout=30, retries=4)
        elevs = payload.get("elevation", [np.nan])
        return float(elevs[0]) if elevs else float("nan")
    except Exception:
        return float("nan")


def _terrain_roughness_proxy(lat: float, lon: float, delta: float = 0.15) -> float:
    pts = [(lat, lon), (lat + delta, lon), (lat - delta, lon), (lat, lon + delta), (lat, lon - delta)]
    vals = []
    for la, lo in pts:
        vals.append(_fetch_openmeteo_elevation(la, lo))
    arr = np.array(vals, dtype=np.float32)
    return float(np.nanstd(arr))


def _nearest_source_distance(lat: float, lon: float, sources: List[Dict]) -> Tuple[float, str]:
    dists = []
    for s in sources:
        d = haversine_km(lat, lon, float(s["lat"]), float(s["lon"]))
        dists.append((d, str(s["name"])))
    dists = sorted(dists, key=lambda x: x[0])
    return float(dists[0][0]), dists[0][1]


def download_all(config: Dict, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    start = config["region"]["start_date"]
    end = config["region"]["end_date"]
    specs = [StationSpec(**x) for x in config["region"]["stations"]]
    sources = config["region"]["dust_source_points"]

    obs_frames: List[pd.DataFrame] = []
    bg_frames: List[pd.DataFrame] = []
    station_meta: List[Dict] = []
    static_rows: List[Dict] = []
    noaa_ok = 0
    openmeteo_fallback = 0

    for spec in specs:
        sid = f"{spec.noaa_station}_{spec.name}" if spec.noaa_station else f"OM_{spec.name}"

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
        bg["station_id"] = sid
        bg["station_name"] = spec.name
        bg["lat"] = spec.lat
        bg["lon"] = spec.lon
        bg_frames.append(bg)

        obs_source = "openmeteo_fallback"
        obs = None
        if spec.noaa_station:
            try:
                obs = _download_noaa_global_hourly(spec.noaa_station, start, end)
                obs_source = "noaa_global_hourly"
                noaa_ok += 1
            except Exception:
                obs = None

        if obs is None or len(obs) < 24 * 7:
            obs = _openmeteo_as_observation(bg)
            openmeteo_fallback += 1
            obs_source = "openmeteo_fallback"

        obs["station_id"] = sid
        obs["station_name"] = spec.name
        obs["lat"] = spec.lat
        obs["lon"] = spec.lon
        obs["obs_source"] = obs_source
        obs_frames.append(obs)

        elevation = _fetch_openmeteo_elevation(spec.lat, spec.lon)
        terrain_roughness = _terrain_roughness_proxy(spec.lat, spec.lon)
        dist_to_source, source_name = _nearest_source_distance(spec.lat, spec.lon, sources)
        source_proximity = float(np.exp(-dist_to_source / 400.0))

        station_meta.append(
            {
                "station_id": sid,
                "station_name": spec.name,
                "lat": spec.lat,
                "lon": spec.lon,
                "elevation": elevation,
                "obs_source": obs_source,
            }
        )
        static_rows.append(
            {
                "station_id": sid,
                "station_name": spec.name,
                "lat": spec.lat,
                "lon": spec.lon,
                "elevation": elevation,
                "terrain_roughness": terrain_roughness,
                "distance_to_source_km": dist_to_source,
                "nearest_source": source_name,
                "source_proximity": source_proximity,
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
                "region_display": config["region"].get("display_name", config["region"]["name"]),
                "start_date": start,
                "end_date": end,
                "stations": station_meta,
                "source": ["NOAA NCEI Global Hourly (priority)", "Open-Meteo Archive API", "Open-Meteo Elevation API"],
                "download_stats": {
                    "noaa_success_station_count": noaa_ok,
                    "openmeteo_fallback_station_count": openmeteo_fallback,
                },
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
