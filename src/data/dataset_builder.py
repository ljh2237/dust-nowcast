from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.downloader import haversine_km


def _wind_to_uv(speed: pd.Series, direction_deg: pd.Series) -> Tuple[pd.Series, pd.Series]:
    rad = np.deg2rad(direction_deg)
    u = -speed * np.sin(rad)
    v = -speed * np.cos(rad)
    return u, v


def _risk_label(score: np.ndarray, thresholds: List[float]) -> np.ndarray:
    out = np.zeros_like(score, dtype=np.int64)
    for i, th in enumerate(thresholds):
        out[score >= th] = i + 1
    return out


def _build_adj(stations: pd.DataFrame) -> np.ndarray:
    n = len(stations)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            d = haversine_km(
                float(stations.iloc[i]["lat"]),
                float(stations.iloc[i]["lon"]),
                float(stations.iloc[j]["lat"]),
                float(stations.iloc[j]["lon"]),
            )
            adj[i, j] = np.exp(-d / 450.0)
    row_sum = adj.sum(axis=1, keepdims=True)
    adj = adj / np.clip(row_sum, 1e-8, None)
    return adj.astype(np.float32)


def build_processed_dataset(config: Dict, raw_dir: str | Path, processed_dir: str | Path) -> None:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    obs = pd.read_parquet(raw_dir / "station_observations.parquet")
    bg = pd.read_parquet(raw_dir / "background_openmeteo.parquet")
    static = pd.read_csv(raw_dir / "static_features.csv")

    obs["station_id"] = obs["station_id"].astype(str)
    bg["station_id"] = bg["station_id"].astype(str)
    static["station_id"] = static["station_id"].astype(str)

    obs["time"] = pd.to_datetime(obs["time"], utc=True)
    bg["time"] = pd.to_datetime(bg["time"], utc=True)

    df = obs.merge(
        bg[
            [
                "time",
                "station_id",
                "bg_temperature",
                "bg_relative_humidity",
                "bg_surface_pressure",
                "bg_wind_speed",
                "bg_wind_dir",
                "bg_wind_gust",
                "bg_soil_moisture",
                "bg_precipitation",
                "bg_visibility",
            ]
        ],
        on=["time", "station_id"],
        how="left",
    )
    df = df.merge(static, on=["station_id", "station_name", "lat", "lon"], how="left")

    if "pressure" not in df.columns:
        df["pressure"] = np.nan
    if "visibility" not in df.columns:
        df["visibility"] = np.nan

    df["wind_speed"] = df["wind_speed"].fillna(df["bg_wind_speed"])
    df["wind_dir"] = df["wind_dir"].fillna(df["bg_wind_dir"])
    df["temperature"] = df["temperature"].fillna(df["bg_temperature"])
    df["relative_humidity"] = df["relative_humidity"].fillna(df["bg_relative_humidity"])
    df["pressure"] = df["pressure"].fillna(df["bg_surface_pressure"])
    df["precipitation"] = df["precipitation"].fillna(df["bg_precipitation"])
    df["visibility"] = df["visibility"].fillna(df["bg_visibility"])

    df["u10"], df["v10"] = _wind_to_uv(df["wind_speed"], df["wind_dir"])
    df["bg_u10"], df["bg_v10"] = _wind_to_uv(df["bg_wind_speed"].fillna(0), df["bg_wind_dir"].fillna(0))
    df["hour_sin"] = np.sin(2 * np.pi * df["time"].dt.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["time"].dt.hour / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["time"].dt.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["time"].dt.dayofyear / 365.25)

    df = df.sort_values(["station_id", "time"]).reset_index(drop=True)
    fill_cols = [
        "wind_speed",
        "wind_dir",
        "temperature",
        "relative_humidity",
        "pressure",
        "precipitation",
        "visibility",
        "bg_wind_speed",
        "bg_relative_humidity",
        "bg_surface_pressure",
        "bg_wind_gust",
        "bg_soil_moisture",
        "bg_visibility",
        "u10",
        "v10",
        "bg_u10",
        "bg_v10",
    ]
    for c in fill_cols:
        df[c] = df.groupby("station_id")[c].transform(lambda s: s.interpolate(limit_direction="both"))
        med = float(df[c].median()) if not np.isnan(df[c].median()) else 0.0
        df[c] = df[c].fillna(med)

    features = [
        "wind_speed",
        "u10",
        "v10",
        "temperature",
        "relative_humidity",
        "pressure",
        "visibility",
        "precipitation",
        "bg_wind_speed",
        "bg_wind_gust",
        "bg_u10",
        "bg_v10",
        "bg_temperature",
        "bg_relative_humidity",
        "bg_surface_pressure",
        "bg_soil_moisture",
        "bg_visibility",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ]

    static_numeric_cols = [
        c
        for c in static.columns
        if c not in {"station_id", "station_name", "nearest_source"} and np.issubdtype(static[c].dtype, np.number)
    ]
    static_features = static_numeric_cols

    stations = df[["station_id", "station_name", "lat", "lon"]].drop_duplicates().reset_index(drop=True)
    station_ids = stations["station_id"].tolist()
    station_to_idx = {s: i for i, s in enumerate(station_ids)}

    all_times = pd.DataFrame({"time": pd.date_range(df["time"].min(), df["time"].max(), freq="1H", tz="UTC")})
    panel = []
    for sid in station_ids:
        sdf = df[df["station_id"] == sid].merge(all_times, on="time", how="right").sort_values("time")
        sdf["station_id"] = sid
        sdf["station_name"] = stations.loc[station_to_idx[sid], "station_name"]
        sdf["lat"] = stations.loc[station_to_idx[sid], "lat"]
        sdf["lon"] = stations.loc[station_to_idx[sid], "lon"]
        for c in features + ["wind_speed", "relative_humidity", "visibility", "bg_soil_moisture"]:
            if c not in sdf.columns:
                sdf[c] = np.nan
            sdf[c] = sdf[c].interpolate(limit_direction="both")
            med = float(sdf[c].median()) if not np.isnan(sdf[c].median()) else 0.0
            sdf[c] = sdf[c].fillna(med)
        static_row = static[static["station_id"] == sid].iloc[0]
        for c in static_features:
            sdf[c] = static_row[c]
        panel.append(sdf)
    panel_df = pd.concat(panel, ignore_index=True)

    horizons = config["dataset"]["horizons"]
    seq_len = int(config["dataset"]["seq_len"])
    wind_warn_th = float(config["dataset"]["wind_warning_threshold"])
    risk_th = [float(x) for x in config["dataset"]["risk_class_thresholds"]]
    spring_months = set(int(x) for x in config["dataset"].get("spring_months", [3, 4, 5]))
    rw = config["dataset"].get("risk_weights", {})

    times = sorted(panel_df["time"].unique())
    n_time = len(times)
    n_station = len(station_ids)
    n_feat = len(features)

    feat_arr = np.zeros((n_time, n_station, n_feat), dtype=np.float32)
    static_arr = np.zeros((n_station, len(static_features)), dtype=np.float32)
    wind_arr = np.zeros((n_time, n_station), dtype=np.float32)
    rh_arr = np.zeros((n_time, n_station), dtype=np.float32)
    vis_arr = np.zeros((n_time, n_station), dtype=np.float32)
    soil_arr = np.zeros((n_time, n_station), dtype=np.float32)

    time_to_idx = {t: i for i, t in enumerate(times)}
    for sid in station_ids:
        si = station_to_idx[sid]
        sdf = panel_df[panel_df["station_id"] == sid].sort_values("time")
        idx = [time_to_idx[t] for t in sdf["time"]]
        feat_arr[idx, si, :] = sdf[features].to_numpy(dtype=np.float32)
        wind_arr[idx, si] = sdf["wind_speed"].to_numpy(dtype=np.float32)
        rh_arr[idx, si] = sdf["relative_humidity"].to_numpy(dtype=np.float32)
        vis_arr[idx, si] = sdf["visibility"].to_numpy(dtype=np.float32)
        soil_arr[idx, si] = sdf["bg_soil_moisture"].to_numpy(dtype=np.float32)
        static_arr[si, :] = sdf[static_features].iloc[0].to_numpy(dtype=np.float32)

    source_col = "source_proximity" if "source_proximity" in static_features else None
    source_vec = None
    if source_col:
        source_vec = static_arr[:, static_features.index(source_col)]
    else:
        source_vec = np.zeros((n_station,), dtype=np.float32)

    X_list = []
    y_wind_list = []
    y_risk_list = []
    y_warn_list = []
    sample_times = []

    max_h = max(horizons)
    for t in range(seq_len, n_time - max_h):
        x = feat_arr[t - seq_len : t, :, :]
        yw = []
        yr = []
        yb = []

        month = pd.Timestamp(times[t]).month
        spring_factor = 1.0 if month in spring_months else 0.0

        for h in horizons:
            fut_wind = wind_arr[t + h, :]
            wind_norm = np.clip(fut_wind / 20.0, 0.0, 1.5)
            dry_score = (rh_arr[t - 1, :] < 35.0).astype(float)
            vis_score = (vis_arr[t - 1, :] < 8000.0).astype(float)
            soil_dry = (soil_arr[t - 1, :] < 0.18).astype(float)
            src = np.clip(source_vec, 0.0, 1.0)
            score = (
                float(rw.get("wind", 0.45)) * wind_norm
                + float(rw.get("dry", 0.16)) * dry_score
                + float(rw.get("visibility", 0.10)) * vis_score
                + float(rw.get("soil", 0.10)) * soil_dry
                + float(rw.get("spring", 0.07)) * spring_factor
                + float(rw.get("source_proximity", 0.12)) * src
            )
            fut_risk = _risk_label(score, risk_th)
            high_risk_level = len(risk_th)
            fut_warn = ((fut_wind >= wind_warn_th) | (fut_risk >= high_risk_level - 1)).astype(np.float32)

            yw.append(fut_wind)
            yr.append(fut_risk)
            yb.append(fut_warn)

        X_list.append(x)
        y_wind_list.append(np.stack(yw, axis=1))
        y_risk_list.append(np.stack(yr, axis=1))
        y_warn_list.append(np.stack(yb, axis=1))
        sample_times.append(times[t])

    X = np.stack(X_list, axis=0).astype(np.float32)
    y_wind = np.stack(y_wind_list, axis=0).astype(np.float32)
    y_risk = np.stack(y_risk_list, axis=0).astype(np.int64)
    y_warn = np.stack(y_warn_list, axis=0).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_wind = np.nan_to_num(y_wind, nan=0.0, posinf=0.0, neginf=0.0)
    y_warn = np.nan_to_num(y_warn, nan=0.0, posinf=0.0, neginf=0.0)

    adj = _build_adj(stations)

    n_samples = X.shape[0]
    i_train = int(n_samples * float(config["dataset"]["split"]["train"]))
    i_val = int(n_samples * (float(config["dataset"]["split"]["train"]) + float(config["dataset"]["split"]["val"])))
    split_idx = {
        "train": np.arange(0, i_train),
        "val": np.arange(i_train, i_val),
        "test": np.arange(i_val, n_samples),
    }

    np.savez_compressed(
        processed_dir / "dataset_tensors.npz",
        X=X,
        X_static=static_arr,
        y_wind=y_wind,
        y_risk=y_risk,
        y_warn=y_warn,
        adj=adj,
        train_idx=split_idx["train"],
        val_idx=split_idx["val"],
        test_idx=split_idx["test"],
    )

    h_idx = horizons.index(3) if 3 in horizons else 0
    rows = []
    for n in range(n_samples):
        for si, sid in enumerate(station_ids):
            seq = X[n, :, si, :]
            row = {
                "sample_idx": n,
                "time": str(sample_times[n]),
                "station_id": sid,
                "y_wind_3h": float(y_wind[n, si, h_idx]),
                "y_risk_3h": int(y_risk[n, si, h_idx]),
                "y_warn_3h": int(y_warn[n, si, h_idx]),
            }
            row.update({f"last_{features[j]}": float(seq[-1, j]) for j in range(n_feat)})
            row.update({f"mean_{features[j]}": float(seq[:, j].mean()) for j in range(n_feat)})
            row.update({f"std_{features[j]}": float(seq[:, j].std()) for j in range(n_feat)})
            row.update({f"static_{static_features[j]}": float(static_arr[si, j]) for j in range(len(static_features))})
            rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_parquet(processed_dir / "dataset_tabular.parquet", index=False)

    risk_flat = y_risk.reshape(-1)
    risk_counts = {str(int(k)): int(v) for k, v in zip(*np.unique(risk_flat, return_counts=True))}

    meta = {
        "region": config["region"]["name"],
        "region_display": config["region"].get("display_name", config["region"]["name"]),
        "features": features,
        "static_features": static_features,
        "horizons": horizons,
        "seq_len": seq_len,
        "stations": stations.to_dict(orient="records"),
        "num_samples": int(n_samples),
        "risk_num_classes": int(risk_flat.max() + 1),
        "risk_label_distribution": risk_counts,
        "tensor_shape": {
            "X": list(X.shape),
            "y_wind": list(y_wind.shape),
            "y_risk": list(y_risk.shape),
            "y_warn": list(y_warn.shape),
        },
    }
    with (processed_dir / "dataset_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
