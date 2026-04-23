from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except Exception:
    HAS_XGB = False


class LSTMBaseline(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, horizons: int, num_risk_classes: int = 3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.reg_head = nn.Linear(hidden_dim, horizons)
        self.risk_head = nn.Linear(hidden_dim, horizons * num_risk_classes)
        self.warn_head = nn.Linear(hidden_dim, horizons)
        self.horizons = horizons

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, T, N, F] -> use target-node independent formulation by flattening nodes
        b, t, n, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * n, t, f)
        h, _ = self.lstm(x)
        z = h[:, -1, :]
        reg = self.reg_head(z).reshape(b, n, self.horizons)
        risk = self.risk_head(z).reshape(b, n, self.horizons, -1)
        warn = self.warn_head(z).reshape(b, n, self.horizons)
        return {"wind": reg, "risk_logits": risk, "warn_logit": warn}


class CNNLSTMBaseline(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, horizons: int, num_risk_classes: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.reg_head = nn.Linear(hidden_dim, horizons)
        self.risk_head = nn.Linear(hidden_dim, horizons * num_risk_classes)
        self.warn_head = nn.Linear(hidden_dim, horizons)
        self.horizons = horizons

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, t, n, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * n, t, f)
        conv_in = x.permute(0, 2, 1)
        conv_out = torch.relu(self.conv(conv_in)).permute(0, 2, 1)
        h, _ = self.lstm(conv_out)
        z = h[:, -1, :]
        reg = self.reg_head(z).reshape(b, n, self.horizons)
        risk = self.risk_head(z).reshape(b, n, self.horizons, -1)
        warn = self.warn_head(z).reshape(b, n, self.horizons)
        return {"wind": reg, "risk_logits": risk, "warn_logit": warn}


@dataclass
class MLBaselineBundle:
    reg_model: object
    risk_model: object
    warn_model: object


def train_xgboost_baseline(X_train: np.ndarray, y_wind: np.ndarray, y_risk: np.ndarray, y_warn: np.ndarray) -> MLBaselineBundle:
    if not HAS_XGB:
        raise RuntimeError("xgboost not installed. Please install xgboost for this baseline.")

    reg = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
    )
    risk = MultiOutputClassifier(
        XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=42,
        )
    )
    warn = MultiOutputClassifier(
        XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
    )

    reg.fit(X_train, y_wind)
    risk.fit(X_train, y_risk)
    warn.fit(X_train, y_warn)
    return MLBaselineBundle(reg_model=reg, risk_model=risk, warn_model=warn)


def train_rf_baseline(X_train: np.ndarray, y_wind: np.ndarray, y_risk: np.ndarray, y_warn: np.ndarray) -> MLBaselineBundle:
    reg = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=1), n_jobs=1)
    risk = MultiOutputClassifier(RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=1), n_jobs=1)
    warn = MultiOutputClassifier(RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=1), n_jobs=1)
    reg.fit(X_train, y_wind)
    risk.fit(X_train, y_risk)
    warn.fit(X_train, y_warn)
    return MLBaselineBundle(reg_model=reg, risk_model=risk, warn_model=warn)
