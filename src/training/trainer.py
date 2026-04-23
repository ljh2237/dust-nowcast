from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.evaluation.metrics import binary_metrics, classification_metrics, confusion, regression_metrics
from src.evaluation.plots import save_confusion_matrix, save_loss_curve, save_pred_vs_true
from src.models.baselines import CNNLSTMBaseline, LSTMBaseline, train_rf_baseline, train_xgboost_baseline
from src.models.dustriskformer import DustRiskFormer, multitask_loss
from src.training.datasets import load_dataset_npz, make_dataloaders


def _device_from_config(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _eval_epoch(model, loader, device, x_static, adj, main_model: bool = True):
    model.eval()
    total = 0.0
    count = 0
    preds_w, trues_w = [], []
    preds_r, trues_r = [], []
    preds_b, trues_b, probs_b = [], [], []

    with torch.no_grad():
        for x, y_w, y_r, y_b in loader:
            x = x.to(device)
            y_w = y_w.to(device)
            y_r = y_r.to(device)
            y_b = y_b.to(device)

            if main_model:
                out = model(x, x_static, adj)
            else:
                out = model(x)

            losses = multitask_loss(out, y_w, y_r, y_b)
            total += float(losses["total"].item()) * x.size(0)
            count += x.size(0)

            preds_w.append(out["wind"].cpu().numpy())
            trues_w.append(y_w.cpu().numpy())
            preds_r.append(out["risk_logits"].argmax(dim=-1).cpu().numpy())
            trues_r.append(y_r.cpu().numpy())
            p = torch.sigmoid(out["warn_logit"]).cpu().numpy()
            probs_b.append(p)
            preds_b.append((p >= 0.5).astype(np.int64))
            trues_b.append(y_b.cpu().numpy().astype(np.int64))

    return {
        "loss": total / max(count, 1),
        "y_w_true": np.concatenate(trues_w, axis=0),
        "y_w_pred": np.concatenate(preds_w, axis=0),
        "y_r_true": np.concatenate(trues_r, axis=0),
        "y_r_pred": np.concatenate(preds_r, axis=0),
        "y_b_true": np.concatenate(trues_b, axis=0),
        "y_b_pred": np.concatenate(preds_b, axis=0),
        "y_b_prob": np.concatenate(probs_b, axis=0),
    }


def train_deep_model(config: Dict, model_name: str = "dustriskformer") -> Dict:
    processed_dir = Path(config["paths"]["processed_dir"])
    results_dir = Path(config["paths"]["results_dir"])
    ckpt_dir = Path(config["paths"]["checkpoints_dir"])
    (results_dir / "plots").mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset_npz(processed_dir / "dataset_tensors.npz")
    train_loader, val_loader, test_loader = make_dataloaders(
        bundle, batch_size=int(config["training"]["batch_size"]), num_workers=int(config["training"]["num_workers"])
    )

    device = _device_from_config(config["training"]["device"])
    x_static = torch.tensor(bundle.X_static, dtype=torch.float32, device=device)
    adj = torch.tensor(bundle.adj, dtype=torch.float32, device=device)

    in_dim = bundle.X.shape[-1]
    horizons = bundle.y_wind.shape[-1]

    if model_name == "dustriskformer":
        model = DustRiskFormer(
            in_dim=in_dim,
            static_dim=bundle.X_static.shape[-1],
            hidden_dim=int(config["model"]["hidden_dim"]),
            num_heads=int(config["model"]["num_heads"]),
            horizons=horizons,
            dropout=float(config["model"]["dropout"]),
        )
        main_model = True
    elif model_name == "lstm":
        model = LSTMBaseline(in_dim=in_dim, hidden_dim=64, horizons=horizons)
        main_model = False
    elif model_name == "cnn_lstm":
        model = CNNLSTMBaseline(in_dim=in_dim, hidden_dim=64, horizons=horizons)
        main_model = False
    else:
        raise ValueError(f"Unknown deep model: {model_name}")

    model = model.to(device)
    optim = Adam(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))

    best_val = float("inf")
    best_path = ckpt_dir / f"{model_name}_best.pt"
    patience = int(config["training"]["patience"])
    wait = 0
    train_losses, val_losses = [], []

    for epoch in range(int(config["training"]["epochs"])):
        model.train()
        run_loss = 0.0
        seen = 0
        for x, y_w, y_r, y_b in train_loader:
            x = x.to(device)
            y_w = y_w.to(device)
            y_r = y_r.to(device)
            y_b = y_b.to(device)

            optim.zero_grad()
            if main_model:
                out = model(x, x_static, adj)
            else:
                out = model(x)
            losses = multitask_loss(out, y_w, y_r, y_b)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            run_loss += float(losses["total"].item()) * x.size(0)
            seen += x.size(0)

        train_loss = run_loss / max(seen, 1)
        val_pack = _eval_epoch(model, val_loader, device, x_static, adj, main_model)
        val_loss = val_pack["loss"]
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"[{model_name}] epoch={epoch+1} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"[{model_name}] early stopping at epoch {epoch+1}")
                break

    if not best_path.exists():
        torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_pack = _eval_epoch(model, test_loader, device, x_static, adj, main_model)

    y_w_true = test_pack["y_w_true"].reshape(-1)
    y_w_pred = test_pack["y_w_pred"].reshape(-1)
    y_r_true = test_pack["y_r_true"].reshape(-1)
    y_r_pred = test_pack["y_r_pred"].reshape(-1)
    y_b_true = test_pack["y_b_true"].reshape(-1)
    y_b_pred = test_pack["y_b_pred"].reshape(-1)
    y_b_prob = test_pack["y_b_prob"].reshape(-1)

    metrics = {
        "model": model_name,
        "val_loss_best": best_val,
        "regression": regression_metrics(y_w_true, y_w_pred),
        "risk_classification": classification_metrics(y_r_true, y_r_pred),
        "warning_binary": binary_metrics(y_b_true, y_b_pred, y_b_prob),
    }

    save_loss_curve(train_losses, val_losses, results_dir / "plots" / f"loss_curve_{model_name}.png")
    save_pred_vs_true(y_w_true, y_w_pred, results_dir / "plots" / f"pred_vs_true_{model_name}.png", f"{model_name} wind prediction")
    cm = confusion(y_r_true, y_r_pred)
    save_confusion_matrix(cm, results_dir / "plots" / f"confusion_risk_{model_name}.png", f"{model_name} risk confusion")

    pred_df = pd.DataFrame(
        {
            "y_wind_true": y_w_true,
            "y_wind_pred": y_w_pred,
            "y_risk_true": y_r_true,
            "y_risk_pred": y_r_pred,
            "y_warn_true": y_b_true,
            "y_warn_pred": y_b_pred,
            "y_warn_prob": y_b_prob,
        }
    )
    pred_df.to_csv(results_dir / f"predictions_{model_name}.csv", index=False)

    with (results_dir / f"metrics_{model_name}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save attention tensors for explainability on main model
    if model_name == "dustriskformer":
        model.eval()
        x0, _, _, _ = next(iter(test_loader))
        x0 = x0[:1].to(device)
        with torch.no_grad():
            out = model(x0, x_static, adj)
        np.save(results_dir / "temporal_attention.npy", out["temporal_attention"].cpu().numpy())
        np.save(results_dir / "graph_attention.npy", out["graph_attention"].cpu().numpy())

    return metrics


def train_ml_baselines(config: Dict) -> Dict:
    processed_dir = Path(config["paths"]["processed_dir"])
    results_dir = Path(config["paths"]["results_dir"])
    (results_dir / "plots").mkdir(parents=True, exist_ok=True)

    tab = pd.read_parquet(processed_dir / "dataset_tabular.parquet")
    tensor = np.load(processed_dir / "dataset_tensors.npz")
    train_idx = set(tensor["train_idx"].tolist())
    test_idx = set(tensor["test_idx"].tolist())

    feats = [c for c in tab.columns if c.startswith("last_") or c.startswith("mean_") or c.startswith("std_") or c.startswith("static_")]
    train_df = tab[tab["sample_idx"].isin(train_idx)]
    test_df = tab[tab["sample_idx"].isin(test_idx)]

    X_train = train_df[feats].to_numpy()
    X_test = test_df[feats].to_numpy()

    y_w_train = train_df[["y_wind_3h"]].to_numpy()
    y_w_test = test_df[["y_wind_3h"]].to_numpy()

    y_r_train = train_df[["y_risk_3h"]].to_numpy()
    y_r_test = test_df[["y_risk_3h"]].to_numpy()

    y_b_train = train_df[["y_warn_3h"]].to_numpy()
    y_b_test = test_df[["y_warn_3h"]].to_numpy()

    bundle_rf = train_rf_baseline(X_train, y_w_train, y_r_train, y_b_train)
    joblib.dump(bundle_rf.reg_model, results_dir / "rf_reg.pkl")
    joblib.dump(bundle_rf.risk_model, results_dir / "rf_risk.pkl")
    joblib.dump(bundle_rf.warn_model, results_dir / "rf_warn.pkl")
    rf_w = bundle_rf.reg_model.predict(X_test).reshape(-1)
    rf_r = bundle_rf.risk_model.predict(X_test).reshape(-1)
    rf_b = bundle_rf.warn_model.predict(X_test).reshape(-1)

    metrics_rf = {
        "model": "random_forest",
        "regression": regression_metrics(y_w_test.reshape(-1), rf_w),
        "risk_classification": classification_metrics(y_r_test.reshape(-1), rf_r),
        "warning_binary": binary_metrics(y_b_test.reshape(-1), rf_b),
    }

    out = {"random_forest": metrics_rf}

    try:
        bundle_xgb = train_xgboost_baseline(X_train, y_w_train, y_r_train, y_b_train)
        joblib.dump(bundle_xgb.reg_model, results_dir / "xgb_reg.pkl")
        joblib.dump(bundle_xgb.risk_model, results_dir / "xgb_risk.pkl")
        joblib.dump(bundle_xgb.warn_model, results_dir / "xgb_warn.pkl")
        xgb_w = bundle_xgb.reg_model.predict(X_test).reshape(-1)
        xgb_r = bundle_xgb.risk_model.predict(X_test).reshape(-1)
        xgb_b = bundle_xgb.warn_model.predict(X_test).reshape(-1)
        out["xgboost"] = {
            "model": "xgboost",
            "regression": regression_metrics(y_w_test.reshape(-1), xgb_w),
            "risk_classification": classification_metrics(y_r_test.reshape(-1), xgb_r),
            "warning_binary": binary_metrics(y_b_test.reshape(-1), xgb_b),
        }
    except Exception as e:
        out["xgboost"] = {"error": str(e)}

    with (results_dir / "metrics_baselines_ml.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out
