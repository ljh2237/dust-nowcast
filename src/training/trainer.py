from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import recall_score
from torch.optim import Adam

from src.evaluation.metrics import binary_metrics, classification_metrics, confusion, regression_metrics
from src.evaluation.plots import save_confusion_matrix, save_loss_curve, save_pred_vs_true
from src.models.baselines import (
    AttentionTCNLSTMBaseline,
    CNNLSTMBaseline,
    LSTMBaseline,
    train_rf_baseline,
    train_xgboost_baseline,
)
from src.models.dustriskformer import DustRiskFormer, multitask_loss
from src.training.datasets import load_dataset_npz, make_dataloaders


def _device_from_config(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _eval_epoch(
    model,
    loader,
    device,
    x_static,
    adj,
    main_model: bool = True,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    risk_class_weights: torch.Tensor | None = None,
    warn_pos_weight: torch.Tensor | None = None,
    warn_focal_gamma: float = 0.0,
):
    model.eval()
    total = 0.0
    count = 0
    preds_w, trues_w = [], []
    preds_r, trues_r = [], []
    preds_b, trues_b, probs_b = [], [], []
    sample_idx = []

    with torch.no_grad():
        for x, y_w, y_r, y_b, idx in loader:
            x = x.to(device)
            y_w = y_w.to(device)
            y_r = y_r.to(device)
            y_b = y_b.to(device)

            if main_model:
                out = model(x, x_static, adj)
            else:
                out = model(x)

            losses = multitask_loss(
                out,
                y_w,
                y_r,
                y_b,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                risk_class_weights=risk_class_weights,
                warn_pos_weight=warn_pos_weight,
                warn_focal_gamma=warn_focal_gamma,
            )
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
            sample_idx.append(idx.numpy())

    return {
        "loss": total / max(count, 1),
        "y_w_true": np.concatenate(trues_w, axis=0),
        "y_w_pred": np.concatenate(preds_w, axis=0),
        "y_r_true": np.concatenate(trues_r, axis=0),
        "y_r_pred": np.concatenate(preds_r, axis=0),
        "y_b_true": np.concatenate(trues_b, axis=0),
        "y_b_pred": np.concatenate(preds_b, axis=0),
        "y_b_prob": np.concatenate(probs_b, axis=0),
        "sample_idx": np.concatenate(sample_idx, axis=0),
    }


def train_deep_model(config: Dict, model_name: str = "dustriskformer") -> Dict:
    processed_dir = Path(config["paths"]["processed_dir"])
    results_dir = Path(config["paths"]["results_dir"])
    ckpt_dir = Path(config["paths"]["checkpoints_dir"])
    (results_dir / "plots").mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset_npz(processed_dir / "dataset_tensors.npz")

    train_idx = bundle.train_idx
    val_idx = bundle.val_idx
    test_idx = bundle.test_idx
    eval_cfg = config.get("evaluation", {})
    test_subset = str(eval_cfg.get("test_subset", "default")).lower()
    if test_subset in {"spring", "spring_3_5", "spring_strict_from_test", "spring_holdout_strict"}:
        tab = pd.read_parquet(processed_dir / "dataset_tabular.parquet", columns=["sample_idx", "time"]).drop_duplicates()
        tt = pd.to_datetime(tab["time"], utc=True, errors="coerce")
        spring_idx = tab.loc[tt.dt.month.isin([3, 4, 5]), "sample_idx"].to_numpy(dtype=np.int64)
        if len(spring_idx) > 0:
            spring_idx = np.unique(spring_idx)
            if test_subset in {"spring", "spring_3_5", "spring_strict_from_test"}:
                # strict: filter within original test split only
                test_idx = np.intersect1d(test_idx, spring_idx)
                print(f"[{model_name}] strict spring subset from original test split, n={len(test_idx)}")
            elif test_subset == "spring_holdout_strict":
                # strict seasonal holdout: train/val exclude spring, test=spring
                train_idx = np.setdiff1d(train_idx, spring_idx)
                val_idx = np.setdiff1d(val_idx, spring_idx)
                test_idx = spring_idx
                print(
                    f"[{model_name}] strict spring holdout split => train={len(train_idx)}, "
                    f"val={len(val_idx)}, test={len(test_idx)}"
                )

    train_loader, val_loader, test_loader = make_dataloaders(
        bundle,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    device = _device_from_config(config["training"]["device"])
    x_static = torch.tensor(bundle.X_static, dtype=torch.float32, device=device)
    adj = torch.tensor(bundle.adj, dtype=torch.float32, device=device)

    in_dim = bundle.X.shape[-1]
    horizons = bundle.y_wind.shape[-1]
    risk_num_classes = int(bundle.y_risk.max() + 1)
    region_name = config["region"].get("display_name", config["region"]["name"])

    if model_name == "dustriskformer":
        model = DustRiskFormer(
            in_dim=in_dim,
            static_dim=bundle.X_static.shape[-1],
            hidden_dim=int(config["model"]["hidden_dim"]),
            num_heads=int(config["model"]["num_heads"]),
            horizons=horizons,
            num_risk_classes=risk_num_classes,
            dropout=float(config["model"]["dropout"]),
        )
        main_model = True
    elif model_name == "lstm":
        model = LSTMBaseline(in_dim=in_dim, hidden_dim=64, horizons=horizons, num_risk_classes=risk_num_classes)
        main_model = False
    elif model_name == "cnn_lstm":
        model = CNNLSTMBaseline(in_dim=in_dim, hidden_dim=64, horizons=horizons, num_risk_classes=risk_num_classes)
        main_model = False
    elif model_name == "attn_tcn_lstm":
        model = AttentionTCNLSTMBaseline(in_dim=in_dim, hidden_dim=64, horizons=horizons, num_risk_classes=risk_num_classes)
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

    loss_cfg = config.get("loss", {})
    alpha = float(loss_cfg.get("alpha", 1.0))
    beta = float(loss_cfg.get("beta", 1.0))
    gamma = float(loss_cfg.get("gamma", 1.0))
    use_balanced = bool(loss_cfg.get("use_balanced_risk_weights", False))
    warn_pos_weight_cfg = float(loss_cfg.get("warn_pos_weight", 1.0))
    warn_focal_gamma = float(loss_cfg.get("warn_focal_gamma", 0.0))

    risk_class_weights_t = None
    if use_balanced:
        yr = bundle.y_risk[bundle.train_idx].reshape(-1)
        counts = np.bincount(yr, minlength=risk_num_classes).astype(np.float32) + 1e-6
        inv = counts.sum() / counts
        inv = inv / inv.mean()
        risk_class_weights_t = torch.tensor(inv, dtype=torch.float32, device=device)

    warn_pos_weight_t = None
    if warn_pos_weight_cfg > 0:
        warn_pos_weight_t = torch.tensor(warn_pos_weight_cfg, dtype=torch.float32, device=device)

    for epoch in range(int(config["training"]["epochs"])):
        model.train()
        run_loss = 0.0
        seen = 0
        for x, y_w, y_r, y_b, _ in train_loader:
            x = x.to(device)
            y_w = y_w.to(device)
            y_r = y_r.to(device)
            y_b = y_b.to(device)

            optim.zero_grad()
            if main_model:
                out = model(x, x_static, adj)
            else:
                out = model(x)
            losses = multitask_loss(
                out,
                y_w,
                y_r,
                y_b,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                risk_class_weights=risk_class_weights_t,
                warn_pos_weight=warn_pos_weight_t,
                warn_focal_gamma=warn_focal_gamma,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            run_loss += float(losses["total"].item()) * x.size(0)
            seen += x.size(0)

        train_loss = run_loss / max(seen, 1)
        val_pack = _eval_epoch(
            model,
            val_loader,
            device,
            x_static,
            adj,
            main_model,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            risk_class_weights=risk_class_weights_t,
            warn_pos_weight=warn_pos_weight_t,
            warn_focal_gamma=warn_focal_gamma,
        )
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
    test_pack = _eval_epoch(
        model,
        test_loader,
        device,
        x_static,
        adj,
        main_model,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        risk_class_weights=risk_class_weights_t,
        warn_pos_weight=warn_pos_weight_t,
        warn_focal_gamma=warn_focal_gamma,
    )

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
    save_pred_vs_true(
        y_w_true,
        y_w_pred,
        results_dir / "plots" / f"pred_vs_true_{model_name}.png",
        f"{region_name} {model_name} wind prediction",
    )
    cm = confusion(y_r_true, y_r_pred)
    save_confusion_matrix(
        cm,
        results_dir / "plots" / f"confusion_risk_{model_name}.png",
        f"{region_name} {model_name} risk confusion",
    )

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
    _export_detailed_predictions(results_dir, processed_dir, model_name, test_pack)
    subset_metrics = _export_subset_metrics(results_dir, processed_dir, model_name)
    metrics["subset_eval"] = subset_metrics

    with (results_dir / f"metrics_{model_name}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save attention tensors for explainability on main model
    if model_name == "dustriskformer":
        model.eval()
        x0, _, _, _, _ = next(iter(test_loader))
        x0 = x0[:1].to(device)
        with torch.no_grad():
            out = model(x0, x_static, adj)
        np.save(results_dir / "temporal_attention.npy", out["temporal_attention"].cpu().numpy())
        np.save(results_dir / "graph_attention.npy", out["graph_attention"].cpu().numpy())

    return metrics


def _export_subset_metrics(results_dir: Path, processed_dir: Path, model_name: str) -> Dict:
    pred = pd.read_csv(results_dir / f"predictions_detailed_{model_name}.csv")
    tab = pd.read_parquet(processed_dir / "dataset_tabular.parquet")[["sample_idx", "time"]].drop_duplicates()
    tab["time"] = pd.to_datetime(tab["time"], utc=True)
    pred = pred.merge(tab, on="sample_idx", how="left")
    pred["month"] = pred["time"].dt.month

    def _pack(df: pd.DataFrame) -> Dict:
        y_w_true = df["y_wind_true"].to_numpy()
        y_w_pred = df["y_wind_pred"].to_numpy()
        y_r_true = df["y_risk_true"].to_numpy()
        y_r_pred = df["y_risk_pred"].to_numpy()
        y_b_true = df["y_warn_true"].to_numpy()
        y_b_pred = df["y_warn_pred"].to_numpy()
        y_b_prob = df["y_warn_prob"].to_numpy()

        high_true = (y_r_true >= 2).astype(int)
        high_pred = (y_r_pred >= 2).astype(int)
        warn_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": None, "pr_auc": None}
        try:
            warn_metrics = binary_metrics(y_b_true, y_b_pred, y_b_prob)
        except Exception:
            warn_metrics = {
                "accuracy": float((y_b_true == y_b_pred).mean()),
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": None,
                "pr_auc": None,
            }

        return {
            "n": int(len(df)),
            "regression": regression_metrics(y_w_true, y_w_pred),
            "risk_f1": float(classification_metrics(y_r_true, y_r_pred)["f1"]),
            "warning": warn_metrics,
            "high_risk_recall": float(recall_score(high_true, high_pred, zero_division=0)),
        }

    spring_df = pred[pred["month"].isin([3, 4, 5])]
    high_df = pred[pred["y_risk_true"] >= 2]
    warn_df = pred[pred["y_warn_true"] == 1]

    out = {
        "all": _pack(pred),
        "spring_3_5": _pack(spring_df) if len(spring_df) > 10 else {"n": int(len(spring_df))},
        "high_risk_subset": _pack(high_df) if len(high_df) > 10 else {"n": int(len(high_df))},
        "warning_subset": _pack(warn_df) if len(warn_df) > 10 else {"n": int(len(warn_df))},
    }
    with (results_dir / f"subset_metrics_{model_name}.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def _export_detailed_predictions(results_dir: Path, processed_dir: Path, model_name: str, test_pack: Dict) -> None:
    meta = json.loads((processed_dir / "dataset_meta.json").read_text(encoding="utf-8"))
    stations = [s["station_id"] for s in meta["stations"]]
    horizons = [int(h) for h in meta["horizons"]]

    y_w_true = test_pack["y_w_true"]
    y_w_pred = test_pack["y_w_pred"]
    y_r_true = test_pack["y_r_true"]
    y_r_pred = test_pack["y_r_pred"]
    y_b_true = test_pack["y_b_true"]
    y_b_pred = test_pack["y_b_pred"]
    y_b_prob = test_pack["y_b_prob"]
    idxs = test_pack["sample_idx"]

    rows = []
    n_sample = y_w_true.shape[0]
    for i in range(n_sample):
        for s_i, sid in enumerate(stations):
            for h_i, h in enumerate(horizons):
                rows.append(
                    {
                        "sample_idx": int(idxs[i]),
                        "station_id": sid,
                        "horizon_hour": h,
                        "y_wind_true": float(y_w_true[i, s_i, h_i]),
                        "y_wind_pred": float(y_w_pred[i, s_i, h_i]),
                        "y_risk_true": int(y_r_true[i, s_i, h_i]),
                        "y_risk_pred": int(y_r_pred[i, s_i, h_i]),
                        "y_warn_true": int(y_b_true[i, s_i, h_i]),
                        "y_warn_pred": int(y_b_pred[i, s_i, h_i]),
                        "y_warn_prob": float(y_b_prob[i, s_i, h_i]),
                    }
                )
    pd.DataFrame(rows).to_csv(results_dir / f"predictions_detailed_{model_name}.csv", index=False)


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
