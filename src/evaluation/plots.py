from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def save_loss_curve(train_losses: list[float], val_losses: list[float], out_path: str | Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, out_path: str | Path, title: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(y_true[:200], label="true")
    plt.plot(y_pred[:200], label="pred")
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Wind Speed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_confusion_matrix(cm: np.ndarray, out_path: str | Path, title: str) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_feature_importance(names: list[str], vals: np.ndarray, out_path: str | Path, title: str) -> None:
    order = np.argsort(vals)[-20:]
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(names)[order], vals[order])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_attention_heatmap(attn: np.ndarray, out_path: str | Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(attn, cmap="magma")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
