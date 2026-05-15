"""MLflow Unterstützung/Hilfe/.. —  config laden ,  tracking URI / experiment, log metadata festlegen."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import mlflow
import numpy as np
import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_project_root"] = str(path.parent)
    return cfg


def _resolve_tracking_uri(uri: str, project_root: Path) -> str:
    """ankern der relative `file:` URIs to project Basis.

    Ohne dies: running a notebook from notebooks/ would create mlruns/
    inside that subfolder instead of at the project root.
    """
    if uri.startswith("file:"):
        raw = uri[len("file:"):]
        p = Path(raw)
        if not p.is_absolute():
            return (project_root / p).resolve().as_uri()
    return uri


def init_mlflow(config: Mapping[str, Any]) -> None:
    project_root = Path(config.get("_project_root", "."))
    uri = _resolve_tracking_uri(config["mlflow"]["tracking_uri"], project_root)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])


def log_split_sizes(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> None:
    mlflow.log_params(
        {
            "n_train": int(X_train.shape[0]),
            "n_val": int(X_val.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]),
            "train_pos_rate": float(np.mean(y_train)),
            "val_pos_rate": float(np.mean(y_val)),
            "test_pos_rate": float(np.mean(y_test)),
        }
    )


def log_dataset_card(card_path: str | Path = "reports/dataset_card.md") -> None:
    card_path = Path(card_path)
    if card_path.exists():
        mlflow.log_artifact(str(card_path), artifact_path="dataset")
