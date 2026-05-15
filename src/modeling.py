"""Der Anomalie-Score ist eine reelle Zahl. Für eine binäre Entscheidung wird eine Schwelle benötigt. Diese Schwelle wird ausschließlich auf dem Validierungs-Split bestimmt, indem über die Precision-Recall-Kurve alle realisierten Schwellenwerte aufgezählt und derjenige mit maximalem F1-Wert ausgewählt wird. Anschließend wird diese Schwelle unverändert auf den Test-Split angewandt:
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from pyod.models.base import BaseDetector
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from sklearn.metrics import roc_curve

try:
    from pyod.models.auto_encoder import AutoEncoder
    _AE_AVAILABLE = True
except ImportError:
    _AE_AVAILABLE = False

from .evaluation import Metrics, best_f1_threshold, evaluate
from .preprocessing import PreparedData


@dataclass
class BaselineResult:
    name: str
    val: Metrics
    test: Metrics


def build_detectors(contamination: float, seed: int, include_ae: bool = True) -> dict[str, BaseDetector]:
    """PyOD detectors for  baseline comparisons

    Contamination should match  empirical anomaly rate in train data so
    PyOD's default thresholding is (at least) sensible; 
    score functions are not affected.

    `include_ae=True` adds PyTorch AutoEncoder (49→32→16→32→49). Wird autom. 
    übersprungen: if torch / `pyod.models.auto_encoder` = not importable.
    """
    detectors: dict[str, BaseDetector] = {
        "iforest": IForest(n_estimators=200, contamination=contamination, random_state=seed, n_jobs=-1),
        "lof": LOF(n_neighbors=35, contamination=contamination, n_jobs=-1),
        "ecod": ECOD(contamination=contamination, n_jobs=-1),
    }
    if include_ae and _AE_AVAILABLE:
        detectors["autoencoder"] = AutoEncoder(
            hidden_neuron_list=[32, 16],
            epoch_num=30,
            batch_size=64,
            dropout_rate=0.2,
            lr=1e-3,
            contamination=contamination,
            random_state=seed,
            verbose=0,
        )
    return detectors


def _plot_confusion(metrics: Metrics, title: str) -> plt.Figure:
    cm = np.array([[metrics.tn, metrics.fp], [metrics.fn, metrics.tp]])
    fig, ax = plt.subplots(figsize=(4, 3.2))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"],
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _plot_roc(y_true: np.ndarray, scores: np.ndarray, title: str) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(4, 3.2))
    ax.plot(fpr, tpr, lw=2)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _log_figure(fig: plt.Figure, artifact_name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / artifact_name
        fig.savefig(path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(str(path), artifact_path="figures")


def run_baseline(
    name: str,
    detector: BaseDetector,
    data: PreparedData,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    model_family: str | None = None,
) -> BaselineResult:
    """Fit, score, tune threshold on val, evaluate on test, log to MLflow.

    `name` wird zu MLflow `run_name` &  artifact filename prefix —
    use a unique value per sweep combo. `model_family` overrides the default
    family tag (which is `name`); pass extra MLflow tags via `tags`.

    Returns the val + test metrics as a `BaselineResult`. 
    """
    params = params or {}

    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model_family", model_family or name)
        if tags:
            mlflow.set_tags(tags)
        mlflow.log_params({"model": model_family or name, **params})

        detector.fit(data.X_train)
        val_scores = detector.decision_function(data.X_val)
        test_scores = detector.decision_function(data.X_test)

        threshold, _ = best_f1_threshold(data.y_val, val_scores)
        val_metrics = evaluate(data.y_val, val_scores, threshold)
        test_metrics = evaluate(data.y_test, test_scores, threshold)

        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.as_dict().items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.as_dict().items()})

        _log_figure(_plot_confusion(test_metrics, f"{name} — test confusion"),
                    f"{name}_confusion_test.png")
        _log_figure(_plot_roc(data.y_test, test_scores, f"{name} — test ROC (AUC={test_metrics.roc_auc:.3f})"),
                    f"{name}_roc_test.png")

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / f"{name}.joblib"
            joblib.dump(detector, model_path)
            mlflow.log_artifact(str(model_path), artifact_path="model")

        card = Path("reports/dataset_card.md")
        if card.exists():
            mlflow.log_artifact(str(card), artifact_path="dataset")

    return BaselineResult(name=name, val=val_metrics, test=test_metrics)
