"""Das gesamte Preprocessing wird ausschließlich auf dem Trainings-Split gefittet und dann mit den eingefrorenen Parametern auf Validierung und Test übertragen. Ein Verstoß gegen diese Regel würde „Data Leakage“ erzeugen und die Testmetriken systematisch zu optimistisch ausweisen.
Konkret heißt das im Code: auf X_train wird fit_transform aufgerufen, auf X_val und X_test ausschließlich transform:

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

ScalerKind = Literal["standard", "robust"]
ArrayTriple = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _sanitize_inf(X: np.ndarray) -> np.ndarray:
    """Replace +/-inf with NaN so the imputer can handle them uniformly."""
    X = np.asarray(X, dtype=float)
    return np.where(np.isfinite(X), X, np.nan)


def handle_missing(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    strategy: Literal["median", "mean", "most_frequent"] = "median",
) -> ArrayTriple:
    """Impute NaN / +-inf in all three splits.

     imputer  fit on training split only & applied to val/test.
    `median` = default (sensor data often heavy-tailed)
    """
    X_train = _sanitize_inf(X_train)
    X_val = _sanitize_inf(X_val)
    X_test = _sanitize_inf(X_test)

    imputer = SimpleImputer(strategy=strategy)
    X_train_i = imputer.fit_transform(X_train)
    X_val_i = imputer.transform(X_val)
    X_test_i = imputer.transform(X_test)
    return X_train_i, X_val_i, X_test_i


def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    kind: ScalerKind = "robust",
) -> ArrayTriple:
    """Fit a scaler on X_train, apply to all three splits.

    Default is `robust` --> vibration / process-sensor features in the
    SmartManuAD datasets tend to be heavy-tailed. outliers should not dominate
    the scaling parameters.
    """
    scaler = StandardScaler() if kind == "standard" else RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s


@dataclass
class PreparedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def prepare(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    impute: Literal["median", "mean", "most_frequent"] = "median",
    scaler: ScalerKind = "robust",
) -> PreparedData:
    """End-to-end preprocessing: impute then scale, fit on train only."""
    X_train_i, X_val_i, X_test_i = handle_missing(X_train, X_val, X_test, strategy=impute)
    X_train_s, X_val_s, X_test_s = scale_features(X_train_i, X_val_i, X_test_i, kind=scaler)
    return PreparedData(
        X_train=X_train_s,
        y_train=np.asarray(y_train).astype(int),
        X_val=X_val_s,
        y_val=np.asarray(y_val).astype(int),
        X_test=X_test_s,
        y_test=np.asarray(y_test).astype(int),
    )
