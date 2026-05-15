"""Daten laden und aufteilen."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

ArrayPair = Tuple[np.ndarray, np.ndarray]
SplitSix = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def load_cfrp(path: str | Path) -> ArrayPair:
    """CFRP .npz Datei aus dem SmartManuAD repository laden.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"CFRP archive not found at {path}. "
            "herunterladen von https://github.com/SmartManuAD/Smart-Manufacturing-AD "
            "und einsetzen in:  data/raw/."
        )

    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        x_key = next((k for k in ("X", "x", "data", "features") if k in keys), None)
        y_key = next((k for k in ("y", "labels", "target")          if k in keys), None)
        if x_key is None or y_key is None:
            raise KeyError(
                f"Unexpected .npz layout. Found keys: {sorted(keys)}. "
                "Expected feature key in {X,x,data,features} and label key in {y,labels,target}."
            )
        X = np.asarray(archive[x_key])
        y = np.asarray(archive[y_key]).ravel()

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Row count mismatch: X has {X.shape[0]}, y has {y.shape[0]}.")
    return X, y


def split_70_20_10(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    stratify: bool = True,
) -> SplitSix:
    """Split into 70% train / 20% val / 10% test.

    Aufteilung ist wie folgt realisiert: zuerst werden 30 % als Holdout abgetrennt, danach das Holdout 2:1 in Validierung und Test geteilt

    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    strat_first = y if stratify else None
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=strat_first
    )

    strat_second = y_hold if stratify else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_hold, y_hold, test_size=1.0 / 3.0, random_state=seed, stratify=strat_second
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
