"""Reproduce the model comparison reported in docs/challenge.md.

Run from the repository root::

    uv run python docs/model_comparison.py

All candidates share the production preprocessing (``DelayModel.preprocess``).
The selected model (balanced XGBoost) is trained through ``DelayModel.fit``
itself, so its numbers describe exactly what ships.
"""

# ruff: noqa: T201  (printing the report is this script's purpose)

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from challenge.model import DelayModel

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "data.csv"
CV_FOLDS = 5

Trainer = Callable[[pd.DataFrame, pd.Series], object]


def _balanced_xgboost(features: pd.DataFrame, labels: pd.Series) -> object:
    model = DelayModel()
    model.fit(features, labels.to_frame())
    return model._model


def _unbalanced_xgboost(features: pd.DataFrame, labels: pd.Series) -> object:
    return xgb.XGBClassifier(random_state=1, learning_rate=0.01).fit(features, labels)


def _balanced_logistic_regression(features: pd.DataFrame, labels: pd.Series) -> object:
    n_negative = int((labels == 0).sum())
    n_positive = int((labels == 1).sum())
    # Same weighting as the notebook (section 6.b.iii).
    weights = {1: n_negative / len(labels), 0: n_positive / len(labels)}
    return LogisticRegression(class_weight=weights).fit(features, labels)


def _unbalanced_logistic_regression(features: pd.DataFrame, labels: pd.Series) -> object:
    return LogisticRegression().fit(features, labels)


SELECTED = "XGBoost, balanced"
RUNNER_UP = "LogisticRegression, balanced"

TRAINERS: dict[str, Trainer] = {
    SELECTED: _balanced_xgboost,
    RUNNER_UP: _balanced_logistic_regression,
    "XGBoost, unbalanced": _unbalanced_xgboost,
    "LogisticRegression, unbalanced": _unbalanced_logistic_regression,
}


def _cv_recall(train: Trainer, features: pd.DataFrame, labels: pd.Series) -> list[float]:
    """Return per-fold class-1 recall over stratified K folds."""
    recalls = []
    folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=0)
    for train_idx, test_idx in folds.split(features, labels):
        model = train(features.iloc[train_idx], labels.iloc[train_idx])
        predictions = model.predict(features.iloc[test_idx])
        recalls.append(recall_score(labels.iloc[test_idx], predictions))
    return recalls


def main() -> None:
    """Print the comparison as a Markdown table, ready to paste into the docs."""
    data = pd.read_csv(DATA_PATH, low_memory=False)
    features, target = DelayModel().preprocess(data, target_column="delay")
    labels = target["delay"]
    # Same split as the notebook and the unit tests.
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.33, random_state=42
    )
    print(f"Delay rate: {labels.mean():.1%} of {len(labels):,} flights\n")
    print("| Model | Class-1 recall | Class-1 F1 | CV class-1 recall | ROC-AUC |")
    print("|---|---|---|---|---|")

    holdout_predictions = {}
    for name, train in TRAINERS.items():
        model = train(x_train, y_train)
        predictions = model.predict(x_test)
        holdout_predictions[name] = predictions
        auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
        cv_recall = _cv_recall(train, features, labels)

        label = f"**{name}**" if name == SELECTED else name
        print(
            f"| {label} | {recall_score(y_test, predictions):.3f} "
            f"| {f1_score(y_test, predictions):.3f} "
            f"| {np.mean(cv_recall):.3f} ± {np.std(cv_recall):.3f} | {auc:.3f} |"
        )

    agreement = np.mean(holdout_predictions[SELECTED] == holdout_predictions[RUNNER_UP])
    print(f"\nBalanced models agree on {agreement:.1%} of holdout predictions")


if __name__ == "__main__":
    main()
