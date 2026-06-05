"""Model evaluation utilities."""

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import (KFold, cross_val_score)

RANDOM_STATE = 42


class Evaluator:
    """Evaluate model performance on test data."""

    def __init__(self, model, X_test, y_test, X_train=None, y_train=None):
        self.model = model
        self.X_test = X_test
        self.y_test = np.asarray(y_test)
        self.X_train = X_train
        self.y_train = y_train
        self.y_pred = np.asarray(self.model.predict(X_test))

    def test_metrics(self):
        """Calculate metrics on holdout test set."""

        return {
            "r2": float(r2_score(self.y_test, self.y_pred)),
            "mae": float(mean_absolute_error(self.y_test, self.y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(self.y_test, self.y_pred))),
        }

    def cross_validation(self, cv_splits=5):
        """Run cross validation on full dataset."""

        if (self.X_train is None or self.y_train is None):
            return None

        X_full = pd.concat([self.X_train, self.X_test])

        y_full = np.concatenate([self.y_train, self.y_test])

        cv = KFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        cv_r2 = cross_val_score(
            self.model.model,
            X_full,
            y_full,
            cv=cv,
            scoring="r2",
        )

        cv_mae = -cross_val_score(
            self.model.model,
            X_full,
            y_full,
            cv=cv,
            scoring="neg_mean_absolute_error",
        )

        return {
            "r2_mean": float(cv_r2.mean()),
            "r2_std": float(cv_r2.std()),
            "mae_mean": float(cv_mae.mean()),
            "mae_std": float(cv_mae.std()),
        }

    def print_summary(self, metrics):
        """Pretty print evaluation results."""

        print("\n[Evaluation]")
        print(f"  R²   : {metrics['r2']:.4f}")
        print(f"  MAE  : {metrics['mae']:.4f}")
        print(f"  RMSE : {metrics['rmse']:.4f}")

        cv = metrics.get("cv")
        if cv is None:
            return

        print(f"  CV R²  : {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}")

        print(f"  CV MAE : {cv['mae_mean']:.4f} ± {cv['mae_std']:.4f}")

    def evaluate(self, cv_splits=5, verbose=True):
        """Run complete evaluation."""

        metrics = self.test_metrics()
        cv_metrics = self.cross_validation(cv_splits=cv_splits)

        if cv_metrics is not None:
            metrics["cv"] = cv_metrics

        if verbose:
            self.print_summary(metrics)

        return metrics