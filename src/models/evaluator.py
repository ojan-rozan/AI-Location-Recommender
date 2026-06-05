"""Model evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score


class Evaluator:
    """Evaluate model performance."""

    def __init__(self, model, X_test, y_test, X_train=None, y_train=None):
        self.model = model
        self.X_test = X_test
        self.y_test = np.array(y_test)
        self.X_train = X_train
        self.y_train = y_train

        # Generate predictions on test set
        self.y_pred = np.array(model.predict(X_test))

    def evaluate(self, cv_splits=5) -> dict:
        # Test set metrics
        metrics = {
            "r2": float(r2_score(self.y_test, self.y_pred)),
            "mae": float(mean_absolute_error(self.y_test, self.y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(self.y_test, self.y_pred))),
        }

        # Cross-validation metrics if training data is available
        if self.X_train is not None and self.y_train is not None:
            X_full = pd.concat([self.X_train, self.X_test])
            y_full = np.concatenate([self.y_train, self.y_test])

            kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

            cv_r2 = cross_val_score(
                self.model.model, X_full, y_full, cv=kf, scoring="r2"
            )

            cv_mae = -cross_val_score(
                self.model.model,
                X_full,
                y_full,
                cv=kf,
                scoring="neg_mean_absolute_error",
            )

            metrics["cv"] = {
                "r2_mean": float(cv_r2.mean()),
                "r2_std": float(cv_r2.std()),
                "mae_mean": float(cv_mae.mean()),
                "mae_std": float(cv_mae.std()),
            }

        # Display results
        print(f"  r2  : {metrics['r2']:.4f}")
        print(f"  mae : {metrics['mae']:.4f}")
        print(f"  rmse: {metrics['rmse']:.4f}")

        if "cv" in metrics:
            cv = metrics["cv"]
            print(f"  cv r2 : {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}")
            print(f"  cv mae: {cv['mae_mean']:.4f} ± {cv['mae_std']:.4f}")

        return metrics