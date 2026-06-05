"""Training pipeline"""

import json
from datetime import datetime

import numpy as np
import mlflow
import mlflow.xgboost
import optuna
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.data.data_loader import ROOT
from src.models.model import DemandModel


model_path = ROOT / "models"
model_path.mkdir(parents=True, exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class Trainer:
    """Training pipeline dengan MLflow tracking + Optuna tuning."""

    experiment_name = "coffee_location_demand"
    n_trials = 100
    timeout_seconds = 300   # 5 menit
    early_stop_after = 50    # stop kalau 50 trials berturut-turut tidak improve

    def __init__(self, df_training):
        self.df = df_training
        mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}")
        mlflow.set_experiment(self.experiment_name)

    def split(self, test_size=0.2, random_state=42):
        feature_cols = self.df.drop(columns=['cafe_id', 'kecamatan', 'kota', 'lat', 'lng', 'target', 'rating', 'reviews_count']).columns.tolist()
        X = self.df[feature_cols].fillna(0)
        y = self.df["target"]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def evaluate(self, model, X_test, y_test) -> dict:
        pred = model.predict(X_test)
        return {
            "r2": float(r2_score(y_test, pred)),
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        }

    def cross_validation(self, model, X, y, n_splits=5) -> dict:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_r2  = cross_val_score(model.model, X, y, cv=kf, scoring="r2")
        cv_mae = -cross_val_score(model.model, X, y, cv=kf,
                                  scoring="neg_mean_absolute_error")
        return {
            "r2_mean": float(cv_r2.mean()),  "r2_std":  float(cv_r2.std()),
            "mae_mean": float(cv_mae.mean()), "mae_std": float(cv_mae.std()),
        }

    def early_stop(self):
        def callback(study, trial):
            if trial.number - study.best_trial.number >= self.early_stop_after:
                study.stop()
        return callback

    def save_meta(self, run_id, params, metrics, cv, n_features):
        meta = {
            "trained_at": datetime.now().isoformat(),
            "mlflow_run_id": run_id,
            "best_params": params,
            "metrics": metrics,
            "cv_metrics": cv,
            "n_train": len(self.df),
            "n_features": n_features,
        }
        (model_path / "model_meta.json").write_text(json.dumps(meta, indent=2))

    def train(self, run_name="train") -> tuple:
        """Optuna tuning pakai CV → train final model dengan best params."""
        X_train, X_test, y_train, y_test = self.split()

        print(f"Running Optuna (max {self.n_trials} trials, timeout {self.timeout_seconds}s, "
              f"early stop after {self.early_stop_after} no-improve)...")

        with mlflow.start_run(run_name=run_name) as run:

            # Optuna tuning
            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
                    "random_state": 42,
                }
                with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                    mlflow.log_params(params)
                    model = DemandModel(params=params)
                    model.fit(X_train, y_train)

                    # CV di train set
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    cv_mae = -cross_val_score(
                        model.model, X_train, y_train, cv=kf,
                        scoring="neg_mean_absolute_error"
                    )
                    mlflow.log_metric("cv_mae", float(cv_mae.mean()))
                return float(cv_mae.mean())

            study = optuna.create_study(direction="minimize")
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout_seconds,
                callbacks=[self.early_stop()],
                show_progress_bar=True,
            )

            best_params = {**study.best_params, "random_state": 42}
            print(f"\n✓ Best trial: #{study.best_trial.number} — CV MAE {study.best_value:.4f}")
            print(f"  Best params: {best_params}")

            # Train final model dengan best params
            print("\nTraining final model dengan best params...")
            final = DemandModel(params=best_params)
            final.fit(X_train, y_train, eval_set=[(X_test, y_test)])

            # Eval di test set
            metrics = self.evaluate(final, X_test, y_test)

            # CV final
            cv = self.cross_validation(final, X_train, y_train)

            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metrics({**metrics, **{f"cv_{k}": v for k, v in cv.items()}})
            mlflow.xgboost.log_model(final.model, "best_model", input_example=X_train.head(2))

            final.save(model_path / "xgb_demand.pkl")
            self.save_meta(run.info.run_id, best_params, metrics, cv, X_train.shape[1])

            print("\n[Final Model]")
            for k, v in metrics.items():
                print(f"  {k:6s}: {v:.4f}")
            print(f"  cv r2 : {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}")
            print(f"  cv mae: {cv['mae_mean']:.4f} ± {cv['mae_std']:.4f}")

        return final, metrics