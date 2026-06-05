"""Build Model"""

import json
import joblib
import xgboost as xgb
import shap
from pathlib import Path


class DemandModel:
    # Load tuned parameters if available
    meta_path = Path(__file__).parent.parent.parent / "models" / "model_meta.json"
    params = (
        json.loads(meta_path.read_text())["best_params"]
        if meta_path.exists()
        else {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "random_state": 42,
        }
    )

    def __init__(self, params=None):
        self.params = params or dict(self.params)
        self.model = None
        self.feature_cols = None
        self.explainer = None

    def fit(self, X, y, eval_set=None):
        # Train model and initialize SHAP explainer
        self.feature_cols = list(X.columns)
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        self.explainer = shap.TreeExplainer(self.model)
        return self

    def predict(self, X):
        # Ensure feature consistency during inference
        return self.model.predict(X[self.feature_cols].fillna(0))

    def predict_with_explanation(self, X, top_n=3) -> list[dict]:
        """Return predictions with top-N SHAP feature contributions."""
        X = X[self.feature_cols].fillna(0)

        raw_pred = self.model.predict(X)
        shap_vals = self.explainer.shap_values(X)

        results = []
        for i in range(len(X)):
            # Select features with the largest SHAP impact
            top = sorted(
                zip(self.feature_cols, shap_vals[i]),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:top_n]

            results.append({
                "raw_prediction": float(raw_pred[i]),
                "score_0_100": float(min(100, max(0, raw_pred[i] * 2.5))),
                "top_factors": [
                    {
                        "feature": f,
                        "shap": float(v),
                        "direction": "positive" if v > 0 else "negative",
                    }
                    for f, v in top
                ],
            })

        return results

    def save(self, path):
        """Save trained model and metadata."""
        joblib.dump(
            {
                "model": self.model,
                "feature_cols": self.feature_cols,
                "params": self.params,
            },
            path,
        )

    @classmethod
    def load(cls, path):
        """Load model and rebuild SHAP explainer."""
        data = joblib.load(path)

        instance = cls(params=data["params"])
        instance.model = data["model"]
        instance.feature_cols = data["feature_cols"]

        # Recreate explainer for interpretability
        instance.explainer = shap.TreeExplainer(instance.model)

        return instance