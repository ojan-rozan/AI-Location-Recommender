"""Build Model"""

import json
import joblib
import xgboost as xgb
import shap
from pathlib import Path


DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "random_state": 42,
}

SCORE_MULTIPLIER = 2.5  # fallback kalau target_min/max gak ada di meta


def _load_meta():
    meta_path = (Path(__file__).parent.parent.parent / "models" / "model_meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def load_default_params():
    return _load_meta().get("best_params", DEFAULT_PARAMS)


def load_score_range():
    """Ambil target_min/target_max dari meta (buat normalisasi skor min-max)."""
    meta = _load_meta()
    return meta.get("target_min"), meta.get("target_max")


class DemandModel:
    def __init__(self, params=None):
        self.params = params or load_default_params()
        self.target_min, self.target_max = load_score_range()

        self.model = None
        self.feature_cols = []
        self.explainer = None

    def norm_score(self, raw):
        """Normalisasi prediksi mentah ke 0-100."""
        if (self.target_min is not None and self.target_max is not None
                and self.target_max > self.target_min):
            s = (raw - self.target_min) / (self.target_max - self.target_min) * 100
        else:
            s = raw * SCORE_MULTIPLIER
        return float(min(100, max(0, s)))

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

        # SHAP opsional — kalau explainer gak ada, top_factors dikosongin
        shap_vals = None
        if self.explainer is not None:
            try:
                shap_vals = self.explainer.shap_values(X)
            except Exception as e:
                print(f"  ⚠️  SHAP gagal: {type(e).__name__}: {str(e)[:80]}")

        results = []
        for i in range(len(X)):
            if shap_vals is not None:
                top = sorted(
                    zip(self.feature_cols, shap_vals[i]),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:top_n]
            else:
                top = []

            results.append({
                "raw_prediction": float(raw_pred[i]),
                "score_0_100": self.norm_score(raw_pred[i]),
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

        try:
            instance.explainer = shap.TreeExplainer(instance.model)
        except Exception as e:
            print(f"  ⚠️  SHAP explainer off: {type(e).__name__}: {str(e)[:80]}")
            instance.explainer = None

        return instance