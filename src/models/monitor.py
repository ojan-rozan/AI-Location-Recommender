"""Model monitoring: drift detection and performance tracking."""

import json
from datetime import datetime
from pathlib import Path

from sklearn.metrics import mean_absolute_error, r2_score

from src.data.data_loader import ROOT
from src.models.model import DemandModel

MODEL_PATH = ROOT / "models" / "xgb_demand.pkl"
META_PATH = ROOT / "models" / "model_meta.json"

LOG_DIR = ROOT / "logs" / "monitoring"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class ModelMonitor:
    """Monitor model performance dan feature drift."""

    def __init__(self, model_path=MODEL_PATH, meta_path=META_PATH, drift_threshold=0.20, perf_drop_threshold=0.05):
        self.drift_threshold = drift_threshold
        self.perf_drop_threshold = perf_drop_threshold

        model_path = Path(model_path)
        meta_path = Path(meta_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        self.model = DemandModel.load(model_path)

        self.baseline_meta = {}

        if meta_path.exists():
            self.baseline_meta = json.loads(
                meta_path.read_text()
            )

    def check_performance(self, current_df):
        """Hitung metric model pada data terbaru."""

        X = current_df[self.model.feature_cols].fillna(0)
        y_true = current_df["target"]

        predictions = self.model.predict(X)

        return {
            "r2": float(
                r2_score(y_true, predictions)
            ),
            "mae": float(
                mean_absolute_error(
                    y_true,
                    predictions,
                )
            ),
            "n_samples": len(current_df),
        }

    def check_drift(self, current_df, baseline_means=None):
        """Bandingkan mean feature saat ini vs baseline."""

        drift_report = {}

        baseline_means = baseline_means or {}

        for feature in self.model.feature_cols:

            if feature not in current_df.columns:
                continue

            current_mean = float(
                current_df[feature].mean()
            )

            reference_mean = baseline_means.get(
                feature
            )

            if reference_mean is None:

                drift_report[feature] = {
                    "current_mean": current_mean,
                    "baseline_mean": None,
                    "pct_change": None,
                    "drift": False,
                }

                continue

            if reference_mean == 0:
                pct_change = 0
            else:
                pct_change = (
                    abs(current_mean - reference_mean)
                    / abs(reference_mean)
                )

            drift_report[feature] = {
                "current_mean": current_mean,
                "baseline_mean": reference_mean,
                "pct_change": float(pct_change),
                "drift": (
                    pct_change
                    > self.drift_threshold
                ),
            }

        return drift_report

    def load_previous_feature_means(self):
        """Ambil baseline feature mean dari log terakhir."""

        logs = sorted(
            LOG_DIR.glob("monitor_*.json")
        )

        if not logs:
            return None

        latest_log = logs[-1]

        payload = json.loads(
            latest_log.read_text()
        )

        return payload.get(
            "feature_means",
            {},
        )

    def build_feature_means(self, current_df):
        """Simpan mean feature untuk baseline monitoring berikutnya."""

        return {
            feature: float(
                current_df[feature].mean()
            )
            for feature in self.model.feature_cols
            if feature in current_df.columns
        }

    def save_log(self, payload):
        """Save monitoring result."""

        filename = (
            "monitor_"+ datetime.now().strftime("%Y%m%d_%H%M%S")+ ".json"
        )

        log_path = LOG_DIR / filename

        log_path.write_text(
            json.dumps(
                payload,
                indent=2,
            )
        )

        return log_path

    def print_summary(self, performance, baseline_r2, r2_drop, drifted_features):
        print("\n" + "=" * 60)
        print(f"MODEL MONITORING — {datetime.now().isoformat()}")
        print("=" * 60)

        if baseline_r2 is not None:
            print(f"Baseline R²: {baseline_r2:.4f}")

        print("\n[Performance Check]")

        print(f"  Current R² : {performance['r2']:.4f}")
        print(f"  Current MAE: {performance['mae']:.4f}")
        print(f"  Samples    : {performance['n_samples']}")

        if (r2_drop is not None and r2_drop > self.perf_drop_threshold):
            print(f"  ⚠️ ALERT: R² dropped by {r2_drop:.3f}")
        else:
            print("  ✓ Performance stable")

        print("\n[Drift Detection]")

        if drifted_features:

            print(f"  ⚠️ Drift detected on {len(drifted_features)} features:")

            for feature in drifted_features:
                print(f"    - {feature}")

        else:
            print("  ✓ No significant drift")

    def run(self, current_df):
        """Run monitoring pipeline."""

        performance = self.check_performance(current_df)

        baseline_metrics = (
            self.baseline_meta.get("metrics",{})
        )

        baseline_r2 = baseline_metrics.get("r2")

        r2_drop = None

        if baseline_r2 is not None:
            r2_drop = (baseline_r2 - performance["r2"])

        perf_alert = (r2_drop is not None and r2_drop > self.perf_drop_threshold)

        baseline_means = (self.load_previous_feature_means())

        drift_report = self.check_drift(current_df, baseline_means)

        drifted_features = [feature for feature, result in drift_report.items() if result["drift"]]

        drift_alert = bool(drifted_features)

        self.print_summary(performance, baseline_r2, r2_drop, drifted_features)

        log_payload = {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": performance,
            "baseline_metrics": baseline_metrics,
            "r2_drop": r2_drop,
            "perf_alert": perf_alert,
            "drift_alert": drift_alert,
            "drifted_features": drifted_features,
            "feature_means": self.build_feature_means(current_df),
        }

        log_path = self.save_log(log_payload)

        print(f"\n✓ Log saved: {log_path.name}")

        alert = (perf_alert or drift_alert)

        return log_payload, alert