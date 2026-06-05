"""Model monitoring: drift detection and performance tracking."""

import json
from datetime import datetime
from pathlib import Path

from sklearn.metrics import r2_score, mean_absolute_error

from src.data.data_loader import ROOT
from src.models.model import DemandModel


model_path = ROOT / "models" / "xgb_demand.pkl"
meta_path = ROOT / "models" / "model_meta.json"
log_dir = ROOT / "logs" / "monitoring"
log_dir.mkdir(parents=True, exist_ok=True)


class ModelMonitor:
    """Cek drift dan performance degradation pada model deployed."""

    drift_threshold = 0.20   # 20% mean shift = drift
    perf_drop_threshold = 0.05   # R² drop > 0.05 = alert

    def __init__(self, model_path=model_path, meta_path=meta_path):
        model_path, meta_path = Path(model_path), Path(meta_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = DemandModel.load(model_path)
        self.baseline_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    def check_performance(self, current_df) -> dict:
        X = current_df[self.model.feature_cols].fillna(0)
        y_true = current_df["target"]
        pred = self.model.predict(X)
        return {
            "r2": float(r2_score(y_true, pred)),
            "mae": float(mean_absolute_error(y_true, pred)),
            "n_samples": len(current_df),
        }

    def check_drift(self, current_df, baseline_means=None) -> dict:
        """Compare feature distribution and baseline means."""
        drift_report = {}
        for col in self.model.feature_cols:
            if col not in current_df.columns:
                continue
            cur_mean = float(current_df[col].mean())
            ref_mean = (baseline_means or {}).get(col)

            if ref_mean:
                pct_change = abs(cur_mean - ref_mean) / abs(ref_mean)
                drift_report[col] = {
                    "current_mean": cur_mean,
                    "baseline_mean": ref_mean,
                    "pct_change": float(pct_change),
                    "drift": pct_change > self.drift_threshold,
                }
            else:
                drift_report[col] = {"current_mean": cur_mean, "baseline_mean": None, "drift": False}

        return drift_report

    def run(self, current_df):
        """Full monitoring run and save log."""
        timestamp = datetime.now().isoformat()
        baseline_metrics = self.baseline_meta.get("metrics", {})
        baseline_r2 = baseline_metrics.get("r2")  # None kalau belum ada baseline

        print(f"\n{'='*60}\nMODEL MONITORING — {timestamp}\n{'='*60}")
        if baseline_r2 is not None:
            print(f"Baseline R²: {baseline_r2:.4f}")

        print("\n[Performance Check]")
        current_perf = self.check_performance(current_df)
        print(f"  Current R² : {current_perf['r2']:.4f}")
        print(f"  Current MAE: {current_perf['mae']:.4f}")
        print(f"  Samples    : {current_perf['n_samples']}")

        r2_drop    = (baseline_r2 - current_perf["r2"]) if baseline_r2 is not None else None
        perf_alert = (r2_drop > self.perf_drop_threshold) if r2_drop is not None else False
        print(f"  ⚠️  ALERT: R² dropped by {r2_drop:.3f}" if perf_alert else "  ✓ Performance stable")

        # Drift
        prev_logs = sorted(log_dir.glob("monitor_*.json"))
        baseline_means = (
            json.loads(prev_logs[-1].read_text()).get("feature_means", {})
            if prev_logs else None
        )

        print("\n[Drift Detection]")
        drift = self.check_drift(current_df, baseline_means=baseline_means)
        drifted = [k for k, v in drift.items() if v.get("drift")]
        drift_alert = bool(drifted)
        if drift_alert:
            print(f"  ⚠️  Drift detected pada {len(drifted)} feature:")
            for f in drifted:
                print(f"    {f}: {drift[f]['pct_change']*100:.1f}% change")
        else:
            print("  ✓ No significant drift")

        log_entry = {
            "timestamp": timestamp,
            "current_metrics": current_perf,
            "baseline_metrics": baseline_metrics,
            "r2_drop": r2_drop,
            "perf_alert": perf_alert,
            "drift_alert": drift_alert,
            "drifted_features": drifted,
            "feature_means": {
                c: float(current_df[c].mean())
                for c in self.model.feature_cols
                if c in current_df.columns
            },
        }

        log_file = log_dir / f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_file.write_text(json.dumps(log_entry, indent=2))
        print(f"\n✓ Log saved: {log_file.name}")

        return log_entry, (perf_alert or drift_alert)