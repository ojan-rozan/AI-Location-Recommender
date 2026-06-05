"""Auto retrain trigger based on data changes and model age."""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.data_loader import ROOT

STATE_FILE = ROOT / "models" / "data_state.json"
META_FILE = ROOT / "models" / "model_meta.json"

DEFAULT_DATA_FILES = [
    ROOT / "data" / "raw" / "cafes_gmaps.csv",
    ROOT / "data" / "helper" / "owner_stores.csv",
]


class RetrainTrigger:
    """Detect dataset updates and trigger model retraining."""

    def __init__(self, data_files=None, state_file=STATE_FILE, row_threshold=0.10, time_threshold_days=30):
        self.data_files = data_files or DEFAULT_DATA_FILES
        self.state_file = Path(state_file)
        self.row_threshold = row_threshold
        self.time_threshold_days = time_threshold_days

    def hash_file(self, path):
        """Generate MD5 hash for a file."""

        path = Path(path)

        md5 = hashlib.md5()

        with open(path, "rb") as file:
            while True:
                chunk = file.read(8192)

                if not chunk:
                    break

                md5.update(chunk)

        return md5.hexdigest()

    def get_file_state(self, path):
        """Collect metadata for a dataset file."""

        path = Path(path)

        if not path.exists():
            return None

        return {
            "hash": self.hash_file(path),
            "rows": len(pd.read_csv(path)),
            "size_kb": path.stat().st_size // 1024,
            "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }

    def load_state(self):
        """Load previously saved dataset state."""

        if not self.state_file.exists():
            return {}

        return json.loads(
            self.state_file.read_text()
        )

    def save_state(self, state):
        """Persist latest dataset state."""

        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        self.state_file.write_text(json.dumps(state, indent=2))

    def detect_changes(self):
        """Compare current files with previous state."""

        previous_state = self.load_state()

        current_state = {}
        changes = []

        for path in self.data_files:

            relative_path = str(Path(path).relative_to(ROOT))

            current_file_state = self.get_file_state(path)

            current_state[relative_path] = current_file_state

            if current_file_state is None:
                continue

            previous_file_state = previous_state.get(relative_path)

            if previous_file_state is None:
                changes.append({
                    "file": relative_path, 
                    "reason": "new_file"
                })
                continue

            if (current_file_state["hash"] != previous_file_state["hash"]):

                row_diff = (current_file_state["rows"] - previous_file_state["rows"])

                row_pct = (row_diff / max(previous_file_state["rows"],1))

                changes.append({
                    "file": relative_path,
                    "reason": "content_changed",
                    "row_diff": row_diff,
                    "row_pct": row_pct,
                    "should_retrain": (abs(row_pct) >= self.row_threshold),
                })

        return current_state, changes

    def days_since_last_train(self):
        """Get model age in days."""

        if not META_FILE.exists():
            return 9999

        metadata = json.loads(META_FILE.read_text())

        trained_at = datetime.fromisoformat(metadata["trained_at"])

        return (datetime.now() - trained_at).days

    def should_retrain(self, changes,model_age_days):
        """Decide whether retraining is needed."""
        reasons = []

        if (model_age_days >= self.time_threshold_days):
            reasons.append(f"model age = {model_age_days} days")

        for change in changes:
            if change["reason"] == "new_file":
                reasons.append(f"new dataset: {change['file']}")
                continue

            if change.get("should_retrain"):
                reasons.append(f"{change['file']} changed {change['row_pct']*100:+.1f}% ({change['row_diff']:+d} rows)")

        return bool(reasons), reasons

    def print_summary(self, model_age_days,changes):
        """Print change summary."""

        print("\n" + "=" * 60)
        print(f"RETRAIN CHECK — {datetime.now().isoformat()}")
        print("=" * 60)

        print(f"\nModel age: {model_age_days} days")

        if not changes:
            print("\nNo dataset changes detected.")
            return

        print("\nDetected changes:")

        for change in changes:

            print(f"\n• {change['file']}")
            print(f"  reason: {change['reason']}")

            if change["reason"] == "content_changed":
                print(f"  row diff: {change['row_diff']:+d}")

                print(f"  row change: {change['row_pct']*100:+.2f}%")

    def run_retrain(self, use_optuna=False,):
        """Execute training pipeline."""

        print("\n🔄 Starting retraining...")

        command = ["python3",str(ROOT/ "test"/ "train_model.py")]

        if use_optuna:
            command.extend(["--optuna", "20"])

        result = subprocess.run(command, cwd=str(ROOT))

        return result.returncode == 0

    def check(self, use_optuna=False):
        """Run retrain trigger workflow."""

        current_state, changes = (self.detect_changes())
        model_age_days = (self.days_since_last_train())
        self.print_summary(model_age_days,changes)
        should_retrain, reasons = (self.should_retrain(changes, model_age_days))

        if not should_retrain:
            print("\n✓ Retraining not required")

            self.save_state(current_state)
            return False

        print("\n🚨 Retraining triggered")

        print("\nReasons:")

        for reason in reasons:
            print(f"  - {reason}")

        success = self.run_retrain(use_optuna=use_optuna)

        if success:
            print("\n✅ Retraining completed")
            self.save_state(current_state)

        else:
            print("\n❌ Retraining failed")

        return success