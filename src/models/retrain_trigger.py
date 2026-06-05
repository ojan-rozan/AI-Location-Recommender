"""Auto-retrain trigger: detect new data and run training."""

import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.data_loader import ROOT


state_file = ROOT / "models" / "data_state.json"
meta_file  = ROOT / "models" / "model_meta.json"

default_data_files = [
    ROOT / "data" / "raw" / "cafes_gmaps.csv",
    ROOT / "data" / "helper" / "owner_stores.csv",
]


class RetrainTrigger:
    """Detect change in data + trigger retrain kalau threshold lampaui."""

    row_threshold  = 0.10   # 10% new data → retrain
    time_threshold_days = 30    # model > 30 hari → retrain

    def __init__(self, data_files=None, state_file=state_file):
        self.data_files = data_files or default_data_files
        self.state_file = Path(state_file or default_data_files)

    def _hash_file(self, path: Path):
        h = hashlib.md5()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def _file_state(self, path: Path):
        path = Path(path)
        if not path.exists():
            return None
        return {
            "hash": self._hash_file(path),
            "rows": len(pd.read_csv(path)),
            "size_kb": path.stat().st_size // 1024,
            "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }

    def _load_state(self):
        return json.loads(self.state_file.read_text()) if self.state_file.exists() else {}

    def _save_state(self, state: dict):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(state, indent=2))

    def detect_changes(self):
        """Compare current data vs last known state."""
        prev = self._load_state()
        current = {}
        changes = []

        for path in self.data_files:
            key = str(Path(path).relative_to(ROOT))
            cur_state = self._file_state(path)
            current[key] = cur_state

            if cur_state is None:
                continue

            prev_state = prev.get(key)
            if prev_state is None:
                changes.append({"file": key, "reason": "new_file"})
                continue

            if cur_state["hash"] != prev_state["hash"]:
                row_diff = cur_state["rows"] - prev_state["rows"]
                row_pct  = row_diff / max(prev_state["rows"], 1)
                changes.append({
                    "file": key,
                    "reason": "content_changed",
                    "row_diff": row_diff,
                    "row_pct": row_pct,
                    "should_retrain": abs(row_pct) >= self.row_threshold,
                })

        return current, changes

    def days_since_last_train(self):
        if not meta_file.exists():
            return 9999
        meta = json.loads(meta_file.read_text())
        trained_at = datetime.fromisoformat(meta["trained_at"])
        return (datetime.now() - trained_at).days


    def run_retrain(self, use_optuna=False):
        """Execute training script."""
        print("\n🔄 TRIGGERING RETRAIN")
        cmd = ["python3", str(ROOT / "test" / "train_model.py")]
        if use_optuna:
            cmd.extend(["--optuna", "20"])
        return subprocess.run(cmd, cwd=str(ROOT)).returncode == 0

    def check_and_trigger(self, use_optuna=False):
        """check and trigger function"""
        timestamp = datetime.now().isoformat()
        print(f"\n{'='*60}\nRETRAIN TRIGGER CHECK — {timestamp}\n{'='*60}")

        current_state, changes = self.detect_changes()
        days_old = self.days_since_last_train()
        print(f"Days since last train: {days_old}")

        should_retrain = False
        reasons = []

        if days_old >= self.time_threshold_days:
            should_retrain = True
            reasons.append(f"model is {days_old} days old (>= {self.time_threshold_days})")

        for c in changes:
            print(f"\nChange in {c['file']}: {c['reason']}")
            if c["reason"] == "new_file":
                should_retrain = True
                reasons.append(f"new file: {c['file']}")
            elif c.get("should_retrain"):
                should_retrain = True
                reasons.append(
                    f"{c['file']} changed {c['row_pct']*100:+.1f}% rows ({c['row_diff']:+d})"
                )
            else:
                print("  → minor change, skip retrain")

        if should_retrain:
            print("\n🚨 RETRAIN TRIGGERED\nReasons:")
            for r in reasons:
                print(f"  - {r}")
            success = self.run_retrain(use_optuna=use_optuna)
            if success:
                print("\n✅ Retrain success — updating state")
                self._save_state(current_state)
            else:
                print("\n❌ Retrain failed — state not updated")
            return success

        print("\n✓ No retrain needed")
        self._save_state(current_state)
        return False