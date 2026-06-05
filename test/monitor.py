"""
Monitor model performance + data drift.

Run periodik (cronjob daily):
    python3 scripts/monitor.py

Exit code:
    0 — healthy
    1 — alert (performance drop OR drift detected)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.data_loader import DataLoader
from src.data.data_processing import DataProcessor
from src.features.extractor import FeatureExtractor
from src.models.monitor import ModelMonitor

JAKARTA_BBOX = {
    "lat_min": -6.40, "lat_max": -6.05,
    "lng_min": 106.65, "lng_max": 107.00,
}

CAFE_KEYWORDS = [
    "cafe", "coffee", "kopi", "café", "espresso",
    "kedai kopi", "coffee shop",
]

def main():
    data = DataLoader().load_all()
    processor = DataProcessor(data, jakarta_bbox=JAKARTA_BBOX, cafe_keywords=CAFE_KEYWORDS)
    current_df = FeatureExtractor(
        cafes=processor.cafes,
        owner=processor.owner,
        poi=processor.poi,
    ).build_training_dataset()

    monitor = ModelMonitor()
    _, has_alert = monitor.run(current_df)

    if has_alert:
        print("\n🚨 Consider retrain — run: python3 test/retrain_trigger.py")
        sys.exit(1)
    else:
        print("\n✅ Model healthy")
        sys.exit(0)


if __name__ == "__main__":
    main()