"""
Full training pipeline CLI.

Flow:
1. Load data (DataLoader)
2. Clean (DataProcessor)
3. Extract features (FeatureExtractor)
4. Train (Trainer)

Run dari project root:
    python3 scripts/train_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.data_loader import DataLoader, ROOT
from src.data.data_processing import DataProcessor
from src.features.extractor import FeatureExtractor
from src.models.trainer import Trainer

JAKARTA_BBOX = {
    "lat_min": -6.40, "lat_max": -6.05,
    "lng_min": 106.65, "lng_max": 107.00,
}

CAFE_KEYWORDS = [
    "cafe", "coffee", "kopi", "café", "espresso",
    "kedai kopi", "coffee shop",
]


def main():
    # Load
    print("\n" + "=" * 60)
    print("LOAD DATA")
    print("=" * 60)
    data = DataLoader().load_all()

    # Cleaning data
    print("\n" + "=" * 60)
    print("\nCLEAN DATA\n")
    print("=" * 60)
    processor = DataProcessor(data, jakarta_bbox=JAKARTA_BBOX, cafe_keywords=CAFE_KEYWORDS)
    processor.save()
    processor.check_owner_overlap(threshold_m=100)

    # Feature extraction
    print("\n" + "=" * 60)
    print("\nFEATURE EXTRACTION\n")
    print("=" * 60)
    extractor = FeatureExtractor(
        cafes=processor.cafes,
        owner=processor.owner,
        poi=processor.poi,
    )
    df_train = extractor.build_training_dataset()

    out = ROOT / "data" / "processed" / "features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(out, index=False)
    print(f"✓ features saved: {out.relative_to(ROOT)}")

    # Train
    print("\n" + "=" * 60)
    print("\nTRAIN MODEL\n")
    print("=" * 60)
    model, metrics = Trainer(df_train).train()

    print("\n" + "=" * 60)
    print("\n")
    print("✅ DONE — model saved ke models/xgb_demand.pkl")
    print("=" * 60)
    print("")
    print(f"   R²  : {metrics['r2']:.4f}")
    print(f"   MAE : {metrics['mae']:.4f}")
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print("\n   MLflow UI: mlflow ui")


if __name__ == "__main__":
    main()