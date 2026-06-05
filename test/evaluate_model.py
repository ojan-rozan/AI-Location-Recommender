"""
Evaluate trained model.

Run dari project root:
    python3 scripts/evaluate_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.data_loader import DataLoader, ROOT
from src.data.data_processing import DataProcessor
from src.features.extractor import FeatureExtractor
from src.models.model import DemandModel
from src.models.evaluator import Evaluator

model_path = ROOT / "models" / "xgb_demand.pkl"
features_path = ROOT / "data" / "processed" / "features.csv"

def main():
    print("=" * 60)
    print("\nMODEL EVALUATION\n")
    print("=" * 60)

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Run: python3 test/train_model.py dulu")
        sys.exit(1)

    print(f"\nLoading model from {model_path.relative_to(ROOT)}")
    model = DemandModel.load(model_path)

    # Load atau re-build features
    if features_path.exists():
        print(f"Loading features from {features_path.relative_to(ROOT)}")
        df = pd.read_csv(features_path)
    else:
        print("Features not found, re-building...")
        data = DataLoader().load_all()
        processor = DataProcessor(data)
        df = FeatureExtractor(
            cafes=processor.cafes,
            owner=processor.owner,
            poi=processor.poi,
        ).build_training_dataset()

    print(f"Total rows: {len(df)}")

    X = df[model.feature_cols].fillna(0)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    ev = Evaluator(model=model, X_test=X_test, y_test=y_test,
                   X_train=X_train, y_train=y_train)
    ev.evaluate()


if __name__ == "__main__":
    main()