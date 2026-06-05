"""
Detect new data + auto-trigger retrain.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.retrain_trigger import RetrainTrigger


def main():
    RetrainTrigger().check()
    sys.exit(0)


if __name__ == "__main__":
    main()