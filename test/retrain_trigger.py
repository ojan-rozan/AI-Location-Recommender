"""
Detect new data + auto-trigger retrain.

Run periodik (cronjob weekly):
    python3 tren/retrain_trigger.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.retrain_trigger import RetrainTrigger


def main():
    RetrainTrigger().check_and_trigger()
    sys.exit(0)


if __name__ == "__main__":
    main()