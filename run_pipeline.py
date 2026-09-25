"""End-to-end entry point: raw CSVs -> daily demand panel -> features -> trained models.

    python run_pipeline.py
"""
from src import data_prep, features, train


def main() -> None:
    print("[1/3] Building daily demand panel from raw transactions...")
    data_prep.main()

    print("\n[2/3] Engineering calendar/lag/rolling features...")
    features.main()

    print("\n[3/3] Training and evaluating XGBoost + PyTorch models...")
    train.main()


if __name__ == "__main__":
    main()
