"""
Update the full football prediction pipeline.

This script:
1. Downloads the latest Kaggle results.csv
2. Cleans the raw match data
3. Builds pre-match features
4. Retrains the pre-match model
"""

import subprocess
import sys


def run_command(command: list[str]) -> None:
    """Run one command and stop if it fails."""

    print()
    print("Running:", " ".join(command))

    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def main() -> None:
    """Run the full update pipeline."""

    # Step 1: Download latest Kaggle data into data/raw/results.csv
    run_command([sys.executable, "scripts/download_kaggle_data.py"])

    # Step 2: Clean raw data and create data/processed/matches_cleaned.csv
    run_command([sys.executable, "-m", "wc_predictor.data.clean_data"])

    # Step 3: Build real pre-match features
    run_command([sys.executable, "-m", "wc_predictor.features.build_prematch_features"])

    # Step 4: Retrain the real prediction model
    # Train the Random Forest model
    run_command([sys.executable, "-m", "wc_predictor.models.random_forest"])

    print()
    print("Pipeline update completed successfully.")


if __name__ == "__main__":
    main()