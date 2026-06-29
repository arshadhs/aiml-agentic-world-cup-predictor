'''
Download Kaggle File
'''
import shutil
from pathlib import Path
import kagglehub

from wc_predictor.utils.paths import RAW_DATA_DIR

DATASET_NAME = "martj42/international-football-results-from-1872-to-2017"

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"

def download_kaggle_results() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download latest version
    dataset_path  = Path(kagglehub.dataset_download(DATASET_NAME))

    print("Path to dataset files:", dataset_path )
    
    source_file = dataset_path / "results.csv"
    destination_file = RAW_DATA_DIR / "results.csv"

    if not source_file.exists():
        raise FileNotFoundError(f"Could not find Kaggle file: {source_file}")

    shutil.copy2(source_file, destination_file)

    print("Kaggle dataset downloaded successfully.")
    print(f"Source: {source_file}")
    print(f"Saved to: {destination_file}")

if __name__ == "__main__":
    download_kaggle_results()