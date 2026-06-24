'''
Load Data
'''

import pandas as pd
from wc_predictor.utils.paths import RAW_DATA_DIR

def load_results(filename: str = "results.csv") -> pd.DataFrame:
    file_path = RAW_DATA_DIR / "filename

    if not file_path.exists()