'''
Clean Data
'''

import pandas as pd
from wc_predictor.data.load_data import load_results
from wc_predictor.utils.paths import PROCESSED_DATA_DIR

def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    required_columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
    ]

    df = df[required_columns]
    df = df.dropna(subset=[
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ])

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df["result"] = df.apply(get_result, axis = 1)

    return df

def get_result(row):
    if row["home_score"] > row["away_score"]:
        return "HOME_WIN"
    elif row["home_score"] < row["away_score"]:
        return "AWAY_WIN"
    else:
        return "DRAW"

if __name__ == "__main__":
    raw_df = load_results()
    clean_df = clean_results(raw_df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "matches_cleaned.csv"

    clean_df.to_csv(output_path, index = False)

    print("Saved cleaned data to: {output_path}")
    print(clean_df.tail())