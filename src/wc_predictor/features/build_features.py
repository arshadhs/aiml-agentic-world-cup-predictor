'''
Features Buildup
'''

import pandas as pd
from wc_predictor.utils.paths import PROCESSED_DATA_DIR

'''
Build real pre-match features → train real model → predict future match

home_form_last_5
away_form_last_5
home_avg_goals_scored_last_5
away_avg_goals_scored_last_5
home_avg_goals_conceded_last_5
away_avg_goals_conceded_last_5
home_matches_played_before
away_matches_played_before
neutral
result
'''


def build_basic_features(df= pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["goal_difference"] = df["home_score"] - df["away_score"]
    df["total_goals"] = df["home_score"] + df["away_score"]

    df["is_draw"] = (df["result"] == "DRAW").astype(int)
    df["home_win"] = (df["result"] == "HOME_WIN").astype(int)
    df["away_win"] = (df["result"] == "AWAY_WIN").astype(int)

    df["neutral"] = df["neutral"].astype(int)

    return df

if __name__ == "__main__":
    input_path = PROCESSED_DATA_DIR / "matches_cleaned.csv"
    df= pd.read_csv(input_path)

    features_df= build_basic_features(df)

    output_path = PROCESSED_DATA_DIR / "features_basic.csv"
    features_df.to_csv(output_path, index = False)

    print(f"Saved features to: {output_path}")
    print(features_df.head())