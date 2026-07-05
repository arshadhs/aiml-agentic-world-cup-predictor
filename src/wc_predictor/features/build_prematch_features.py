'''
Build real pre-match features.

Historical results -> pre-match features -> real prediction model

Only use information available before the match starts.
Do NOT use home_score, away_score, goal_difference, or total_goals as model inputs.
'''

import pandas as pd

from wc_predictor.utils.paths import PROCESSED_DATA_DIR


def get_team_history(df: pd.DataFrame, team: str, match_date) -> pd.DataFrame:
    '''
    Return all matches played by a team before a given match date.
    This prevents the model from using future information.
    '''

    previous_matches = df[
        (
            (df["home_team"] == team) |
            (df["away_team"] == team)
        ) &
        (df["date"] < match_date)
    ].copy()

    # Sort by date so the latest matches are at the bottom
    return previous_matches.sort_values("date")


def calculate_team_features(
        df: pd.DataFrame,
        team: str,
        match_date,
        last_n: int = 5
    ) -> dict:
    '''
    Calculate pre-match statistics for one team.

    Example:
    Before England vs Ghana, calculate England's form using only
    matches England played before that date.
    '''

    # Get previous matches for this team
    history = get_team_history(df, team, match_date)

    # If the team has no previous matches, return default values
    if history.empty:
        return {
            "form_last_5": 0,
            "avg_goals_scored_last_5": 0,
            "avg_goals_conceded_last_5": 0,
            "matches_played_before": 0,
        }

    # Take only the most recent N matches
    recent = history.tail(last_n)

    points = []
    goals_scored = []
    goals_conceded = []

    # Go through each recent match and calculate stats from this team's view
    for _, row in recent.iterrows():

        # If the team was home team
        if row["home_team"] == team:
            team_goals = row["home_score"]
            opponent_goals = row["away_score"]

        # If the team was away team
        else:
            team_goals = row["away_score"]
            opponent_goals = row["home_score"]

        goals_scored.append(team_goals)
        goals_conceded.append(opponent_goals)

        # Win = 3 points, draw = 1 point, loss = 0 points
        if team_goals > opponent_goals:
            points.append(3)
        elif team_goals == opponent_goals:
            points.append(1)
        else:
            points.append(0)

    return {
        "form_last_5": sum(points),
        "avg_goals_scored_last_5": sum(goals_scored) / len(goals_scored),
        "avg_goals_conceded_last_5": sum(goals_conceded) / len(goals_conceded),
        "matches_played_before": len(history),
    }


def build_prematch_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Build one row of pre-match features for every match.
    Each row uses only historical data before that match date.
    '''

    # Make a copy so we do not modify the original dataframe
    df = df.copy()

    # Convert date column to datetime so date comparisons work correctly
    df["date"] = pd.to_datetime(df["date"])

    # Sort matches by date from oldest to newest
    df = df.sort_values("date")

    rows = []

    # Build features match by match
    for _, row in df.iterrows():

        # Use current row values, not whole dataframe columns
        match_date = row["date"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        # Calculate home team stats before this match
        home_features = calculate_team_features(df, home_team, match_date)

        # Calculate away team stats before this match
        away_features = calculate_team_features(df, away_team, match_date)

        # Add one row of features for this match
        rows.append({
            # Match information
            "date": row["date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "tournament": row["tournament"],
            "neutral": row["neutral"],

            # Home team pre-match features
            "home_form_last_5": home_features["form_last_5"],
            "home_avg_goals_scored_last_5": home_features["avg_goals_scored_last_5"],
            "home_avg_goals_conceded_last_5": home_features["avg_goals_conceded_last_5"],
            "home_matches_played_before": home_features["matches_played_before"],

            # Away team pre-match features
            "away_form_last_5": away_features["form_last_5"],
            "away_avg_goals_scored_last_5": away_features["avg_goals_scored_last_5"],
            "away_avg_goals_conceded_last_5": away_features["avg_goals_conceded_last_5"],
            "away_matches_played_before": away_features["matches_played_before"],

            # Target label the model will learn to predict
            "result": row["result"],
        })

    # Convert list of dictionaries into a dataframe
    return pd.DataFrame(rows)


if __name__ == "__main__":

    # Load cleaned match data
    input_path = PROCESSED_DATA_DIR / "matches_cleaned.csv"
    df = pd.read_csv(input_path)

    # Build pre-match feature dataset
    features_df = build_prematch_features(df)

    # Save the new feature file
    output_path = PROCESSED_DATA_DIR / "features_prematch.csv"
    features_df.to_csv(output_path, index=False)

    print(f"Saved pre-match features to: {output_path}")
    print(features_df.head())
    print(features_df.info())