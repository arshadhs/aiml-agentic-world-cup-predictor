'''
Predict results for a match

python -m wc_predictor.models.predict_match --home England --away Ghana --neutral 1
'''

import argparse
import pickle
import pandas as pd

from wc_predictor.features.build_prematch_features import calculate_team_features
from wc_predictor.utils.paths import PROCESSED_DATA_DIR, MODELS_DIR


# Same features used during model training.
# Order must match train_prematch_model.py
FEATURE_COLUMNS = [
    "neutral",
    "home_form_last_5",
    "away_form_last_5",
    "home_avg_goals_scored_last_5",
    "away_avg_goals_scored_last_5",
    "home_avg_goals_conceded_last_5",
    "away_avg_goals_conceded_last_5",
    "home_matches_played_before",
    "away_matches_played_before",
]

def load_model_and_encoder():
    """
    Load the trained model and label encoder from the models folder.
    """

    model_path = MODELS_DIR / "prematch_model.pkl"
    encoder_path = MODELS_DIR / "prematch_label_encoder.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run: python -m wc_predictor.models.train_prematch_model"
        )

    if not encoder_path.exists():
        raise FileNotFoundError(
            f"Label encoder not found: {encoder_path}\n"
            "Run: python -m wc_predictor.models.train_prematch_model"
        )

    # Load trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load label encoder so we can convert numeric prediction back to text
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    return model, label_encoder


def resolve_team_name(df: pd.DataFrame, team_name: str) -> str:
    """
    Find the correct team name from the dataset.

    This allows the user to type:
    - AUSTRALIA
    - australia
    - Australia

    and still match the dataset value:
    - Australia
    """

    # Get all unique team names from both home_team and away_team columns
    all_teams = pd.concat([
        df["home_team"],
        df["away_team"]
    ]).dropna().unique()

    # Try to find a case-insensitive match
    for existing_team in all_teams:
        if existing_team.lower() == team_name.lower():
            return existing_team

    # If no match is found, show a clear error
    raise ValueError(
        f"Team not found in dataset: {team_name}\n"
        "Please check the spelling or use the team name as it appears in the dataset."
    )


def build_match_features(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    neutral: int,
    match_date=None
) -> pd.DataFrame:
    """
    Build one row of pre-match features for a future match.

    We calculate team form using historical matches before match_date.
    If match_date is not given, we use today's date.
    """

    # Make a copy so the original dataframe is not changed
    df = df.copy()

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])

    # If no match date is provided, use today's date
    if match_date is None:
        match_date = pd.Timestamp.today()
    else:
        match_date = pd.to_datetime(match_date)

    # Calculate home team features using previous matches
    home_features = calculate_team_features(
        df=df,
        team=home_team,
        match_date=match_date
    )

    # Calculate away team features using previous matches
    away_features = calculate_team_features(
        df=df,
        team=away_team,
        match_date=match_date
    )

    # Create one prediction row
    match_features = {
        "neutral": int(neutral),

        "home_form_last_5": home_features["form_last_5"],
        "away_form_last_5": away_features["form_last_5"],

        "home_avg_goals_scored_last_5": home_features["avg_goals_scored_last_5"],
        "away_avg_goals_scored_last_5": away_features["avg_goals_scored_last_5"],

        "home_avg_goals_conceded_last_5": home_features["avg_goals_conceded_last_5"],
        "away_avg_goals_conceded_last_5": away_features["avg_goals_conceded_last_5"],

        "home_matches_played_before": home_features["matches_played_before"],
        "away_matches_played_before": away_features["matches_played_before"],
    }

    # Convert dictionary into a dataframe because sklearn expects tabular input
    return pd.DataFrame([match_features], columns=FEATURE_COLUMNS)


def format_result_label(label: str, home_team: str, away_team: str) -> str:
    """
    Convert model result labels into user-friendly wording.

    Example:
    HOME_WIN -> France win
    AWAY_WIN -> Sweden win
    DRAW     -> Draw
    """

    if label == "HOME_WIN":
        return f"{home_team} win"

    if label == "AWAY_WIN":
        return f"{away_team} win"

    return "Draw"


def predict_match(
    home_team: str,
    away_team: str,
    neutral: int,
    match_date=None
):
    """
    Predict result probabilities for one match.
    """

    # Load cleaned historical data
    data_path = PROCESSED_DATA_DIR / "matches_cleaned.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found: {data_path}\n"
            "Run: python -m wc_predictor.data.clean_data"
        )

    df = pd.read_csv(data_path)

    home_team = resolve_team_name(df, home_team)
    away_team = resolve_team_name(df, away_team)

    # Load trained model and label encoder
    model, label_encoder = load_model_and_encoder()

    # Build features for this specific match
    X_match = build_match_features(
        df=df,
        home_team=home_team,
        away_team=away_team,
        neutral=neutral,
        match_date=match_date
    )

    # Predict numeric class
    predicted_class = model.predict(X_match)[0]

    # Convert numeric class back to text label
    predicted_result = label_encoder.inverse_transform([predicted_class])[0]

    # Predict probabilities for all result classes
    probabilities = model.predict_proba(X_match)[0]

    # Convert probabilities into readable labels
    probability_dict = {
        label: probability
        for label, probability in zip(label_encoder.classes_, probabilities)
    }

    return predicted_result, probability_dict, X_match, home_team, away_team


def main():
    '''
    Command line entry point.
    '''

    parser = argparse.ArgumentParser(description="Predict a football match result using pre-match model.")

    # Home Team Arg
    parser.add_argument(
        "--home",
        required=True,
        help="Home team name, e.g. England"
    )

    # Away Team Arg
    parser.add_argument(
        "--away",
        required=True,
        help="Away team name, e.g. Brazil"
    )

    # Neutral venue flag: 1 means neutral, 0 means not neutral
    parser.add_argument(
        "--neutral",
        type=int,
        default=1,
        choices=[0,1],
        help="Use 1 for neutral venuem 0 for non neutral venue"
    )

    # Match Date (optional)
    parser.add_argument(
        "--date",
        default=None,
        help="Match date in YYYY-MM-DD format - (optional)"
    )

    args = parser.parse_args()

    predicted_result, probability_dict, features, home_team, away_team = predict_match(
        home_team=args.home,
        away_team=args.away,
        neutral=args.neutral,
        match_date=args.date
    )

    print()
    print(f"Match: {home_team} vs {away_team}")

    # Convert model label into user-friendly wording
    if predicted_result == "HOME_WIN":
        friendly_result = f"{args.home} win"
    elif predicted_result == "AWAY_WIN":
        friendly_result = f"{args.away} win"
    else:
        friendly_result = "Draw"
    # Convert HOME_WIN/AWAY_WIN/DRAW into readable wording
    friendly_prediction = format_result_label(
        label=predicted_result,
        home_team=home_team,
        away_team=away_team
    )

    print(f"Predicted result: {friendly_prediction}")
    print()

    # Print probability for each possible result
    print("Probabilities:")

    # Print each probability using team names instead of HOME_WIN/AWAY_WIN
    for label, probability in probability_dict.items():
        friendly_label = format_result_label(
            label=label,
            home_team=home_team,
            away_team=away_team
        )

        print(f"- {friendly_label}: {probability:.2%}")

    print()
    print("Features used:")

    # Convert the one-row DataFrame into a dictionary
    feature_values = features.iloc[0].to_dict()

    # Friendly feature labels using actual team names
    feature_labels = {
        "neutral": "Neutral venue",
        "home_form_last_5": f"{home_team} form last 5",
        "away_form_last_5": f"{away_team} form last 5",
        "home_avg_goals_scored_last_5": f"{home_team} avg goals scored last 5",
        "away_avg_goals_scored_last_5": f"{away_team} avg goals scored last 5",
        "home_avg_goals_conceded_last_5": f"{home_team} avg goals conceded last 5",
        "away_avg_goals_conceded_last_5": f"{away_team} avg goals conceded last 5",
        "home_matches_played_before": f"{home_team} matches played before",
        "away_matches_played_before": f"{away_team} matches played before",
    }

    # Print each feature on a separate line
    for feature_name, feature_value in feature_values.items():
        label = feature_labels.get(feature_name, feature_name)

        # Clean up numbers:
        # 12.0 becomes 12, but 2.8 stays 2.8
        if isinstance(feature_value, float) and feature_value.is_integer():
            feature_value = int(feature_value)

        print(f"- {label}: {feature_value}")


if __name__ == "__main__":
    main()