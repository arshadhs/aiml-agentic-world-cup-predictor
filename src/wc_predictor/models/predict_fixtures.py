"""
Predict multiple upcoming football fixtures.

This script:
1. Loads upcoming fixtures from data/raw/upcoming_fixtures.csv
2. Uses the trained pre-match model
3. Predicts each match result
4. Saves predictions to data/processed/fixture_predictions.csv
"""

import pandas as pd

from wc_predictor.models.predict_match import predict_match, format_result_label
from wc_predictor.utils.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR

def get_confidence_label(max_probability: float) -> str:
    """
    Convert the highest probability into a simple confidence label.
    """

    if max_probability >= 0.60:
        return "High"

    if max_probability >= 0.45:
        return "Medium"

    return "Low"

def convert_neutral_value(value) -> int:
    """
    Convert neutral venue values into 1 or 0.

    The Kaggle file may contain values like:
    TRUE, FALSE, True, False, 1, 0
    """

    # If the value is already True or False, convert it directly
    if isinstance(value, bool):
        return int(value)

    # Convert value to lowercase text for easier checking
    value = str(value).strip().lower()

    # Neutral venue means yes
    if value in ["true", "1", "yes", "y"]:
        return 1

    # Not neutral means no
    if value in ["false", "0", "no", "n"]:
        return 0

    # Stop the program if the value is unknown
    raise ValueError(f"Invalid neutral value: {value}")


def predict_fixtures() -> pd.DataFrame:
    """
    Predict results for all matches in upcoming_fixtures.csv.
    """

    # Input file containing future matches
    fixtures_path = RAW_DATA_DIR / "upcoming_fixtures.csv"

    # Check if upcoming_fixtures.csv exists
    if not fixtures_path.exists():
        raise FileNotFoundError(
            f"Could not find fixtures file: {fixtures_path}\n"
            "Please create data/raw/upcoming_fixtures.csv first."
        )

    # Load upcoming fixtures
    fixtures_df = pd.read_csv(fixtures_path)

    # Store all prediction rows here
    predictions = []

    # Loop through each upcoming fixture
    for _, row in fixtures_df.iterrows():

        # Read basic fixture information
        match_date = row["date"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        # Convert TRUE/FALSE neutral value into 1/0
        neutral = convert_neutral_value(row["neutral"])

        # Predict this match using your existing single-match predictor
        predicted_result, probability_dict, features, resolved_home, resolved_away = predict_match(
            home_team=home_team,
            away_team=away_team,
            neutral=neutral,
            match_date=match_date
        )

        # Convert HOME_WIN / AWAY_WIN / DRAW into friendly wording
        friendly_prediction = format_result_label(
            label=predicted_result,
            home_team=resolved_home,
            away_team=resolved_away
        )

        # Get probabilities from the model output
        home_win_probability = probability_dict.get("HOME_WIN", 0)
        draw_probability = probability_dict.get("DRAW", 0)
        away_win_probability = probability_dict.get("AWAY_WIN", 0)

        # Find the highest probability for this match
        max_probability = max(
            home_win_probability,
            draw_probability,
            away_win_probability
        )

        # Convert probability into confidence label
        confidence = get_confidence_label(max_probability)

        # Add prediction result for this fixture
        predictions.append({
            "date": match_date,
            "home_team": resolved_home,
            "away_team": resolved_away,

            # These fields are useful for dashboard display
            "tournament": row.get("tournament", ""),
            "city": row.get("city", ""),
            "country": row.get("country", ""),

            # Model input
            "neutral": neutral,

            # Model prediction
            "predicted_result": friendly_prediction,
            "confidence": confidence,            
            "home_win_probability": round(home_win_probability, 4),
            "draw_probability": round(draw_probability, 4),
            "away_win_probability": round(away_win_probability, 4),
        })

    # Convert list of prediction dictionaries into a DataFrame
    return pd.DataFrame(predictions)


if __name__ == "__main__":

    # Predict all fixtures
    predictions_df = predict_fixtures()

    # Make sure processed data folder exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save predictions
    output_path = PROCESSED_DATA_DIR / "fixture_predictions.csv"
    predictions_df.to_csv(output_path, index=False)

    print(f"Saved fixture predictions to: {output_path}")
    print()
    print(predictions_df.to_string(index=False))