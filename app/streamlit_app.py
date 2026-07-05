"""
Streamlit dashboard for the AI World Cup Predictor.

This dashboard:
1. Shows batch predictions from fixture_predictions.csv
2. Highlights confidence level
3. Allows single match prediction
"""

import pandas as pd
import streamlit as st

from wc_predictor.models.predict_match import predict_match, format_result_label
from wc_predictor.utils.paths import PROCESSED_DATA_DIR


def get_confidence_label(max_probability: float) -> str:
    """
    Convert highest probability into a simple confidence label.
    """

    if max_probability >= 0.60:
        return "High"

    if max_probability >= 0.45:
        return "Medium"

    return "Low"


def load_fixture_predictions() -> pd.DataFrame:
    """
    Load saved fixture predictions from CSV.
    """

    # Path to prediction output file
    predictions_path = PROCESSED_DATA_DIR / "fixture_predictions.csv"

    # If the file does not exist, show an error
    if not predictions_path.exists():
        st.error(
            "fixture_predictions.csv not found. "
            "Run: python -m wc_predictor.models.predict_fixtures"
        )
        return pd.DataFrame()

    # Load predictions
    df = pd.read_csv(predictions_path)

    # Add confidence label if not already available
    if "confidence" not in df.columns:
        probabilities = df[
            [
                "home_win_probability",
                "draw_probability",
                "away_win_probability",
            ]
        ]

        df["max_probability"] = probabilities.max(axis=1)
        df["confidence"] = df["max_probability"].apply(get_confidence_label)

    return df


def show_fixture_predictions(df: pd.DataFrame) -> None:
    """
    Show upcoming fixture predictions in a clean table.
    """

    st.subheader("Upcoming Fixture Predictions")

    if df.empty:
        return

    # Create a display-friendly copy
    display_df = df.copy()

    # Convert probabilities to percentages
    display_df["Home Win %"] = (display_df["home_win_probability"] * 100).round(1)
    display_df["Draw %"] = (display_df["draw_probability"] * 100).round(1)
    display_df["Away Win %"] = (display_df["away_win_probability"] * 100).round(1)

    # Select useful columns for dashboard
    display_df = display_df[
        [
            "date",
            "home_team",
            "away_team",
            "city",
            "country",
            "predicted_result",
            "confidence",
            "Home Win %",
            "Draw %",
            "Away Win %",
        ]
    ]

    # Show table
    st.dataframe(display_df, use_container_width=True)


def show_single_match_predictor() -> None:
    """
    Show form where user can predict one match.
    """

    st.subheader("Single Match Predictor")

    # Layout columns
    col1, col2 = st.columns(2)

    with col1:
        home_team = st.text_input("First team", value="England")

    with col2:
        away_team = st.text_input("Second team", value="France")

    col3, col4 = st.columns(2)

    with col3:
        match_date = st.date_input("Match date")

    with col4:
        neutral = st.checkbox("Neutral venue", value=True)

    # Predict button
    if st.button("Predict Match"):

        # Convert checkbox value into 1 or 0
        neutral_value = 1 if neutral else 0

        # Run prediction
        predicted_result, probability_dict, features, resolved_home, resolved_away = predict_match(
            home_team=home_team,
            away_team=away_team,
            neutral=neutral_value,
            match_date=str(match_date),
        )

        # Convert prediction label into friendly wording
        friendly_prediction = format_result_label(
            label=predicted_result,
            home_team=resolved_home,
            away_team=resolved_away,
        )

        # Read probabilities
        home_probability = probability_dict.get("HOME_WIN", 0)
        draw_probability = probability_dict.get("DRAW", 0)
        away_probability = probability_dict.get("AWAY_WIN", 0)

        # Find confidence
        max_probability = max(
            home_probability,
            draw_probability,
            away_probability,
        )

        confidence = get_confidence_label(max_probability)

        # Display result
        st.success(f"Prediction: {friendly_prediction}")
        st.info(f"Confidence: {confidence}")

        # Show probability metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric(f"{resolved_home} win", f"{home_probability:.1%}")

        with metric_col2:
            st.metric("Draw", f"{draw_probability:.1%}")

        with metric_col3:
            st.metric(f"{resolved_away} win", f"{away_probability:.1%}")

        # Show probability bar chart
        chart_df = pd.DataFrame({
            "Outcome": [
                f"{resolved_home} win",
                "Draw",
                f"{resolved_away} win",
            ],
            "Probability": [
                home_probability,
                draw_probability,
                away_probability,
            ],
        })

        st.bar_chart(
            chart_df,
            x="Outcome",
            y="Probability",
        )

        # Show features used by model
        st.subheader("Features Used by Model")
        st.dataframe(features, use_container_width=True)


def main() -> None:
    """
    Main Streamlit app.
    """

    st.set_page_config(
        page_title="AI World Cup Predictor",
        page_icon="⚽",
        layout="wide",
    )

    st.title("⚽ AI World Cup Predictor")
    st.write(
        "Predict football match outcomes using historical international match data "
        "and machine learning."
    )

    # Load batch predictions
    predictions_df = load_fixture_predictions()

    # Show batch predictions table
    show_fixture_predictions(predictions_df)

    st.divider()

    # Show single match predictor
    show_single_match_predictor()


if __name__ == "__main__":
    main()