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


def load_team_names() -> list[str]:
    """
    Load all team names from matches_cleaned.csv.

    This gives the dashboard a dropdown list of valid teams.
    """

    # Path to cleaned historical match data
    data_path = PROCESSED_DATA_DIR / "matches_cleaned.csv"

    # If cleaned data is missing, return a small fallback list
    if not data_path.exists():
        return ["England", "France", "Brazil", "Norway", "Argentina", "Spain"]

    # Load cleaned match data
    df = pd.read_csv(data_path)

    # Combine home and away team names
    teams = pd.concat([
        df["home_team"],
        df["away_team"],
    ]).dropna().unique()

    # Sort teams alphabetically
    return sorted(teams)


def highlight_predicted_winner(row):
    """
    Highlight the team that matches the predicted result.
    """

    # Default style for all columns
    styles = [""] * len(row)

    home_team = row["Home Team"]
    away_team = row["Away Team"]
    predicted_result = row["Predicted Result"]

    # Highlight home team if predicted result says home team will win
    if predicted_result == f"{home_team} win":
        styles[row.index.get_loc("Home Team")] = "font-weight: bold; background-color: #d4edda;"

    # Highlight away team if predicted result says away team will win
    elif predicted_result == f"{away_team} win":
        styles[row.index.get_loc("Away Team")] = "font-weight: bold; background-color: #d4edda;"

    # Highlight predicted result if draw
    elif predicted_result == "Draw":
        styles[row.index.get_loc("Predicted Result")] = "font-weight: bold; background-color: #fff3cd;"

    return styles


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


def format_date_with_suffix(date_value) -> str:
    """
    Format date like:
    Sat, 26th July 2026
    """

    # Convert value to datetime
    date = pd.to_datetime(date_value)

    # Get day number
    day = date.day

    # Work out day suffix: st, nd, rd, th
    if 11 <= day <= 13:
        suffix = "th"
    elif day % 10 == 1:
        suffix = "st"
    elif day % 10 == 2:
        suffix = "nd"
    elif day % 10 == 3:
        suffix = "rd"
    else:
        suffix = "th"

    # Format final date string
    return date.strftime(f"%a, {day}{suffix} %B %Y")


def show_fixture_predictions(df: pd.DataFrame) -> None:
    """
    Show upcoming fixture predictions in a clean user-friendly table.
    """

    st.subheader("Upcoming Fixture Predictions")

    if df.empty:
        return

    # Create a display-friendly copy
    display_df = df.copy()

    # Create formatted date column
    display_df["Date"] = display_df["date"].apply(format_date_with_suffix)

    # Create location column from city and country
    display_df["Location"] = (
        display_df["city"].fillna("").astype(str)
        + ", "
        + display_df["country"].fillna("").astype(str)
    )

    # Clean up cases where city/country may be missing
    display_df["Location"] = display_df["Location"].str.strip(", ")

    # Convert probabilities to percentages
    display_df["Home Win Probability"] = (
        display_df["home_win_probability"] * 100
    ).round(1).astype(str) + "%"

    display_df["Draw Probability"] = (
        display_df["draw_probability"] * 100
    ).round(1).astype(str) + "%"

    display_df["Away Win Probability"] = (
        display_df["away_win_probability"] * 100
    ).round(1).astype(str) + "%"

    # Rename columns to user-friendly labels
    display_df = display_df.rename(
        columns={
            "home_team": "Home Team",
            "away_team": "Away Team",
            "predicted_result": "Predicted Result",
            "confidence": "Confidence",
        }
    )

    # Select and order columns exactly how you want them shown
    display_df = display_df[
        [
            "Date",
            "Home Team",
            "Away Team",
            "Predicted Result",
            "Confidence",
            "Home Win Probability",
            "Draw Probability",
            "Away Win Probability",
            "Location",
        ]
    ]

    # Apply row-wise highlighting
    styled_df = display_df.style.apply(
        highlight_predicted_winner,
        axis=1,
    )

    # Show styled table without index
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
    )


def show_single_match_predictor() -> None:
    """
    Show form where user can predict one match.
    """

    st.subheader("Single Match Predictor")

    # Load valid team names for dropdowns
    teams = load_team_names()

    col1, col2 = st.columns(2)

    with col1:
        home_team = st.selectbox(
            "First team",
            teams,
            index=teams.index("England") if "England" in teams else 0,
        )

    with col2:
        away_team = st.selectbox(
            "Second team",
            teams,
            index=teams.index("France") if "France" in teams else 1,
        )

    col3, col4 = st.columns(2)

    with col3:
        match_date = st.date_input("Match date")

    with col4:
        neutral = st.checkbox("Neutral venue", value=True)

    if home_team == away_team:
        st.warning("Please choose two different teams.")
        return

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

    with st.expander("How the model works"):
        st.markdown(
            """
            Data → Cleaning → Feature Engineering → Model Training → Single-Match Prediction → Batch Fixture Prediction
    
            I am using historical international football results dataset
            for training the machine learning model.
    
            During feature engineering,
            - the **model learns from many historical matches**,
            - but **each individual prediction** is based on **pre-match features** calculated
            from **recent team history**.
    
            For each fixture, the prediction uses features such as:
    
            - Recent form from the team's last **5 matches**
            - Average goals scored in the last **5 matches**
            - Average goals conceded in the last **5 matches**
            - Number of previous matches played in the historical dataset
            - Whether the match is played at a neutral venue
      
            The features are passed into a trained machine learning classifier,
            currently **Random Forest**, which estimates the probability of each outcome:
            **first team win**, **draw**, or **second team win**.
            """
    )

    # Load batch predictions
    predictions_df = load_fixture_predictions()

    # Show batch predictions table
    show_fixture_predictions(predictions_df)

    st.divider()

    # Show single match predictor
    show_single_match_predictor()

    st.markdown("---")
    
    st.markdown(
        """
        **AI World Cup Predictor**  
        Built by **Arshad Siddiqui**
    
        [LinkedIn](https://www.linkedin.com/in/arshadsiddiqui/) |
        """
    )

if __name__ == "__main__":
    main()
