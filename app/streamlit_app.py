"""
Streamlit dashboard for the AI World Cup Predictor.

This dashboard:
1. Shows upcoming fixture predictions
2. Allows model selection: Random Forest or XGBoost
3. Compares Random Forest and XGBoost predictions on demand
4. Allows single match prediction
5. Highlights the predicted winning team
"""

import pandas as pd
import streamlit as st

from wc_predictor.models.predict_match import predict_match, format_result_label
from wc_predictor.models.predict_fixtures import predict_fixtures
from wc_predictor.utils.paths import PROCESSED_DATA_DIR


# Available models in the dashboard
MODEL_OPTIONS = ["random_forest", "xgboost"]

# User-friendly model names
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def format_model_name(model_name: str) -> str:
    """
    Convert internal model name into user-friendly label.
    """

    return MODEL_LABELS.get(model_name, model_name)


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


def get_confidence_label(max_probability: float) -> str:
    """
    Convert highest probability into a simple confidence label.
    """

    if max_probability >= 0.60:
        return "High"

    if max_probability >= 0.45:
        return "Medium"

    return "Low"


@st.cache_data(show_spinner=False)
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
    teams = pd.concat(
        [
            df["home_team"],
            df["away_team"],
        ]
    ).dropna().unique()

    # Sort teams alphabetically
    return sorted(teams)


@st.cache_data(show_spinner=False)
def load_predictions_for_model(model_name: str) -> pd.DataFrame:
    """
    Generate fixture predictions for one selected model.

    Cached so Streamlit does not rerun predictions on every page refresh.
    """

    return predict_fixtures(
        model_name=model_name,
        verbose=False,
    )


@st.cache_data(show_spinner=False)
def load_model_comparison_predictions() -> pd.DataFrame:
    """
    Generate fixture predictions from both Random Forest and XGBoost.

    Cached so the comparison is not regenerated repeatedly.
    """

    prediction_frames = []

    # Run predictions for each model
    for model_name in MODEL_OPTIONS:
        try:
            model_df = predict_fixtures(
                model_name=model_name,
                verbose=False,
            )

            prediction_frames.append(model_df)

        except Exception as error:
            st.warning(
                f"Could not load predictions for {format_model_name(model_name)}: {error}"
            )

    # If no model could generate predictions, return empty dataframe
    if not prediction_frames:
        return pd.DataFrame()

    # Combine model outputs
    combined_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    return combined_df


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
        styles[row.index.get_loc("Home Team")] = (
            "font-weight: bold; background-color: #d4edda;"
        )

    # Highlight away team if predicted result says away team will win
    elif predicted_result == f"{away_team} win":
        styles[row.index.get_loc("Away Team")] = (
            "font-weight: bold; background-color: #d4edda;"
        )

    # Highlight predicted result if draw
    elif predicted_result == "Draw":
        styles[row.index.get_loc("Predicted Result")] = (
            "font-weight: bold; background-color: #fff3cd;"
        )

    return styles


def prepare_predictions_for_display(
    df: pd.DataFrame,
    include_model: bool = False,
) -> pd.DataFrame:
    """
    Convert raw prediction output into a user-friendly display table.
    """

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

    # Add user-friendly model name if needed
    if include_model:
        display_df["Model"] = display_df["model_used"].apply(format_model_name)

    # Rename columns to user-friendly labels
    display_df = display_df.rename(
        columns={
            "home_team": "Home Team",
            "away_team": "Away Team",
            "predicted_result": "Predicted Result",
            "confidence": "Confidence",
        }
    )

    # Select and order columns
    if include_model:
        display_df = display_df[
            [
                "Date",
                "Home Team",
                "Away Team",
                "Model",
                "Predicted Result",
                "Confidence",
                "Home Win Probability",
                "Draw Probability",
                "Away Win Probability",
                "Location",
            ]
        ]
    else:
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

    return display_df


def show_fixture_predictions(df: pd.DataFrame) -> None:
    """
    Show upcoming fixture predictions in a clean user-friendly table.
    """

    st.subheader("Upcoming Fixture Predictions")

    if df.empty:
        st.warning("No fixture predictions available.")
        return

    # Prepare display table
    display_df = prepare_predictions_for_display(
        df=df,
        include_model=False,
    )

    # Apply row-wise highlighting
    styled_df = display_df.style.apply(
        highlight_predicted_winner,
        axis=1,
    )

    # Show styled table without index
    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
    )


def show_model_comparison_predictions(df: pd.DataFrame) -> None:
    """
    Show Random Forest and XGBoost predictions in one comparison table.
    """

    st.subheader("Model Comparison")

    if df.empty:
        st.warning("No comparison predictions available.")
        return

    # Prepare display table with model column
    display_df = prepare_predictions_for_display(
        df=df,
        include_model=True,
    )

    # Apply row-wise highlighting
    styled_df = display_df.style.apply(
        highlight_predicted_winner,
        axis=1,
    )

    # Show table without index
    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
    )


def show_single_match_predictor() -> None:
    """
    Show form where user can predict one match.
    """

    st.subheader("Single Match Predictor")

    # Load valid team names for dropdowns
    teams = load_team_names()

    # Choose model for single match prediction
    selected_model = st.selectbox(
        "Choose model for single match",
        MODEL_OPTIONS,
        index=0,
        format_func=format_model_name,
        key="single_match_model",
    )

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

        try:
            # Run prediction
            predicted_result, probability_dict, features, resolved_home, resolved_away = predict_match(
                home_team=home_team,
                away_team=away_team,
                neutral=neutral_value,
                match_date=str(match_date),
                model_name=selected_model,
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.exception(error)
            return

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
        st.info(
            f"Model: {format_model_name(selected_model)} | Confidence: {confidence}"
        )

        # Show probability metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric(f"{resolved_home} win", f"{home_probability:.1%}")

        with metric_col2:
            st.metric("Draw", f"{draw_probability:.1%}")

        with metric_col3:
            st.metric(f"{resolved_away} win", f"{away_probability:.1%}")

        # Show probability bar chart
        chart_df = pd.DataFrame(
            {
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
            }
        )

        st.bar_chart(
            chart_df,
            x="Outcome",
            y="Probability",
        )

        # Show features used by model
        st.subheader("Features Used by Model")

        st.dataframe(
            features,
            width="stretch",
            hide_index=True,
        )


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
            **Pipeline**

            Data → Cleaning → Feature Engineering → Model Training → Single-Match Prediction → Batch Fixture Prediction

            I am using a historical international football results dataset for training the machine learning model.

            During feature engineering:

            - the model learns from many historical matches
            - each individual prediction is based on **pre-match features**
            - these features are calculated from recent team history

            For each fixture, the prediction uses features such as:

            - Recent form from the team's last **5 matches**
            - Average goals scored in the last **5 matches**
            - Average goals conceded in the last **5 matches**
            - Number of previous matches played in the historical dataset
            - Whether the match is played at a neutral venue
            - Elo-style team strength rating before the match

            The model does **not** use the final score of the fixture being predicted.
            It only uses information that would be available before the match.

            The features are passed into trained machine learning classifiers such as
            **Random Forest** and **XGBoost**, which estimate the probability of each outcome:
            **first team win**, **draw**, or **second team win**.
            """
        )

    selected_model = st.selectbox(
        "Choose model for fixture predictions",
        MODEL_OPTIONS,
        index=0,
        format_func=format_model_name,
        key="fixture_model",
    )

    try:
        with st.spinner(
            f"Generating predictions using {format_model_name(selected_model)}..."
        ):
            predictions_df = load_predictions_for_model(
                model_name=selected_model,
            )

        show_fixture_predictions(predictions_df)

    except Exception as error:
        st.error(
            f"Could not generate fixture predictions using "
            f"{format_model_name(selected_model)}."
        )
        st.exception(error)

    with st.expander("Compare Random Forest and XGBoost"):
        st.write(
            "Click the button below to generate and compare predictions from both models."
        )

        if st.button("Generate model comparison"):
            with st.spinner("Generating model comparison..."):
                comparison_df = load_model_comparison_predictions()

            show_model_comparison_predictions(comparison_df)

    st.divider()

    show_single_match_predictor()

    st.markdown("---")

    st.markdown(
        """
        **AI World Cup Predictor**  
        Built by **Arshad Siddiqui** | [LinkedIn](https://www.linkedin.com/in/arshadsiddiqui/)
        """
    )


if __name__ == "__main__":
    main()