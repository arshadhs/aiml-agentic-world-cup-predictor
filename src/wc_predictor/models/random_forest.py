'''
Model based on Pre-match data
'''

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from wc_predictor.utils.paths import PROCESSED_DATA_DIR, MODELS_DIR

def train_prematch_model():
    df = pd.read_csv(PROCESSED_DATA_DIR / "features_prematch.csv")

    features = [
        "neutral",
        "home_form_last_5",
        "away_form_last_5",
        "home_avg_goals_scored_last_5",
        "away_avg_goals_scored_last_5",
        "home_avg_goals_conceded_last_5",
        "away_avg_goals_conceded_last_5",
        "home_matches_played_before",
        "away_matches_played_before",

        # Elo rating features
        "home_elo_before",
        "away_elo_before",
        "elo_difference",
    ]
    target = "result"

    x = df[features]
    y = df[target]
    
    # Show which features are being used for training
    print("Training features:")
    print(x.columns.tolist())

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y_encoded,
        test_size = 0.2,
        random_state = 42,
        stratify = y_encoded
    )

    model = RandomForestClassifier(
        n_estimators = 200,
        random_state = 42,
        max_depth=10
    )

    model.fit(X_train, y_train)
    # Confirm which feature names were stored inside the trained model
    print("Model trained with features:")
    print(model.feature_names_in_)
    
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classifiation:\n", classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save trained random_forest model
    model_path = MODELS_DIR / "random_forest_model.pkl"
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    # Save label encoder
    encoder_path = MODELS_DIR / "random_forest_label_encoder.pkl"
    with open(encoder_path, "wb") as file:
        pickle.dump(label_encoder, file)


if __name__ == "__main__":
    train_prematch_model()