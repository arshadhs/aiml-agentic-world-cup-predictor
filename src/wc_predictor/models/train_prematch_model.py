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
    ]
    target = "result"

    x = df[features]
    y = df[target]

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
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classifiation:\n", classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODELS_DIR / "prematch_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(MODELS_DIR / "prematch_label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    print("Saved prematch model.")

if __name__ == "__main__":
    train_prematch_model()