'''
Train XGBoost Model for Football Match Prediction
'''

import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from wc_predictor.utils.paths import PROCESSED_DATA_DIR, MODELS_DIR

# Input Features
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

def train_xgboost_model() -> None:
    '''
    Train and save XGBoost Model
    '''

    # Load pre-match feature dataset
    input_path = PROCESSED_DATA_DIR / "features_prematch.csv"
    df = pd.read_csv(input_path)

    # Keep only rows where all required feature values exist
    df = df.dropna(subset=FEATURE_COLUMNS + ["result"])

    # Model Inputs
    X = df[FEATURE_COLUMNS]

    # Model target: HOME_WIN, DRAW, AWAY_WIN
    y = df["result"]

    # Convert text labels into numbers because XGBoost needs numeric labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size = 0.2,
        random_state = 42,      # random_state makes the result repeatable
        stratify = y_encoded,   # stratify keeps class distribution similar in train/test sets
    )

    # Create XGBoost classifier
    model = XGBClassifier(
        n_estimators=300,          # Number of boosting trees
        max_depth=4,               # Limit tree depth to reduce overfitting
        learning_rate=0.05,        # Smaller learning rate = slower but safer learning
        subsample=0.8,             # Use 80% of rows per tree
        colsample_bytree=0.8,      # Use 80% of features per tree
        objective="multi:softprob",# Multi-class probability prediction
        eval_metric="mlogloss",    # Good metric for multi-class probabilities
        random_state=42,
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("XGBoost model trained.")
    print()
    print(f"Accuracy: {accuracy:.4f}")
    print()

    # Show confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

    # Show precision, recall and F1-score
    print("Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
        )
    )

    # Make sure models folder exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save trained XGBoost model
    model_path = MODELS_DIR / "xgboost_model.pkl"
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    # Save label encoder
    encoder_path = MODELS_DIR / "xgboost_label_encoder.pkl"
    with open(encoder_path, "wb") as file:
        pickle.dump(label_encoder, file)

    print(f"Saved XGBoost model to: {model_path}")
    print(f"Saved label encoder to: {encoder_path}")

if __name__ == "__main__":
    train_xgboost_model()
