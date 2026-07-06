'''
Baseline Model

Note:
This is an early learning baseline and may use score-derived features.
The real prediction model is trained in train_random_forest_model.py.
'''

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from wc_predictor.utils.paths import PROCESSED_DATA_DIR, MODELS_DIR

def train_baseline_model():
    df = pd.read_csv(PROCESSED_DATA_DIR / "features_basic.csv")

    features = [
        "neutral",
        "goal_difference",
        "total_goals"
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
        n_estimators = 100,
        random_state = 42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classifiation:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODELS_DIR / "baseline_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(MODELS_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    print("Saved baseline model.")

if __name__ == "__main__":
    train_baseline_model()