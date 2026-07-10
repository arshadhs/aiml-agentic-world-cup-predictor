'''
train.py

Main training entry point.

Example usage:

python -m wc_predictor.features.build_prematch_features
python -m wc_predictor.models.random_forest
python -m wc_predictor.models.predict_fixtures --model random_forest
streamlit run app\streamlit_app.py

python -m wc_predictor.features.build_prematch_features
python -m wc_predictor.models.xgboost_model
python -m wc_predictor.models.predict_fixtures --model xgboost
streamlit run app\streamlit_app.py

python -m wc_predictor.models.train --model random_forest
python -m wc_predictor.models.train --model xgboost
python -m wc_predictor.models.train --model logistic
'''