# Agentic AI World Cup Predictor

An AI-assisted probabilistic football prediction and simulation system.

## Overview

## Problem Statement

## Key Features

## Architecture

## Dataset

## ML Models

## Agentic AI Workflow

## RAG Component

## Simulation Engine

## Dashboard

## Installation

## How to Run

## Model Evaluation

## Future Improvements


---

```text
world-cup-ai-predictor/
│
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── requirements.txt
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
│
├── data/
│   ├── raw/
│   │   ├── historical_results.csv
│   │   ├── world_cup_fixtures.csv
│   │   ├── fifa_rankings.csv
│   │   └── elo_ratings.csv
│   │
│   ├── processed/
│   │   ├── matches_cleaned.csv
│   │   ├── team_features.csv
│   │   └── training_dataset.csv
│   │
│   └── external/
│       └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_xgboost_model.ipynb
│   ├── 05_poisson_score_model.ipynb
│   └── 06_world_cup_simulation.ipynb
│
├── src/
│   └── wc_predictor/
│       │
│       ├── __init__.py
│       │
│       ├── config/
│       │   ├── settings.py
│       │   └── constants.py
│       │
│       ├── data/
│       │   ├── load_data.py
│       │   ├── clean_data.py
│       │   ├── update_fixtures.py
│       │   └── validate_data.py
│       │
│       ├── features/
│       │   ├── build_features.py
│       │   ├── elo_features.py
│       │   ├── form_features.py
│       │   ├── ranking_features.py
│       │   └── team_strength.py
│       │
│       ├── models/
│       │   ├── baseline_model.py
│       │   ├── logistic_regression.py
│       │   ├── random_forest.py
│       │   ├── xgboost_model.py
│       │   ├── poisson_model.py
│       │   ├── train.py
│       │   ├── predict.py
│       │   └── evaluate.py
│       │
│       ├── simulation/
│       │   ├── match_simulator.py
│       │   ├── group_stage.py
│       │   ├── knockout_stage.py
│       │   ├── tournament_simulator.py
│       │   └── probability_summary.py
│       │
│       ├── agents/
│       │   ├── data_agent.py
│       │   ├── feature_agent.py
│       │   ├── prediction_agent.py
│       │   ├── simulation_agent.py
│       │   ├── news_agent.py
│       │   ├── explanation_agent.py
│       │   ├── evaluation_agent.py
│       │   └── agent_graph.py
│       │
│       ├── rag/
│       │   ├── ingest_news.py
│       │   ├── vector_store.py
│       │   ├── retriever.py
│       │   └── football_context_rag.py
│       │
│       ├── api/
│       │   ├── main.py
│       │   ├── routes.py
│       │   └── schemas.py
│       │
│       └── utils/
│           ├── logger.py
│           ├── helpers.py
│           └── metrics.py
│
├── app/
│   ├── streamlit_app.py
│   ├── pages/
│   │   ├── 1_Today_Matches.py
│   │   ├── 2_Predictions.py
│   │   ├── 3_Tournament_Simulation.py
│   │   ├── 4_Model_Performance.py
│   │   └── 5_Agent_Explanation.py
│   │
│   └── assets/
│       ├── logo.png
│       └── styles.css
│
├── models/
│   ├── baseline_model.pkl
│   ├── xgboost_model.pkl
│   ├── poisson_model.pkl
│   └── model_metadata.json
│
├── reports/
│   ├── figures/
│   ├── model_comparison.md
│   ├── prediction_results.md
│   └── final_project_report.md
│
├── tests/
│   ├── test_data_loading.py
│   ├── test_feature_engineering.py
│   ├── test_models.py
│   ├── test_simulation.py
│   └── test_agents.py
│
├── scripts/
│   ├── download_data.py
│   ├── train_model.py
│   ├── run_predictions.py
│   ├── run_simulation.py
│   └── update_live_results.py
│
└── docs/
    ├── project_plan.md
    ├── architecture.md
    ├── data_sources.md
    ├── model_design.md
    ├── agentic_ai_design.md
    ├── evaluation_strategy.md
    └── demo_script.md
```

---