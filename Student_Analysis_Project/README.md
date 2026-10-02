# Student Performance Analytics & ML Platform

A comprehensive, full-stack data analytics and machine learning application recreated from `Students_Performance.ipynb`.

In this version, opaque binary model files have been replaced with **pure Python model implementations**:
* `models/linear_regression_model.py` (replaces `linear_regression_model.joblib`)
* `models/random_forest_grade_model.py` (replaces `random_forest_grade_model.joblib`)

---

## 🏛️ Project Directory Structure

```
Student_Analysis_Project/
├── data/                                # Dataset storage
│   └── Expanded_data_with_more_features.csv  # 30,641-record student dataset
├── models/                              # Pure Python model implementations (.py)
│   ├── __init__.py                      # Package entry point
│   ├── linear_regression_model.py       # Standalone Linear Regression model class
│   ├── random_forest_grade_model.py     # Standalone Random Forest grade classifier class
│   ├── linear_regression_weights.json   # Model weights and coefficients
│   └── random_forest_grade_meta.json    # Encodings and evaluation metrics
├── src/                                 # Analytics & pipeline modules
│   ├── config.py                        # Constants, schema, and paths
│   ├── dataset_generator.py             # Realistic 30k-row dataset synthesizer
│   ├── data_loader.py                   # Data ingestion, stats, and EDA engine
│   ├── preprocessor.py                  # Encoders, splitters, feature engineering
│   ├── stats_engine.py                  # Hypothesis testing (t-test, ANOVA, Cohen's d)
│   ├── train.py                         # Training pipeline invoking models/*.py
│   └── inference.py                     # Inference service for predictions
├── app/                                 # FastAPI Web Application & UI
│   ├── main.py                          # REST API and router
│   ├── static/                          # CSS styling, Chart.js clients
│   │   ├── css/styles.css
│   │   └── js/dashboard.js
│   └── templates/                       # Single-page dashboard UI
│       └── index.html
├── notebooks/                           # Clean recreated Jupyter Notebook
│   └── Students_Performance_Recreated.ipynb
├── tests/                               # Comprehensive unit and integration test suite
│   ├── conftest.py
│   ├── test_data_loader.py
│   ├── test_stats_engine.py
│   ├── test_models.py
│   └── test_api.py
├── run.py                               # Single-command application launcher
├── pytest.ini                           # Pytest configuration
├── requirements.txt                     # Dependencies
├── .env.example                         # Environment configuration template
└── README.md                            # Documentation
```

---

## 🚀 How to Run

1. Navigate to the project directory:
```bash
cd C:\Users\ASUS\.gemini\antigravity\scratch\Student_Analysis_Project
```

2. Activate virtual environment or install requirements:
```bash
pip install -r requirements.txt
```

3. Launch the application:
```bash
python run.py
```

4. Open in browser:
* Dashboard UI: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧪 Testing

Execute the test suite:
```bash
pytest tests/ -v
```
All 11 tests verify data loading, descriptive summaries, hypothesis testing, model training & prediction, and API endpoints.
