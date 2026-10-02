"""
Unit tests for Python models in models/ folder.
"""
import pytest
from src.data_loader import DataLoader
from models.linear_regression_model import LinearRegressionModel
from models.random_forest_grade_model import RandomForestGradeModel

def test_score_regression_python_model():
    loader = DataLoader()
    df = loader.load_data().head(1000) # Quick subset for test speed
    model = LinearRegressionModel(target_column="MathScore")
    summary = model.fit(df)
    
    assert "metrics" in summary
    assert "r2_score" in summary["metrics"]
    assert "mean_squared_error" in summary["metrics"]
    
    # Inference test
    sample_student = {
        "Gender": "female",
        "EthnicGroup": "group C",
        "ParentEduc": "some college",
        "LunchType": "standard",
        "ReadingScore": 75,
        "WritingScore": 78
    }
    pred = model.predict(sample_student)
    assert "predicted_score" in pred
    assert 0 <= pred["predicted_score"] <= 100

def test_grade_classification_python_model():
    loader = DataLoader()
    df = loader.load_data().head(1000)
    model = RandomForestGradeModel(target_score_column="MathScore")
    summary = model.fit(df)
    
    assert "metrics" in summary
    assert "accuracy" in summary["metrics"]
    
    sample_student = {
        "PracticeSport": "regularly",
        "TestPrep": "completed",
        "WklyStudyHours": "> 10",
        "NrSiblings": 1
    }
    pred = model.predict(sample_student)
    assert "predicted_grade" in pred
    assert pred["predicted_grade"] in ['A', 'B', 'C', 'D', 'F']
    assert "probabilities" in pred
