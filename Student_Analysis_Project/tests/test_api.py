"""
API Endpoint tests using FastAPI TestClient.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"

def test_summary_api():
    response = client.get("/api/data/summary")
    assert response.status_code == 200
    data = response.json()
    assert "mean" in data
    assert "total_records" in data

def test_hypothesis_test_api():
    payload = {
        "group_column": "LunchType",
        "group1_val": "standard",
        "group2_val": "free/reduced",
        "score_column": "MathScore",
        "alpha": 0.05
    }
    response = client.post("/api/analytics/hypothesis-test", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "t_statistic" in data
    assert "p_value" in data

def test_predict_score_api():
    payload = {
        "Gender": "male",
        "EthnicGroup": "group C",
        "ParentEduc": "some college",
        "LunchType": "standard",
        "ReadingScore": 80.0,
        "WritingScore": 82.0
    }
    response = client.post("/api/predict/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_score" in data

def test_predict_grade_api():
    payload = {
        "PracticeSport": "regularly",
        "TestPrep": "completed",
        "WklyStudyHours": "5 - 10",
        "NrSiblings": 2.0
    }
    response = client.post("/api/predict/grade", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_grade" in data
