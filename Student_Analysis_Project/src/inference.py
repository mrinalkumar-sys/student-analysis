"""
Inference Service for loading and querying trained Python models.
"""
from typing import Dict, Any, Optional
from pathlib import Path
from src.config import DATASET_FILE
from models.linear_regression_model import LinearRegressionModel
from models.random_forest_grade_model import RandomForestGradeModel
from src.data_loader import DataLoader

class InferenceService:
    def __init__(self):
        self.regression_model: Optional[LinearRegressionModel] = None
        self.classification_model: Optional[RandomForestGradeModel] = None
        self.ensure_models_loaded()

    def ensure_models_loaded(self):
        if self.regression_model is None:
            self.regression_model = LinearRegressionModel(target_column="MathScore")
            # Try to load saved weights, else train from dataset
            if not self.regression_model.load_weights():
                loader = DataLoader(DATASET_FILE)
                self.regression_model.fit(loader.load_data())

        if self.classification_model is None:
            self.classification_model = RandomForestGradeModel(target_score_column="MathScore")
            loader = DataLoader(DATASET_FILE)
            self.classification_model.fit(loader.load_data())

    def predict_score(self, student_input: Dict[str, Any]) -> Dict[str, Any]:
        self.ensure_models_loaded()
        if self.regression_model is None:
            raise RuntimeError("Score Regression model is not available.")
        return self.regression_model.predict(student_input)

    def predict_grade(self, student_input: Dict[str, Any]) -> Dict[str, Any]:
        self.ensure_models_loaded()
        if self.classification_model is None:
            raise RuntimeError("Grade Classification model is not available.")
        return self.classification_model.predict(student_input)

    def get_models_info(self) -> Dict[str, Any]:
        self.ensure_models_loaded()
        return {
            "regression": {
                "algorithm": "LinearRegression (models/linear_regression_model.py)",
                "target": self.regression_model.target_column if self.regression_model else "MathScore",
                "metrics": self.regression_model.metrics if self.regression_model else {},
                "top_features": self.regression_model.feature_importance if self.regression_model else {}
            },
            "classification": {
                "algorithm": "RandomForestClassifier (models/random_forest_grade_model.py)",
                "target": "Grade (A, B, C, D, F)",
                "metrics": self.classification_model.metrics if self.classification_model else {},
                "feature_importance": self.classification_model.feature_importance if self.classification_model else {}
            }
        }
