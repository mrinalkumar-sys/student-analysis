"""
Linear Regression Model for Student Exam Score Prediction.
Recreates and enhances the regression pipeline from Cell 58 of Students_Performance.ipynb.
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.preprocessor import DataPreprocessor

class ScoreRegressionModel:
    def __init__(self, target_column: str = "MathScore"):
        self.target_column = target_column
        self.model = LinearRegression()
        self.preprocessor = DataPreprocessor()
        self.metrics: Dict[str, float] = {}
        self.feature_importance: Dict[str, float] = {}

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Trains linear regression model and calculates evaluation metrics."""
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_regression_data(
            df, target_col=self.target_column
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        self.metrics = {
            "mean_squared_error": round(mse, 4),
            "root_mean_squared_error": round(rmse, 4),
            "mean_absolute_error": round(mae, 4),
            "r2_score": round(r2, 4)
        }

        # Coefficient feature weights
        coefs = self.model.coef_
        feat_names = self.preprocessor.feature_names
        importance = dict(sorted(
            zip(feat_names, [round(float(c), 4) for c in coefs]),
            key=lambda item: abs(item[1]),
            reverse=True
        )[:15])
        self.feature_importance = importance

        return {
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "intercept": round(float(self.model.intercept_), 4)
        }

    def predict(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """Inference for a single student."""
        X_input = self.preprocessor.transform_single_regression_input(student_data)
        predicted_score = float(self.model.predict(X_input)[0])
        # Clip to valid grade bounds [0, 100]
        clipped_score = round(max(0.0, min(100.0, predicted_score)), 2)

        return {
            "target": self.target_column,
            "predicted_score": clipped_score,
            "raw_score": round(predicted_score, 2)
        }
