"""
Linear Regression Model for Student Exam Score Prediction.
Python module implementation replacing linear_regression_model.joblib.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

WEIGHTS_FILE = Path(__file__).parent / "linear_regression_weights.json"

class LinearRegressionModel:
    """
    Self-contained Linear Regression model for student score prediction.
    Recreates and expands Cell 58 of Students_Performance.ipynb.
    """
    def __init__(self, target_column: str = "MathScore"):
        self.target_column = target_column
        self.model = LinearRegression()
        self.feature_names: List[str] = []
        self.metrics: Dict[str, float] = {}
        self.feature_importance: Dict[str, float] = {}
        self.intercept: float = 0.0

    def fit(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """Trains the Linear Regression model on student exam data."""
        data = df.copy()
        if "Unnamed: 0" in data.columns:
            data = data.drop(columns=["Unnamed: 0"])

        data = data.dropna()
        y = data[self.target_column]
        X = data.drop(columns=[self.target_column])

        # One-hot encode categorical variables
        X_encoded = pd.get_dummies(X, drop_first=True)
        self.feature_names = list(X_encoded.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, random_state=random_state
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

        self.intercept = float(self.model.intercept_)

        # Coefficient feature importances
        coefs = self.model.coef_
        importance = dict(sorted(
            zip(self.feature_names, [round(float(c), 4) for c in coefs]),
            key=lambda item: abs(item[1]),
            reverse=True
        )[:15])
        self.feature_importance = importance

        self.save_weights()

        return {
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "intercept": round(self.intercept, 4)
        }

    def save_weights(self, filepath: Optional[Path] = None):
        """Saves model parameters and coefficients to human-readable JSON."""
        path = filepath or WEIGHTS_FILE
        payload = {
            "target_column": self.target_column,
            "intercept": self.intercept,
            "feature_names": self.feature_names,
            "coefficients": [float(c) for c in self.model.coef_] if hasattr(self.model, "coef_") else [],
            "metrics": self.metrics,
            "feature_importance": self.feature_importance
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_weights(self, filepath: Optional[Path] = None) -> bool:
        """Loads model parameters from JSON."""
        path = filepath or WEIGHTS_FILE
        if not path.exists():
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.target_column = data.get("target_column", "MathScore")
        self.intercept = data.get("intercept", 0.0)
        self.feature_names = data.get("feature_names", [])
        self.metrics = data.get("metrics", {})
        self.feature_importance = data.get("feature_importance", {})

        coefs = np.array(data.get("coefficients", []))
        if len(coefs) > 0 and len(self.feature_names) == len(coefs):
            self.model.coef_ = coefs
            self.model.intercept_ = self.intercept
            return True
        return False

    def predict(self, student_input: Dict[str, Any]) -> Dict[str, Any]:
        """Runs score inference for a single student."""
        # Build one-hot encoded vector aligned with training features
        input_df = pd.DataFrame([student_input])
        input_encoded = pd.get_dummies(input_df)

        aligned_row = np.zeros(len(self.feature_names))
        for i, name in enumerate(self.feature_names):
            if name in input_encoded.columns:
                aligned_row[i] = input_encoded[name].iloc[0]
            elif name in student_input:
                try:
                    aligned_row[i] = float(student_input[name])
                except (ValueError, TypeError):
                    pass

        raw_pred = float(self.model.intercept_ + np.dot(aligned_row, self.model.coef_))
        clipped_score = round(max(0.0, min(100.0, raw_pred)), 2)

        return {
            "target": self.target_column,
            "predicted_score": clipped_score,
            "raw_score": round(raw_pred, 2)
        }

if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "data" / "Expanded_data_with_more_features.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        model = LinearRegressionModel()
        summary = model.fit(df)
        print("Model trained successfully.")
        print(f"MSE: {summary['metrics']['mean_squared_error']}, R²: {summary['metrics']['r2_score']}")
        demo_pred = model.predict({"Gender": "female", "ReadingScore": 85, "WritingScore": 88})
        print("Demo prediction:", demo_pred)
