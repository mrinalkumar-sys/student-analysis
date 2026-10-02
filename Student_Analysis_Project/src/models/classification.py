"""
Random Forest Classification Model for Student Grade Categorization (A, B, C, D, F).
Recreates and elevates the classification pipeline from Cell 60 of Students_Performance.ipynb.
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.config import GRADE_LABELS
from src.preprocessor import DataPreprocessor

class GradeClassificationModel:
    def __init__(self, target_score_column: str = "MathScore"):
        self.target_score_column = target_score_column
        self.model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        self.preprocessor = DataPreprocessor()
        self.feature_columns: List[str] = [
            "Gender", "EthnicGroup", "ParentEduc", "LunchType",
            "TestPrep", "ParentMaritalStatus", "PracticeSport",
            "IsFirstChild", "NrSiblings", "WklyStudyHours"
        ]
        self.metrics: Dict[str, Any] = {}
        self.feature_importance: Dict[str, float] = {}

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Trains Random Forest Classifier and computes performance metrics."""
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_classification_data(
            df,
            feature_cols=self.feature_columns,
            target_score_col=self.target_score_column
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        accuracy = float(accuracy_score(y_test, y_pred))
        grade_le = self.preprocessor.label_encoders["Grade"]
        
        # Report
        report = classification_report(
            y_test, y_pred,
            target_names=[str(c) for c in grade_le.classes_],
            output_dict=True,
            zero_division=0
        )
        
        cm = confusion_matrix(y_test, y_pred).tolist()

        self.metrics = {
            "accuracy": round(accuracy, 4),
            "macro_avg_f1": round(report["macro avg"]["f1-score"], 4),
            "weighted_avg_f1": round(report["weighted avg"]["f1-score"], 4),
            "classification_report": report,
            "confusion_matrix": cm,
            "classes": [str(c) for c in grade_le.classes_]
        }

        # Feature importances
        importances = self.model.feature_importances_
        self.feature_importance = dict(sorted(
            zip(self.feature_columns, [round(float(imp), 4) for imp in importances]),
            key=lambda x: x[1],
            reverse=True
        ))

        return {
            "metrics": self.metrics,
            "feature_importance": self.feature_importance
        }

    def predict(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """Inference for predicting student's academic letter grade."""
        X_input = self.preprocessor.transform_single_classification_input(
            student_data, self.feature_columns
        )
        
        pred_encoded = self.model.predict(X_input)[0]
        grade_le = self.preprocessor.label_encoders["Grade"]
        predicted_grade = str(grade_le.inverse_transform([pred_encoded])[0])
        
        # Class probabilities
        probs = self.model.predict_proba(X_input)[0]
        prob_dict = {
            str(cls_name): round(float(prob), 4)
            for cls_name, prob in zip(grade_le.classes_, probs)
        }

        return {
            "predicted_grade": predicted_grade,
            "probabilities": prob_dict,
            "confidence": round(float(np.max(probs)), 4)
        }
