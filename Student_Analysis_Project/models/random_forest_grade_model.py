"""
Random Forest Grade Classification Model.
Python module implementation replacing random_forest_grade_model.joblib.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

GRADE_BINS = [0, 60, 70, 80, 90, 100]
GRADE_LABELS = ['F', 'D', 'C', 'B', 'A']
META_FILE = Path(__file__).parent / "random_forest_grade_meta.json"

class RandomForestGradeModel:
    """
    Self-contained Random Forest Classifier for student grade categorization.
    Recreates and expands Cell 60 of Students_Performance.ipynb.
    """
    def __init__(self, target_score_column: str = "MathScore"):
        self.target_score_column = target_score_column
        self.model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        self.feature_columns: List[str] = [
            "Gender", "EthnicGroup", "ParentEduc", "LunchType",
            "TestPrep", "ParentMaritalStatus", "PracticeSport",
            "IsFirstChild", "NrSiblings", "WklyStudyHours"
        ]
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.metrics: Dict[str, Any] = {}
        self.feature_importance: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """Trains the Random Forest model to predict academic letter grade."""
        data = df.copy()
        if "Unnamed: 0" in data.columns:
            data = data.drop(columns=["Unnamed: 0"])

        # Discretize target into letter grades (Cell 60)
        data["Grade"] = pd.cut(data[self.target_score_column], bins=GRADE_BINS, labels=GRADE_LABELS, include_lowest=True)
        data = data.dropna(subset=["Grade"])

        # Encode categorical features
        for col in self.feature_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                mode_val = data[col].mode()[0] if not data[col].mode().empty else "Unknown"
                data[col] = data[col].fillna(mode_val)
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le
            else:
                med_val = data[col].median()
                data[col] = data[col].fillna(med_val)

        # Encode target grade
        grade_le = LabelEncoder()
        grade_le.fit(GRADE_LABELS)
        data["GradeEncoded"] = grade_le.transform(data["Grade"].astype(str))
        self.label_encoders["Grade"] = grade_le

        X = data[self.feature_columns]
        y = data["GradeEncoded"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        accuracy = float(accuracy_score(y_test, y_pred))
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

        self.save_metadata()

        return {
            "metrics": self.metrics,
            "feature_importance": self.feature_importance
        }

    def save_metadata(self, filepath: Optional[Path] = None):
        """Saves label encodings and performance metrics to JSON."""
        path = filepath or META_FILE
        payload = {
            "target_score_column": self.target_score_column,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "encoder_classes": {
                col: le.classes_.tolist() for col, le in self.label_encoders.items()
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_metadata(self, filepath: Optional[Path] = None) -> bool:
        """Loads label encodings and model metrics from JSON."""
        path = filepath or META_FILE
        if not path.exists():
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.target_score_column = data.get("target_score_column", "MathScore")
        self.feature_columns = data.get("feature_columns", self.feature_columns)
        self.metrics = data.get("metrics", {})
        self.feature_importance = data.get("feature_importance", {})

        for col, classes in data.get("encoder_classes", {}).items():
            le = LabelEncoder()
            le.classes_ = np.array(classes)
            self.label_encoders[col] = le

        return True

    def predict(self, student_input: Dict[str, Any]) -> Dict[str, Any]:
        """Predicts academic letter grade for a single student."""
        transformed = {}
        for col in self.feature_columns:
            val = student_input.get(col)
            if col in self.label_encoders:
                le = self.label_encoders[col]
                val_str = str(val) if val is not None else le.classes_[0]
                if val_str in le.classes_:
                    transformed[col] = le.transform([val_str])[0]
                else:
                    transformed[col] = 0
            else:
                try:
                    transformed[col] = float(val) if val is not None else 0.0
                except (ValueError, TypeError):
                    transformed[col] = 0.0

        X_input = pd.DataFrame([transformed])
        pred_encoded = self.model.predict(X_input)[0]

        grade_le = self.label_encoders.get("Grade")
        if grade_le:
            predicted_grade = str(grade_le.inverse_transform([pred_encoded])[0])
            classes = grade_le.classes_
        else:
            predicted_grade = GRADE_LABELS[pred_encoded]
            classes = GRADE_LABELS

        probs = self.model.predict_proba(X_input)[0]
        prob_dict = {
            str(cls_name): round(float(prob), 4)
            for cls_name, prob in zip(classes, probs)
        }

        return {
            "predicted_grade": predicted_grade,
            "probabilities": prob_dict,
            "confidence": round(float(np.max(probs)), 4)
        }

if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "data" / "Expanded_data_with_more_features.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        model = RandomForestGradeModel()
        summary = model.fit(df)
        print("Grade Classification Model trained successfully.")
        print("Accuracy:", summary["metrics"]["accuracy"])
        demo_pred = model.predict({"PracticeSport": "regularly", "TestPrep": "completed", "WklyStudyHours": "> 10"})
        print("Demo prediction:", demo_pred)
