"""
Data Preprocessing & Feature Engineering Module.
Recreates the preprocessing pipelines from Cells 58 & 60 of Students_Performance.ipynb.
"""
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.config import GRADE_BINS, GRADE_LABELS, CATEGORICAL_FEATURES, SCORE_COLUMNS

class DataPreprocessor:
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.dummy_columns: List[str] = []
        self.feature_names: List[str] = []

    @staticmethod
    def assign_grades(df: pd.DataFrame, score_col: str = "MathScore") -> pd.Series:
        """Converts numerical score to letter grade (A, B, C, D, F) as in Cell 60."""
        return pd.cut(df[score_col], bins=GRADE_BINS, labels=GRADE_LABELS, include_lowest=True)

    def prepare_regression_data(
        self,
        df: pd.DataFrame,
        target_col: str = "MathScore",
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Recreates Regression data preparation from Cell 58:
        df = df.dropna()
        df = pd.get_dummies(df)
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        """
        data = df.copy()
        if "Unnamed: 0" in data.columns:
            data = data.drop(columns=["Unnamed: 0"])

        data = data.dropna()
        
        y = data[target_col]
        X = data.drop(columns=[target_col])
        
        # One-hot encode categorical features
        X_encoded = pd.get_dummies(X, drop_first=True)
        self.dummy_columns = list(X_encoded.columns)
        self.feature_names = self.dummy_columns

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test

    def transform_single_regression_input(self, input_dict: Dict[str, Any]) -> pd.DataFrame:
        """Encodes a single student record ensuring matching columns with trained model."""
        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df)
        
        # Align with training columns
        aligned_df = pd.DataFrame(0, index=[0], columns=self.dummy_columns)
        for col in input_encoded.columns:
            if col in aligned_df.columns:
                aligned_df[col] = input_encoded[col].values

        # Fill any missing numerical columns
        for col in self.dummy_columns:
            if col in input_dict and aligned_df[col].iloc[0] == 0:
                try:
                    aligned_df[col] = float(input_dict[col])
                except (ValueError, TypeError):
                    pass

        return aligned_df

    def prepare_classification_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None,
        target_score_col: str = "MathScore",
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Recreates Classification data preparation from Cell 60:
        Grade discretization + Label Encoding
        """
        data = df.copy()
        if "Unnamed: 0" in data.columns:
            data = data.drop(columns=["Unnamed: 0"])

        data["Grade"] = self.assign_grades(data, score_col=target_score_col)
        data = data.dropna(subset=["Grade"])
        
        if feature_cols is None:
            # Full comprehensive feature set: all categorical factors + siblings + study hours
            feature_cols = [
                "Gender", "EthnicGroup", "ParentEduc", "LunchType",
                "TestPrep", "ParentMaritalStatus", "PracticeSport",
                "IsFirstChild", "NrSiblings", "WklyStudyHours"
            ]
        
        feature_cols = [c for c in feature_cols if c in data.columns]
        
        # Fill missing values for robust training
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(data[col]):
                mode_val = data[col].mode()[0] if not data[col].mode().empty else "Unknown"
                data[col] = data[col].fillna(mode_val)
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le
            else:
                med_val = data[col].median()
                data[col] = data[col].fillna(med_val)

        grade_le = LabelEncoder()
        # Ensure ordered classes ['F', 'D', 'C', 'B', 'A']
        grade_le.fit(GRADE_LABELS)
        data["GradeEncoded"] = grade_le.transform(data["Grade"].astype(str))
        self.label_encoders["Grade"] = grade_le

        X = data[feature_cols]
        y = data["GradeEncoded"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test

    def transform_single_classification_input(
        self, input_dict: Dict[str, Any], feature_cols: List[str]
    ) -> pd.DataFrame:
        """Transforms a single input dictionary for classification inference."""
        transformed = {}
        for col in feature_cols:
            val = input_dict.get(col)
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

        return pd.DataFrame([transformed])
