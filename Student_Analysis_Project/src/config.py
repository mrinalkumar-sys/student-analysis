"""
Configuration and constants for Student Analysis application.
"""
from pathlib import Path
import os

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"

DATASET_FILE = DATA_DIR / "Expanded_data_with_more_features.csv"
REGRESSION_MODEL_PY = MODELS_DIR / "linear_regression_model.py"
CLASSIFICATION_MODEL_PY = MODELS_DIR / "random_forest_grade_model.py"

# Data schema definitions matching Students_Performance.ipynb
NUMERICAL_FEATURES = ["MathScore", "ReadingScore", "WritingScore", "NrSiblings"]
SCORE_COLUMNS = ["MathScore", "ReadingScore", "WritingScore"]

CATEGORICAL_FEATURES = [
    "Gender",
    "EthnicGroup",
    "ParentEduc",
    "LunchType",
    "TestPrep",
    "ParentMaritalStatus",
    "PracticeSport",
    "IsFirstChild",
    "TransportMeans",
    "WklyStudyHours"
]

ALL_COLUMNS = [
    "Gender",
    "EthnicGroup",
    "ParentEduc",
    "LunchType",
    "TestPrep",
    "ParentMaritalStatus",
    "PracticeSport",
    "IsFirstChild",
    "NrSiblings",
    "TransportMeans",
    "WklyStudyHours",
    "MathScore",
    "ReadingScore",
    "WritingScore"
]

# Grade bins and labels as defined in Students_Performance.ipynb
GRADE_BINS = [0, 60, 70, 80, 90, 100]
GRADE_LABELS = ['F', 'D', 'C', 'B', 'A']

# Server Configuration
HOST = os.getenv("STUDENT_ANALYSIS_HOST", "127.0.0.1")
PORT = int(os.getenv("STUDENT_ANALYSIS_PORT", "8000"))
DEBUG = os.getenv("STUDENT_ANALYSIS_DEBUG", "True").lower() == "true"
