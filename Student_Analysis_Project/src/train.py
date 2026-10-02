"""
Model Training & Artifact Serialization Pipeline.
Trains both Score Regression and Grade Classification models using the standalone
Python modules in the models/ folder.
"""
from pathlib import Path
from src.config import DATASET_FILE, MODELS_DIR
from src.data_loader import DataLoader
from models.linear_regression_model import LinearRegressionModel
from models.random_forest_grade_model import RandomForestGradeModel

def train_all_models(data_path: Path = DATASET_FILE) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(data_path)
    df = loader.load_data()
    print(f"Loaded dataset with {len(df)} rows for model training.")

    # 1. Train Score Regression Model
    print("Training Score Regression Model (models/linear_regression_model.py)...")
    reg_model = LinearRegressionModel(target_column="MathScore")
    reg_summary = reg_model.fit(df)
    print(f"Regression Model trained successfully.")
    print(f"MSE: {reg_summary['metrics']['mean_squared_error']}, R²: {reg_summary['metrics']['r2_score']}")

    # 2. Train Grade Classification Model
    print("Training Grade Classification Model (models/random_forest_grade_model.py)...")
    clf_model = RandomForestGradeModel(target_score_column="MathScore")
    clf_summary = clf_model.fit(df)
    print(f"Classification Model trained successfully.")
    print(f"Accuracy: {clf_summary['metrics']['accuracy']}, Macro F1: {clf_summary['metrics']['macro_avg_f1']}")

    return {
        "regression": reg_summary,
        "classification": clf_summary
    }

if __name__ == "__main__":
    train_all_models()
