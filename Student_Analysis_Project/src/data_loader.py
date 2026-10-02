"""
Data Loader & Statistics Module for Student Performance Analysis.
Recreates the exploratory data analysis, descriptive statistics, and demographic
breakdowns from Students_Performance.ipynb.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import DATASET_FILE, NUMERICAL_FEATURES, SCORE_COLUMNS, CATEGORICAL_FEATURES
from src.dataset_generator import generate_student_dataset

class DataLoader:
    def __init__(self, data_path: Path = DATASET_FILE):
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        self.load_data()

    def load_data(self, reload: bool = False) -> pd.DataFrame:
        if self.df is not None and not reload:
            return self.df

        if not self.data_path.exists():
            print(f"Dataset not found at {self.data_path}. Generating realistic dataset...")
            self.df = generate_student_dataset(output_path=self.data_path)
        else:
            self.df = pd.read_csv(self.data_path)

        # Drop index column 'Unnamed: 0' if it exists (reproducing Cell 6 in notebook)
        if "Unnamed: 0" in self.df.columns:
            self.df = self.df.drop(columns=["Unnamed: 0"])

        return self.df

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculates mean, median, mode, std dev matching Cell 9 of the notebook."""
        df = self.load_data()
        stats_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
        
        subset = df[stats_cols]
        mean = subset.mean().round(4).to_dict()
        median = subset.median().round(4).to_dict()
        
        # Mode might return multiple rows, take first row
        mode_df = subset.mode()
        mode = mode_df.iloc[0].round(4).to_dict() if not mode_df.empty else {}
        std_dev = subset.std().round(4).to_dict()
        
        # Full describe
        desc = subset.describe().round(2).to_dict()

        return {
            "mean": mean,
            "median": median,
            "mode": mode,
            "std_dev": std_dev,
            "describe": desc,
            "total_records": len(df),
            "columns": list(df.columns)
        }

    def get_missing_values(self) -> Dict[str, Any]:
        """Missing values count and percentage (reproducing Cell 5 in notebook)."""
        df = self.load_data()
        null_counts = df.isnull().sum().to_dict()
        null_pct = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        return {
            "counts": null_counts,
            "percentages": null_pct,
            "total_rows": len(df)
        }

    def get_correlation_matrix(self) -> Dict[str, Any]:
        """Calculates correlation matrix for exam scores (reproducing Cell 10 in notebook)."""
        df = self.load_data()
        corr_cols = [c for c in SCORE_COLUMNS if c in df.columns]
        corr_matrix = df[corr_cols].corr().round(4)
        return {
            "columns": corr_cols,
            "matrix": corr_matrix.to_dict()
        }

    def get_demographic_breakdown(self, column: str) -> Dict[str, Any]:
        """
        Grouped analysis of scores by categorical factor
        (reproducing Grouped Analysis cells in notebook).
        """
        df = self.load_data()
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not in dataset.")

        grouped = df.groupby(column)[SCORE_COLUMNS].agg(['mean', 'count']).round(2)
        
        categories = list(grouped.index)
        data = {
            "categories": [str(c) for c in categories],
            "counts": grouped[("MathScore", "count")].tolist(),
            "math_mean": grouped[("MathScore", "mean")].tolist(),
            "reading_mean": grouped[("ReadingScore", "mean")].tolist(),
            "writing_mean": grouped[("WritingScore", "mean")].tolist(),
            "overall_mean": [
                round((m + r + w) / 3, 2)
                for m, r, w in zip(
                    grouped[("MathScore", "mean")].tolist(),
                    grouped[("ReadingScore", "mean")].tolist(),
                    grouped[("WritingScore", "mean")].tolist()
                )
            ]
        }
        return data

    def get_all_demographic_analyses(self) -> Dict[str, Any]:
        """Aggregates all demographic factors explored in the notebook."""
        factors = [
            "Gender",
            "EthnicGroup",
            "ParentEduc",
            "LunchType",
            "PracticeSport",
            "ParentMaritalStatus",
            "WklyStudyHours",
            "TestPrep"
        ]
        results = {}
        for f in factors:
            if f in self.load_data().columns:
                results[f] = self.get_demographic_breakdown(f)
        return results

    def get_records(self, page: int = 1, page_size: int = 20, search: Optional[str] = None) -> Dict[str, Any]:
        """Returns paginated, searchable records for data table views."""
        df = self.load_data()
        
        filtered = df
        if search:
            search_str = str(search).lower()
            mask = df.astype(str).apply(lambda row: row.str.lower().str.contains(search_str).any(), axis=1)
            filtered = df[mask]

        total = len(filtered)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        records = filtered.iloc[start_idx:end_idx].replace({np.nan: None}).to_dict(orient="records")
        
        return {
            "records": records,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": int(np.ceil(total / page_size)) if page_size > 0 else 1
        }
