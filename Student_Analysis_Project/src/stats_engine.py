"""
Statistical Analysis & Hypothesis Testing Module.
Recreates the t-test hypothesis testing from Cell 53 of Students_Performance.ipynb
and adds ANOVA and effect size calculations.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
from src.config import DATASET_FILE
from src.data_loader import DataLoader

class StatsEngine:
    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.data_loader = data_loader or DataLoader(DATASET_FILE)

    def run_two_sample_ttest(
        self,
        group_column: str = "LunchType",
        group1_val: str = "standard",
        group2_val: str = "free/reduced",
        score_column: str = "MathScore",
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Executes an independent two-sample t-test as featured in Students_Performance.ipynb.
        Tests: H0: mu1 = mu2 vs H1: mu1 != mu2
        """
        df = self.data_loader.load_data()
        
        if group_column not in df.columns or score_column not in df.columns:
            raise ValueError(f"Columns {group_column} or {score_column} not found in dataset.")

        data1 = df[df[group_column] == group1_val][score_column].dropna()
        data2 = df[df[group_column] == group2_val][score_column].dropna()

        if len(data1) < 2 or len(data2) < 2:
            raise ValueError("Insufficient sample size for one or both groups.")

        # Compute t-test (Welch's t-test for unequal variances)
        t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)

        mean1, mean2 = float(np.mean(data1)), float(np.mean(data2))
        std1, std2 = float(np.std(data1, ddof=1)), float(np.std(data2, ddof=1))
        n1, n2 = len(data1), len(data2)

        # Cohen's d effect size
        pooled_std = np.sqrt(((n1 - 1) * (std1 ** 2) + (n2 - 1) * (std2 ** 2)) / (n1 + n2 - 2))
        cohens_d = float((mean1 - mean2) / pooled_std) if pooled_std != 0 else 0.0

        # Conclusion
        reject_null = bool(p_val < alpha)
        if reject_null:
            conclusion = (
                f"Reject Null Hypothesis (p < {alpha}). There is a statistically significant "
                f"difference in {score_column} between {group_column} '{group1_val}' "
                f"(Mean: {mean1:.2f}) and '{group2_val}' (Mean: {mean2:.2f})."
            )
        else:
            conclusion = (
                f"Fail to Reject Null Hypothesis (p >= {alpha}). There is no statistically "
                f"significant difference in {score_column} between '{group1_val}' and '{group2_val}'."
            )

        return {
            "group_column": group_column,
            "score_column": score_column,
            "group1": {
                "name": group1_val,
                "n": n1,
                "mean": round(mean1, 2),
                "std": round(std1, 2)
            },
            "group2": {
                "name": group2_val,
                "n": n2,
                "mean": round(mean2, 2),
                "std": round(std2, 2)
            },
            "mean_difference": round(mean1 - mean2, 2),
            "t_statistic": round(float(t_stat), 4),
            "p_value": float(p_val),
            "p_value_formatted": f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
            "alpha": alpha,
            "cohens_d": round(cohens_d, 3),
            "reject_null": reject_null,
            "conclusion": conclusion
        }

    def run_one_way_anova(
        self,
        group_column: str = "ParentEduc",
        score_column: str = "MathScore",
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Runs One-way ANOVA across multiple groups."""
        df = self.data_loader.load_data()
        
        valid = df[[group_column, score_column]].dropna()
        groups = [group[score_column].values for _, group in valid.groupby(group_column)]
        
        f_stat, p_val = stats.f_oneway(*groups)
        
        reject_null = bool(p_val < alpha)
        return {
            "group_column": group_column,
            "score_column": score_column,
            "f_statistic": round(float(f_stat), 4),
            "p_value": float(p_val),
            "p_value_formatted": f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
            "reject_null": reject_null,
            "conclusion": f"Difference across categories is {'statistically significant' if reject_null else 'not statistically significant'}."
        }
