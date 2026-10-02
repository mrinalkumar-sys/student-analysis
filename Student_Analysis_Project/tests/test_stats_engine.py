"""
Unit tests for Statistical Engine and Hypothesis Testing.
"""
import pytest
from src.stats_engine import StatsEngine
from src.data_loader import DataLoader

def test_ttest_lunch_type():
    engine = StatsEngine()
    result = engine.run_two_sample_ttest(
        group_column="LunchType",
        group1_val="standard",
        group2_val="free/reduced",
        score_column="MathScore"
    )
    
    assert "t_statistic" in result
    assert "p_value" in result
    assert "reject_null" in result
    assert "conclusion" in result
    assert isinstance(result["reject_null"], bool)
    # Standard lunch students score significantly higher in the dataset
    assert result["mean_difference"] > 0
