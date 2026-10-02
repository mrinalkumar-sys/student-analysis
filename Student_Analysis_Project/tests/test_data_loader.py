"""
Unit tests for DataLoader and descriptive statistics.
"""
import pytest
import pandas as pd
from src.data_loader import DataLoader
from src.config import DATASET_FILE

def test_data_loader_initialization():
    loader = DataLoader(DATASET_FILE)
    df = loader.load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "Unnamed: 0" not in df.columns
    assert "MathScore" in df.columns
    assert "ReadingScore" in df.columns
    assert "WritingScore" in df.columns

def test_summary_statistics():
    loader = DataLoader(DATASET_FILE)
    stats = loader.get_summary_statistics()
    assert "mean" in stats
    assert "median" in stats
    assert "mode" in stats
    assert "std_dev" in stats
    assert "MathScore" in stats["mean"]
    assert 50 <= stats["mean"]["MathScore"] <= 85

def test_correlation_matrix():
    loader = DataLoader(DATASET_FILE)
    corr = loader.get_correlation_matrix()
    assert "columns" in corr
    assert "matrix" in corr
    # High correlation between reading and writing
    rw_corr = corr["matrix"]["ReadingScore"]["WritingScore"]
    assert rw_corr > 0.7
