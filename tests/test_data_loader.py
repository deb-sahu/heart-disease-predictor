"""
Unit tests for data loader module.
"""
import pytest
import pandas as pd
from pathlib import Path

from src.data_loader import (
    save_raw_data,
    load_raw_data,
    get_feature_target_split,
    validate_data
)
from src.config import TARGET_COLUMN, ALL_FEATURES


class TestDataLoader:
    """Tests for data loading functions."""
    
    def test_save_and_load_raw_data(self, sample_dataframe, temp_data_dir):
        """Test saving and loading raw data."""
        filepath = temp_data_dir / "test_data.csv"
        
        # Split into X and y
        X = sample_dataframe.drop(columns=[TARGET_COLUMN])
        y = sample_dataframe[[TARGET_COLUMN]]
        
        # Save
        save_raw_data(X, y, filepath)
        
        # Verify file exists
        assert filepath.exists()
        
        # Load and verify
        loaded_df = load_raw_data(filepath)
        assert len(loaded_df) == len(sample_dataframe)
        assert TARGET_COLUMN in loaded_df.columns
    
    def test_load_nonexistent_file(self, temp_data_dir):
        """Test loading a file that doesn't exist."""
        filepath = temp_data_dir / "nonexistent.csv"
        
        with pytest.raises(FileNotFoundError):
            load_raw_data(filepath)
    
    def test_get_feature_target_split(self, sample_dataframe):
        """Test splitting DataFrame into features and target."""
        X, y = get_feature_target_split(sample_dataframe)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert TARGET_COLUMN not in X.columns
        assert len(X) == len(y)
        assert len(X) == len(sample_dataframe)
    
    def test_get_feature_target_split_missing_target(self, sample_dataframe):
        """Test splitting when target column is missing."""
        df_no_target = sample_dataframe.drop(columns=[TARGET_COLUMN])
        
        with pytest.raises(ValueError, match="Target column"):
            get_feature_target_split(df_no_target)
    
    def test_validate_data_valid(self, sample_dataframe):
        """Test validation with valid data."""
        result = validate_data(sample_dataframe)
        assert result is True
    
    def test_validate_data_missing_columns(self, sample_dataframe):
        """Test validation with missing columns."""
        df_incomplete = sample_dataframe.drop(columns=['age', 'sex'])
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_data(df_incomplete)
    
    def test_validate_data_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        empty_df = pd.DataFrame(columns=ALL_FEATURES + [TARGET_COLUMN])
        
        with pytest.raises(ValueError, match="empty"):
            validate_data(empty_df)

