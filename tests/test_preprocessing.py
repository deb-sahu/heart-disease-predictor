"""
Unit tests for preprocessing module.
"""
import pytest
import pandas as pd
import numpy as np

from src.preprocessing import (
    handle_missing_values,
    convert_target_to_binary,
    create_preprocessor,
    preprocess_data,
    save_preprocessor,
    load_preprocessor
)
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN


class TestPreprocessing:
    """Tests for preprocessing functions."""
    
    def test_handle_missing_values(self, sample_dataframe_with_missing):
        """Test handling missing values."""
        df_cleaned = handle_missing_values(sample_dataframe_with_missing)
        
        # Check no missing values in result
        assert df_cleaned['ca'].isnull().sum() == 0
        assert df_cleaned['thal'].isnull().sum() == 0
        
        # Check original DataFrame is not modified
        assert sample_dataframe_with_missing['ca'].isnull().sum() == 2
        assert sample_dataframe_with_missing['thal'].isnull().sum() == 1
    
    def test_handle_missing_values_no_missing(self, sample_dataframe):
        """Test handling when there are no missing values."""
        df_cleaned = handle_missing_values(sample_dataframe)
        
        # Should return a copy with no changes to data
        assert df_cleaned.shape == sample_dataframe.shape
        pd.testing.assert_frame_equal(
            df_cleaned.reset_index(drop=True),
            sample_dataframe.reset_index(drop=True)
        )
    
    def test_convert_target_to_binary(self, sample_dataframe):
        """Test converting target to binary."""
        df_binary = convert_target_to_binary(sample_dataframe)
        
        # Check target is binary
        assert set(df_binary[TARGET_COLUMN].unique()).issubset({0, 1})
        
        # Original values 1,2 should become 1
        # Original value 0 should stay 0
        assert df_binary[TARGET_COLUMN].iloc[0] == 1  # was 1
        assert df_binary[TARGET_COLUMN].iloc[1] == 1  # was 2
        assert df_binary[TARGET_COLUMN].iloc[2] == 0  # was 0
    
    def test_convert_target_already_binary(self):
        """Test converting target when already binary."""
        df = pd.DataFrame({
            TARGET_COLUMN: [0, 1, 0, 1, 1]
        })
        
        df_result = convert_target_to_binary(df)
        assert list(df_result[TARGET_COLUMN]) == [0, 1, 0, 1, 1]
    
    def test_create_preprocessor(self):
        """Test creating the preprocessor."""
        preprocessor = create_preprocessor()
        
        assert preprocessor is not None
        assert hasattr(preprocessor, 'fit_transform')
        assert hasattr(preprocessor, 'transform')
    
    def test_preprocess_data_fit(self, sample_dataframe):
        """Test preprocessing with fitting."""
        X_processed, y, preprocessor = preprocess_data(sample_dataframe, fit=True)
        
        # Check output shapes
        assert X_processed.shape[0] == len(sample_dataframe)
        assert len(y) == len(sample_dataframe)
        
        # Check target is binary
        assert set(y.unique()).issubset({0, 1})
        
        # Check preprocessor is fitted
        assert hasattr(preprocessor, 'transformers_')
    
    def test_preprocess_data_transform_only(self, sample_dataframe):
        """Test preprocessing without fitting (transform only)."""
        # First fit the preprocessor
        _, _, fitted_preprocessor = preprocess_data(sample_dataframe, fit=True)
        
        # Now transform new data
        df_new = sample_dataframe.copy()
        X_processed, y, _ = preprocess_data(
            df_new,
            fit=False,
            preprocessor=fitted_preprocessor
        )
        
        assert X_processed.shape[0] == len(df_new)
    
    def test_preprocess_data_no_preprocessor_error(self, sample_dataframe):
        """Test that fit=False without preprocessor raises error."""
        with pytest.raises(ValueError, match="Preprocessor must be provided"):
            preprocess_data(sample_dataframe, fit=False, preprocessor=None)
    
    def test_save_and_load_preprocessor(self, sample_dataframe, temp_model_dir):
        """Test saving and loading preprocessor."""
        # Fit preprocessor
        _, _, preprocessor = preprocess_data(sample_dataframe, fit=True)
        
        # Save
        filepath = temp_model_dir / "test_preprocessor.pkl"
        save_preprocessor(preprocessor, str(filepath))
        
        assert filepath.exists()
        
        # Load
        loaded_preprocessor = load_preprocessor(str(filepath))
        
        # Verify it works
        X = sample_dataframe.drop(columns=[TARGET_COLUMN])
        result = loaded_preprocessor.transform(X)
        
        assert result.shape[0] == len(X)

