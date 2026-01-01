"""
Data preprocessing and feature engineering pipeline.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    PREPROCESSOR_PATH
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    # Handle missing values for numeric columns
    for col in NUMERIC_FEATURES:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Filled {col} missing values with median: {median_val}")
    
    # Handle missing values for categorical columns
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info(f"Filled {col} missing values with mode: {mode_val}")
    
    return df


def convert_target_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert target variable to binary classification.
    0 -> No heart disease
    1-4 -> Heart disease present (converted to 1)
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with binary target
    """
    df = df.copy()
    
    if TARGET_COLUMN in df.columns:
        original_values = df[TARGET_COLUMN].unique()
        df[TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype(int)
        logger.info(f"Converted target from {original_values} to binary")
    
    return df


def create_preprocessor() -> ColumnTransformer:
    """
    Create the sklearn preprocessing pipeline.
    
    Returns:
        ColumnTransformer with numeric scaling and categorical encoding
    """
    # Numeric features pipeline: imputation + scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features pipeline: imputation + one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'  # Drop any other columns
    )
    
    return preprocessor


def preprocess_data(df: pd.DataFrame, fit: bool = True, preprocessor: Optional[ColumnTransformer] = None) -> Tuple[np.ndarray, Optional[pd.Series], ColumnTransformer]:
    """
    Preprocess the dataset for model training/prediction.
    
    Args:
        df: Input DataFrame
        fit: Whether to fit the preprocessor (True for training, False for inference)
        preprocessor: Existing preprocessor to use (required if fit=False)
        
    Returns:
        Tuple of (processed features array, target series if present, fitted preprocessor)
    """
    df = df.copy()
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Convert target to binary if present
    if TARGET_COLUMN in df.columns:
        df = convert_target_to_binary(df)
        y = df[TARGET_COLUMN]
        X = df.drop(columns=[TARGET_COLUMN])
    else:
        y = None
        X = df
    
    # Create or use existing preprocessor
    if fit:
        preprocessor = create_preprocessor()
        X_processed = preprocessor.fit_transform(X)
        logger.info(f"Fitted preprocessor. Output shape: {X_processed.shape}")
    else:
        if preprocessor is None:
            raise ValueError("Preprocessor must be provided when fit=False")
        X_processed = preprocessor.transform(X)
        logger.info(f"Transformed data. Output shape: {X_processed.shape}")
    
    return X_processed, y, preprocessor


def save_preprocessor(preprocessor: ColumnTransformer, filepath: Optional[str] = None) -> str:
    """
    Save the fitted preprocessor to disk.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        filepath: Output path
        
    Returns:
        Path to saved preprocessor
    """
    filepath = filepath or str(PREPROCESSOR_PATH)
    joblib.dump(preprocessor, filepath)
    logger.info(f"Saved preprocessor to {filepath}")
    return filepath


def load_preprocessor(filepath: Optional[str] = None) -> ColumnTransformer:
    """
    Load a fitted preprocessor from disk.
    
    Args:
        filepath: Path to saved preprocessor
        
    Returns:
        Fitted ColumnTransformer
    """
    filepath = filepath or str(PREPROCESSOR_PATH)
    preprocessor = joblib.load(filepath)
    logger.info(f"Loaded preprocessor from {filepath}")
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Get feature names after preprocessing.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        
    Returns:
        List of feature names
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        # Fallback for older sklearn versions
        numeric_names = NUMERIC_FEATURES
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        return numeric_names + cat_names

