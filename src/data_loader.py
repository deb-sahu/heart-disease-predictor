"""
Data acquisition and loading utilities for Heart Disease dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

from src.config import (
    DATASET_UCI_ID, 
    RAW_DATA_PATH, 
    PROCESSED_DATA_PATH,
    TARGET_COLUMN
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_dataset_from_uci() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch the Heart Disease dataset from UCI ML Repository.
    
    Returns:
        Tuple of (features DataFrame, targets DataFrame)
    """
    try:
        from ucimlrepo import fetch_ucirepo
        
        logger.info(f"Fetching Heart Disease dataset (UCI ID: {DATASET_UCI_ID})")
        heart_disease = fetch_ucirepo(id=DATASET_UCI_ID)
        
        X = heart_disease.data.features
        y = heart_disease.data.targets
        
        logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
        
    except ImportError:
        raise ImportError("ucimlrepo package is required. Install with: pip install ucimlrepo")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch dataset from UCI: {e}")


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw data from a CSV file.
    
    Args:
        filepath: Path to the CSV file. Defaults to RAW_DATA_PATH.
        
    Returns:
        DataFrame with raw data
    """
    filepath = filepath or RAW_DATA_PATH
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows from {filepath}")
    return df


def save_raw_data(X: pd.DataFrame, y: pd.DataFrame, filepath: Optional[Path] = None) -> Path:
    """
    Save raw data to CSV file.
    
    Args:
        X: Features DataFrame
        y: Targets DataFrame
        filepath: Output path. Defaults to RAW_DATA_PATH.
        
    Returns:
        Path to saved file
    """
    filepath = filepath or RAW_DATA_PATH
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Combine features and target
    df = X.copy()
    df[TARGET_COLUMN] = y.values
    
    df.to_csv(filepath, index=False)
    logger.info(f"Saved raw data to {filepath}")
    return filepath


def prepare_dataset() -> pd.DataFrame:
    """
    Main function to prepare the dataset.
    Fetches from UCI if local file doesn't exist, otherwise loads from disk.
    
    Returns:
        DataFrame with features and target
    """
    if RAW_DATA_PATH.exists():
        logger.info("Loading existing dataset from disk")
        return load_raw_data(RAW_DATA_PATH)
    
    logger.info("Fetching dataset from UCI repository")
    X, y = fetch_dataset_from_uci()
    save_raw_data(X, y)
    
    df = X.copy()
    df[TARGET_COLUMN] = y.values
    return df


def get_feature_target_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target.
    
    Args:
        df: DataFrame with features and target column
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame")
    
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    return X, y


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the dataset structure and content.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError if validation fails
    """
    from src.config import ALL_FEATURES
    
    # Check required columns
    required_columns = ALL_FEATURES + [TARGET_COLUMN]
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for empty DataFrame
    if len(df) == 0:
        raise ValueError("DataFrame is empty")
    
    # Check target values are binary (0 or 1) or can be converted
    unique_targets = df[TARGET_COLUMN].dropna().unique()
    if not all(t in [0, 1, 2, 3, 4] for t in unique_targets):
        raise ValueError(f"Invalid target values: {unique_targets}")
    
    logger.info("Data validation passed")
    return True

