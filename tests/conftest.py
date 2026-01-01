"""
Pytest configuration and fixtures.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_patient_data():
    """Sample patient data for testing."""
    return {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0.0,
        "thal": 1.0
    }


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        "age": [63, 67, 37, 41, 56],
        "sex": [1, 1, 1, 0, 1],
        "cp": [3, 0, 1, 1, 1],
        "trestbps": [145, 160, 130, 130, 120],
        "chol": [233, 286, 250, 204, 236],
        "fbs": [1, 0, 0, 0, 0],
        "restecg": [0, 0, 1, 0, 1],
        "thalach": [150, 108, 187, 172, 178],
        "exang": [0, 1, 0, 0, 0],
        "oldpeak": [2.3, 1.5, 3.5, 1.4, 0.8],
        "slope": [0, 1, 0, 2, 2],
        "ca": [0.0, 3.0, 0.0, 0.0, 0.0],
        "thal": [1.0, 2.0, 2.0, 2.0, 2.0],
        "target": [1, 2, 0, 0, 0]
    })


@pytest.fixture
def sample_dataframe_with_missing():
    """Sample DataFrame with missing values for testing."""
    return pd.DataFrame({
        "age": [63, 67, 37, 41, 56],
        "sex": [1, 1, 1, 0, 1],
        "cp": [3, 0, 1, 1, 1],
        "trestbps": [145, 160, 130, 130, 120],
        "chol": [233, 286, 250, 204, 236],
        "fbs": [1, 0, 0, 0, 0],
        "restecg": [0, 0, 1, 0, 1],
        "thalach": [150, 108, 187, 172, 178],
        "exang": [0, 1, 0, 0, 0],
        "oldpeak": [2.3, 1.5, 3.5, 1.4, 0.8],
        "slope": [0, 1, 0, 2, 2],
        "ca": [0.0, np.nan, 0.0, np.nan, 0.0],  # Missing values
        "thal": [1.0, 2.0, np.nan, 2.0, 2.0],   # Missing values
        "target": [1, 2, 0, 0, 0]
    })


@pytest.fixture
def temp_data_dir():
    """Temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_model_dir():
    """Temporary directory for test models."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

