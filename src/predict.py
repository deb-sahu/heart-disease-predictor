"""
Prediction utilities for the Heart Disease Predictor.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
import logging
import joblib
from pathlib import Path

from src.config import (
    PIPELINE_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeartDiseasePredictor:
    """
    Heart Disease Prediction model wrapper for inference.
    """
    
    def __init__(self, pipeline_path: Optional[str] = None):
        """
        Initialize the predictor with a trained pipeline.
        
        Args:
            pipeline_path: Path to saved pipeline. Defaults to PIPELINE_PATH.
        """
        self.pipeline_path = Path(pipeline_path) if pipeline_path else PIPELINE_PATH
        self.pipeline = None
        self.feature_names = ALL_FEATURES
        
    def load(self) -> "HeartDiseasePredictor":
        """
        Load the trained pipeline from disk.
        
        Returns:
            self for method chaining
        """
        if not self.pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline not found at {self.pipeline_path}")
        
        self.pipeline = joblib.load(self.pipeline_path)
        logger.info(f"Loaded pipeline from {self.pipeline_path}")
        return self
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """
        Validate input data has all required features.
        
        Args:
            data: Input data dictionary
            
        Returns:
            True if valid
            
        Raises:
            ValueError if invalid
        """
        missing_features = set(self.feature_names) - set(data.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Validate numeric features are numbers
        for feature in NUMERIC_FEATURES:
            if feature in data and not isinstance(data[feature], (int, float)):
                try:
                    float(data[feature])
                except (ValueError, TypeError):
                    raise ValueError(f"Feature '{feature}' must be numeric, got: {data[feature]}")
        
        return True
    
    def predict(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Make a prediction for the given input data.
        
        Args:
            data: Input data as dict or DataFrame
            
        Returns:
            Dictionary with prediction and probability
        """
        if self.pipeline is None:
            self.load()
        
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            self.validate_input(data)
            df = pd.DataFrame([data])
        else:
            df = data
        
        # Ensure correct column order
        df = df[self.feature_names]
        
        # Make prediction
        prediction = self.pipeline.predict(df)[0]
        probability = self.pipeline.predict_proba(df)[0]
        
        result = {
            "prediction": int(prediction),
            "prediction_label": "Heart Disease Present" if prediction == 1 else "No Heart Disease",
            "probability_no_disease": float(probability[0]),
            "probability_disease": float(probability[1]),
            "confidence": float(max(probability))
        }
        
        logger.info(f"Prediction: {result['prediction_label']} (confidence: {result['confidence']:.2%})")
        return result
    
    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for a batch of inputs.
        
        Args:
            data: DataFrame with input features
            
        Returns:
            DataFrame with predictions added
        """
        if self.pipeline is None:
            self.load()
        
        # Ensure correct column order
        df = data[self.feature_names].copy()
        
        # Make predictions
        predictions = self.pipeline.predict(df)
        probabilities = self.pipeline.predict_proba(df)
        
        result_df = data.copy()
        result_df['prediction'] = predictions
        result_df['probability_no_disease'] = probabilities[:, 0]
        result_df['probability_disease'] = probabilities[:, 1]
        result_df['prediction_label'] = result_df['prediction'].map({
            0: "No Heart Disease",
            1: "Heart Disease Present"
        })
        
        return result_df


def load_predictor(pipeline_path: Optional[str] = None) -> HeartDiseasePredictor:
    """
    Convenience function to load a predictor.
    
    Args:
        pipeline_path: Optional path to pipeline
        
    Returns:
        Loaded HeartDiseasePredictor instance
    """
    predictor = HeartDiseasePredictor(pipeline_path)
    predictor.load()
    return predictor


def predict_single(data: Dict[str, Any], pipeline_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for single prediction.
    
    Args:
        data: Input features as dictionary
        pipeline_path: Optional path to pipeline
        
    Returns:
        Prediction result dictionary
    """
    predictor = load_predictor(pipeline_path)
    return predictor.predict(data)


# Sample input for testing
SAMPLE_INPUT = {
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
    "ca": 0,
    "thal": 1
}


if __name__ == "__main__":
    # Test prediction with sample input
    print("Testing prediction with sample input...")
    print(f"Input: {SAMPLE_INPUT}")
    
    try:
        result = predict_single(SAMPLE_INPUT)
        print(f"Result: {result}")
    except FileNotFoundError:
        print("Model not found. Please run training first: python -m src.train")

