"""
Unit tests for model training and prediction.
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import TARGET_COLUMN
from src.preprocessing import create_preprocessor, preprocess_data
from src.train import create_full_pipeline, evaluate_model, get_models


class TestModelTraining:
    """Tests for model training functions."""

    def test_get_models(self):
        """Test getting model dictionary."""
        models = get_models()

        assert "LogisticRegression" in models
        assert "RandomForest" in models
        assert isinstance(models["LogisticRegression"], LogisticRegression)
        assert isinstance(models["RandomForest"], RandomForestClassifier)

    def test_model_training(self, sample_dataframe):
        """Test that models can be trained on sample data."""
        # Preprocess data
        X_processed, y, preprocessor = preprocess_data(sample_dataframe, fit=True)

        models = get_models()

        for name, model in models.items():
            # Should not raise any errors
            model.fit(X_processed, y)

            # Should be able to predict
            predictions = model.predict(X_processed)
            assert len(predictions) == len(y)

            # Should be able to get probabilities
            probas = model.predict_proba(X_processed)
            assert probas.shape == (len(y), 2)

    def test_evaluate_model(self, sample_dataframe):
        """Test model evaluation."""
        # Preprocess and train
        X_processed, y, preprocessor = preprocess_data(sample_dataframe, fit=True)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_processed, y)

        # Evaluate
        metrics = evaluate_model(model, X_processed, y)

        # Check all expected metrics are present
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics

        # Check metrics are valid
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"{metric_name} should be between 0 and 1"

    def test_create_full_pipeline(self, sample_dataframe):
        """Test creating full pipeline."""
        # Create and fit preprocessor
        preprocessor = create_preprocessor()
        X = sample_dataframe.drop(columns=[TARGET_COLUMN])
        y = (sample_dataframe[TARGET_COLUMN] > 0).astype(int)

        preprocessor.fit(X)

        # Create model
        model = LogisticRegression(max_iter=1000)
        X_processed = preprocessor.transform(X)
        model.fit(X_processed, y)

        # Create pipeline
        pipeline = create_full_pipeline(preprocessor, model)

        assert isinstance(pipeline, Pipeline)

        # Pipeline should work end-to-end
        predictions = pipeline.predict(X)
        assert len(predictions) == len(X)

    def test_pipeline_save_and_load(self, sample_dataframe, temp_model_dir):
        """Test saving and loading pipeline."""
        # Create pipeline
        preprocessor = create_preprocessor()
        X = sample_dataframe.drop(columns=[TARGET_COLUMN])
        y = (sample_dataframe[TARGET_COLUMN] > 0).astype(int)

        preprocessor.fit(X)
        model = LogisticRegression(max_iter=1000)
        X_processed = preprocessor.transform(X)
        model.fit(X_processed, y)

        pipeline = create_full_pipeline(preprocessor, model)

        # Save
        filepath = temp_model_dir / "test_pipeline.pkl"
        joblib.dump(pipeline, filepath)

        assert filepath.exists()

        # Load
        loaded_pipeline = joblib.load(filepath)

        # Verify predictions match
        original_preds = pipeline.predict(X)
        loaded_preds = loaded_pipeline.predict(X)

        np.testing.assert_array_equal(original_preds, loaded_preds)
