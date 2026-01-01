"""
Model training with MLflow experiment tracking.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
import joblib
import json
from datetime import datetime

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_PATH,
    PIPELINE_PATH,
    ARTIFACTS_DIR,
    MODELS_DIR
)
from src.data_loader import prepare_dataset, get_feature_target_split
from src.preprocessing import (
    preprocess_data,
    create_preprocessor,
    save_preprocessor,
    handle_missing_values,
    convert_target_to_binary
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_mlflow():
    """Configure MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")


def get_models() -> Dict[str, Any]:
    """
    Get dictionary of models to train.
    
    Returns:
        Dictionary mapping model names to model instances
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver='lbfgs'
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    
    return metrics


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str
) -> str:
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        
    Returns:
        Path to saved plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['No Disease', 'Disease'],
        yticklabels=['No Disease', 'Disease']
    )
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    filepath = ARTIFACTS_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(filepath)


def plot_roc_curve(
    y_true: pd.Series,
    y_prob: np.ndarray,
    model_name: str
) -> str:
    """
    Plot and save ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        model_name: Name of the model
        
    Returns:
        Path to saved plot
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    
    filepath = ARTIFACTS_DIR / f"roc_curve_{model_name}.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(filepath)


def train_and_log_model(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: Any
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a model and log to MLflow.
    
    Args:
        model_name: Name of the model
        model: Model instance
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        preprocessor: Fitted preprocessor
        
    Returns:
        Tuple of (trained model, metrics dict)
    """
    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log model parameters
        params = model.get_params()
        mlflow.log_params(params)
        
        # Log training configuration
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("random_state", RANDOM_STATE)
        
        # Train model
        logger.info(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='roc_auc')
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
        
        # Evaluate on test set
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Generate and log plots
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
        roc_path = plot_roc_curve(y_test, y_prob, model_name)
        
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)
        
        # Log classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = ARTIFACTS_DIR / f"classification_report_{model_name}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(str(report_path))
        
        # Log the model with signature and input example
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train[:1]  # First row as example
        mlflow.sklearn.log_model(
            model, 
            f"model_{model_name}",
            signature=signature,
            input_example=input_example
        )
        
        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
        
    return model, metrics


def create_full_pipeline(preprocessor: Any, model: Any) -> Pipeline:
    """
    Create a full sklearn pipeline with preprocessing and model.
    
    Args:
        preprocessor: Fitted preprocessor
        model: Trained model
        
    Returns:
        Complete pipeline
    """
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])


def save_model(model: Any, filepath: Optional[str] = None) -> str:
    """
    Save model to disk.
    
    Args:
        model: Model to save
        filepath: Output path
        
    Returns:
        Path to saved model
    """
    filepath = filepath or str(MODEL_PATH)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Saved model to {filepath}")
    return filepath


def save_pipeline(pipeline: Pipeline, filepath: Optional[str] = None) -> str:
    """
    Save full pipeline to disk.
    
    Args:
        pipeline: Pipeline to save
        filepath: Output path
        
    Returns:
        Path to saved pipeline
    """
    filepath = filepath or str(PIPELINE_PATH)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, filepath)
    logger.info(f"Saved pipeline to {filepath}")
    return filepath


def train_all_models() -> Dict[str, Any]:
    """
    Main training function - trains all models and returns results.
    
    Returns:
        Dictionary with training results
    """
    # Setup MLflow
    setup_mlflow()
    
    # Load and prepare data
    logger.info("Loading dataset...")
    df = prepare_dataset()
    
    # Preprocess data
    df = handle_missing_values(df)
    df = convert_target_to_binary(df)
    
    X, y = get_feature_target_split(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Create and fit preprocessor
    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save preprocessor
    save_preprocessor(preprocessor)
    
    # Train models
    models = get_models()
    results = {}
    best_model = None
    best_metrics = None
    best_model_name = None
    
    for model_name, model in models.items():
        trained_model, metrics = train_and_log_model(
            model_name,
            model,
            X_train_processed,
            X_test_processed,
            y_train,
            y_test,
            preprocessor
        )
        
        results[model_name] = {
            "model": trained_model,
            "metrics": metrics
        }
        
        # Track best model by ROC-AUC
        if best_metrics is None or metrics['roc_auc'] > best_metrics['roc_auc']:
            best_model = trained_model
            best_metrics = metrics
            best_model_name = model_name
    
    # Save best model
    logger.info(f"Best model: {best_model_name} (ROC-AUC: {best_metrics['roc_auc']:.4f})")
    save_model(best_model)
    
    # Create and save full pipeline
    full_pipeline = create_full_pipeline(preprocessor, best_model)
    save_pipeline(full_pipeline)
    
    return {
        "results": results,
        "best_model_name": best_model_name,
        "best_metrics": best_metrics
    }


if __name__ == "__main__":
    results = train_all_models()
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best Model: {results['best_model_name']}")
    print(f"Best ROC-AUC: {results['best_metrics']['roc_auc']:.4f}")

