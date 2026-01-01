#!/usr/bin/env python3
"""
Script to run model training with MLflow tracking.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import train_all_models


def main():
    """Run the training pipeline."""
    print("=" * 60)
    print("Heart Disease Predictor - Model Training")
    print("=" * 60)
    
    try:
        results = train_all_models()
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        
        print(f"\nBest Model: {results['best_model_name']}")
        print(f"Best Metrics:")
        for metric, value in results['best_metrics'].items():
            print(f"  - {metric}: {value:.4f}")
        
        print("\nModel artifacts saved to: models/")
        print("MLflow runs saved to: mlruns/")
        print("\nTo view MLflow UI, run: mlflow ui --port 5000")
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

