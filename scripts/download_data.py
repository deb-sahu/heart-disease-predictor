#!/usr/bin/env python3
"""
Script to download the Heart Disease UCI dataset.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import fetch_dataset_from_uci, save_raw_data
from src.config import RAW_DATA_PATH


def main():
    """Download and save the Heart Disease dataset."""
    print("=" * 50)
    print("Heart Disease Dataset Download Script")
    print("=" * 50)
    
    try:
        print("\nFetching dataset from UCI ML Repository...")
        X, y = fetch_dataset_from_uci()
        
        print(f"Dataset size: {len(X)} samples, {X.shape[1]} features")
        print(f"Target distribution:\n{y.value_counts()}")
        
        print(f"\nSaving to {RAW_DATA_PATH}...")
        save_raw_data(X, y)
        
        print("\n✓ Dataset downloaded and saved successfully!")
        print(f"  Location: {RAW_DATA_PATH}")
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

