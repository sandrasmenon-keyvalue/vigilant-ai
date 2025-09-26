#!/usr/bin/env python3
"""
Quick script to train the enhanced drowsiness model with exactly 5,000 images per category.
This is a wrapper around the enhanced retrain script for easy usage.
"""

import os
import sys
from pathlib import Path

def main():
    print("🎯 ENHANCED DROWSINESS MODEL TRAINING")
    print("=" * 50)
    print("Training with LIMITED DATASET:")
    print("  • 5,000 drowsy images")
    print("  • 5,000 non-drowsy images")
    print("  • Total: 10,000 balanced images")
    print()
    
    # Check if retrain script exists
    retrain_script = Path(__file__).parent / "retrain_enhanced_model.py"
    if not retrain_script.exists():
        print("❌ retrain_enhanced_model.py not found!")
        return
    
    print("🚀 Starting enhanced training with limited dataset...")
    print("   This will prompt you for your dataset path.")
    print()
    
    # Run the enhanced retrain script
    os.system(f"python {retrain_script}")

if __name__ == "__main__":
    main()
