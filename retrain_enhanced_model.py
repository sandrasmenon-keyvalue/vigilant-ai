#!/usr/bin/env python3
"""
Quick script to retrain the drowsiness model with enhanced features.
Uses the existing training data but with MediaPipe Face Landmarker and optimized extraction.
"""

import sys
import os
from pathlib import Path

# Add ai-module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai-module', 'vision-model'))

from enhanced_training_pipeline import EnhancedTrainingPipeline

def main():
    print("🚀 RETRAINING WITH ENHANCED DROWSINESS FEATURES")
    print("=" * 60)
    print("This will create a new model using:")
    print("✅ MediaPipe Face Landmarker (478 3D landmarks)")
    print("✅ Facial expression blendshapes")
    print("✅ Optimized drowsiness feature extraction")
    print("✅ Better response to clear fatigue signs")
    print()
    
    # Get training data path from user
    data_path = ""
    
    if not data_path:
        print("❌ No path provided!")
        return
    
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"❌ Directory not found: {data_path}")
        return
    
    # Check if it has the right structure
    drowsy_dir = data_path / "drowsy"
    # alert_dir = data_path / "alert"
    not_drowsy_dir = data_path / "notdrowsy"
    
    if not drowsy_dir.exists():
        print(f"❌ 'drowsy' directory not found in {data_path}")
        print("Expected structure:")
        print("  training_data/")
        print("  ├── drowsy/")
        print("  └── alert/ (or not_drowsy/)")
        return
    
    # if not (alert_dir.exists() or not_drowsy_dir.exists()):
    #     print(f"❌ Neither 'alert' nor 'not_drowsy' directory found in {data_path}")
    #     print("Expected structure:")
    #     print("  training_data/")
    #     print("  ├── drowsy/")
    #     print("  └── alert/ (or not_drowsy/)")
    #     return
    
    print(f"✅ Training data found: {data_path}")
    
    # Output directory
    output_dir = "trained_models/enhanced_vision_training"
    
    print(f"💾 Enhanced model will be saved to: {output_dir}")
    print()
    
    response = input("🚀 Start enhanced training? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Training cancelled.")
        return
    
    try:
        # Create enhanced training pipeline
        pipeline = EnhancedTrainingPipeline(
            resources_dir=str(data_path),
            output_dir=output_dir,
            model_type='xgboost'
        )
        
        # Run enhanced training
        success = pipeline.run_enhanced_training()
        
        if success:
            print("\n🎉 ENHANCED MODEL TRAINING SUCCESS!")
            print("=" * 50)
            print("Your enhanced model features:")
            print("  🎯 MediaPipe Face Landmarker (478 landmarks)")
            print("  🎭 Facial expression blendshapes") 
            print("  🔧 Optimized drowsiness detection")
            print("  📈 Better response to clear fatigue signs")
            print(f"\n💾 New model saved to: {output_dir}/models/drowsiness_model.pkl")
            print(f"📋 Training report: {output_dir}/enhanced_training_report.json")
            
            print("\n🔄 NEXT STEPS:")
            print("1. The inference_api.py will automatically use the new model")
            print("2. Restart your services to load the enhanced model:")
            print("   python start_all_services.py")
            print("3. Test with live camera - should be much more responsive!")
            
        else:
            print("\n❌ ENHANCED TRAINING FAILED!")
            print("Check the error messages above.")
            
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
