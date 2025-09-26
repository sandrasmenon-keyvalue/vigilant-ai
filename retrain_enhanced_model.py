#!/usr/bin/env python3
"""
Enhanced script to retrain the drowsiness model with limited dataset size.
Uses exactly 5,000 images each of drowsy and non-drowsy categories (10,000 total).
"""

import sys
import os
import random
import numpy as np
from pathlib import Path
from typing import List, Dict

# Add ai-module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai_module', 'vision_model'))

from ai_module.vision_model.enhanced_training_pipeline import EnhancedTrainingPipeline


class LimitedDatasetTrainingPipeline(EnhancedTrainingPipeline):
    """
    Enhanced training pipeline with dataset size limiting functionality.
    Limits to exactly 5,000 images per category for balanced training.
    """
    
    def __init__(self, resources_dir: str, output_dir: str, model_type: str = 'xgboost', 
                 max_images_per_category: int = 5000):
        """
        Initialize limited dataset training pipeline.
        
        Args:
            resources_dir: Directory containing image files
            output_dir: Directory to save training outputs
            model_type: Type of model to train
            max_images_per_category: Maximum images per category (default: 5000)
        """
        super().__init__(resources_dir, output_dir, model_type)
        self.max_images_per_category = max_images_per_category
        
        # Set random seed for reproducible sampling
        random.seed(42)
        np.random.seed(42)
        
        print(f"🎯 Dataset limiting enabled: {max_images_per_category} images per category")
    
    def load_images_with_labels(self) -> List[Dict]:
        """
        Load images with labels, limiting to specified number per category.
        
        Returns:
            List of dictionaries with image info (limited dataset)
        """
        print(f"📂 Loading images with dataset limiting ({self.max_images_per_category} per category)...")
        
        image_data = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Load drowsy images (with limiting)
        drowsy_dir = self.resources_dir / "drowsy"
        drowsy_images = []
        for img_file in drowsy_dir.iterdir():
            if img_file.suffix.lower() in image_extensions:
                drowsy_images.append(img_file)
        
        # Randomly sample drowsy images
        if len(drowsy_images) > self.max_images_per_category:
            print(f"🎲 Found {len(drowsy_images)} drowsy images, randomly sampling {self.max_images_per_category}")
            drowsy_images = random.sample(drowsy_images, self.max_images_per_category)
        else:
            print(f"📊 Found {len(drowsy_images)} drowsy images (using all)")
        
        for img_file in drowsy_images:
            image_data.append({
                'image_path': str(img_file),
                'label': 'drowsy',
                'image_name': img_file.name,
                'binary_label': 1  # 1 for drowsy
            })
        
        # Load non-drowsy images (with limiting)
        not_drowsy_dir = self.resources_dir / "non_drowsy"
        non_drowsy_label = "non_drowsy"
        
        # Try different directory names
        if not not_drowsy_dir.exists():
            alt_dirs = ["not_drowsy", "notdrowsy", "alert", "awake", "normal"]
            for alt_name in alt_dirs:
                alt_dir = self.resources_dir / alt_name
                if alt_dir.exists():
                    not_drowsy_dir = alt_dir
                    non_drowsy_label = alt_name
                    break
            else:
                raise ValueError(f"Non-drowsy directory not found! Tried: {alt_dirs}")
        
        non_drowsy_images = []
        for img_file in not_drowsy_dir.iterdir():
            if img_file.suffix.lower() in image_extensions:
                non_drowsy_images.append(img_file)
        
        # Randomly sample non-drowsy images
        if len(non_drowsy_images) > self.max_images_per_category:
            print(f"🎲 Found {len(non_drowsy_images)} {non_drowsy_label} images, randomly sampling {self.max_images_per_category}")
            non_drowsy_images = random.sample(non_drowsy_images, self.max_images_per_category)
        else:
            print(f"📊 Found {len(non_drowsy_images)} {non_drowsy_label} images (using all)")
        
        for img_file in non_drowsy_images:
            image_data.append({
                'image_path': str(img_file),
                'label': non_drowsy_label,
                'image_name': img_file.name,
                'binary_label': 0  # 0 for not drowsy
            })
        
        # Shuffle for better training
        random.shuffle(image_data)
        
        self.stats['images_processed'] = len(image_data)
        drowsy_count = len([d for d in image_data if d['binary_label'] == 1])
        alert_count = len([d for d in image_data if d['binary_label'] == 0])
        
        print(f"✅ Limited dataset loaded: {len(image_data)} images")
        print(f"   💤 Drowsy: {drowsy_count}")
        print(f"   😊 Alert: {alert_count}")
        print(f"   📊 Class balance: {drowsy_count/(drowsy_count+alert_count):.2%} drowsy")
        print(f"   🎯 Target per category: {self.max_images_per_category}")
        
        return image_data


def get_dataset_path() -> str:
    """Get dataset path from user input with validation."""
    print("📁 DATASET PATH CONFIGURATION")
    print("=" * 40)
    print("Please provide the path to your training dataset.")
    print("Expected structure:")
    print("  your_dataset/")
    print("  ├── drowsy/")
    print("  │   ├── image1.jpg")
    print("  │   ├── image2.jpg")
    print("  │   └── ...")
    print("  └── non_drowsy/ (or alert/, awake/, normal/)")
    print("      ├── image1.jpg")
    print("      ├── image2.jpg")
    print("      └── ...")
    print()
    
    while True:
        data_path = input("Enter dataset path: ").strip()
        
        if not data_path:
            print("❌ Please provide a valid path!")
            continue
        
        data_path = Path(data_path)
        if not data_path.exists():
            print(f"❌ Directory not found: {data_path}")
            continue
        
        # Check for drowsy directory
        drowsy_dir = data_path / "drowsy"
        if not drowsy_dir.exists():
            print(f"❌ 'drowsy' directory not found in {data_path}")
            continue
        
        # Check for non-drowsy directory (try multiple names)
        non_drowsy_found = False
        possible_dirs = ["non_drowsy", "not_drowsy", "notdrowsy", "alert", "awake", "normal"]
        for dir_name in possible_dirs:
            if (data_path / dir_name).exists():
                non_drowsy_found = True
                break
        
        if not non_drowsy_found:
            print(f"❌ Non-drowsy directory not found in {data_path}")
            print(f"   Tried: {possible_dirs}")
            continue
        
        # Count images in each directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        drowsy_count = len([f for f in drowsy_dir.iterdir() if f.suffix.lower() in image_extensions])
        
        non_drowsy_count = 0
        for dir_name in possible_dirs:
            test_dir = data_path / dir_name
            if test_dir.exists():
                non_drowsy_count = len([f for f in test_dir.iterdir() if f.suffix.lower() in image_extensions])
                break
        
        print(f"✅ Dataset found: {data_path}")
        print(f"   💤 Drowsy images: {drowsy_count}")
        print(f"   😊 Non-drowsy images: {non_drowsy_count}")
        print(f"   📊 Total images: {drowsy_count + non_drowsy_count}")
        
        if drowsy_count < 5000 or non_drowsy_count < 5000:
            print(f"⚠️  Warning: Less than 5,000 images in one or both categories")
            print(f"   Will use all available images for training")
        
        return str(data_path)


def main():
    print("🚀 ENHANCED DROWSINESS MODEL TRAINING")
    print("=" * 60)
    print("This will create a new model using:")
    print("✅ MediaPipe Face Landmarker (478 3D landmarks)")
    print("✅ Facial expression blendshapes")
    print("✅ Optimized drowsiness feature extraction")
    print("✅ Better response to clear fatigue signs")
    print("🎯 LIMITED DATASET: 5,000 images per category (10,000 total)")
    print()
    
    # Get training data path from user
    data_path = get_dataset_path()
    
    # Output directory
    output_dir = "trained_models/enhanced_vision_training_limited"
    
    print(f"\n💾 Enhanced model will be saved to: {output_dir}")
    print(f"🎯 Training with maximum 5,000 images per category")
    print()
    
    response = input("🚀 Start enhanced training with limited dataset? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Training cancelled.")
        return
    
    try:
        # Create limited dataset training pipeline
        pipeline = LimitedDatasetTrainingPipeline(
            resources_dir=data_path,
            output_dir=output_dir,
            model_type='xgboost',
            max_images_per_category=5000
        )
        
        # Run enhanced training with limited dataset
        success = pipeline.run_enhanced_training()
        
        if success:
            print("\n🎉 ENHANCED MODEL TRAINING SUCCESS!")
            print("=" * 50)
            print("Your enhanced model features:")
            print("  🎯 MediaPipe Face Landmarker (478 landmarks)")
            print("  🎭 Facial expression blendshapes") 
            print("  🔧 Optimized drowsiness detection")
            print("  📈 Better response to clear fatigue signs")
            print("  🎲 Trained on balanced 10,000 image dataset")
            print(f"\n💾 New model saved to: {output_dir}/models/drowsiness_model.pkl")
            print(f"📋 Training report: {output_dir}/enhanced_training_report.json")
            
            print("\n🔄 NEXT STEPS:")
            print("1. Copy the new model to replace the current one:")
            print(f"   cp {output_dir}/models/drowsiness_model.pkl ai_module/vision_model/")
            print(f"   cp {output_dir}/models/feature_scaler.pkl ai_module/vision_model/")
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
