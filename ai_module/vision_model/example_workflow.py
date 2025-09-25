"""
Example workflow demonstrating the complete drowsiness detection pipeline.
"""

import os
import sys
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from setup_resources import create_resources_structure
from train_from_images import ImageTrainingPipeline
from inference_single_image import SingleImageInference


def run_example_workflow():
    """Run the complete example workflow."""
    print("🚗 Vigilant AI - Complete Workflow Example")
    print("=" * 60)
    
    # Step 1: Setup resources directory
    print("\n🔹 Step 1: Setting up resources directory...")
    resources_dir = "example_resources"
    create_resources_structure(resources_dir)
    
    print(f"\n📁 Resources directory created at: {resources_dir}")
    print("📝 Please add your image files to:")
    print(f"   - {resources_dir}/drowsy/ (images of drowsy drivers)")
    print(f"   - {resources_dir}/not_drowsy/ (images of not drowsy drivers)")
    
    # Check if user wants to continue
    response = input("\nHave you added image files to the resources directory? (y/n): ").lower()
    
    if response != 'y':
        print("⏭️  Skipping training. You can run this script again after adding images.")
        return
    
    # Step 2: Train model
    print("\n🔹 Step 2: Training model from images...")
    output_dir = "example_training_output"
    
    pipeline = ImageTrainingPipeline(
        resources_dir=resources_dir,
        output_dir=output_dir,
        fps=5.0
    )
    
    success = pipeline.run_complete_training()
    
    if not success:
        print("❌ Training failed. Please check your image files and try again.")
        return
    
    # Step 3: Test inference
    print("\n🔹 Step 3: Testing inference...")
    model_path = f"{output_dir}/models/drowsiness_model.pkl"
    scaler_path = f"{output_dir}/models/feature_scaler.pkl"
    
    if not Path(model_path).exists() or not Path(scaler_path).exists():
        print("❌ Model files not found. Training may have failed.")
        return
    
    # Initialize inference
    inference = SingleImageInference(model_path, scaler_path)
    
    # Ask user for inference method
    print("\nChoose inference method:")
    print("1. Single image")
    print("2. Directory of images")
    print("3. Skip inference")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        image_path = input("Enter path to image file: ").strip()
        if Path(image_path).exists():
            print(f"\n📸 Processing image: {image_path}")
            result = inference.process_image(image_path)
            print(f"✅ Drowsiness Score: {result['drowsiness_score']:.3f}")
            print(f"✅ Status: {'DROWSY' if result['drowsiness_score'] > 0.5 else 'ALERT'}")
        else:
            print(f"❌ Image file not found: {image_path}")
    
    elif choice == "2":
        directory_path = input("Enter path to directory: ").strip()
        if Path(directory_path).exists():
            print(f"\n📁 Processing directory: {directory_path}")
            results = inference.process_directory(directory_path)
            inference.print_summary(results)
        else:
            print(f"❌ Directory not found: {directory_path}")
    
    else:
        print("⏭️  Skipping inference.")
    
    # Final summary
    print("\n🎉 Example workflow completed!")
    print("=" * 60)
    print(f"📁 Resources directory: {resources_dir}")
    print(f"📁 Training output: {output_dir}")
    print(f"📁 Model files: {output_dir}/models/")
    print("\nTo use the trained model:")
    print(f"  python inference.py --model {model_path} --scaler {scaler_path}")


def main():
    """Main function."""
    try:
        run_example_workflow()
    except KeyboardInterrupt:
        print("\n\n👋 Workflow interrupted by user.")
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
