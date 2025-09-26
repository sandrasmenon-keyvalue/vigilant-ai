# 🚗 Enhanced Drowsiness Model Training with Limited Dataset

This guide explains how to train the enhanced drowsiness detection model using exactly **5,000 images per category** (10,000 total images) for optimal performance and balanced training.

## 🎯 Features

- **Dataset Limiting**: Automatically limits to 5,000 images per category
- **Random Sampling**: Ensures diverse representation from your full dataset
- **Enhanced Features**: Uses MediaPipe Face Landmarker with 478 3D landmarks
- **Blendshape Integration**: Includes facial expression blendshapes
- **Optimized Extraction**: Drowsiness-specific feature optimization
- **Balanced Training**: Perfect 50/50 class balance

## 📁 Dataset Structure

Your dataset should be organized as follows:

```
your_dataset/
├── drowsy/
│   ├── drowsy_001.jpg
│   ├── drowsy_002.jpg
│   ├── drowsy_003.jpg
│   └── ... (5,000+ images)
└── non_drowsy/  (or alert/, awake/, normal/)
    ├── alert_001.jpg
    ├── alert_002.jpg
    ├── alert_003.jpg
    └── ... (5,000+ images)
```

### Supported Directory Names for Non-Drowsy:
- `non_drowsy/`
- `not_drowsy/`
- `notdrowsy/`
- `alert/`
- `awake/`
- `normal/`

### Supported Image Formats:
- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.tiff`, `.tif`

## 🚀 Quick Start

### Method 1: Using the Enhanced Retrain Script

```bash
python retrain_enhanced_model.py
```

This will:
1. Prompt you for your dataset path
2. Validate the dataset structure
3. Show you the image counts
4. Train with exactly 5,000 images per category

### Method 2: Using the Wrapper Script

```bash
python train_with_limited_dataset.py
```

## 📊 Training Process

### 1. Dataset Validation
- Checks for required directory structure
- Counts available images in each category
- Warns if less than 5,000 images available

### 2. Random Sampling
- If more than 5,000 images available, randomly samples exactly 5,000
- Uses fixed random seed (42) for reproducible results
- Maintains class balance

### 3. Feature Extraction
- Processes each image with MediaPipe Face Landmarker
- Extracts 478 3D facial landmarks
- Includes facial expression blendshapes
- Uses optimized drowsiness feature extraction

### 4. Model Training
- Trains XGBoost model on extracted features
- Uses train/test split (80/20)
- Performs hyperparameter optimization
- Generates comprehensive performance metrics

### 5. Model Saving
- Saves trained model and feature scaler
- Generates detailed training report
- Creates human-readable summary

## 📈 Expected Output

```
🎯 Dataset limiting enabled: 5000 images per category
🎲 Found 15000 drowsy images, randomly sampling 5000
🎲 Found 12000 non_drowsy images, randomly sampling 5000
✅ Limited dataset loaded: 10000 images
   💤 Drowsy: 5000
   😊 Alert: 5000
   📊 Class balance: 50.00% drowsy
   🎯 Target per category: 5000

📊 Progress: 10000/10000 (100.0%) - Faces: 9847, Blendshapes: 9654
✅ Enhanced feature extraction completed!
   📈 Success rate: 98.5% (9847/10000)
   🎭 Blendshapes detected: 98.0% (9654/9847)

🎯 Enhanced model trained!
   📊 Accuracy: 0.924
   📈 AUC Score: 0.967
   ⏱️  Training Time: 45.2s
```

## 📋 Output Files

After successful training, you'll find:

```
trained_models/enhanced_vision_training_limited/
├── models/
│   ├── drowsiness_model.pkl          # Main model file
│   ├── feature_scaler.pkl            # Feature scaler
│   ├── enhanced_drowsiness_model_20241201_143022.pkl  # Timestamped backup
│   └── enhanced_feature_scaler_20241201_143022.pkl    # Timestamped backup
├── enhanced_training_report.json     # Detailed training metrics
├── training_summary.txt              # Human-readable summary
└── feature_statistics.json          # Feature analysis
```

## 🔄 Deploying the New Model

### Option 1: Manual Copy (Recommended)
```bash
# Copy the new model files
cp trained_models/enhanced_vision_training_limited/models/drowsiness_model.pkl ai_module/vision_model/
cp trained_models/enhanced_vision_training_limited/models/feature_scaler.pkl ai_module/vision_model/

# Restart services
python start_all_services.py
```

### Option 2: Update Model Path
Update your inference configuration to point to the new model location.

## 🎯 Performance Optimization

### Dataset Size Recommendations:
- **Minimum**: 1,000 images per category
- **Recommended**: 5,000 images per category (current default)
- **Maximum**: No limit (will be sampled down to 5,000)

### Quality Tips:
1. **Diverse Lighting**: Include various lighting conditions
2. **Multiple Angles**: Different head poses and camera angles  
3. **Various Demographics**: Different ages, ethnicities, genders
4. **Clear Faces**: Ensure faces are clearly visible and unobstructed
5. **Balanced Drowsiness**: Include various levels of drowsiness

## 🔧 Customization

### Change Dataset Size Limit

Edit `retrain_enhanced_model.py`:

```python
# Change the default limit
pipeline = LimitedDatasetTrainingPipeline(
    resources_dir=data_path,
    output_dir=output_dir,
    model_type='xgboost',
    max_images_per_category=3000  # Change this value
)
```

### Change Model Type

Supported model types:
- `'xgboost'` (default, recommended)
- `'logistic'` (faster, simpler)
- `'random_forest'` (good baseline)

## 🐛 Troubleshooting

### Common Issues:

1. **"No face detected" warnings**: Normal for some images, aim for >95% success rate
2. **Memory errors**: Reduce batch size or use fewer images
3. **Low accuracy**: Check dataset quality and balance
4. **Import errors**: Ensure all dependencies are installed

### Performance Expectations:
- **Accuracy**: >90% (excellent), >85% (good), <80% (check data quality)
- **AUC Score**: >0.95 (excellent), >0.90 (good), <0.85 (check data quality)
- **Processing Speed**: ~2-5 images/second during training

## 📞 Support

If you encounter issues:
1. Check the training logs for specific error messages
2. Verify your dataset structure matches the expected format
3. Ensure you have sufficient disk space (>2GB recommended)
4. Check that all dependencies are properly installed

---

**Happy Training! 🚀**

Your enhanced model will provide much better drowsiness detection with the optimized features and balanced dataset.
