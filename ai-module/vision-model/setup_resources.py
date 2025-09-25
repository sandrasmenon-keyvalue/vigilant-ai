"""
Setup script to create resources directory structure for training.
"""

import os
import argparse
from pathlib import Path


def create_resources_structure(resources_dir: str):
    """
    Create the expected resources directory structure.
    
    Args:
        resources_dir: Path to resources directory
    """
    resources_path = Path(resources_dir)
    
    # Create main directories
    resources_path.mkdir(parents=True, exist_ok=True)
    (resources_path / "drowsy").mkdir(exist_ok=True)
    (resources_path / "not_drowsy").mkdir(exist_ok=True)
    
    # Create README files
    readme_content = """# Drowsiness Detection Training Data

## Directory Structure

```
resources/
├── drowsy/          # Images of drowsy drivers
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── not_drowsy/      # Images of not drowsy drivers
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Image Requirements

- **Format**: JPG, JPEG, PNG, BMP, TIFF
- **Quality**: 720p or higher recommended
- **Content**: Clear view of driver's face
- **Lighting**: Good lighting conditions
- **Resolution**: At least 640x480 pixels

## Data Collection Tips

### Drowsy Images
- Capture when driver is actually tired/sleepy
- Include yawning, eye rubbing, head nodding
- Show eyes closing or heavy eyelids
- Capture micro-sleeps or brief eye closures
- Take multiple angles and expressions

### Not Drowsy Images
- Capture when driver is fully awake and alert
- Include normal driving behavior
- Show regular blinking patterns
- Capture attentive driving posture
- Take multiple angles and expressions

## Usage

1. Place your image files in the appropriate directories
2. Run the training script:
   ```bash
   python train_from_images.py --resources ./resources --output ./training_output
   ```
"""
    
    with open(resources_path / "README.md", 'w') as f:
        f.write(readme_content)
    
    # Create placeholder files
    drowsy_readme = """# Drowsy Images

Place images of drowsy drivers here.

Each image should show:
- Driver appearing tired or sleepy
- Yawning, eye rubbing, or head nodding
- Eyes closing or heavy eyelids
- Micro-sleeps or brief eye closures

Supported formats: JPG, JPEG, PNG, BMP, TIFF
"""
    
    not_drowsy_readme = """# Not Drowsy Images

Place images of not drowsy drivers here.

Each image should show:
- Driver appearing fully awake and alert
- Normal driving behavior
- Regular blinking patterns
- Attentive driving posture

Supported formats: JPG, JPEG, PNG, BMP, TIFF
"""
    
    with open(resources_path / "drowsy" / "README.md", 'w') as f:
        f.write(drowsy_readme)
    
    with open(resources_path / "not_drowsy" / "README.md", 'w') as f:
        f.write(not_drowsy_readme)
    
    print(f"✅ Created resources directory structure at: {resources_path}")
    print(f"📁 Drowsy images directory: {resources_path / 'drowsy'}")
    print(f"📁 Not_drowsy images directory: {resources_path / 'not_drowsy'}")
    print("\nNext steps:")
    print("1. Add your image files to the appropriate directories")
    print("2. Run training: python train_from_images.py --resources ./resources")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Setup resources directory for training')
    parser.add_argument('--resources', default='resources', 
                       help='Path to resources directory (default: resources)')
    
    args = parser.parse_args()
    
    create_resources_structure(args.resources)


if __name__ == "__main__":
    main()
