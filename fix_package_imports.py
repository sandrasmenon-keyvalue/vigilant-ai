#!/usr/bin/env python3
"""
Fix Package Imports Script
Creates missing __init__.py files and fixes all import issues.
"""

import os
from pathlib import Path

def create_init_files():
    """Create missing __init__.py files for all packages."""
    print("🔧 Creating Missing __init__.py Files")
    print("-" * 40)
    
    # Define packages and their contents
    packages = {
        'websocket': {
            'file': 'websocket/__init__.py',
            'content': '''"""
WebSocket package for Vigilant AI
Contains WebSocket client for receiving vitals data.
"""

from .websocket_client import VitalsWebSocketClient

__all__ = ['VitalsWebSocketClient']
'''
        },
        'vitals_processing': {
            'file': 'vitals_processing/__init__.py',
            'content': '''"""
Vitals Processing package for Vigilant AI
Contains vitals processing and WebSocket integration.
"""

from .vitals_processor import VitalsProcessor
from .vitals_websocket_processor import VitalsWebSocketProcessor

__all__ = ['VitalsProcessor', 'VitalsWebSocketProcessor']
'''
        },
        'inference': {
            'file': 'inference/__init__.py',
            'content': '''"""
Inference package for Vigilant AI
Contains synchronized inference engine for DV and HV data processing.
"""

from .synchronized_inference_engine import SynchronizedInferenceEngine

__all__ = ['SynchronizedInferenceEngine']
'''
        },
        'actions': {
            'file': 'actions/__init__.py',
            'content': '''"""
Actions package for Vigilant AI
Contains alert prediction and health score calculation.
"""
'''
        },
        'actions/alert_prediction': {
            'file': 'actions/alert_prediction/__init__.py',
            'content': '''"""
Alert Prediction module for Vigilant AI
Contains alert creation and management.
"""
'''
        },
        'actions/health_score_prediction': {
            'file': 'actions/health_score_prediction/__init__.py',
            'content': '''"""
Health Score Prediction module for Vigilant AI
Contains health score calculation.
"""
'''
        }
    }
    
    created_files = []
    for package_name, package_info in packages.items():
        file_path = Path(__file__).parent / package_info['file']
        
        if not file_path.exists():
            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the __init__.py file
            with open(file_path, 'w') as f:
                f.write(package_info['content'])
            
            created_files.append(file_path)
            print(f"✅ Created {package_info['file']}")
        else:
            print(f"✅ {package_info['file']} already exists")
    
    return created_files

def test_imports():
    """Test all imports after creating __init__.py files."""
    print("\n🧪 Testing All Imports")
    print("-" * 30)
    
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Test imports
    tests = [
        ("websocket.websocket_client", "VitalsWebSocketClient"),
        ("vitals_processing.vitals_processor", "VitalsProcessor"),
        ("vitals_processing.vitals_websocket_processor", "VitalsWebSocketProcessor"),
        ("inference.synchronized_inference_engine", "SynchronizedInferenceEngine"),
        ("actions.alert_prediction.alert_prediction", "Alert"),
        ("inference_api", "process_frame_inference")
    ]
    
    all_passed = True
    for module, class_name in tests:
        try:
            module_obj = __import__(module, fromlist=[class_name])
            getattr(module_obj, class_name)
            print(f"✅ {module}.{class_name}")
        except ImportError as e:
            print(f"❌ {module}.{class_name} - {e}")
            all_passed = False
        except AttributeError as e:
            print(f"❌ {module}.{class_name} - {e}")
            all_passed = False
    
    return all_passed

def main():
    """Main fix function."""
    print("🚗 Vigilant AI - Fix Package Imports")
    print("=" * 50)
    
    # Create __init__.py files
    created_files = create_init_files()
    
    if created_files:
        print(f"\n📁 Created {len(created_files)} __init__.py files")
    
    # Test imports
    if test_imports():
        print("\n✅ All imports working correctly!")
        print("\n📋 Next steps:")
        print("   1. Run: python start_system.py")
        print("   2. Or run: python start_integrated_system.py")
        return True
    else:
        print("\n❌ Some imports still failing")
        print("   You may need to install missing dependencies:")
        print("   pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    main()
