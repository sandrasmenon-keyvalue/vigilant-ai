#!/usr/bin/env python3
"""
Fix All Imports Script
Fixes all import path issues in the Vigilant AI system.
"""

import os
import sys
from pathlib import Path

def fix_vitals_websocket_processor():
    """Fix imports in vitals_websocket_processor.py"""
    print("🔧 Fixing vitals_websocket_processor.py imports...")
    
    file_path = Path(__file__).parent / "vitals_processing" / "vitals_websocket_processor.py"
    
    if not file_path.exists():
        print("❌ File not found: vitals_websocket_processor.py")
        return False
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if "from vitals_processing.vitals_processor import VitalsProcessor" in content:
        print("✅ vitals_websocket_processor.py already fixed")
        return True
    
    # Fix the imports
    old_imports = """import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable
from vitals_processor import VitalsProcessor
from websocket_client import VitalsWebSocketClient"""
    
    new_imports = """import asyncio
import logging
import time
import sys
import os
from typing import Dict, Any, Optional, Callable

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitals_processing.vitals_processor import VitalsProcessor
from websocket.websocket_client import VitalsWebSocketClient"""
    
    if old_imports in content:
        content = content.replace(old_imports, new_imports)
        
        # Write back the file
        with open(file_path, 'w') as f:
            f.write(content)
        
        print("✅ Fixed vitals_websocket_processor.py imports")
        return True
    else:
        print("⚠️  Could not find expected import pattern in vitals_websocket_processor.py")
        return False

def test_all_imports():
    """Test all critical imports"""
    print("\n🧪 Testing All Imports")
    print("-" * 30)
    
    # Add project root to path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Test imports
    tests = [
        ("inference.synchronized_inference_engine", "SynchronizedInferenceEngine"),
        ("vitals_processing.vitals_processor", "VitalsProcessor"),
        ("vitals_processing.vitals_websocket_processor", "VitalsWebSocketProcessor"),
        ("websocket.websocket_client", "VitalsWebSocketClient"),
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
    """Main fix function"""
    print("🚗 Vigilant AI - Fix All Imports")
    print("=" * 40)
    
    # Fix vitals websocket processor
    if not fix_vitals_websocket_processor():
        print("❌ Failed to fix vitals_websocket_processor.py")
        return False
    
    # Test all imports
    if not test_all_imports():
        print("\n❌ Some imports still failing")
        print("   You may need to install missing dependencies:")
        print("   pip install -r requirements.txt")
        return False
    
    print("\n✅ All imports fixed successfully!")
    print("\n📋 Next steps:")
    print("   1. Run: python setup_and_run.py")
    print("   2. Or run: python start_integrated_system.py")
    
    return True

if __name__ == "__main__":
    main()
