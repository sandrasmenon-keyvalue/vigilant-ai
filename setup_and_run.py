#!/usr/bin/env python3
"""
Setup and Run Script
Handles dependency installation and system startup for Vigilant AI.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(command, description, check=True):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"⚠️  {description} completed with warnings")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing Dependencies")
    print("-" * 30)
    
    # Try to install from requirements.txt
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        success = run_command(f"{sys.executable} -m pip install -r {requirements_file}", 
                            "Installing from requirements.txt")
        if success:
            return True
    
    # If that fails, try installing core packages individually
    print("🔄 Trying individual package installation...")
    
    core_packages = [
        "opencv-python==4.8.1.78",
        "mediapipe==0.10.7", 
        "numpy==1.24.3",
        "pandas==2.0.3",
        "scikit-learn==1.3.0",
        "xgboost==1.7.6",
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "websockets==12.0",
        "aiohttp==3.9.0",
        "python-multipart==0.0.6"
    ]
    
    failed_packages = []
    for package in core_packages:
        if not run_command(f"{sys.executable} -m pip install {package}", 
                         f"Installing {package.split('==')[0]}", check=False):
            failed_packages.append(package)
    
    if failed_packages:
        print(f"⚠️  Some packages failed to install: {failed_packages}")
        print("   You may need to install them manually")
    
    return len(failed_packages) == 0

def verify_installation():
    """Verify that key packages are installed."""
    print("\n🔍 Verifying Installation")
    print("-" * 30)
    
    # Run the comprehensive import test
    try:
        import subprocess
        result = subprocess.run([sys.executable, "test_imports.py"], 
                              capture_output=True, text=True, cwd=Path(__file__).parent)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def start_system():
    """Start the integrated system."""
    print("\n🚀 Starting Integrated System")
    print("-" * 30)
    
    # Check if start script exists
    start_script = Path(__file__).parent / "start_integrated_system.py"
    if not start_script.exists():
        print("❌ start_integrated_system.py not found")
        return False
    
    print("🎉 Starting all services...")
    print("   This will start:")
    print("   • Inference API (port 8002)")
    print("   • Vitals WebSocket Server (port 8765)")
    print("   • Integrated Live Stream Service (port 8001)")
    print("\n   Press Ctrl+C to stop all services")
    print("=" * 50)
    
    try:
        # Start the integrated system
        subprocess.run([sys.executable, str(start_script)], check=True)
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to start system: {e}")
        return False
    
    return True

def main():
    """Main setup and run function."""
    print("🚗 Vigilant AI - Setup and Run")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\n⚠️  Some dependencies failed to install")
        print("   You may need to install them manually")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Verify installation
    if not verify_installation():
        print("\n⚠️  Some required packages are missing")
        print("   The system may not work properly")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Start the system
    print("\n🎯 Ready to start the integrated system!")
    response = input("   Start now? (y/n): ")
    if response.lower() == 'y':
        start_system()
    else:
        print("\n📋 To start manually later:")
        print("   python start_integrated_system.py")
        print("\n📋 To test the system:")
        print("   python test_integrated_system.py")
    
    print("\n✅ Setup completed!")

if __name__ == "__main__":
    main()
