#!/usr/bin/env python3
"""
Start System Script
Handles all import fixes and starts the Vigilant AI integrated system.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def fix_imports():
    """Fix all import issues."""
    print("🔧 Fixing Import Issues")
    print("-" * 30)
    
    # Run the comprehensive package import fix
    return run_command(f"{sys.executable} fix_package_imports.py", "Fixing package imports")

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing Dependencies")
    print("-" * 30)
    
    return run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing dependencies")

def start_integrated_system():
    """Start the integrated system."""
    print("\n🚀 Starting Integrated System")
    print("-" * 30)
    
    print("🎉 Starting all services...")
    print("   This will start:")
    print("   • Inference API (port 8002)")
    print("   • Vitals WebSocket Server (port 8765)")
    print("   • Integrated Live Stream Service (port 8001)")
    print("\n   Press Ctrl+C to stop all services")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, "start_integrated_system.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to start system: {e}")
        return False
    
    return True

def main():
    """Main function."""
    print("🚗 Vigilant AI - Start System")
    print("=" * 50)
    
    # Step 1: Install dependencies
    if not install_dependencies():
        print("⚠️  Some dependencies failed to install, continuing...")
    
    # Step 2: Fix imports
    if not fix_imports():
        print("❌ Failed to fix imports")
        return False
    
    # Step 3: Start system
    print("\n🎯 Ready to start the integrated system!")
    response = input("   Start now? (y/n): ")
    if response.lower() == 'y':
        start_integrated_system()
    else:
        print("\n📋 To start manually later:")
        print("   python start_integrated_system.py")
    
    print("\n✅ Setup completed!")

if __name__ == "__main__":
    main()
