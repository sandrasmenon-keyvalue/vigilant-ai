#!/usr/bin/env python3
"""
Fix and Start Script
One-command fix for all import issues and start the system.
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

def main():
    """Main function."""
    print("🚗 Vigilant AI - Fix and Start")
    print("=" * 50)
    
    # Step 1: Install dependencies
    print("📦 Installing Dependencies...")
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing dependencies"):
        print("⚠️  Some dependencies failed to install, continuing...")
    
    # Step 2: Fix package imports
    print("\n🔧 Fixing Package Imports...")
    if not run_command(f"{sys.executable} fix_package_imports.py", "Fixing package imports"):
        print("❌ Failed to fix package imports")
        return False
    
    # Step 3: Start system
    print("\n🚀 Starting Integrated System...")
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

if __name__ == "__main__":
    main()
