#!/usr/bin/env python3
"""
Quick run script for Privacy-Preserving ML Dashboard
Handles virtual environment activation and server startup
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_venv():
    """Check if virtual environment exists and is activated"""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("❌ Virtual environment not found!")
        print("🔧 Run: python setup.py")
        return False
    
    # Check if we're in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is active")
        return True
    else:
        print("⚠️  Virtual environment exists but not activated")
        return False

def activate_and_run():
    """Activate virtual environment and run the dashboard"""
    if platform.system() == "Windows":
        python_path = "venv\\Scripts\\python.exe"
        activate_cmd = "venv\\Scripts\\activate.bat"
    else:
        python_path = "venv/bin/python"
        activate_cmd = "source venv/bin/activate"
    
    # Check if Python exists in venv
    if not Path(python_path).exists():
        print("❌ Virtual environment Python not found!")
        print("🔧 Run: python setup.py")
        return False
    
    print("🚀 Starting Privacy-Preserving ML Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("🔧 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run the Flask app using venv Python
        subprocess.run([python_path, "backend/app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start dashboard: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        return True
    
    return True

def main():
    """Main function"""
    print("🎯 Privacy-Preserving ML Dashboard Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("backend/app.py").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Check virtual environment
    if not check_venv():
        print("\n💡 To set up the project:")
        print("1. Run: python setup.py")
        print("2. Then run this script again")
        sys.exit(1)
    
    # If venv exists but not activated, run with venv python
    if not (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)):
        activate_and_run()
    else:
        # Already in venv, just run directly
        print("🚀 Starting Privacy-Preserving ML Dashboard...")
        try:
            import backend.app
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("🔧 Try: pip install -r requirements-minimal.txt")
            sys.exit(1)

if __name__ == "__main__":
    main()
