#!/usr/bin/env python3
"""
Setup script for Privacy-Preserving ML Project
Automated installation and environment setup
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr}")
        return None

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_virtual_environment():
    """Create and activate virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("📁 Virtual environment already exists")
        return True
    
    # Create virtual environment
    if run_command("python -m venv venv", "Creating virtual environment"):
        print("✅ Virtual environment created successfully")
        return True
    return False

def get_activation_command():
    """Get the correct activation command for the platform"""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def install_requirements(minimal=False):
    """Install Python requirements"""
    req_file = "requirements-minimal.txt" if minimal else "requirements.txt"
    
    if platform.system() == "Windows":
        pip_command = "venv\\Scripts\\pip install -r " + req_file
    else:
        pip_command = "venv/bin/pip install -r " + req_file
    
    return run_command(pip_command, f"Installing requirements from {req_file}")

def create_directories():
    """Create necessary project directories"""
    directories = [
        "data/synthetic",
        "data/federated", 
        "experiments/results",
        "experiments/plots",
        "experiments/reports",
        "models/saved",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Project directories created")

def generate_sample_data():
    """Generate sample dataset for testing"""
    if platform.system() == "Windows":
        python_cmd = "venv\\Scripts\\python"
    else:
        python_cmd = "venv/bin/python"
    
    return run_command(f"{python_cmd} data/generate_dataset.py", "Generating sample dataset")

def test_installation():
    """Test if installation was successful"""
    if platform.system() == "Windows":
        python_cmd = "venv\\Scripts\\python"
    else:
        python_cmd = "venv/bin/python"
    
    test_script = """
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import flask
print("✅ All core dependencies imported successfully")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"Flask: {flask.__version__}")
"""
    
    with open("test_imports.py", "w") as f:
        f.write(test_script)
    
    result = run_command(f"{python_cmd} test_imports.py", "Testing installation")
    os.remove("test_imports.py")
    
    return result is not None

def main():
    """Main setup function"""
    print("🚀 Privacy-Preserving ML Project Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        print("❌ Failed to create virtual environment")
        sys.exit(1)
    
    # Install requirements
    print("\n📦 Installing Python packages...")
    
    # Try minimal requirements first
    if install_requirements(minimal=True):
        print("✅ Minimal requirements installed successfully")
    else:
        print("❌ Failed to install minimal requirements")
        print("Try running: pip install -r requirements-minimal.txt")
        sys.exit(1)
    
    # Create project directories
    create_directories()
    
    # Test installation
    if test_installation():
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print(f"1. Activate virtual environment: {get_activation_command()}")
        print("2. Generate sample data: python data/generate_dataset.py")
        print("3. Run baseline models: python models/baseline.py")
        print("4. Start web dashboard: python backend/app.py")
        print("5. Open browser to: http://localhost:5000")
        
        # Create activation script
        create_activation_script()
        
    else:
        print("❌ Installation test failed")
        sys.exit(1)

def create_activation_script():
    """Create platform-specific activation script"""
    if platform.system() == "Windows":
        script_content = """@echo off
echo Activating Privacy-Preserving ML Environment...
call venv\\Scripts\\activate.bat
echo Environment activated! 
echo.
echo Available commands:
echo   python data/generate_dataset.py    - Generate sample data
echo   python models/baseline.py          - Train baseline models  
echo   python backend/app.py              - Start web dashboard
echo   python demo.py                     - Run interactive demo
echo.
cmd /k
"""
        with open("activate.bat", "w") as f:
            f.write(script_content)
        print("✅ Created activate.bat for Windows")
        
    else:
        script_content = """#!/bin/bash
echo "Activating Privacy-Preserving ML Environment..."
source venv/bin/activate
echo "Environment activated!"
echo ""
echo "Available commands:"
echo "  python data/generate_dataset.py    - Generate sample data"
echo "  python models/baseline.py          - Train baseline models"
echo "  python backend/app.py              - Start web dashboard" 
echo "  python demo.py                     - Run interactive demo"
echo ""
exec "$SHELL"
"""
        with open("activate.sh", "w") as f:
            f.write(script_content)
        os.chmod("activate.sh", 0o755)
        print("✅ Created activate.sh for Unix/Linux")

if __name__ == "__main__":
    main()
