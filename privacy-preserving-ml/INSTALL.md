# 🚀 Installation Guide - Privacy-Preserving ML Dashboard

## Quick Start (Recommended)

### Option 1: Automated Setup
```bash
# 1. Download/clone the project
cd privacy-preserving-ml

# 2. Run automated setup
python setup.py

# 3. Launch dashboard
python run.py
```

### Option 2: Manual Setup
```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements-minimal.txt

# 4. Launch dashboard
python backend/app.py
```

## Step-by-Step Guide

### Step 1: Prerequisites
- **Python 3.8+** (Check with `python --version`)
- **pip** package manager
- **4GB RAM** minimum
- **2GB storage** space

### Step 2: Download Project
```bash
# If using Git:
git clone <repository-url>
cd privacy-preserving-ml

# If downloaded as ZIP:
# Extract and navigate to folder
cd privacy-preserving-ml
```

### Step 3: Environment Setup

#### Windows Users:
```cmd
# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate

# Install packages
pip install -r requirements-minimal.txt

# Launch dashboard
python backend\app.py
```

#### macOS/Linux Users:
```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Install packages
pip install -r requirements-minimal.txt

# Launch dashboard
python backend/app.py
```

### Step 4: Verify Installation
Open browser to: **http://localhost:5000**

You should see the Privacy-Preserving ML Dashboard!

## Package Requirements

### Minimal Installation (Recommended)
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
flask==2.3.2
flask-cors==4.0.0
tqdm==4.65.0
faker==19.3.0
joblib==1.3.1
```

### Full Installation (Advanced Features)
Includes PyTorch for advanced differential privacy:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### Common Issues

**❌ "Python not found"**
```bash
# Solution: Install Python 3.8+ from python.org
# Verify: python --version
```

**❌ "pip not found"**
```bash
# Windows: Use py -m pip instead of pip
# macOS: Use python3 -m pip instead of pip
```

**❌ "Permission denied"**
```bash
# Solution: Use --user flag
pip install --user -r requirements-minimal.txt
```

**❌ "Port 5000 already in use"**
```bash
# Windows: 
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# macOS/Linux:
lsof -ti:5000 | xargs kill -9
```

**❌ "Import errors"**
```bash
# Solution: Ensure virtual environment is activated
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
```

**❌ "Memory errors"**
```bash
# Solution: Use minimal requirements
pip install -r requirements-minimal.txt
```

### Getting Help

1. **Check Python version**: `python --version` (need 3.8+)
2. **Verify pip**: `pip --version`
3. **Test imports**: `python -c "import numpy, pandas, sklearn"`
4. **Check ports**: `netstat -an | grep 5000`

## Alternative Launch Methods

### Method 1: Direct Python
```bash
cd privacy-preserving-ml
python backend/app.py
```

### Method 2: Using Run Script
```bash
cd privacy-preserving-ml
python run.py
```

### Method 3: Batch/Shell Script
```bash
# Windows: Double-click activate.bat
# Linux/Mac: ./activate.sh
```

## Project Structure After Setup
```
privacy-preserving-ml/
├── venv/                    # Virtual environment
├── backend/app.py          # Flask server
├── frontend/               # Web dashboard files
├── data/                   # Dataset storage
├── models/                 # ML models
├── experiments/            # Results storage
├── requirements-minimal.txt # Core dependencies
├── setup.py               # Automated installer
├── run.py                 # Quick launcher
└── README.md              # Full documentation
```

## Next Steps After Installation

1. **Generate Data**: `python data/generate_dataset.py`
2. **Train Models**: `python models/baseline.py`
3. **Run Experiments**: `python privacy/simple_dp_experiment.py`
4. **View Dashboard**: Open http://localhost:5000

## Advanced Setup

### Docker Installation (Optional)
```dockerfile
# Create Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements-minimal.txt .
RUN pip install -r requirements-minimal.txt
COPY . .
EXPOSE 5000
CMD ["python", "backend/app.py"]

# Build and run
docker build -t privacy-ml .
docker run -p 5000:5000 privacy-ml
```

### Production Deployment
```bash
# Install production server
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 backend.app:app
```

---

**🎉 Installation complete! Your Privacy-Preserving ML Dashboard is ready!**
