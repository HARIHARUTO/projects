# Privacy-Preserving Machine Learning Dashboard

A comprehensive framework for implementing privacy-preserving machine learning techniques with an interactive web dashboard. Features Differential Privacy, Federated Learning, and real-time privacy-utility analysis.

## 🎯 Project Overview

This project demonstrates how to build machine learning models while preserving privacy through:
- **Differential Privacy (DP)**: Adding calibrated noise to protect individual privacy
- **Federated Learning (FL)**: Training models without centralizing sensitive data
- **Interactive Dashboard**: Web-based visualization and experiment management
- **Privacy-Utility Analysis**: Real-time measurement of privacy-performance trade-offs

## ✨ Key Features

- 🔒 **Privacy-First Design**: (ε,δ)-differential privacy with configurable budgets
- 🌐 **Interactive Web Dashboard**: Modern React-style frontend with real-time charts
- 🤖 **Multiple ML Models**: Logistic Regression, Random Forest, Gradient Boosting, SVM
- 📊 **Real-time Visualization**: Privacy-utility tradeoffs, model comparisons, convergence plots
- 🏥 **Healthcare Focus**: Synthetic patient data with realistic medical features
- 🔄 **Federated Simulation**: Multi-client distributed learning environment
- 📈 **Comprehensive Analytics**: Detailed privacy cost analysis and recommendations

## 🚀 Quick Start Guide

### Step 1: System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space

### Step 2: Download and Setup

1. **Download the project**
```bash
git clone <repository-url>
cd privacy-preserving-ml
```

2. **Automated Setup (Recommended)**
```bash
python setup.py
```

This will:
- ✅ Check Python version compatibility
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Create project directories
- ✅ Test installation
- ✅ Generate activation scripts

### Step 3: Manual Setup (Alternative)

If automated setup fails, follow these manual steps:

1. **Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

2. **Install Dependencies**
```bash
# Install minimal requirements (recommended for most users)
pip install -r requirements-minimal.txt

# OR install full requirements (includes PyTorch for advanced features)
pip install -r requirements.txt
```

3. **Create Project Directories**
```bash
mkdir -p data/synthetic data/federated experiments/results experiments/plots models/saved
```

### Step 4: Generate Sample Data
```bash
python data/generate_dataset.py
```

### Step 5: Launch Dashboard
```bash
python backend/app.py
```

Open your browser to: **http://localhost:5000**

## 🖥️ Using the Dashboard

### Overview Tab
- **Statistics Cards**: Best model performance, privacy methods available
- **Recent Experiments**: Latest experiment results and status
- **Quick Navigation**: Jump to detailed analysis sections

### Privacy-Utility Tab
- **Interactive Charts**: Drag, zoom, and explore privacy-utility tradeoffs
- **Privacy Budget Controls**: Adjust ε and δ parameters in real-time
- **Utility Analysis**: See exact performance drops for different privacy levels
- **Recommendations**: AI-powered suggestions for optimal privacy settings

### Models Tab
- **Model Comparison**: Side-by-side performance analysis
- **Detailed Metrics**: Accuracy, ROC-AUC, F1-score, precision, recall
- **Model Selection**: Switch between Baseline, DP, and Federated Learning models
- **Performance Tables**: Sortable results with export functionality

### Experiments Tab
- **Run New Experiments**: Launch privacy experiments with custom parameters
- **Progress Tracking**: Real-time experiment status and progress bars
- **Results History**: Browse all completed experiments
- **Export Results**: Download experiment data in JSON format

### Settings Tab
- **Privacy Parameters**: Default ε and δ values
- **Visualization**: Chart themes and animation settings
- **Dataset Configuration**: Training split ratios and data paths
- **Advanced Options**: Budget tracking and refresh intervals

## 📁 Project Structure

```
privacy-preserving-ml/
├── 🌐 frontend/              # Web dashboard
│   ├── index.html           # Main dashboard page
│   ├── style.css            # Modern CSS styling
│   └── script.js            # Interactive JavaScript
├── 🔧 backend/               # Flask API server
│   └── app.py               # REST API endpoints
├── 📊 data/                  # Dataset management
│   ├── generate_dataset.py  # Synthetic data generator
│   ├── synthetic/           # Generated datasets
│   └── federated/           # Client data splits
├── 🤖 models/                # Machine learning models
│   ├── baseline.py          # Non-private models
│   └── saved/               # Model artifacts
├── 🔒 privacy/               # Privacy mechanisms
│   ├── differential_privacy.py  # DP algorithms
│   ├── dp_experiment.py     # DP experiments
│   └── simple_dp_experiment.py  # Lightweight DP
├── 🌍 federated/             # Federated learning
│   ├── federated_learning.py   # FL framework
│   └── simple_fl_experiment.py # Lightweight FL
├── 📈 evaluation/            # Analysis tools
│   └── analyze_results.py   # Results processing
├── 🧪 experiments/           # Experiment outputs
│   ├── results/             # JSON experiment data
│   ├── plots/               # Generated visualizations
│   └── reports/             # Analysis reports
├── 🔧 utils/                 # Utility functions
│   └── privacy_utils.py     # Privacy calculations
├── 📋 requirements.txt       # Full dependencies
├── 📋 requirements-minimal.txt # Core dependencies only
├── 🚀 setup.py              # Automated setup script
├── 🎮 demo.py               # Interactive CLI demo
├── ▶️ main.py               # Full experiment pipeline
└── 📖 README.md             # This file
```

## 🎯 Experiment Types

### 1. Baseline Models (No Privacy)
```bash
python models/baseline.py
```
- Trains 4 ML models without privacy protection
- Establishes performance benchmarks
- Results: ~98.86% ROC-AUC (Gradient Boosting)

### 2. Differential Privacy
```bash
python privacy/simple_dp_experiment.py
```
- Tests multiple privacy budgets (ε = 0.1 to 10.0)
- Measures privacy-utility tradeoffs
- Results: 78-96% ROC-AUC depending on ε

### 3. Federated Learning
```bash
python federated/simple_fl_experiment.py
```
- Simulates 5 healthcare institutions
- Preserves data locality
- Results: ~92% ROC-AUC with privacy

### 4. Full Pipeline
```bash
python main.py
```
- Runs all experiments sequentially
- Generates comprehensive analysis
- Creates publication-ready reports

## 📊 Expected Results

### Baseline Performance
| Model | Accuracy | ROC-AUC | F1-Score |
|-------|----------|---------|----------|
| Gradient Boosting | 95.05% | **98.86%** | 83.12% |
| Random Forest | 94.85% | 98.75% | 82.45% |
| Logistic Regression | 94.20% | 98.20% | 81.36% |
| SVM | 93.95% | 97.98% | 80.89% |

### Privacy-Utility Tradeoffs
| Privacy Budget (ε) | ROC-AUC | Utility Drop | Privacy Level |
|-------------------|---------|--------------|---------------|
| 0.1 | 78.45% | 20.41% | **Strong** |
| 1.0 | 89.23% | 9.63% | **Balanced** |
| 5.0 | 94.56% | 4.30% | Moderate |
| 10.0 | 95.67% | 3.19% | Weak |

### Federated Learning Results
- **Convergence**: 8-10 rounds
- **Final ROC-AUC**: 91.87%
- **Data Locality**: ✅ Preserved
- **Communication Efficiency**: ~85% reduction vs centralized

## 🔧 Advanced Configuration

### Custom Privacy Budgets
```python
# In privacy/dp_experiment.py
EPSILON_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
DELTA = 1e-5
```

### Federated Learning Settings
```python
# In federated/simple_fl_experiment.py
NUM_CLIENTS = 5
NUM_ROUNDS = 10
CLIENT_FRACTION = 1.0  # Fraction of clients per round
```

### Dashboard Configuration
```python
# In backend/app.py
DEBUG_MODE = True
HOST = '0.0.0.0'
PORT = 5000
```

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Solution: Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

**2. Port Already in Use**
```bash
# Solution: Kill process on port 5000
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

**3. Memory Issues**
```bash
# Solution: Use minimal requirements
pip install -r requirements-minimal.txt
```

**4. PyTorch Installation Issues**
```bash
# Solution: Skip PyTorch-dependent features
# The project works without PyTorch using scikit-learn only
```

### Getting Help

1. **Check Logs**: Look at console output for detailed error messages
2. **Test Installation**: Run `python test_system.py` to verify setup
3. **Minimal Setup**: Use `requirements-minimal.txt` for basic functionality
4. **Clean Install**: Delete `venv/` folder and run `python setup.py` again

## 🚀 Deployment Options

### Local Development
```bash
python backend/app.py
# Access at http://localhost:5000
```

### Production Deployment
```bash
# Using Gunicorn (Linux/Mac)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 backend.app:app

# Using Waitress (Windows)
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 backend.app:app
```

## 📄 License & Citation

**License**: MIT License - see LICENSE file

**Citation**:
```bibtex
@misc{privacy-preserving-ml-dashboard,
  title={Privacy-Preserving Machine Learning Dashboard},
  author={Privacy ML Team},
  year={2024},
  url={https://github.com/your-username/privacy-preserving-ml}
}
```

---

**🎉 Ready to explore privacy-preserving ML? Run `python setup.py` to get started!**
  - Baseline models (pickle format)
  - Feature scalers
  - Model performance metrics

## Advanced Usage

### Custom Privacy Budgets
```python
from privacy.differential_privacy import DPLogisticRegression

# Custom privacy parameters
model = DPLogisticRegression(epsilon=0.5, delta=1e-6)
model.fit(X_train, y_train)
```

### Custom Federated Setup
```python
from federated.federated_learning import FederatedLearningExperiment

experiment = FederatedLearningExperiment()
experiment.setup_federated_environment(input_dim=10, num_clients=10)
experiment.run_federated_experiment(
    num_rounds=20,
    local_epochs=5,
    use_dp=True,
    epsilon=1.0
)
```

### Privacy Budget Tracking
```python
from privacy.differential_privacy import PrivacyBudgetTracker

tracker = PrivacyBudgetTracker(total_epsilon=5.0, total_delta=1e-5)
tracker.spend_budget(1.0, 1e-6, "Model training")
tracker.print_budget_status()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{privacy_preserving_ml,
  title={Privacy-Preserving Analytics: Secure ML on Sensitive Data},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/privacy-preserving-ml}
}
```

## License

MIT License
