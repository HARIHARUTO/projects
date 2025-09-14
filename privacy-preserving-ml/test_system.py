"""
System Validation and Testing Script
Validates the complete privacy-preserving ML system
"""

import os
import sys
import traceback
import importlib.util
from pathlib import Path

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing imports...")
    
    required_modules = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('sklearn', 'scikit-learn'),
        ('torch', 'pytorch'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn')
    ]
    
    missing_modules = []
    
    for module_name, package_name in required_modules:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name} - Missing")
            missing_modules.append(package_name)
    
    if missing_modules:
        print(f"\n⚠ Missing modules: {', '.join(missing_modules)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All required modules available")
    return True

def test_project_structure():
    """Test project structure"""
    print("\n🏗️ Testing project structure...")
    
    required_files = [
        'main.py',
        'demo.py',
        'requirements.txt',
        'README.md',
        'data/__init__.py',
        'data/generate_dataset.py',
        'models/__init__.py',
        'models/baseline.py',
        'privacy/__init__.py',
        'privacy/differential_privacy.py',
        'privacy/dp_experiment.py',
        'federated/__init__.py',
        'federated/federated_learning.py',
        'evaluation/__init__.py',
        'evaluation/analyze_results.py',
        'utils/__init__.py',
        'utils/privacy_utils.py',
        'experiments/__init__.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ Project structure complete")
    return True

def test_data_generation():
    """Test synthetic data generation"""
    print("\n📊 Testing data generation...")
    
    try:
        from data.generate_dataset import HealthcareDataGenerator
        
        generator = HealthcareDataGenerator(seed=42)
        df = generator.generate_patient_data(n_patients=100)
        
        # Validate dataset
        assert len(df) == 100, "Incorrect number of samples"
        assert 'high_risk' in df.columns, "Missing target column"
        assert df['high_risk'].isin([0, 1]).all(), "Invalid target values"
        
        print("  ✓ Dataset generation successful")
        print(f"  ✓ Generated {len(df)} samples with {len(df.columns)} features")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data generation failed: {e}")
        return False

def test_baseline_model():
    """Test baseline model training"""
    print("\n🤖 Testing baseline model...")
    
    try:
        # First ensure we have data
        if not os.path.exists('data/train_data.csv'):
            from data.generate_dataset import HealthcareDataGenerator
            generator = HealthcareDataGenerator(seed=42)
            df = generator.generate_patient_data(n_patients=500)
            
            train_df = df.sample(frac=0.8, random_state=42)
            test_df = df.drop(train_df.index)
            
            os.makedirs('data', exist_ok=True)
            train_df.to_csv('data/train_data.csv', index=False)
            test_df.to_csv('data/test_data.csv', index=False)
        
        from models.baseline import BaselineModels
        
        baseline = BaselineModels()
        baseline.load_data()
        
        # Train one model for testing
        model = baseline.models['logistic_regression']
        model.fit(baseline.X_train_scaled, baseline.y_train)
        
        # Test prediction
        y_pred = model.predict(baseline.X_test_scaled[:10])  # Test on small subset
        
        assert len(y_pred) == 10, "Prediction length mismatch"
        assert all(pred in [0, 1] for pred in y_pred), "Invalid predictions"
        
        print("  ✓ Baseline model training successful")
        print("  ✓ Model predictions working correctly")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Baseline model test failed: {e}")
        traceback.print_exc()
        return False

def test_differential_privacy():
    """Test differential privacy implementation"""
    print("\n🔐 Testing differential privacy...")
    
    try:
        from privacy.differential_privacy import DPLogisticRegression, PrivacyBudgetTracker
        
        # Test privacy budget tracker
        tracker = PrivacyBudgetTracker(total_epsilon=1.0, total_delta=1e-5)
        tracker.spend_budget(0.5, 1e-6, "Test query")
        remaining = tracker.get_remaining_budget()
        
        assert remaining['epsilon'] == 0.5, "Budget tracking error"
        
        print("  ✓ Privacy budget tracking working")
        
        # Test DP model (if data available)
        if os.path.exists('data/train_data.csv'):
            import pandas as pd
            
            train_df = pd.read_csv('data/train_data.csv').head(100)  # Small subset for testing
            
            feature_cols = ['age', 'bmi', 'systolic_bp', 'cholesterol', 'glucose']
            train_df['gender'] = train_df['gender'].map({'M': 1, 'F': 0})
            
            X_train = train_df[feature_cols]
            y_train = train_df['high_risk']
            
            dp_model = DPLogisticRegression(epsilon=1.0, delta=1e-5)
            dp_model.fit(X_train, y_train)
            
            # Test prediction
            y_pred = dp_model.predict(X_train[:5])
            assert len(y_pred) == 5, "DP prediction length error"
            
            print("  ✓ DP model training and prediction successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Differential privacy test failed: {e}")
        traceback.print_exc()
        return False

def test_federated_learning():
    """Test federated learning components"""
    print("\n🌐 Testing federated learning...")
    
    try:
        from federated.federated_learning import FederatedClient, FederatedServer, FederatedModel
        import torch
        
        # Test model creation
        model = FederatedModel(input_dim=5)
        assert model is not None, "Model creation failed"
        
        print("  ✓ Federated model creation successful")
        
        # Test client creation
        client = FederatedClient("test_client", model)
        assert client.client_id == "test_client", "Client initialization failed"
        
        print("  ✓ Federated client creation successful")
        
        # Test server creation
        server = FederatedServer(model)
        server.register_client(client)
        assert "test_client" in server.clients, "Client registration failed"
        
        print("  ✓ Federated server and client registration successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Federated learning test failed: {e}")
        traceback.print_exc()
        return False

def test_utilities():
    """Test utility functions"""
    print("\n🛠️ Testing utilities...")
    
    try:
        from utils.privacy_utils import PrivacyMetrics, UtilityMetrics, validate_privacy_parameters
        
        # Test privacy parameter validation
        assert validate_privacy_parameters(1.0, 1e-5) == True, "Valid parameters rejected"
        
        try:
            validate_privacy_parameters(-1.0, 1e-5)
            assert False, "Invalid epsilon accepted"
        except ValueError:
            pass  # Expected
        
        print("  ✓ Privacy parameter validation working")
        
        # Test utility calculations
        baseline_metrics = {'accuracy': 0.85, 'roc_auc': 0.90}
        private_metrics = {'accuracy': 0.80, 'roc_auc': 0.85}
        
        utility_drop = UtilityMetrics.calculate_utility_drop(baseline_metrics, private_metrics)
        assert 'accuracy_drop' in utility_drop, "Utility drop calculation failed"
        
        print("  ✓ Utility metrics calculation working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Utilities test failed: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run comprehensive system test"""
    print("🧪 COMPREHENSIVE SYSTEM VALIDATION")
    print("=" * 50)
    
    tests = [
        ("Import Dependencies", test_imports),
        ("Project Structure", test_project_structure),
        ("Data Generation", test_data_generation),
        ("Baseline Models", test_baseline_model),
        ("Differential Privacy", test_differential_privacy),
        ("Federated Learning", test_federated_learning),
        ("Utility Functions", test_utilities)
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("VALIDATION SUMMARY")
    print(f"{'='*50}")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for use.")
        print("\nNext steps:")
        print("1. Run interactive demo: python demo.py")
        print("2. Run full experiments: python main.py")
        print("3. Explore the documentation in README.md")
    else:
        print(f"\n⚠ {total-passed} tests failed. Please address issues before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
