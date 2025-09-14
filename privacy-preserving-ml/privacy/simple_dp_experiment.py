"""
Simplified Differential Privacy Experiments
Runs DP experiments without PyTorch dependencies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import json
import os
from datetime import datetime

class SimpleDPLogisticRegression:
    """Simple DP Logistic Regression using output perturbation"""
    
    def __init__(self, epsilon=1.0, delta=1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        """Train DP logistic regression"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # Add Gaussian noise to coefficients for privacy
        sensitivity = 1.0 / len(X)  # Simplified sensitivity
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon
        
        # Add noise to coefficients
        noise = np.random.normal(0, sigma, size=self.model.coef_.shape)
        self.model.coef_ += noise
        
        # Add noise to intercept
        intercept_noise = np.random.normal(0, sigma)
        self.model.intercept_ += intercept_noise
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

def run_simple_dp_experiments():
    """Run simplified DP experiments"""
    print("Starting Differential Privacy Experiments...")
    
    # Load data
    train_df = pd.read_csv('data/train_data.csv')
    test_df = pd.read_csv('data/test_data.csv')
    
    # Prepare features
    feature_cols = ['age', 'gender', 'bmi', 'systolic_bp', 'diastolic_bp', 
                   'cholesterol', 'glucose', 'smoking', 'exercise_hours', 'family_history']
    
    train_df['gender'] = train_df['gender'].map({'M': 1, 'F': 0})
    test_df['gender'] = test_df['gender'].map({'M': 1, 'F': 0})
    
    X_train = train_df[feature_cols]
    y_train = train_df['high_risk']
    X_test = test_df[feature_cols]
    y_test = test_df['high_risk']
    
    # Test different privacy budgets
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = {}
    
    print(f"\nTesting DP Logistic Regression with different privacy budgets:")
    print(f"{'Epsilon':<8} {'Accuracy':<10} {'ROC-AUC':<10} {'F1-Score':<10}")
    print("-" * 45)
    
    for epsilon in epsilons:
        # Train DP model
        dp_model = SimpleDPLogisticRegression(epsilon=epsilon, delta=1e-5)
        dp_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = dp_model.predict(X_test)
        y_pred_proba = dp_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        results[epsilon] = metrics
        
        print(f"{epsilon:<8} {metrics['accuracy']:<10.4f} {metrics['roc_auc']:<10.4f} {metrics['f1_score']:<10.4f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot privacy-utility tradeoff
    plt.subplot(2, 2, 1)
    epsilons_list = list(results.keys())
    accuracies = [results[eps]['accuracy'] for eps in epsilons_list]
    plt.plot(epsilons_list, accuracies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Privacy Budget (ε)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Privacy Budget')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.subplot(2, 2, 2)
    aucs = [results[eps]['roc_auc'] for eps in epsilons_list]
    plt.plot(epsilons_list, aucs, 's-', linewidth=2, markersize=8, color='orange')
    plt.xlabel('Privacy Budget (ε)')
    plt.ylabel('ROC-AUC')
    plt.title('ROC-AUC vs Privacy Budget')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.subplot(2, 2, 3)
    f1s = [results[eps]['f1_score'] for eps in epsilons_list]
    plt.plot(epsilons_list, f1s, '^-', linewidth=2, markersize=8, color='green')
    plt.xlabel('Privacy Budget (ε)')
    plt.ylabel('F1-Score')
    plt.title('F1-Score vs Privacy Budget')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    # Privacy vs Utility scatter
    plt.subplot(2, 2, 4)
    privacy_levels = [1/eps for eps in epsilons_list]
    plt.scatter(aucs, privacy_levels, s=100, alpha=0.7, c=range(len(epsilons_list)), cmap='viridis')
    plt.xlabel('ROC-AUC (Utility)')
    plt.ylabel('Privacy Level (1/ε)')
    plt.title('Privacy vs Utility')
    plt.grid(True, alpha=0.3)
    
    for i, eps in enumerate(epsilons_list):
        plt.annotate(f'ε={eps}', (aucs[i], privacy_levels[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('experiments/plots', exist_ok=True)
    plt.savefig('experiments/plots/dp_privacy_utility_tradeoffs.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    os.makedirs('experiments/results', exist_ok=True)
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'experiment_type': 'differential_privacy',
        'results': results
    }
    
    with open('experiments/results/simple_dp_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n✅ Differential Privacy experiments completed!")
    print(f"📊 Results saved to experiments/results/simple_dp_results.json")
    print(f"📈 Plots saved to experiments/plots/dp_privacy_utility_tradeoffs.png")
    
    return results

if __name__ == "__main__":
    results = run_simple_dp_experiments()
