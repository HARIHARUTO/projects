"""
Differential Privacy Experiments
Compare different DP mechanisms and privacy budgets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score
import json
import os
from datetime import datetime

from differential_privacy import (
    DPLogisticRegression, DPNeuralNetworkTrainer, DPRandomForest,
    PrivacyBudgetTracker, evaluate_dp_model
)

class DPExperimentRunner:
    """Run comprehensive DP experiments"""
    
    def __init__(self):
        self.results = {}
        self.privacy_budgets = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        self.delta = 1e-5
        
    def load_data(self):
        """Load training and test data"""
        self.train_df = pd.read_csv('data/train_data.csv')
        self.test_df = pd.read_csv('data/test_data.csv')
        
        # Prepare features
        feature_cols = ['age', 'gender', 'bmi', 'systolic_bp', 'diastolic_bp', 
                       'cholesterol', 'glucose', 'smoking', 'exercise_hours', 'family_history']
        
        # Convert gender to numeric
        self.train_df['gender'] = self.train_df['gender'].map({'M': 1, 'F': 0})
        self.test_df['gender'] = self.test_df['gender'].map({'M': 1, 'F': 0})
        
        self.X_train = self.train_df[feature_cols]
        self.y_train = self.train_df['high_risk']
        self.X_test = self.test_df[feature_cols]
        self.y_test = self.test_df['high_risk']
        
        print(f"Data loaded: {len(self.X_train)} training, {len(self.X_test)} test samples")
        
    def run_dp_logistic_regression_experiments(self):
        """Run DP Logistic Regression with different epsilon values"""
        print("\n" + "="*50)
        print("DIFFERENTIAL PRIVACY LOGISTIC REGRESSION EXPERIMENTS")
        print("="*50)
        
        lr_results = {}
        
        for epsilon in self.privacy_budgets:
            print(f"\nTraining DP Logistic Regression with ε = {epsilon}")
            
            # Initialize privacy budget tracker
            budget_tracker = PrivacyBudgetTracker(epsilon, self.delta)
            
            # Train DP model
            dp_lr = DPLogisticRegression(epsilon=epsilon, delta=self.delta)
            dp_lr.fit(self.X_train, self.y_train)
            
            # Spend privacy budget
            budget_tracker.spend_budget(epsilon, self.delta, f"DP Logistic Regression training")
            
            # Evaluate
            metrics = evaluate_dp_model(dp_lr, self.X_test, self.y_test, 
                                      f"DP Logistic Regression (ε={epsilon})")
            
            lr_results[epsilon] = {
                'metrics': metrics,
                'privacy_spent': {'epsilon': epsilon, 'delta': self.delta}
            }
        
        self.results['dp_logistic_regression'] = lr_results
        
    def run_dp_neural_network_experiments(self):
        """Run DP Neural Network with different epsilon values"""
        print("\n" + "="*50)
        print("DIFFERENTIAL PRIVACY NEURAL NETWORK EXPERIMENTS")
        print("="*50)
        
        nn_results = {}
        
        for epsilon in self.privacy_budgets:
            print(f"\nTraining DP Neural Network with ε = {epsilon}")
            
            try:
                # Train DP neural network
                dp_nn = DPNeuralNetworkTrainer(
                    input_dim=self.X_train.shape[1],
                    epsilon=epsilon,
                    delta=self.delta
                )
                dp_nn.fit(self.X_train, self.y_train, epochs=30, batch_size=64)
                
                # Evaluate
                metrics = evaluate_dp_model(dp_nn, self.X_test, self.y_test, 
                                          f"DP Neural Network (ε={epsilon})")
                
                nn_results[epsilon] = {
                    'metrics': metrics,
                    'privacy_spent': {'epsilon': dp_nn.final_epsilon, 'delta': self.delta}
                }
                
            except Exception as e:
                print(f"Error training DP Neural Network with ε={epsilon}: {e}")
                nn_results[epsilon] = {'error': str(e)}
        
        self.results['dp_neural_network'] = nn_results
        
    def run_dp_random_forest_experiments(self):
        """Run DP Random Forest with different epsilon values"""
        print("\n" + "="*50)
        print("DIFFERENTIAL PRIVACY RANDOM FOREST EXPERIMENTS")
        print("="*50)
        
        rf_results = {}
        
        for epsilon in self.privacy_budgets:
            print(f"\nTraining DP Random Forest with ε = {epsilon}")
            
            # Train DP Random Forest
            dp_rf = DPRandomForest(epsilon=epsilon, n_estimators=20, subsample_ratio=0.1)
            dp_rf.fit(self.X_train, self.y_train)
            
            # Evaluate
            metrics = evaluate_dp_model(dp_rf, self.X_test, self.y_test, 
                                      f"DP Random Forest (ε={epsilon})")
            
            rf_results[epsilon] = {
                'metrics': metrics,
                'privacy_spent': {'epsilon': epsilon, 'delta': 0}  # No delta for Laplace mechanism
            }
        
        self.results['dp_random_forest'] = rf_results
        
    def analyze_privacy_utility_tradeoffs(self):
        """Analyze privacy-utility tradeoffs"""
        print("\n" + "="*50)
        print("PRIVACY-UTILITY TRADEOFF ANALYSIS")
        print("="*50)
        
        # Create plots directory
        os.makedirs('experiments/plots', exist_ok=True)
        
        # Prepare data for plotting
        plot_data = []
        
        for model_type, model_results in self.results.items():
            for epsilon, result in model_results.items():
                if 'metrics' in result:
                    plot_data.append({
                        'model': model_type,
                        'epsilon': epsilon,
                        'accuracy': result['metrics']['accuracy'],
                        'roc_auc': result['metrics']['roc_auc'],
                        'f1_score': result['metrics']['f1_score']
                    })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create privacy-utility tradeoff plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Privacy-Utility Tradeoffs in Differential Privacy', fontsize=16)
        
        # Accuracy vs Epsilon
        for model in plot_df['model'].unique():
            model_data = plot_df[plot_df['model'] == model]
            axes[0, 0].plot(model_data['epsilon'], model_data['accuracy'], 
                           marker='o', label=model.replace('_', ' ').title())
        axes[0, 0].set_xlabel('Privacy Budget (ε)')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy vs Privacy Budget')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # ROC-AUC vs Epsilon
        for model in plot_df['model'].unique():
            model_data = plot_df[plot_df['model'] == model]
            axes[0, 1].plot(model_data['epsilon'], model_data['roc_auc'], 
                           marker='o', label=model.replace('_', ' ').title())
        axes[0, 1].set_xlabel('Privacy Budget (ε)')
        axes[0, 1].set_ylabel('ROC-AUC')
        axes[0, 1].set_title('ROC-AUC vs Privacy Budget')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1-Score vs Epsilon
        for model in plot_df['model'].unique():
            model_data = plot_df[plot_df['model'] == model]
            axes[1, 0].plot(model_data['epsilon'], model_data['f1_score'], 
                           marker='o', label=model.replace('_', ' ').title())
        axes[1, 0].set_xlabel('Privacy Budget (ε)')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_title('F1-Score vs Privacy Budget')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Heatmap of performance across models and epsilon values
        pivot_data = plot_df.pivot(index='model', columns='epsilon', values='roc_auc')
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=axes[1, 1])
        axes[1, 1].set_title('ROC-AUC Heatmap: Models vs Privacy Budget')
        axes[1, 1].set_xlabel('Privacy Budget (ε)')
        axes[1, 1].set_ylabel('Model')
        
        plt.tight_layout()
        plt.savefig('experiments/plots/privacy_utility_tradeoffs.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\nSummary Statistics:")
        summary_stats = plot_df.groupby('model').agg({
            'accuracy': ['mean', 'std', 'min', 'max'],
            'roc_auc': ['mean', 'std', 'min', 'max'],
            'f1_score': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        print(summary_stats)
        
        return plot_df
        
    def compare_with_baseline(self):
        """Compare DP results with non-private baseline"""
        print("\n" + "="*50)
        print("COMPARISON WITH NON-PRIVATE BASELINE")
        print("="*50)
        
        # Load baseline results
        try:
            with open('models/saved/baseline_results.json', 'r') as f:
                baseline_data = json.load(f)
                baseline_results = baseline_data['results']
        except FileNotFoundError:
            print("Baseline results not found. Please run baseline experiments first.")
            return
        
        # Get best baseline performance
        best_baseline_model = max(baseline_results.keys(), 
                                key=lambda x: baseline_results[x]['roc_auc'])
        best_baseline_auc = baseline_results[best_baseline_model]['roc_auc']
        
        print(f"Best baseline model: {best_baseline_model}")
        print(f"Best baseline ROC-AUC: {best_baseline_auc:.4f}")
        
        # Compare with DP results
        print("\nPrivacy-Utility Analysis:")
        print("Model | Epsilon | ROC-AUC | Accuracy Drop | Privacy Gain")
        print("-" * 65)
        
        for model_type, model_results in self.results.items():
            for epsilon, result in model_results.items():
                if 'metrics' in result:
                    dp_auc = result['metrics']['roc_auc']
                    auc_drop = best_baseline_auc - dp_auc
                    privacy_gain = f"ε = {epsilon}"
                    
                    print(f"{model_type[:15]:15} | {epsilon:7.1f} | {dp_auc:7.4f} | "
                          f"{auc_drop:12.4f} | {privacy_gain}")
        
    def save_results(self):
        """Save experimental results"""
        os.makedirs('experiments/results', exist_ok=True)
        
        results_with_metadata = {
            'timestamp': datetime.now().isoformat(),
            'experiment_config': {
                'privacy_budgets': self.privacy_budgets,
                'delta': self.delta,
                'dataset_size': {'train': len(self.X_train), 'test': len(self.X_test)}
            },
            'results': self.results
        }
        
        with open('experiments/results/dp_experiment_results.json', 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"\nResults saved to experiments/results/dp_experiment_results.json")

def main():
    """Run comprehensive DP experiments"""
    print("Starting Differential Privacy Experiments...")
    
    # Initialize experiment runner
    runner = DPExperimentRunner()
    
    # Load data
    runner.load_data()
    
    # Run experiments
    runner.run_dp_logistic_regression_experiments()
    runner.run_dp_neural_network_experiments()
    runner.run_dp_random_forest_experiments()
    
    # Analyze results
    runner.analyze_privacy_utility_tradeoffs()
    runner.compare_with_baseline()
    
    # Save results
    runner.save_results()
    
    print("\nDifferential Privacy experiments completed!")
    
    return runner

if __name__ == "__main__":
    runner = main()
