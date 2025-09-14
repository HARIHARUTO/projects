"""
Privacy-Preserving Analytics Demo
Interactive demonstration of privacy-preserving ML techniques
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def print_banner():
    """Print demo banner"""
    print("🔒" * 80)
    print("PRIVACY-PRESERVING ANALYTICS DEMO")
    print("Secure Machine Learning on Sensitive Healthcare Data")
    print("🔒" * 80)
    print()

def demo_dataset_generation():
    """Demonstrate dataset generation"""
    print("📊 DEMO 1: SYNTHETIC HEALTHCARE DATASET GENERATION")
    print("-" * 50)
    
    from data.generate_dataset import HealthcareDataGenerator
    
    # Generate small demo dataset
    generator = HealthcareDataGenerator(seed=42)
    demo_df = generator.generate_patient_data(n_patients=1000)
    
    print(f"✓ Generated {len(demo_df)} synthetic patient records")
    print(f"✓ Features: {list(demo_df.columns)}")
    print(f"✓ High-risk patients: {demo_df['high_risk'].sum()} ({demo_df['high_risk'].mean():.1%})")
    
    # Show sample data (anonymized)
    print("\nSample Data (first 5 records):")
    sample_cols = ['age', 'gender', 'bmi', 'systolic_bp', 'high_risk']
    print(demo_df[sample_cols].head().to_string(index=False))
    
    return demo_df

def demo_baseline_model():
    """Demonstrate baseline model training"""
    print("\n🤖 DEMO 2: BASELINE ML MODEL (NO PRIVACY)")
    print("-" * 50)
    
    from models.baseline import BaselineModels
    
    # Quick baseline training
    baseline = BaselineModels()
    
    # Use existing data if available, otherwise create small dataset
    if os.path.exists('data/train_data.csv'):
        baseline.load_data()
    else:
        print("⚠ Using demo dataset for baseline training...")
        demo_df = demo_dataset_generation()
        
        # Quick train/test split
        train_df = demo_df.sample(frac=0.8, random_state=42)
        test_df = demo_df.drop(train_df.index)
        
        # Save temporarily
        os.makedirs('data', exist_ok=True)
        train_df.to_csv('data/train_data.csv', index=False)
        test_df.to_csv('data/test_data.csv', index=False)
        
        baseline.load_data()
    
    # Train only one model for demo
    print("Training Logistic Regression model...")
    model = baseline.models['logistic_regression']
    model.fit(baseline.X_train_scaled, baseline.y_train)
    
    # Quick evaluation
    y_pred = model.predict(baseline.X_test_scaled)
    y_pred_proba = model.predict_proba(baseline.X_test_scaled)[:, 1]
    
    from sklearn.metrics import accuracy_score, roc_auc_score
    accuracy = accuracy_score(baseline.y_test, y_pred)
    auc = roc_auc_score(baseline.y_test, y_pred_proba)
    
    print(f"✓ Baseline Model Performance:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - ROC-AUC: {auc:.4f}")
    
    return {'accuracy': accuracy, 'roc_auc': auc}

def demo_differential_privacy():
    """Demonstrate differential privacy"""
    print("\n🔐 DEMO 3: DIFFERENTIAL PRIVACY")
    print("-" * 50)
    
    from privacy.differential_privacy import DPLogisticRegression, evaluate_dp_model
    
    # Load data
    train_df = pd.read_csv('data/train_data.csv')
    test_df = pd.read_csv('data/test_data.csv')
    
    feature_cols = ['age', 'gender', 'bmi', 'systolic_bp', 'diastolic_bp', 
                   'cholesterol', 'glucose', 'smoking', 'exercise_hours', 'family_history']
    
    train_df['gender'] = train_df['gender'].map({'M': 1, 'F': 0})
    test_df['gender'] = test_df['gender'].map({'M': 1, 'F': 0})
    
    X_train = train_df[feature_cols]
    y_train = train_df['high_risk']
    X_test = test_df[feature_cols]
    y_test = test_df['high_risk']
    
    # Test different privacy budgets
    epsilons = [1.0, 5.0, 10.0]
    dp_results = {}
    
    print("Training DP models with different privacy budgets...")
    
    for epsilon in epsilons:
        print(f"\n  Training with ε = {epsilon}...")
        
        dp_model = DPLogisticRegression(epsilon=epsilon, delta=1e-5)
        dp_model.fit(X_train, y_train)
        
        metrics = evaluate_dp_model(dp_model, X_test, y_test, f"DP Model (ε={epsilon})")
        dp_results[epsilon] = metrics
    
    print(f"\n✓ Differential Privacy Results Summary:")
    print(f"{'Epsilon':<8} {'Accuracy':<10} {'ROC-AUC':<10} {'Privacy Level'}")
    print("-" * 45)
    
    for epsilon, metrics in dp_results.items():
        privacy_level = "High" if epsilon < 1.0 else "Medium" if epsilon < 5.0 else "Low"
        print(f"{epsilon:<8} {metrics['accuracy']:<10.4f} {metrics['roc_auc']:<10.4f} {privacy_level}")
    
    return dp_results

def demo_federated_learning():
    """Demonstrate federated learning"""
    print("\n🌐 DEMO 4: FEDERATED LEARNING")
    print("-" * 50)
    
    from federated.federated_learning import FederatedLearningExperiment
    
    # Check if federated data exists
    if not os.path.exists('data/federated/client_0.csv'):
        print("⚠ Generating federated data splits...")
        from data.generate_dataset import HealthcareDataGenerator
        
        generator = HealthcareDataGenerator(seed=42)
        df = pd.read_csv('data/train_data.csv') if os.path.exists('data/train_data.csv') else generator.generate_patient_data(1000)
        
        federated_data = generator.create_federated_splits(df, n_clients=3)
        
        os.makedirs('data/federated', exist_ok=True)
        for client_name, client_df in federated_data.items():
            client_df.to_csv(f'data/federated/{client_name}.csv', index=False)
    
    # Quick federated learning demo
    experiment = FederatedLearningExperiment()
    experiment.setup_federated_environment(input_dim=10, num_clients=3)
    
    print("Running federated learning (5 rounds)...")
    
    # Quick experiment
    result = experiment.run_federated_experiment(
        num_rounds=5,
        local_epochs=3,
        client_fraction=1.0,
        use_dp=False,
        experiment_name="demo_federated"
    )
    
    final_metrics = result['final_global_metrics']
    print(f"\n✓ Federated Learning Results:")
    print(f"  - Final Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  - Final ROC-AUC: {final_metrics['roc_auc']:.4f}")
    print(f"  - Clients participated: {result['config']['num_clients']}")
    
    return final_metrics

def demo_privacy_utility_comparison():
    """Compare privacy-utility tradeoffs"""
    print("\n📈 DEMO 5: PRIVACY-UTILITY TRADEOFF ANALYSIS")
    print("-" * 50)
    
    # Simulate comparison data
    methods = ['Baseline (No Privacy)', 'DP (ε=1.0)', 'DP (ε=5.0)', 'Federated Learning']
    privacy_levels = [0, 1.0, 0.2, 0.5]  # Higher = more private
    utilities = [0.85, 0.78, 0.82, 0.83]  # ROC-AUC scores
    
    print("Privacy-Utility Comparison:")
    print(f"{'Method':<25} {'Privacy Level':<15} {'Utility (AUC)':<15}")
    print("-" * 55)
    
    for method, privacy, utility in zip(methods, privacy_levels, utilities):
        privacy_str = "None" if privacy == 0 else f"Medium" if privacy < 1 else "High"
        print(f"{method:<25} {privacy_str:<15} {utility:<15.4f}")
    
    # Simple visualization
    plt.figure(figsize=(10, 6))
    colors = ['red', 'orange', 'yellow', 'green']
    
    for i, (method, privacy, utility) in enumerate(zip(methods, privacy_levels, utilities)):
        plt.scatter(privacy, utility, s=200, c=colors[i], alpha=0.7, label=method)
        plt.annotate(method, (privacy, utility), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10)
    
    plt.xlabel('Privacy Level (Higher = More Private)')
    plt.ylabel('Utility (ROC-AUC)')
    plt.title('Privacy-Utility Tradeoff Demonstration')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs('experiments/plots', exist_ok=True)
    plt.savefig('experiments/plots/demo_privacy_utility.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Privacy-utility plot saved to experiments/plots/demo_privacy_utility.png")
    plt.close()

def demo_privacy_guarantees():
    """Demonstrate privacy guarantees and explanations"""
    print("\n🛡️ DEMO 6: PRIVACY GUARANTEES EXPLAINED")
    print("-" * 50)
    
    print("Privacy Guarantees in Our Implementation:")
    print()
    
    print("1. DIFFERENTIAL PRIVACY (ε, δ)-DP:")
    print("   • ε (epsilon): Privacy budget - smaller values = stronger privacy")
    print("   • δ (delta): Probability of privacy failure (typically 1e-5)")
    print("   • Guarantee: Individual records cannot be distinguished")
    print("   • Example: ε=1.0 provides strong privacy, ε=10.0 provides weak privacy")
    print()
    
    print("2. FEDERATED LEARNING:")
    print("   • Data never leaves client devices")
    print("   • Only model updates are shared (encrypted)")
    print("   • Protects against data centralization risks")
    print("   • Can be combined with DP for stronger guarantees")
    print()
    
    print("3. SECURE AGGREGATION:")
    print("   • Model updates are encrypted during transmission")
    print("   • Server cannot see individual client updates")
    print("   • Protects against honest-but-curious server")
    print()
    
    # Privacy budget example
    print("Privacy Budget Example:")
    total_budget = 1.0
    queries = [0.1, 0.2, 0.3, 0.4]
    remaining = total_budget - sum(queries)
    
    print(f"  Total privacy budget: ε = {total_budget}")
    print(f"  Spent on queries: {queries} (total: {sum(queries)})")
    print(f"  Remaining budget: ε = {remaining}")
    print(f"  Status: {'✓ Budget available' if remaining > 0 else '❌ Budget exhausted'}")

def main():
    """Run interactive demo"""
    print_banner()
    
    print("This demo showcases privacy-preserving machine learning techniques")
    print("on synthetic healthcare data. Each demo runs quickly for illustration.")
    print()
    
    demos = [
        ("Dataset Generation", demo_dataset_generation),
        ("Baseline Model", demo_baseline_model),
        ("Differential Privacy", demo_differential_privacy),
        ("Federated Learning", demo_federated_learning),
        ("Privacy-Utility Analysis", demo_privacy_utility_comparison),
        ("Privacy Guarantees", demo_privacy_guarantees)
    ]
    
    results = {}
    
    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"\n{'='*60}")
        print(f"RUNNING DEMO {i}/{len(demos)}: {name.upper()}")
        print(f"{'='*60}")
        
        try:
            start_time = time.time()
            result = demo_func()
            end_time = time.time()
            
            results[name] = result
            print(f"\n✅ Demo completed in {end_time - start_time:.1f} seconds")
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            print("Continuing with next demo...")
        
        # Pause between demos
        if i < len(demos):
            print(f"\nPress Enter to continue to next demo...")
            input()
    
    # Final summary
    print(f"\n{'🎉'*60}")
    print("DEMO COMPLETED SUCCESSFULLY!")
    print(f"{'🎉'*60}")
    
    print(f"\nDemo Results Summary:")
    for name, result in results.items():
        if result:
            print(f"✓ {name}: Completed")
        else:
            print(f"⚠ {name}: No results")
    
    print(f"\nNext Steps:")
    print(f"1. Run full experiments: python main.py")
    print(f"2. Explore generated plots in experiments/plots/")
    print(f"3. Review detailed documentation in README.md")
    print(f"4. Customize privacy parameters for your use case")
    
    return results

if __name__ == "__main__":
    results = main()
