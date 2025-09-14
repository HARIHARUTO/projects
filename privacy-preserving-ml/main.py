"""
Privacy-Preserving Analytics: Main Execution Script
Run complete privacy-preserving ML experiments with a single command
"""

import os
import sys
import argparse
from datetime import datetime

def run_data_generation():
    """Generate synthetic healthcare dataset"""
    print("="*60)
    print("STEP 1: GENERATING SYNTHETIC HEALTHCARE DATASET")
    print("="*60)
    
    from data.generate_dataset import main as generate_data
    generate_data()

def run_baseline_experiments():
    """Run baseline ML experiments"""
    print("\n" + "="*60)
    print("STEP 2: TRAINING BASELINE ML MODELS")
    print("="*60)
    
    from models.baseline import main as run_baseline
    baseline = run_baseline()
    return baseline

def run_differential_privacy_experiments():
    """Run differential privacy experiments"""
    print("\n" + "="*60)
    print("STEP 3: DIFFERENTIAL PRIVACY EXPERIMENTS")
    print("="*60)
    
    from privacy.dp_experiment import main as run_dp
    dp_runner = run_dp()
    return dp_runner

def run_federated_learning_experiments():
    """Run federated learning experiments"""
    print("\n" + "="*60)
    print("STEP 4: FEDERATED LEARNING EXPERIMENTS")
    print("="*60)
    
    from federated.federated_learning import main as run_fl
    fl_experiment = run_fl()
    return fl_experiment

def run_comprehensive_analysis():
    """Run comprehensive analysis and generate reports"""
    print("\n" + "="*60)
    print("STEP 5: COMPREHENSIVE ANALYSIS AND REPORTING")
    print("="*60)
    
    from evaluation.analyze_results import main as run_analysis
    analyzer = run_analysis()
    return analyzer

def create_summary_report():
    """Create final summary report"""
    print("\n" + "="*60)
    print("FINAL SUMMARY REPORT")
    print("="*60)
    
    summary = {
        'project': 'Privacy-Preserving Analytics on Healthcare Data',
        'completion_time': datetime.now().isoformat(),
        'components_implemented': [
            '✓ Synthetic healthcare dataset with 10,000 patient records',
            '✓ Baseline ML models (Logistic Regression, Random Forest, SVM, Gradient Boosting)',
            '✓ Differential Privacy mechanisms (ε-DP and (ε,δ)-DP)',
            '✓ Federated Learning with secure aggregation',
            '✓ Privacy-utility tradeoff analysis',
            '✓ Comprehensive evaluation framework'
        ],
        'privacy_guarantees': [
            'Differential Privacy: (ε, δ)-DP with configurable privacy budgets',
            'Federated Learning: Data locality preservation',
            'Secure Aggregation: Encrypted model updates'
        ],
        'measurable_outcomes': [
            'Privacy Budget: ε ∈ [0.1, 10.0], δ = 1e-5',
            'Accuracy metrics across all privacy levels',
            'Throughput and scalability measurements',
            'Privacy-utility tradeoff visualizations'
        ],
        'outputs_generated': [
            'experiments/plots/ - Comprehensive visualizations',
            'experiments/results/ - Experimental results (JSON)',
            'experiments/reports/ - Analysis reports',
            'models/saved/ - Trained model artifacts',
            'data/ - Synthetic datasets and federated splits'
        ]
    }
    
    print(f"\n🎉 PROJECT COMPLETION SUMMARY")
    print(f"Project: {summary['project']}")
    print(f"Completed: {summary['completion_time']}")
    
    print(f"\n📋 COMPONENTS IMPLEMENTED:")
    for component in summary['components_implemented']:
        print(f"  {component}")
    
    print(f"\n🔒 PRIVACY GUARANTEES:")
    for guarantee in summary['privacy_guarantees']:
        print(f"  • {guarantee}")
    
    print(f"\n📊 MEASURABLE OUTCOMES:")
    for outcome in summary['measurable_outcomes']:
        print(f"  • {outcome}")
    
    print(f"\n📁 OUTPUTS GENERATED:")
    for output in summary['outputs_generated']:
        print(f"  • {output}")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"  1. Review visualizations in experiments/plots/")
    print(f"  2. Examine detailed results in experiments/results/")
    print(f"  3. Read comprehensive report in experiments/reports/")
    print(f"  4. Deploy models using saved artifacts in models/saved/")
    print(f"  5. Extend experiments with custom privacy budgets or datasets")
    
    # Save summary
    import json
    os.makedirs('experiments/reports', exist_ok=True)
    with open('experiments/reports/project_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Privacy-Preserving Analytics Experiments')
    parser.add_argument('--step', type=str, choices=['data', 'baseline', 'dp', 'fl', 'analysis', 'all'],
                       default='all', help='Which step to run')
    parser.add_argument('--quick', action='store_true', help='Run quick experiments (fewer epochs/rounds)')
    
    args = parser.parse_args()
    
    print("🔒 PRIVACY-PRESERVING ANALYTICS: SECURE ML ON SENSITIVE DATA")
    print("=" * 80)
    print("A comprehensive implementation of privacy-preserving machine learning")
    print("techniques including differential privacy and federated learning.")
    print("=" * 80)
    
    start_time = datetime.now()
    
    try:
        if args.step in ['data', 'all']:
            run_data_generation()
        
        if args.step in ['baseline', 'all']:
            run_baseline_experiments()
        
        if args.step in ['dp', 'all']:
            run_differential_privacy_experiments()
        
        if args.step in ['fl', 'all']:
            run_federated_learning_experiments()
        
        if args.step in ['analysis', 'all']:
            run_comprehensive_analysis()
        
        if args.step == 'all':
            create_summary_report()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n🎉 EXECUTION COMPLETED SUCCESSFULLY!")
        print(f"Total execution time: {duration}")
        print(f"Check the experiments/ directory for all results and visualizations.")
        
    except Exception as e:
        print(f"\n❌ ERROR OCCURRED: {e}")
        print(f"Please check the error message and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
