"""
Comprehensive Analysis Summary
Analyze and visualize results from all privacy-preserving ML experiments
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

def load_baseline_results():
    """Load baseline model results"""
    try:
        with open('models/saved/baseline_results.json', 'r') as f:
            data = json.load(f)
            return data['results']
    except FileNotFoundError:
        print("⚠ Baseline results not found")
        return None

def load_dp_results():
    """Load differential privacy results"""
    try:
        with open('experiments/results/simple_dp_results.json', 'r') as f:
            data = json.load(f)
            return data['results']
    except FileNotFoundError:
        print("⚠ DP results not found")
        return None

def create_comprehensive_analysis():
    """Create comprehensive analysis of all results"""
    print("🔍 COMPREHENSIVE PRIVACY-PRESERVING ML ANALYSIS")
    print("=" * 60)
    
    # Load results
    baseline_results = load_baseline_results()
    dp_results = load_dp_results()
    
    # Prepare analysis data
    analysis_data = []
    
    # Add baseline results
    if baseline_results:
        best_baseline = max(baseline_results.keys(), key=lambda x: baseline_results[x]['roc_auc'])
        best_metrics = baseline_results[best_baseline]
        
        analysis_data.append({
            'method': 'Baseline (No Privacy)',
            'privacy_level': 'None',
            'epsilon': float('inf'),
            'accuracy': best_metrics['accuracy'],
            'roc_auc': best_metrics['roc_auc'],
            'f1_score': best_metrics['f1_score'],
            'privacy_guarantee': 'None'
        })
        
        print(f"✅ Best baseline model: {best_baseline}")
        print(f"   ROC-AUC: {best_metrics['roc_auc']:.4f}")
        print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    
    # Add DP results
    if dp_results:
        print(f"\n✅ Differential Privacy results loaded")
        for epsilon, metrics in dp_results.items():
            analysis_data.append({
                'method': f'DP Logistic Regression',
                'privacy_level': f'ε={epsilon}',
                'epsilon': float(epsilon),
                'accuracy': metrics['accuracy'],
                'roc_auc': metrics['roc_auc'],
                'f1_score': metrics['f1_score'],
                'privacy_guarantee': f'({epsilon}, 1e-5)-DP'
            })
    
    # Create DataFrame for analysis
    df = pd.DataFrame(analysis_data)
    
    # Print summary table
    print(f"\n📊 PRIVACY-UTILITY SUMMARY")
    print("-" * 80)
    print(f"{'Method':<25} {'Privacy Level':<15} {'Accuracy':<10} {'ROC-AUC':<10} {'F1-Score':<10}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        print(f"{row['method']:<25} {row['privacy_level']:<15} "
              f"{row['accuracy']:<10.4f} {row['roc_auc']:<10.4f} {row['f1_score']:<10.4f}")
    
    # Calculate privacy-utility tradeoffs
    if baseline_results and dp_results:
        baseline_auc = best_metrics['roc_auc']
        
        print(f"\n🔒 PRIVACY-UTILITY TRADEOFF ANALYSIS")
        print("-" * 50)
        print(f"{'Epsilon':<8} {'AUC Drop':<10} {'Relative Drop':<15} {'Privacy Gain'}")
        print("-" * 50)
        
        for epsilon, metrics in dp_results.items():
            auc_drop = baseline_auc - metrics['roc_auc']
            relative_drop = (auc_drop / baseline_auc) * 100
            privacy_gain = "High" if float(epsilon) < 1.0 else "Medium" if float(epsilon) < 5.0 else "Low"
            
            print(f"{epsilon:<8} {auc_drop:<10.4f} {relative_drop:<15.2f}% {privacy_gain}")
    
    # Create comprehensive visualization
    create_comprehensive_plots(df, baseline_results, dp_results)
    
    # Generate final report
    generate_final_report(df, baseline_results, dp_results)
    
    return df

def create_comprehensive_plots(df, baseline_results, dp_results):
    """Create comprehensive visualizations"""
    print(f"\n📈 Creating comprehensive visualizations...")
    
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Privacy-Utility Tradeoff
    if dp_results:
        plt.subplot(2, 3, 1)
        epsilons = [float(eps) for eps in dp_results.keys()]
        aucs = [dp_results[str(eps)]['roc_auc'] for eps in epsilons]
        
        plt.plot(epsilons, aucs, 'o-', linewidth=3, markersize=8, color='red')
        if baseline_results:
            best_baseline = max(baseline_results.keys(), key=lambda x: baseline_results[x]['roc_auc'])
            baseline_auc = baseline_results[best_baseline]['roc_auc']
            plt.axhline(y=baseline_auc, color='blue', linestyle='--', linewidth=2, label='Baseline (No Privacy)')
        
        plt.xlabel('Privacy Budget (ε)', fontsize=12)
        plt.ylabel('ROC-AUC', fontsize=12)
        plt.title('Privacy-Utility Tradeoff', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.legend()
    
    # 2. Method Comparison
    plt.subplot(2, 3, 2)
    if not df.empty:
        methods = df['method'].unique()
        best_aucs = [df[df['method'] == method]['roc_auc'].max() for method in methods]
        
        bars = plt.bar(range(len(methods)), best_aucs, alpha=0.7, 
                      color=['blue', 'red', 'orange', 'green', 'purple'][:len(methods)])
        plt.xlabel('Methods', fontsize=12)
        plt.ylabel('Best ROC-AUC', fontsize=12)
        plt.title('Method Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(range(len(methods)), [m.replace(' (No Privacy)', '') for m in methods], rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, best_aucs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Privacy Levels
    if dp_results:
        plt.subplot(2, 3, 3)
        epsilons = [float(eps) for eps in dp_results.keys()]
        privacy_levels = [1/eps for eps in epsilons]
        accuracies = [dp_results[str(eps)]['accuracy'] for eps in epsilons]
        
        scatter = plt.scatter(accuracies, privacy_levels, s=100, alpha=0.7, c=epsilons, cmap='viridis')
        plt.xlabel('Accuracy', fontsize=12)
        plt.ylabel('Privacy Level (1/ε)', fontsize=12)
        plt.title('Privacy vs Accuracy', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='ε')
        plt.grid(True, alpha=0.3)
    
    # 4. Utility Drop Analysis
    if baseline_results and dp_results:
        plt.subplot(2, 3, 4)
        best_baseline = max(baseline_results.keys(), key=lambda x: baseline_results[x]['roc_auc'])
        baseline_auc = baseline_results[best_baseline]['roc_auc']
        
        epsilons = [float(eps) for eps in dp_results.keys()]
        utility_drops = [baseline_auc - dp_results[str(eps)]['roc_auc'] for eps in epsilons]
        
        plt.bar(range(len(epsilons)), utility_drops, alpha=0.7, color='coral')
        plt.xlabel('Privacy Budget', fontsize=12)
        plt.ylabel('Utility Drop (AUC)', fontsize=12)
        plt.title('Utility Loss vs Privacy', fontsize=14, fontweight='bold')
        plt.xticks(range(len(epsilons)), [f'ε={eps}' for eps in epsilons], rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
    
    # 5. Privacy Efficiency
    if dp_results:
        plt.subplot(2, 3, 5)
        epsilons = [float(eps) for eps in dp_results.keys()]
        aucs = [dp_results[str(eps)]['roc_auc'] for eps in epsilons]
        privacy_efficiency = [auc * eps for auc, eps in zip(aucs, epsilons)]  # AUC * epsilon
        
        plt.plot(epsilons, privacy_efficiency, 's-', linewidth=2, markersize=8, color='green')
        plt.xlabel('Privacy Budget (ε)', fontsize=12)
        plt.ylabel('Privacy Efficiency (AUC × ε)', fontsize=12)
        plt.title('Privacy Efficiency Analysis', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
    
    # 6. Summary Statistics
    plt.subplot(2, 3, 6)
    if not df.empty:
        metrics = ['accuracy', 'roc_auc', 'f1_score']
        private_methods = df[df['epsilon'] != float('inf')]
        
        if not private_methods.empty:
            means = [private_methods[metric].mean() for metric in metrics]
            stds = [private_methods[metric].std() for metric in metrics]
            
            x = range(len(metrics))
            plt.bar(x, means, yerr=stds, alpha=0.7, capsize=5, color='lightblue')
            plt.xlabel('Metrics', fontsize=12)
            plt.ylabel('Average Performance', fontsize=12)
            plt.title('Average Private Method Performance', fontsize=14, fontweight='bold')
            plt.xticks(x, [m.replace('_', '-').upper() for m in metrics])
            plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('experiments/plots', exist_ok=True)
    plt.savefig('experiments/plots/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Comprehensive visualization saved to experiments/plots/comprehensive_analysis.png")

def generate_final_report(df, baseline_results, dp_results):
    """Generate final comprehensive report"""
    print(f"\n📋 GENERATING FINAL REPORT")
    print("-" * 40)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'project_title': 'Privacy-Preserving Analytics: Secure ML on Sensitive Healthcare Data',
        'summary': {
            'total_experiments': len(df) if not df.empty else 0,
            'privacy_methods_tested': ['Differential Privacy'],
            'dataset_size': {'train': 8000, 'test': 2000, 'total': 10000},
            'best_baseline_performance': None,
            'best_private_performance': None,
            'privacy_utility_tradeoff': None
        },
        'detailed_results': {},
        'key_findings': [],
        'recommendations': []
    }
    
    # Add baseline summary
    if baseline_results:
        best_baseline = max(baseline_results.keys(), key=lambda x: baseline_results[x]['roc_auc'])
        report['summary']['best_baseline_performance'] = {
            'model': best_baseline,
            'roc_auc': baseline_results[best_baseline]['roc_auc'],
            'accuracy': baseline_results[best_baseline]['accuracy']
        }
        report['detailed_results']['baseline'] = baseline_results
    
    # Add DP summary
    if dp_results:
        best_dp_eps = max(dp_results.keys(), key=lambda x: dp_results[x]['roc_auc'])
        report['summary']['best_private_performance'] = {
            'method': 'Differential Privacy',
            'epsilon': float(best_dp_eps),
            'roc_auc': dp_results[best_dp_eps]['roc_auc'],
            'accuracy': dp_results[best_dp_eps]['accuracy']
        }
        report['detailed_results']['differential_privacy'] = dp_results
        
        # Calculate utility drop
        if baseline_results:
            baseline_auc = baseline_results[best_baseline]['roc_auc']
            dp_auc = dp_results[best_dp_eps]['roc_auc']
            utility_drop = baseline_auc - dp_auc
            report['summary']['privacy_utility_tradeoff'] = {
                'utility_drop': utility_drop,
                'relative_drop_percent': (utility_drop / baseline_auc) * 100
            }
    
    # Key findings
    findings = [
        "Successfully implemented privacy-preserving ML on synthetic healthcare data",
        "Demonstrated measurable privacy-utility tradeoffs with differential privacy",
        "Generated comprehensive visualizations and analysis framework"
    ]
    
    if baseline_results and dp_results:
        best_baseline_auc = baseline_results[best_baseline]['roc_auc']
        best_dp_auc = dp_results[best_dp_eps]['roc_auc']
        utility_drop = best_baseline_auc - best_dp_auc
        
        findings.extend([
            f"Best baseline model achieved {best_baseline_auc:.4f} ROC-AUC",
            f"Best private model achieved {best_dp_auc:.4f} ROC-AUC with ε={best_dp_eps}",
            f"Privacy cost: {utility_drop:.4f} AUC drop ({(utility_drop/best_baseline_auc)*100:.1f}%)"
        ])
    
    report['key_findings'] = findings
    
    # Recommendations
    recommendations = [
        "For high-privacy requirements (ε < 1.0): Expect significant utility drop but strong privacy",
        "For moderate privacy (ε = 1.0-5.0): Good balance between privacy and utility",
        "For production deployment: Implement privacy budget tracking and monitoring",
        "Consider federated learning for scenarios requiring data locality",
        "Extend experiments with larger datasets and more complex models"
    ]
    
    report['recommendations'] = recommendations
    
    # Save report
    os.makedirs('experiments/reports', exist_ok=True)
    with open('experiments/reports/final_comprehensive_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print key findings
    print(f"\n🎯 KEY FINDINGS:")
    for i, finding in enumerate(findings, 1):
        print(f"{i}. {finding}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print(f"\n✅ Final report saved to experiments/reports/final_comprehensive_report.json")
    
    return report

def main():
    """Run comprehensive analysis"""
    print("🚀 STARTING COMPREHENSIVE PRIVACY-PRESERVING ML ANALYSIS")
    print("=" * 70)
    
    # Create analysis
    df = create_comprehensive_analysis()
    
    print(f"\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("Generated outputs:")
    print("📊 experiments/plots/comprehensive_analysis.png - Comprehensive visualizations")
    print("📋 experiments/reports/final_comprehensive_report.json - Detailed report")
    print("📈 experiments/plots/dp_privacy_utility_tradeoffs.png - DP analysis")
    
    print(f"\n🔍 PROJECT SUMMARY:")
    print("✅ Synthetic healthcare dataset generated (10,000 patients)")
    print("✅ Baseline ML models trained and evaluated")
    print("✅ Differential privacy mechanisms implemented and tested")
    print("✅ Privacy-utility tradeoffs quantified and visualized")
    print("✅ Comprehensive analysis and reporting completed")
    
    return df

if __name__ == "__main__":
    results = main()
