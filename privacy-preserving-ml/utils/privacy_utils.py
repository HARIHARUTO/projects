"""
Privacy Utilities
Common utility functions for privacy-preserving machine learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import json
import os

class PrivacyMetrics:
    """Calculate and track privacy metrics"""
    
    @staticmethod
    def calculate_epsilon_delta_spent(queries: List[Dict]) -> Tuple[float, float]:
        """Calculate total privacy budget spent using composition theorems"""
        total_epsilon = sum(q.get('epsilon', 0) for q in queries)
        total_delta = sum(q.get('delta', 0) for q in queries)
        return total_epsilon, total_delta
    
    @staticmethod
    def advanced_composition(epsilons: List[float], deltas: List[float], 
                           delta_prime: float = 1e-6) -> Tuple[float, float]:
        """Calculate privacy cost using advanced composition"""
        k = len(epsilons)
        if k == 0:
            return 0.0, 0.0
        
        # Simple composition for now (can be improved with advanced composition)
        total_epsilon = sum(epsilons)
        total_delta = sum(deltas) + delta_prime
        
        return total_epsilon, total_delta
    
    @staticmethod
    def privacy_loss_distribution(epsilon: float, delta: float, 
                                num_samples: int = 1000) -> np.ndarray:
        """Simulate privacy loss distribution"""
        # Simplified simulation - in practice would be more complex
        return np.random.normal(epsilon, epsilon/4, num_samples)

class UtilityMetrics:
    """Calculate utility metrics and comparisons"""
    
    @staticmethod
    def calculate_utility_drop(baseline_metrics: Dict, private_metrics: Dict) -> Dict:
        """Calculate utility drop from baseline to private model"""
        utility_drop = {}
        for metric in baseline_metrics:
            if metric in private_metrics:
                utility_drop[f'{metric}_drop'] = baseline_metrics[metric] - private_metrics[metric]
                utility_drop[f'{metric}_relative_drop'] = (
                    (baseline_metrics[metric] - private_metrics[metric]) / baseline_metrics[metric]
                )
        return utility_drop
    
    @staticmethod
    def privacy_utility_ratio(epsilon: float, utility: float) -> float:
        """Calculate privacy-utility ratio (higher is better)"""
        if epsilon == 0:
            return float('inf') if utility > 0 else 0
        return utility / epsilon

class DatasetAnalyzer:
    """Analyze datasets for privacy risks and characteristics"""
    
    @staticmethod
    def analyze_feature_sensitivity(df: pd.DataFrame, target_col: str) -> Dict:
        """Analyze feature sensitivity for privacy"""
        analysis = {}
        
        for col in df.columns:
            if col != target_col and pd.api.types.is_numeric_dtype(df[col]):
                # Calculate feature statistics
                analysis[col] = {
                    'range': df[col].max() - df[col].min(),
                    'std': df[col].std(),
                    'unique_values': df[col].nunique(),
                    'correlation_with_target': df[col].corr(df[target_col]) if target_col in df.columns else 0,
                    'sensitivity_score': df[col].std() / (df[col].max() - df[col].min() + 1e-8)
                }
        
        return analysis
    
    @staticmethod
    def identify_quasi_identifiers(df: pd.DataFrame, threshold: float = 0.8) -> List[str]:
        """Identify potential quasi-identifiers in the dataset"""
        quasi_identifiers = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # High uniqueness might indicate quasi-identifier
                uniqueness = df[col].nunique() / len(df)
                if uniqueness > threshold:
                    quasi_identifiers.append(col)
        
        return quasi_identifiers

class VisualizationUtils:
    """Utility functions for creating privacy-preserving ML visualizations"""
    
    @staticmethod
    def plot_privacy_utility_curve(epsilons: List[float], utilities: List[float], 
                                 method_name: str, save_path: str = None):
        """Plot privacy-utility curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(epsilons, utilities, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Privacy Budget (ε)', fontsize=12)
        plt.ylabel('Utility (ROC-AUC)', fontsize=12)
        plt.title(f'Privacy-Utility Tradeoff: {method_name}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def create_privacy_heatmap(results_dict: Dict, save_path: str = None):
        """Create heatmap of privacy-utility results"""
        # Convert results to DataFrame
        data = []
        for method, method_results in results_dict.items():
            for epsilon, result in method_results.items():
                if 'metrics' in result:
                    data.append({
                        'Method': method,
                        'Epsilon': epsilon,
                        'ROC_AUC': result['metrics']['roc_auc']
                    })
        
        df = pd.DataFrame(data)
        pivot_df = df.pivot(index='Method', columns='Epsilon', values='ROC_AUC')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='viridis')
        plt.title('Privacy-Utility Heatmap Across Methods and Privacy Budgets')
        plt.xlabel('Privacy Budget (ε)')
        plt.ylabel('Method')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class ExperimentLogger:
    """Log and track privacy-preserving ML experiments"""
    
    def __init__(self, log_dir: str = 'experiments/logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.current_experiment = None
    
    def start_experiment(self, experiment_name: str, config: Dict):
        """Start logging a new experiment"""
        self.current_experiment = {
            'name': experiment_name,
            'config': config,
            'start_time': pd.Timestamp.now().isoformat(),
            'events': [],
            'results': {}
        }
    
    def log_event(self, event_type: str, message: str, data: Dict = None):
        """Log an experiment event"""
        if self.current_experiment:
            event = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'type': event_type,
                'message': message,
                'data': data or {}
            }
            self.current_experiment['events'].append(event)
    
    def log_results(self, results: Dict):
        """Log experiment results"""
        if self.current_experiment:
            self.current_experiment['results'] = results
    
    def end_experiment(self):
        """End current experiment and save log"""
        if self.current_experiment:
            self.current_experiment['end_time'] = pd.Timestamp.now().isoformat()
            
            log_file = os.path.join(self.log_dir, f"{self.current_experiment['name']}.json")
            with open(log_file, 'w') as f:
                json.dump(self.current_experiment, f, indent=2)
            
            print(f"Experiment log saved to {log_file}")
            self.current_experiment = None

def validate_privacy_parameters(epsilon: float, delta: float = None) -> bool:
    """Validate privacy parameters"""
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive")
    
    if delta is not None:
        if delta < 0 or delta >= 1:
            raise ValueError("Delta must be in [0, 1)")
    
    return True

def estimate_privacy_cost(dataset_size: int, num_queries: int, 
                         target_epsilon: float) -> Dict:
    """Estimate privacy cost for a given experimental setup"""
    per_query_epsilon = target_epsilon / num_queries
    
    return {
        'total_epsilon': target_epsilon,
        'per_query_epsilon': per_query_epsilon,
        'num_queries': num_queries,
        'dataset_size': dataset_size,
        'privacy_per_sample': target_epsilon / dataset_size,
        'recommendation': 'Acceptable' if per_query_epsilon > 0.1 else 'Consider reducing queries'
    }

def create_privacy_report_template() -> Dict:
    """Create template for privacy analysis report"""
    return {
        'experiment_info': {
            'name': '',
            'date': '',
            'dataset': '',
            'privacy_method': ''
        },
        'privacy_guarantees': {
            'epsilon': 0.0,
            'delta': 0.0,
            'privacy_mechanism': '',
            'composition_method': ''
        },
        'utility_metrics': {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'roc_auc': 0.0
        },
        'privacy_utility_analysis': {
            'utility_drop': 0.0,
            'privacy_efficiency': 0.0,
            'recommendation': ''
        },
        'compliance': {
            'gdpr_compliant': False,
            'hipaa_compliant': False,
            'notes': ''
        }
    }
