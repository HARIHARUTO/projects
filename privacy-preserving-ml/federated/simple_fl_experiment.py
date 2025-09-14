"""
Simplified Federated Learning Experiment
Runs FL experiments without PyTorch dependencies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import json
import os
from datetime import datetime
import copy

class SimpleFederatedClient:
    """Simple federated learning client using sklearn"""
    
    def __init__(self, client_id, data_path):
        self.client_id = client_id
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        self.load_data(data_path)
        
    def load_data(self, data_path):
        """Load client's local data"""
        self.data = pd.read_csv(data_path)
        
        feature_cols = ['age', 'gender', 'bmi', 'systolic_bp', 'diastolic_bp', 
                       'cholesterol', 'glucose', 'smoking', 'exercise_hours', 'family_history']
        
        self.data['gender'] = self.data['gender'].map({'M': 1, 'F': 0})
        
        self.X = self.data[feature_cols]
        self.y = self.data['high_risk']
        
        # Scale features
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print(f"Client {self.client_id}: Loaded {len(self.data)} samples, {self.y.mean():.1%} high-risk")
        
    def train_local_model(self, global_params=None):
        """Train model on local data"""
        if global_params is not None:
            # Initialize with global parameters
            self.model.coef_ = global_params['coef_'].copy()
            self.model.intercept_ = global_params['intercept_'].copy()
        
        # Train on local data
        self.model.fit(self.X_scaled, self.y)
        
        # Return model parameters
        return {
            'coef_': self.model.coef_.copy(),
            'intercept_': self.model.intercept_.copy(),
            'num_samples': len(self.data)
        }
    
    def evaluate_local_model(self):
        """Evaluate model on local data"""
        y_pred = self.model.predict(self.X_scaled)
        y_pred_proba = self.model.predict_proba(self.X_scaled)[:, 1]
        
        return {
            'accuracy': accuracy_score(self.y, y_pred),
            'roc_auc': roc_auc_score(self.y, y_pred_proba),
            'f1_score': f1_score(self.y, y_pred),
            'num_samples': len(self.y)
        }

class SimpleFederatedServer:
    """Simple federated learning server"""
    
    def __init__(self):
        self.global_model = LogisticRegression(random_state=42, max_iter=1000)
        self.clients = {}
        self.round_history = []
        
    def register_client(self, client):
        """Register a client"""
        self.clients[client.client_id] = client
        
    def federated_round(self, selected_clients=None):
        """Execute one round of federated learning"""
        if selected_clients is None:
            selected_clients = list(self.clients.keys())
        
        print(f"Round with clients: {selected_clients}")
        
        # Get current global parameters
        if hasattr(self.global_model, 'coef_'):
            global_params = {
                'coef_': self.global_model.coef_.copy(),
                'intercept_': self.global_model.intercept_.copy()
            }
        else:
            global_params = None
        
        # Collect client updates
        client_updates = []
        client_weights = []
        
        for client_id in selected_clients:
            client = self.clients[client_id]
            update = client.train_local_model(global_params)
            client_updates.append(update)
            client_weights.append(update['num_samples'])
        
        # Aggregate parameters (weighted average)
        total_samples = sum(client_weights)
        weights = [w / total_samples for w in client_weights]
        
        # Weighted average of coefficients
        aggregated_coef = np.zeros_like(client_updates[0]['coef_'])
        aggregated_intercept = np.zeros_like(client_updates[0]['intercept_'])
        
        for update, weight in zip(client_updates, weights):
            aggregated_coef += weight * update['coef_']
            aggregated_intercept += weight * update['intercept_']
        
        # Update global model
        self.global_model.coef_ = aggregated_coef
        self.global_model.intercept_ = aggregated_intercept
        
        return {
            'selected_clients': selected_clients,
            'total_samples': total_samples
        }
    
    def evaluate_global_model(self, test_data_path):
        """Evaluate global model on test data"""
        test_df = pd.read_csv(test_data_path)
        
        feature_cols = ['age', 'gender', 'bmi', 'systolic_bp', 'diastolic_bp', 
                       'cholesterol', 'glucose', 'smoking', 'exercise_hours', 'family_history']
        
        test_df['gender'] = test_df['gender'].map({'M': 1, 'F': 0})
        X_test = test_df[feature_cols]
        y_test = test_df['high_risk']
        
        # Use first client's scaler (approximation)
        first_client = list(self.clients.values())[0]
        X_test_scaled = first_client.scaler.transform(X_test)
        
        y_pred = self.global_model.predict(X_test_scaled)
        y_pred_proba = self.global_model.predict_proba(X_test_scaled)[:, 1]
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'f1_score': f1_score(y_test, y_pred),
            'num_samples': len(y_test)
        }

def run_simple_fl_experiment():
    """Run simplified federated learning experiment"""
    print("Starting Federated Learning Experiment...")
    
    # Initialize server
    server = SimpleFederatedServer()
    
    # Register clients
    for i in range(5):
        client_data_path = f'data/federated/client_{i}.csv'
        if os.path.exists(client_data_path):
            client = SimpleFederatedClient(f'client_{i}', client_data_path)
            server.register_client(client)
    
    print(f"\nRegistered {len(server.clients)} clients")
    
    # Run federated learning rounds
    num_rounds = 10
    round_metrics = []
    
    print(f"\nRunning {num_rounds} federated learning rounds...")
    print(f"{'Round':<6} {'Accuracy':<10} {'ROC-AUC':<10} {'F1-Score':<10}")
    print("-" * 45)
    
    for round_num in range(num_rounds):
        # Execute federated round
        round_result = server.federated_round()
        
        # Evaluate global model
        global_metrics = server.evaluate_global_model('data/test_data.csv')
        
        round_metrics.append({
            'round': round_num + 1,
            'global_metrics': global_metrics
        })
        
        print(f"{round_num + 1:<6} {global_metrics['accuracy']:<10.4f} "
              f"{global_metrics['roc_auc']:<10.4f} {global_metrics['f1_score']:<10.4f}")
    
    # Evaluate client models
    print(f"\nFinal client model performance:")
    print(f"{'Client':<10} {'Accuracy':<10} {'ROC-AUC':<10} {'F1-Score':<10}")
    print("-" * 45)
    
    client_metrics = {}
    for client_id, client in server.clients.items():
        metrics = client.evaluate_local_model()
        client_metrics[client_id] = metrics
        print(f"{client_id:<10} {metrics['accuracy']:<10.4f} "
              f"{metrics['roc_auc']:<10.4f} {metrics['f1_score']:<10.4f}")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Global model convergence
    rounds = [m['round'] for m in round_metrics]
    accuracies = [m['global_metrics']['accuracy'] for m in round_metrics]
    aucs = [m['global_metrics']['roc_auc'] for m in round_metrics]
    f1s = [m['global_metrics']['f1_score'] for m in round_metrics]
    
    plt.subplot(2, 2, 1)
    plt.plot(rounds, accuracies, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Federated Round')
    plt.ylabel('Global Model Accuracy')
    plt.title('Global Model Accuracy Convergence')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(rounds, aucs, 's-', linewidth=2, markersize=6, color='orange')
    plt.xlabel('Federated Round')
    plt.ylabel('Global Model ROC-AUC')
    plt.title('Global Model ROC-AUC Convergence')
    plt.grid(True, alpha=0.3)
    
    # Client performance comparison
    plt.subplot(2, 2, 3)
    client_ids = list(client_metrics.keys())
    client_aucs = [client_metrics[cid]['roc_auc'] for cid in client_ids]
    plt.bar(client_ids, client_aucs, alpha=0.7)
    plt.xlabel('Client ID')
    plt.ylabel('Local Model ROC-AUC')
    plt.title('Client Model Performance')
    plt.xticks(rotation=45)
    
    # Final comparison
    plt.subplot(2, 2, 4)
    final_global_auc = aucs[-1]
    avg_client_auc = np.mean(client_aucs)
    
    plt.bar(['Global Model', 'Avg Client Model'], [final_global_auc, avg_client_auc], 
            color=['blue', 'green'], alpha=0.7)
    plt.ylabel('ROC-AUC')
    plt.title('Global vs Average Client Performance')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('experiments/plots', exist_ok=True)
    plt.savefig('experiments/plots/fl_convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    os.makedirs('experiments/results', exist_ok=True)
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'experiment_type': 'federated_learning',
        'num_rounds': num_rounds,
        'num_clients': len(server.clients),
        'round_metrics': round_metrics,
        'client_metrics': client_metrics,
        'final_global_metrics': round_metrics[-1]['global_metrics']
    }
    
    with open('experiments/results/simple_fl_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n✅ Federated Learning experiment completed!")
    print(f"📊 Results saved to experiments/results/simple_fl_results.json")
    print(f"📈 Plots saved to experiments/plots/fl_convergence_analysis.png")
    
    return results_data

if __name__ == "__main__":
    results = run_simple_fl_experiment()
