"""
Federated Learning Framework
Implements federated learning with secure aggregation for privacy-preserving ML
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import copy
import os
from typing import List, Dict, Tuple
import json
from datetime import datetime

class FederatedModel(nn.Module):
    """Neural network model for federated learning"""
    
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x).squeeze()

class FederatedClient:
    """Federated learning client"""
    
    def __init__(self, client_id: str, model: nn.Module, learning_rate: float = 0.001):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def load_data(self, data_path: str):
        """Load client's local data"""
        self.data = pd.read_csv(data_path)
        
        # Prepare features
        feature_cols = ['age', 'gender', 'bmi', 'systolic_bp', 'diastolic_bp', 
                       'cholesterol', 'glucose', 'smoking', 'exercise_hours', 'family_history']
        
        # Convert gender to numeric
        self.data['gender'] = self.data['gender'].map({'M': 1, 'F': 0})
        
        self.X = self.data[feature_cols]
        self.y = self.data['high_risk']
        
        # Scale features
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print(f"Client {self.client_id}: Loaded {len(self.data)} samples")
        
    def train_local_model(self, epochs: int = 10, batch_size: int = 32) -> Dict:
        """Train model on local data"""
        self.model.train()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(self.X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(self.y.values).to(self.device)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        total_loss = 0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return {
            'client_id': self.client_id,
            'num_samples': len(self.data),
            'avg_loss': avg_loss,
            'epochs': epochs
        }
    
    def get_model_parameters(self) -> Dict:
        """Get current model parameters"""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_model_parameters(self, parameters: Dict):
        """Set model parameters"""
        for name, param in self.model.named_parameters():
            param.data.copy_(parameters[name])
    
    def evaluate_local_model(self) -> Dict:
        """Evaluate model on local data"""
        self.model.eval()
        
        X_tensor = torch.FloatTensor(self.X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(self.y.values).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = (outputs > 0.5).float()
            probabilities = outputs.cpu().numpy()
        
        y_true = self.y.values
        y_pred = predictions.cpu().numpy()
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, probabilities),
            'f1_score': f1_score(y_true, y_pred),
            'num_samples': len(y_true)
        }
        
        return metrics

class SecureAggregator:
    """Secure aggregation for federated learning"""
    
    def __init__(self, noise_scale: float = 0.1):
        self.noise_scale = noise_scale
        
    def aggregate_parameters(self, client_parameters: List[Dict], 
                           client_weights: List[float] = None) -> Dict:
        """Securely aggregate client parameters"""
        if client_weights is None:
            # Equal weighting
            client_weights = [1.0 / len(client_parameters)] * len(client_parameters)
        
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # Initialize aggregated parameters
        aggregated_params = {}
        
        # Get parameter names from first client
        param_names = list(client_parameters[0].keys())
        
        for param_name in param_names:
            # Weighted average of parameters
            weighted_sum = torch.zeros_like(client_parameters[0][param_name])
            
            for client_params, weight in zip(client_parameters, client_weights):
                weighted_sum += weight * client_params[param_name]
            
            # Add noise for privacy (simple Gaussian noise)
            if self.noise_scale > 0:
                noise = torch.normal(0, self.noise_scale, size=weighted_sum.shape)
                weighted_sum += noise
            
            aggregated_params[param_name] = weighted_sum
        
        return aggregated_params
    
    def add_differential_privacy(self, parameters: Dict, epsilon: float = 1.0, 
                               sensitivity: float = 1.0) -> Dict:
        """Add differential privacy noise to aggregated parameters"""
        dp_parameters = {}
        
        for param_name, param_tensor in parameters.items():
            # Calculate noise scale for differential privacy
            noise_scale = sensitivity / epsilon
            
            # Add Laplace noise (approximated with Gaussian for simplicity)
            noise = torch.normal(0, noise_scale, size=param_tensor.shape)
            dp_parameters[param_name] = param_tensor + noise
        
        return dp_parameters

class FederatedServer:
    """Federated learning server"""
    
    def __init__(self, model: nn.Module, aggregator: SecureAggregator = None):
        self.global_model = model
        self.aggregator = aggregator or SecureAggregator()
        self.clients = {}
        self.round_history = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model.to(self.device)
        
    def register_client(self, client: FederatedClient):
        """Register a client with the server"""
        self.clients[client.client_id] = client
        print(f"Registered client: {client.client_id}")
        
    def federated_round(self, selected_clients: List[str] = None, 
                       local_epochs: int = 5, use_dp: bool = False, 
                       epsilon: float = 1.0) -> Dict:
        """Execute one round of federated learning"""
        if selected_clients is None:
            selected_clients = list(self.clients.keys())
        
        print(f"\nFederated Round - Selected clients: {selected_clients}")
        
        # Send global model to selected clients
        global_params = {name: param.data.clone() 
                        for name, param in self.global_model.named_parameters()}
        
        client_updates = []
        client_weights = []
        training_results = []
        
        # Each client trains locally
        for client_id in selected_clients:
            client = self.clients[client_id]
            
            # Set global model parameters
            client.set_model_parameters(global_params)
            
            # Train locally
            training_result = client.train_local_model(epochs=local_epochs)
            training_results.append(training_result)
            
            # Get updated parameters
            updated_params = client.get_model_parameters()
            client_updates.append(updated_params)
            
            # Weight by number of samples
            client_weights.append(training_result['num_samples'])
        
        # Aggregate client updates
        aggregated_params = self.aggregator.aggregate_parameters(
            client_updates, client_weights
        )
        
        # Add differential privacy if requested
        if use_dp:
            aggregated_params = self.aggregator.add_differential_privacy(
                aggregated_params, epsilon=epsilon
            )
        
        # Update global model
        for name, param in self.global_model.named_parameters():
            param.data.copy_(aggregated_params[name])
        
        # Record round results
        round_result = {
            'selected_clients': selected_clients,
            'training_results': training_results,
            'total_samples': sum(client_weights),
            'avg_loss': np.mean([r['avg_loss'] for r in training_results]),
            'use_dp': use_dp,
            'epsilon': epsilon if use_dp else None
        }
        
        self.round_history.append(round_result)
        
        return round_result
    
    def evaluate_global_model(self, test_data_path: str) -> Dict:
        """Evaluate global model on test data"""
        # Load test data
        test_df = pd.read_csv(test_data_path)
        
        # Prepare features
        feature_cols = ['age', 'gender', 'bmi', 'systolic_bp', 'diastolic_bp', 
                       'cholesterol', 'glucose', 'smoking', 'exercise_hours', 'family_history']
        
        test_df['gender'] = test_df['gender'].map({'M': 1, 'F': 0})
        X_test = test_df[feature_cols]
        y_test = test_df['high_risk']
        
        # Scale features (using first client's scaler as approximation)
        first_client = list(self.clients.values())[0]
        X_test_scaled = first_client.scaler.transform(X_test)
        
        # Evaluate
        self.global_model.eval()
        X_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = self.global_model(X_tensor)
            predictions = (outputs > 0.5).float()
            probabilities = outputs.cpu().numpy()
        
        y_true = y_test.values
        y_pred = predictions.cpu().numpy()
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, probabilities),
            'f1_score': f1_score(y_true, y_pred),
            'num_samples': len(y_true)
        }
        
        return metrics
    
    def evaluate_client_models(self) -> Dict:
        """Evaluate all client models on their local data"""
        client_evaluations = {}
        
        for client_id, client in self.clients.items():
            metrics = client.evaluate_local_model()
            client_evaluations[client_id] = metrics
        
        return client_evaluations

class FederatedLearningExperiment:
    """Orchestrate federated learning experiments"""
    
    def __init__(self, data_dir: str = 'data/federated'):
        self.data_dir = data_dir
        self.results = {}
        
    def setup_federated_environment(self, input_dim: int, num_clients: int = 5):
        """Setup federated learning environment"""
        # Create global model
        global_model = FederatedModel(input_dim)
        
        # Create secure aggregator
        aggregator = SecureAggregator(noise_scale=0.01)
        
        # Create server
        self.server = FederatedServer(global_model, aggregator)
        
        # Create and register clients
        for i in range(num_clients):
            client_id = f'client_{i}'
            client = FederatedClient(client_id, global_model)
            
            # Load client data
            client_data_path = os.path.join(self.data_dir, f'{client_id}.csv')
            if os.path.exists(client_data_path):
                client.load_data(client_data_path)
                self.server.register_client(client)
            else:
                print(f"Warning: Data file not found for {client_id}")
        
        print(f"Federated environment setup complete with {len(self.server.clients)} clients")
        
    def run_federated_experiment(self, num_rounds: int = 10, local_epochs: int = 5,
                                client_fraction: float = 1.0, use_dp: bool = False,
                                epsilon: float = 1.0, experiment_name: str = "federated_baseline"):
        """Run federated learning experiment"""
        print(f"\nStarting federated learning experiment: {experiment_name}")
        print(f"Rounds: {num_rounds}, Local epochs: {local_epochs}, Client fraction: {client_fraction}")
        if use_dp:
            print(f"Using differential privacy with ε = {epsilon}")
        
        # Calculate number of clients to select each round
        total_clients = len(self.server.clients)
        clients_per_round = max(1, int(total_clients * client_fraction))
        
        round_metrics = []
        
        for round_num in range(num_rounds):
            print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
            
            # Select clients for this round
            selected_clients = np.random.choice(
                list(self.server.clients.keys()), 
                size=clients_per_round, 
                replace=False
            ).tolist()
            
            # Execute federated round
            round_result = self.server.federated_round(
                selected_clients=selected_clients,
                local_epochs=local_epochs,
                use_dp=use_dp,
                epsilon=epsilon
            )
            
            # Evaluate global model
            global_metrics = self.server.evaluate_global_model('data/test_data.csv')
            
            # Evaluate client models
            client_metrics = self.server.evaluate_client_models()
            
            # Record metrics
            round_metrics.append({
                'round': round_num + 1,
                'global_metrics': global_metrics,
                'client_metrics': client_metrics,
                'training_info': round_result
            })
            
            print(f"Global model - Accuracy: {global_metrics['accuracy']:.4f}, "
                  f"ROC-AUC: {global_metrics['roc_auc']:.4f}")
        
        # Store experiment results
        self.results[experiment_name] = {
            'config': {
                'num_rounds': num_rounds,
                'local_epochs': local_epochs,
                'client_fraction': client_fraction,
                'use_dp': use_dp,
                'epsilon': epsilon if use_dp else None,
                'num_clients': total_clients
            },
            'round_metrics': round_metrics,
            'final_global_metrics': round_metrics[-1]['global_metrics']
        }
        
        print(f"\nExperiment {experiment_name} completed!")
        print(f"Final global model performance:")
        final_metrics = round_metrics[-1]['global_metrics']
        for metric, value in final_metrics.items():
            if metric != 'num_samples':
                print(f"  {metric}: {value:.4f}")
        
        return self.results[experiment_name]
    
    def save_results(self, output_dir: str = 'experiments/results'):
        """Save experiment results"""
        os.makedirs(output_dir, exist_ok=True)
        
        results_with_metadata = {
            'timestamp': datetime.now().isoformat(),
            'experiments': self.results
        }
        
        output_path = os.path.join(output_dir, 'federated_learning_results.json')
        with open(output_path, 'w') as f:
            json.dump(results_with_metadata, f, indent=2, default=str)
        
        print(f"Results saved to {output_path}")

def main():
    """Run federated learning experiments"""
    print("Starting Federated Learning Experiments...")
    
    # Initialize experiment
    experiment = FederatedLearningExperiment()
    
    # Setup federated environment (10 features)
    experiment.setup_federated_environment(input_dim=10, num_clients=5)
    
    # Run baseline federated learning
    experiment.run_federated_experiment(
        num_rounds=15,
        local_epochs=5,
        client_fraction=1.0,
        use_dp=False,
        experiment_name="federated_baseline"
    )
    
    # Run federated learning with differential privacy
    experiment.run_federated_experiment(
        num_rounds=15,
        local_epochs=5,
        client_fraction=1.0,
        use_dp=True,
        epsilon=1.0,
        experiment_name="federated_dp_eps1"
    )
    
    # Run with partial client participation
    experiment.run_federated_experiment(
        num_rounds=20,
        local_epochs=5,
        client_fraction=0.6,
        use_dp=False,
        experiment_name="federated_partial_clients"
    )
    
    # Save results
    experiment.save_results()
    
    print("Federated Learning experiments completed!")
    
    return experiment

if __name__ == "__main__":
    experiment = main()
