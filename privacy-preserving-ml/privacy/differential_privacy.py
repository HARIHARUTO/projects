"""
Differential Privacy Implementation
Implements various DP mechanisms for privacy-preserving machine learning
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import warnings
warnings.filterwarnings('ignore')

class DPMechanisms:
    """Differential Privacy mechanisms"""
    
    @staticmethod
    def laplace_mechanism(value, sensitivity, epsilon):
        """Add Laplace noise for differential privacy"""
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    @staticmethod
    def gaussian_mechanism(value, sensitivity, epsilon, delta):
        """Add Gaussian noise for (ε, δ)-differential privacy"""
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        noise = np.random.normal(0, sigma)
        return value + noise
    
    @staticmethod
    def exponential_mechanism(scores, sensitivity, epsilon):
        """Exponential mechanism for selecting from discrete set"""
        probabilities = np.exp(epsilon * scores / (2 * sensitivity))
        probabilities = probabilities / np.sum(probabilities)
        return np.random.choice(len(scores), p=probabilities)

class DPLogisticRegression:
    """Differentially Private Logistic Regression using output perturbation"""
    
    def __init__(self, epsilon=1.0, delta=1e-5, regularization=0.1):
        self.epsilon = epsilon
        self.delta = delta
        self.regularization = regularization
        self.model = LogisticRegression(C=1/regularization, random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        """Train DP logistic regression"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train non-private model
        self.model.fit(X_scaled, y)
        
        # Add noise to coefficients
        sensitivity = 1.0 / (self.regularization * len(X))  # L2 sensitivity
        
        # Add Gaussian noise to coefficients
        noisy_coef = np.zeros_like(self.model.coef_)
        for i in range(self.model.coef_.shape[1]):
            noisy_coef[0, i] = DPMechanisms.gaussian_mechanism(
                self.model.coef_[0, i], sensitivity, self.epsilon, self.delta
            )
        
        # Add noise to intercept
        noisy_intercept = DPMechanisms.gaussian_mechanism(
            self.model.intercept_[0], sensitivity, self.epsilon, self.delta
        )
        
        # Update model parameters
        self.model.coef_ = noisy_coef
        self.model.intercept_ = np.array([noisy_intercept])
        
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

class DPNeuralNetwork(nn.Module):
    """Simple neural network for DP training with Opacus"""
    
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x).squeeze()

class DPNeuralNetworkTrainer:
    """Trainer for DP Neural Networks using Opacus"""
    
    def __init__(self, input_dim, epsilon=1.0, delta=1e-5, max_grad_norm=1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = DPNeuralNetwork(input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        self.scaler = StandardScaler()
        
        # Privacy engine
        self.privacy_engine = PrivacyEngine()
        
    def fit(self, X, y, epochs=50, batch_size=32):
        """Train DP neural network"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y.values).to(self.device)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Attach privacy engine
        self.model, self.optimizer, dataloader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=dataloader,
            epochs=epochs,
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            max_grad_norm=self.max_grad_norm,
        )
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {epoch_loss/len(dataloader):.4f}')
        
        # Get final privacy spent
        self.final_epsilon = self.privacy_engine.get_epsilon(self.delta)
        print(f'Final privacy spent: ε = {self.final_epsilon:.2f}, δ = {self.delta}')
        
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            probabilities = self.model(X_tensor).cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)
        
        return predictions
    
    def predict_proba(self, X):
        """Predict probabilities"""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            probabilities = self.model(X_tensor).cpu().numpy()
        
        # Return in sklearn format [prob_class_0, prob_class_1]
        return np.column_stack([1 - probabilities, probabilities])

class DPRandomForest:
    """Differentially Private Random Forest using subsample and aggregate"""
    
    def __init__(self, epsilon=1.0, n_estimators=10, subsample_ratio=0.1):
        self.epsilon = epsilon
        self.n_estimators = n_estimators
        self.subsample_ratio = subsample_ratio
        self.models = []
        
    def fit(self, X, y):
        """Train DP Random Forest"""
        n_samples = len(X)
        subsample_size = int(n_samples * self.subsample_ratio)
        
        # Train multiple models on disjoint subsamples
        for i in range(self.n_estimators):
            # Random subsample
            indices = np.random.choice(n_samples, subsample_size, replace=False)
            X_sub = X.iloc[indices]
            y_sub = y.iloc[indices]
            
            # Train model
            model = RandomForestClassifier(n_estimators=10, random_state=i)
            model.fit(X_sub, y_sub)
            self.models.append(model)
    
    def predict_proba(self, X):
        """Aggregate predictions with noise"""
        # Get predictions from all models
        all_predictions = []
        for model in self.models:
            pred_proba = model.predict_proba(X)[:, 1]  # Probability of class 1
            all_predictions.append(pred_proba)
        
        # Average predictions
        avg_predictions = np.mean(all_predictions, axis=0)
        
        # Add Laplace noise for privacy
        sensitivity = 1.0 / self.n_estimators  # Sensitivity of average
        noisy_predictions = np.array([
            DPMechanisms.laplace_mechanism(pred, sensitivity, self.epsilon)
            for pred in avg_predictions
        ])
        
        # Clip to [0, 1]
        noisy_predictions = np.clip(noisy_predictions, 0, 1)
        
        # Return in sklearn format
        return np.column_stack([1 - noisy_predictions, noisy_predictions])
    
    def predict(self, X):
        """Make binary predictions"""
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities > 0.5).astype(int)

class PrivacyBudgetTracker:
    """Track privacy budget consumption across multiple queries"""
    
    def __init__(self, total_epsilon, total_delta):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.spent_epsilon = 0
        self.spent_delta = 0
        self.queries = []
    
    def spend_budget(self, epsilon, delta, description=""):
        """Spend privacy budget"""
        if self.spent_epsilon + epsilon > self.total_epsilon:
            raise ValueError(f"Epsilon budget exceeded: {self.spent_epsilon + epsilon} > {self.total_epsilon}")
        
        if self.spent_delta + delta > self.total_delta:
            raise ValueError(f"Delta budget exceeded: {self.spent_delta + delta} > {self.total_delta}")
        
        self.spent_epsilon += epsilon
        self.spent_delta += delta
        
        self.queries.append({
            'epsilon': epsilon,
            'delta': delta,
            'description': description,
            'remaining_epsilon': self.total_epsilon - self.spent_epsilon,
            'remaining_delta': self.total_delta - self.spent_delta
        })
    
    def get_remaining_budget(self):
        """Get remaining privacy budget"""
        return {
            'epsilon': self.total_epsilon - self.spent_epsilon,
            'delta': self.total_delta - self.spent_delta
        }
    
    def print_budget_status(self):
        """Print current budget status"""
        print(f"\nPrivacy Budget Status:")
        print(f"Total ε: {self.total_epsilon}, Spent: {self.spent_epsilon:.4f}, Remaining: {self.total_epsilon - self.spent_epsilon:.4f}")
        print(f"Total δ: {self.total_delta}, Spent: {self.spent_delta:.6f}, Remaining: {self.total_delta - self.spent_delta:.6f}")
        
        if self.queries:
            print(f"\nQuery History:")
            for i, query in enumerate(self.queries):
                print(f"  {i+1}. ε={query['epsilon']:.2f}, δ={query['delta']:.6f} - {query['description']}")

def evaluate_dp_model(model, X_test, y_test, model_name="DP Model"):
    """Evaluate a differential privacy model"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n{model_name} Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return metrics
