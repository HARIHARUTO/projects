"""
Baseline ML Models for Health Risk Prediction
Non-private baseline models for comparison with privacy-preserving versions
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import json
from datetime import datetime

class BaselineModels:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42)
        }
        self.scaler = StandardScaler()
        self.trained_models = {}
        self.results = {}
        
    def load_data(self, train_path='data/train_data.csv', test_path='data/test_data.csv'):
        """Load training and test data"""
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        # Prepare features (exclude patient_id and target variables)
        feature_cols = ['age', 'gender', 'bmi', 'systolic_bp', 'diastolic_bp', 
                       'cholesterol', 'glucose', 'smoking', 'exercise_hours', 'family_history']
        
        # Convert gender to numeric
        self.train_df['gender'] = self.train_df['gender'].map({'M': 1, 'F': 0})
        self.test_df['gender'] = self.test_df['gender'].map({'M': 1, 'F': 0})
        
        self.X_train = self.train_df[feature_cols]
        self.y_train = self.train_df['high_risk']
        self.X_test = self.test_df[feature_cols]
        self.y_test = self.test_df['high_risk']
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training data shape: {self.X_train.shape}")
        print(f"Test data shape: {self.X_test.shape}")
        print(f"Class distribution - High risk: {self.y_train.mean():.2%}")
        
    def train_models(self):
        """Train all baseline models"""
        print("Training baseline models...")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled features for SVM and Logistic Regression
            if name in ['svm', 'logistic_regression']:
                X_train = self.X_train_scaled
                X_test = self.X_test_scaled
            else:
                X_train = self.X_train
                X_test = self.X_test
            
            # Train model
            model.fit(X_train, self.y_train)
            self.trained_models[name] = model
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1_score': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
            }
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, self.y_train, cv=5, scoring='roc_auc')
            metrics['cv_auc_mean'] = cv_scores.mean()
            metrics['cv_auc_std'] = cv_scores.std()
            
            self.results[name] = metrics
            
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            
    def save_models(self, model_dir='models/saved'):
        """Save trained models and results"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for name, model in self.trained_models.items():
            joblib.dump(model, f'{model_dir}/{name}_baseline.pkl')
        
        # Save scaler
        joblib.dump(self.scaler, f'{model_dir}/scaler.pkl')
        
        # Save results
        results_with_timestamp = {
            'timestamp': datetime.now().isoformat(),
            'results': self.results
        }
        
        with open(f'{model_dir}/baseline_results.json', 'w') as f:
            json.dump(results_with_timestamp, f, indent=2)
            
        print(f"\nModels and results saved to {model_dir}/")
        
    def get_best_model(self):
        """Get the best performing model based on ROC-AUC"""
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['roc_auc'])
        best_model = self.trained_models[best_model_name]
        best_score = self.results[best_model_name]['roc_auc']
        
        print(f"\nBest model: {best_model_name} (ROC-AUC: {best_score:.4f})")
        return best_model_name, best_model, best_score
    
    def print_results_summary(self):
        """Print comprehensive results summary"""
        print("\n" + "="*60)
        print("BASELINE MODELS PERFORMANCE SUMMARY")
        print("="*60)
        
        # Create results DataFrame for easy viewing
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        
        print("\nDetailed Results:")
        print(results_df.to_string())
        
        # Ranking by different metrics
        print(f"\nModel Rankings:")
        print(f"By ROC-AUC: {sorted(self.results.keys(), key=lambda x: self.results[x]['roc_auc'], reverse=True)}")
        print(f"By Accuracy: {sorted(self.results.keys(), key=lambda x: self.results[x]['accuracy'], reverse=True)}")
        print(f"By F1-Score: {sorted(self.results.keys(), key=lambda x: self.results[x]['f1_score'], reverse=True)}")
        
        # Best model
        best_name, _, best_score = self.get_best_model()
        print(f"\nBest Overall Model: {best_name} (ROC-AUC: {best_score:.4f})")

def main():
    """Train and evaluate baseline models"""
    print("Starting baseline model training...")
    
    # Initialize baseline models
    baseline = BaselineModels()
    
    # Load data
    baseline.load_data()
    
    # Train models
    baseline.train_models()
    
    # Print results
    baseline.print_results_summary()
    
    # Save models
    baseline.save_models()
    
    print("\nBaseline model training completed!")
    
    return baseline

if __name__ == "__main__":
    baseline = main()
