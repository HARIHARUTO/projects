"""
Synthetic Healthcare Dataset Generation
Generates realistic patient data for privacy-preserving ML experiments
"""

import numpy as np
import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta
import os

class HealthcareDataGenerator:
    def __init__(self, seed=42):
        self.fake = Faker()
        Faker.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_patient_data(self, n_patients=10000):
        """Generate synthetic patient records"""
        patients = []
        
        for i in range(n_patients):
            # Basic demographics
            age = np.random.normal(45, 15)
            age = max(18, min(90, int(age)))  # Clamp between 18-90
            
            gender = np.random.choice(['M', 'F'], p=[0.48, 0.52])
            
            # Health metrics with realistic correlations
            bmi_base = np.random.normal(26, 4)
            bmi = max(15, min(45, bmi_base))
            
            # Blood pressure correlated with age and BMI
            systolic_base = 90 + (age - 18) * 0.5 + (bmi - 25) * 1.2 + np.random.normal(0, 10)
            systolic = max(80, min(200, int(systolic_base)))
            diastolic = max(50, min(120, int(systolic * 0.65 + np.random.normal(0, 5))))
            
            # Cholesterol correlated with age and BMI
            cholesterol_base = 150 + (age - 18) * 1.5 + (bmi - 25) * 2 + np.random.normal(0, 20)
            cholesterol = max(100, min(350, int(cholesterol_base)))
            
            # Blood glucose
            glucose_base = 85 + (age - 18) * 0.3 + (bmi - 25) * 1.5 + np.random.normal(0, 10)
            glucose = max(60, min(200, int(glucose_base)))
            
            # Lifestyle factors
            smoking = np.random.choice([0, 1], p=[0.75, 0.25])
            exercise_hours = max(0, np.random.exponential(3))
            
            # Medical history
            family_history = np.random.choice([0, 1], p=[0.7, 0.3])
            
            # Calculate risk score (target variable)
            risk_score = self._calculate_risk_score(
                age, bmi, systolic, diastolic, cholesterol, 
                glucose, smoking, exercise_hours, family_history
            )
            
            # High risk if score > 0.6
            high_risk = 1 if risk_score > 0.6 else 0
            
            patient = {
                'patient_id': f'P{i:06d}',
                'age': age,
                'gender': gender,
                'bmi': round(bmi, 2),
                'systolic_bp': systolic,
                'diastolic_bp': diastolic,
                'cholesterol': cholesterol,
                'glucose': glucose,
                'smoking': smoking,
                'exercise_hours': round(exercise_hours, 2),
                'family_history': family_history,
                'risk_score': round(risk_score, 4),
                'high_risk': high_risk
            }
            
            patients.append(patient)
            
        return pd.DataFrame(patients)
    
    def _calculate_risk_score(self, age, bmi, systolic, diastolic, 
                            cholesterol, glucose, smoking, exercise_hours, family_history):
        """Calculate cardiovascular risk score"""
        score = 0.0
        
        # Age factor
        if age > 65:
            score += 0.3
        elif age > 50:
            score += 0.15
        elif age > 35:
            score += 0.05
            
        # BMI factor
        if bmi > 30:
            score += 0.2
        elif bmi > 25:
            score += 0.1
            
        # Blood pressure factor
        if systolic > 140 or diastolic > 90:
            score += 0.25
        elif systolic > 130 or diastolic > 80:
            score += 0.1
            
        # Cholesterol factor
        if cholesterol > 240:
            score += 0.2
        elif cholesterol > 200:
            score += 0.1
            
        # Glucose factor
        if glucose > 126:
            score += 0.15
        elif glucose > 100:
            score += 0.05
            
        # Lifestyle factors
        if smoking:
            score += 0.2
            
        if exercise_hours < 2:
            score += 0.1
        elif exercise_hours > 5:
            score -= 0.05
            
        # Family history
        if family_history:
            score += 0.15
            
        # Add some noise
        score += np.random.normal(0, 0.05)
        
        return max(0, min(1, score))
    
    def create_federated_splits(self, df, n_clients=5):
        """Split data for federated learning simulation"""
        # Shuffle the data
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
        
        # Create non-IID splits (different distributions per client)
        client_data = {}
        n_samples = len(df_shuffled)
        
        for i in range(n_clients):
            start_idx = i * n_samples // n_clients
            end_idx = (i + 1) * n_samples // n_clients
            
            client_df = df_shuffled.iloc[start_idx:end_idx].copy()
            
            # Add some bias to make it non-IID
            if i % 2 == 0:
                # Bias towards older patients
                age_bias = client_df['age'] > client_df['age'].median()
                client_df = client_df[age_bias].sample(frac=0.8).reset_index(drop=True)
            else:
                # Bias towards younger patients
                age_bias = client_df['age'] <= client_df['age'].median()
                client_df = client_df[age_bias].sample(frac=0.8).reset_index(drop=True)
            
            client_data[f'client_{i}'] = client_df
            
        return client_data

def main():
    """Generate and save synthetic healthcare dataset"""
    print("Generating synthetic healthcare dataset...")
    
    generator = HealthcareDataGenerator(seed=42)
    
    # Generate main dataset
    df = generator.generate_patient_data(n_patients=10000)
    
    # Save main dataset
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/healthcare_data.csv', index=False)
    
    # Create train/test splits
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    train_df.to_csv('data/train_data.csv', index=False)
    test_df.to_csv('data/test_data.csv', index=False)
    
    # Create federated splits
    federated_data = generator.create_federated_splits(train_df, n_clients=5)
    
    os.makedirs('data/federated', exist_ok=True)
    for client_name, client_df in federated_data.items():
        client_df.to_csv(f'data/federated/{client_name}.csv', index=False)
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"High-risk patients: {df['high_risk'].sum()} ({df['high_risk'].mean():.2%})")
    print(f"\nFeature statistics:")
    print(df.describe())
    
    print(f"\nFederated splits:")
    for client_name, client_df in federated_data.items():
        high_risk_pct = client_df['high_risk'].mean()
        print(f"{client_name}: {len(client_df)} samples, {high_risk_pct:.2%} high-risk")
    
    print("\nDataset generation completed!")

if __name__ == "__main__":
    main()
