"""
Contamination Detection Model
Uses historical sensor data to predict contamination risk in bioreactors
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ContaminationDetectionModel:
    """
    Machine Learning model for detecting contamination in bioreactor cultures
    
    Features:
    - pH level and variance
    - Dissolved Oxygen (DO) level and trends
    - CO2 concentration
    - Temperature stability
    - Rate of change metrics
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the contamination detection model
        
        Args:
            model_type (str): Type of model ('random_forest' or 'gradient_boosting')
        """
        self.model_type = model_type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        
        self.scaler = StandardScaler()
        self.feature_names = [
            'pH', 'DO', 'CO2', 'temperature',
            'pH_change', 'DO_change', 'CO2_change', 'temp_change',
            'pH_variance', 'DO_variance', 'pH_rolling_mean',
            'DO_rolling_mean', 'pH_deviation', 'DO_trend'
        ]
        self.is_trained = False
    
    def generate_synthetic_data(self, n_samples=1000, contamination_ratio=0.2):
        """
        Generate synthetic sensor data for training
        
        Args:
            n_samples (int): Number of samples to generate
            contamination_ratio (float): Ratio of contaminated samples
        
        Returns:
            pd.DataFrame: Synthetic sensor data
        """
        np.random.seed(42)
        
        n_contaminated = int(n_samples * contamination_ratio)
        n_normal = n_samples - n_contaminated
        
        # Normal conditions
        normal_data = {
            'pH': np.random.normal(7.0, 0.15, n_normal),
            'DO': np.random.normal(42, 5, n_normal),
            'CO2': np.random.normal(5, 1, n_normal),
            'temperature': np.random.normal(37, 0.8, n_normal),
            'contaminated': [0] * n_normal
        }
        
        # Contaminated conditions (more variance and drift)
        contaminated_data = {
            'pH': np.random.normal(7.3, 0.4, n_contaminated) + np.random.uniform(-0.5, 0.5, n_contaminated),
            'DO': np.random.normal(35, 8, n_contaminated) - np.random.uniform(0, 10, n_contaminated),
            'CO2': np.random.normal(7, 2, n_contaminated) + np.random.uniform(0, 3, n_contaminated),
            'temperature': np.random.normal(37.5, 1.5, n_contaminated),
            'contaminated': [1] * n_contaminated
        }
        
        # Combine and shuffle
        df_normal = pd.DataFrame(normal_data)
        df_contaminated = pd.DataFrame(contaminated_data)
        df = pd.concat([df_normal, df_contaminated], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def create_features(self, df):
        """
        Create engineered features from raw sensor data
        
        Args:
            df (pd.DataFrame): Raw sensor data with columns: pH, DO, CO2, temperature
        
        Returns:
            pd.DataFrame: Feature matrix
        """
        df = df.copy()
        
        # Rate of change features
        df['pH_change'] = df['pH'].diff().fillna(0)
        df['DO_change'] = df['DO'].diff().fillna(0)
        df['CO2_change'] = df['CO2'].diff().fillna(0)
        df['temp_change'] = df['temperature'].diff().fillna(0)
        
        # Rolling statistics (window of 10)
        df['pH_variance'] = df['pH'].rolling(window=10, min_periods=1).std().fillna(0)
        df['DO_variance'] = df['DO'].rolling(window=10, min_periods=1).std().fillna(0)
        df['pH_rolling_mean'] = df['pH'].rolling(window=10, min_periods=1).mean().fillna(df['pH'])
        df['DO_rolling_mean'] = df['DO'].rolling(window=10, min_periods=1).mean().fillna(df['DO'])
        
        # Deviation from optimal ranges
        df['pH_deviation'] = np.abs(df['pH'] - 7.0)
        df['DO_trend'] = df['DO'].rolling(window=5, min_periods=1).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        ).fillna(0)
        
        return df[self.feature_names]
    
    def train(self, X, y, test_size=0.2):
        """
        Train the contamination detection model
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target labels (0=normal, 1=contaminated)
            test_size (float): Proportion of test set
        
        Returns:
            dict: Training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate
        print("\n" + "="*60)
        print("CONTAMINATION DETECTION MODEL PERFORMANCE")
        print("="*60)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Contaminated']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"\nCross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # ROC-AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 5 Most Important Features:")
            print(feature_importance.head())
        
        self.is_trained = True
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm)
        
        # Plot ROC curve
        self.plot_roc_curve(y_test, y_pred_proba)
        
        return {
            'accuracy': (y_pred == y_test).mean(),
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def predict(self, X):
        """
        Predict contamination probability
        
        Args:
            X (pd.DataFrame): Feature matrix
        
        Returns:
            np.array: Contamination probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict_with_risk_level(self, X):
        """
        Predict contamination with risk levels
        
        Args:
            X (pd.DataFrame): Feature matrix
        
        Returns:
            pd.DataFrame: Predictions with risk levels
        """
        probabilities = self.predict(X)
        
        risk_levels = []
        for prob in probabilities:
            if prob < 0.3:
                risk_levels.append('Low')
            elif prob < 0.6:
                risk_levels.append('Medium')
            else:
                risk_levels.append('High')
        
        return pd.DataFrame({
            'contamination_probability': probabilities,
            'risk_level': risk_levels
        })
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Contaminated'],
                    yticklabels=['Normal', 'Contaminated'])
        plt.title('Contamination Detection - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix saved as 'confusion_matrix.png'")
        plt.close()
    
    def plot_roc_curve(self, y_true, y_pred_proba):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        print("ROC curve saved as 'roc_curve.png'")
        plt.close()
    
    def save_model(self, filepath='contamination_model.pkl'):
        """
        Save trained model to disk
        
        Args:
            filepath (str): Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath='contamination_model.pkl'):
        """
        Load trained model from disk
        
        Args:
            filepath (str): Path to load model from
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = True
        print(f"Model loaded from {filepath}")
        print(f"Model trained on: {model_data['timestamp']}")


def main():
    """Example usage of ContaminationDetectionModel"""
    
    print("BioSensor Contamination Detection Model")
    print("="*60)
    
    # Initialize model
    model = ContaminationDetectionModel(model_type='random_forest')
    
    # Generate synthetic training data
    print("\nGenerating synthetic training data...")
    df = model.generate_synthetic_data(n_samples=2000, contamination_ratio=0.25)
    print(f"Generated {len(df)} samples")
    print(f"Contaminated samples: {df['contaminated'].sum()} ({df['contaminated'].mean()*100:.1f}%)")
    
    # Create features
    print("\nEngineering features...")
    X = model.create_features(df)
    y = df['contaminated']
    print(f"Created {len(model.feature_names)} features")
    
    # Train model
    metrics = model.train(X, y, test_size=0.2)
    
    # Save model
    model.save_model('contamination_model.pkl')
    
    # Example prediction
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    
    # Normal sample
    normal_sample = pd.DataFrame({
        'pH': [7.0], 'DO': [42.0], 'CO2': [5.0], 'temperature': [37.0]
    })
    normal_features = model.create_features(normal_sample)
    normal_pred = model.predict_with_risk_level(normal_features)
    print("\nNormal Sample:")
    print(f"  pH: 7.0, DO: 42%, CO2: 5%, Temp: 37°C")
    print(f"  Contamination Probability: {normal_pred['contamination_probability'].values[0]:.3f}")
    print(f"  Risk Level: {normal_pred['risk_level'].values[0]}")
    
    # Contaminated sample
    contaminated_sample = pd.DataFrame({
        'pH': [7.5], 'DO': [28.0], 'CO2': [9.0], 'temperature': [38.5]
    })
    contaminated_features = model.create_features(contaminated_sample)
    contaminated_pred = model.predict_with_risk_level(contaminated_features)
    print("\nPotentially Contaminated Sample:")
    print(f"  pH: 7.5, DO: 28%, CO2: 9%, Temp: 38.5°C")
    print(f"  Contamination Probability: {contaminated_pred['contamination_probability'].values[0]:.3f}")
    print(f"  Risk Level: {contaminated_pred['risk_level'].values[0]}")
    
    print("\n" + "="*60)
    print("Model training complete!")
    print("="*60)


if __name__ == "__main__":
    main()