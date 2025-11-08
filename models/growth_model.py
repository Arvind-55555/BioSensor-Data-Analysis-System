"""
Cell Growth Prediction Model
Predicts cell growth phases and rates from bioreactor sensor data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class CellGrowthModel:
    """
    Machine Learning model for predicting cell growth phases and rates
    
    Predicts:
    - Growth phase (Lag, Exponential, Stationary, Death, Stressed)
    - Growth rate (μ in h⁻¹)
    
    Based on environmental conditions and time-series features
    """
    
    def __init__(self):
        """Initialize cell growth model"""
        # Phase classifier
        self.phase_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        # Growth rate regressor
        self.rate_regressor = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self.feature_names = [
            'pH_mean', 'DO_mean', 'CO2_mean', 'temp_mean',
            'pH_variance', 'DO_variance', 'time_elapsed',
            'DO_trend', 'pH_stability', 'nutrient_depletion_indicator',
            'metabolic_activity_score', 'stress_indicator'
        ]
        
        self.phases = ['Lag', 'Exponential', 'Stationary', 'Death', 'Stressed']
        self.is_trained = False
    
    def generate_synthetic_data(self, n_samples=1500):
        """
        Generate synthetic cell growth data
        
        Args:
            n_samples (int): Number of samples to generate
        
        Returns:
            pd.DataFrame: Synthetic growth data
        """
        np.random.seed(42)
        data = []
        
        for _ in range(n_samples):
            # Random time point (0-120 hours)
            time_elapsed = np.random.uniform(0, 120)
            
            # Determine phase based on time and conditions
            if time_elapsed < 10:
                phase = 'Lag'
                base_rate = np.random.uniform(0.05, 0.15)
                pH_mean = np.random.normal(7.0, 0.15)
                DO_mean = np.random.normal(45, 5)
            elif time_elapsed < 40:
                phase = 'Exponential'
                base_rate = np.random.uniform(0.25, 0.40)
                pH_mean = np.random.normal(7.0, 0.12)
                DO_mean = np.random.normal(40, 6)
            elif time_elapsed < 80:
                phase = 'Stationary'
                base_rate = np.random.uniform(-0.02, 0.08)
                pH_mean = np.random.normal(7.1, 0.2)
                DO_mean = np.random.normal(38, 7)
            elif time_elapsed < 120:
                phase = 'Death'
                base_rate = np.random.uniform(-0.15, -0.01)
                pH_mean = np.random.normal(7.2, 0.25)
                DO_mean = np.random.normal(42, 8)
            
            # Random chance of stressed conditions
            if np.random.random() < 0.15:
                phase = 'Stressed'
                base_rate = np.random.uniform(-0.05, 0.05)
                pH_mean = np.random.normal(7.4, 0.4)
                DO_mean = np.random.normal(28, 10)
            
            # Generate other parameters
            CO2_mean = np.random.normal(5.5, 1.2)
            temp_mean = np.random.normal(37, 0.8)
            
            pH_variance = np.random.uniform(0.01, 0.3)
            DO_variance = np.random.uniform(1, 6)
            
            DO_trend = np.random.uniform(-2, 1) if phase in ['Exponential', 'Death'] else np.random.uniform(-0.5, 0.5)
            pH_stability = max(0, 1 - pH_variance / 0.3)
            
            # Nutrient depletion indicator (increases with time)
            nutrient_depletion = min(1, time_elapsed / 80) + np.random.normal(0, 0.1)
            
            # Metabolic activity score
            metabolic_activity = 1 - abs(pH_mean - 7.0) / 1.0
            metabolic_activity *= (DO_mean / 50)
            metabolic_activity = max(0, min(1, metabolic_activity))
            
            # Stress indicator
            stress_indicator = 0
            if pH_mean < 6.5 or pH_mean > 7.8:
                stress_indicator += 0.3
            if DO_mean < 30:
                stress_indicator += 0.3
            if CO2_mean > 8:
                stress_indicator += 0.2
            if abs(temp_mean - 37) > 2:
                stress_indicator += 0.2
            stress_indicator = min(1, stress_indicator)
            
            # Add noise to growth rate
            growth_rate = base_rate + np.random.normal(0, 0.02)
            
            data.append({
                'pH_mean': pH_mean,
                'DO_mean': DO_mean,
                'CO2_mean': CO2_mean,
                'temp_mean': temp_mean,
                'pH_variance': pH_variance,
                'DO_variance': DO_variance,
                'time_elapsed': time_elapsed,
                'DO_trend': DO_trend,
                'pH_stability': pH_stability,
                'nutrient_depletion_indicator': nutrient_depletion,
                'metabolic_activity_score': metabolic_activity,
                'stress_indicator': stress_indicator,
                'growth_phase': phase,
                'growth_rate': growth_rate
            })
        
        return pd.DataFrame(data)
    
    def calculate_features_from_timeseries(self, sensor_df, time_elapsed):
        """
        Calculate features from time-series sensor data
        
        Args:
            sensor_df (pd.DataFrame): Time-series sensor data
            time_elapsed (float): Time since inoculation (hours)
        
        Returns:
            pd.DataFrame: Feature vector
        """
        features = {}
        
        # Mean values
        features['pH_mean'] = sensor_df['pH'].mean()
        features['DO_mean'] = sensor_df['DO'].mean()
        features['CO2_mean'] = sensor_df['CO2'].mean()
        features['temp_mean'] = sensor_df['temperature'].mean()
        
        # Variance
        features['pH_variance'] = sensor_df['pH'].var()
        features['DO_variance'] = sensor_df['DO'].var()
        
        # Time elapsed
        features['time_elapsed'] = time_elapsed
        
        # DO trend (linear regression slope)
        if len(sensor_df) > 1:
            x = np.arange(len(sensor_df))
            features['DO_trend'] = np.polyfit(x, sensor_df['DO'].values, 1)[0]
        else:
            features['DO_trend'] = 0
        
        # pH stability
        features['pH_stability'] = 1 / (1 + features['pH_variance'])
        
        # Nutrient depletion indicator
        features['nutrient_depletion_indicator'] = min(1, time_elapsed / 80)
        
        # Metabolic activity score
        pH_opt_score = 1 - abs(features['pH_mean'] - 7.0) / 1.0
        DO_score = features['DO_mean'] / 50
        features['metabolic_activity_score'] = max(0, min(1, pH_opt_score * DO_score))
        
        # Stress indicator
        stress = 0
        if features['pH_mean'] < 6.5 or features['pH_mean'] > 7.8:
            stress += 0.3
        if features['DO_mean'] < 30:
            stress += 0.3
        if features['CO2_mean'] > 8:
            stress += 0.2
        if abs(features['temp_mean'] - 37) > 2:
            stress += 0.2
        features['stress_indicator'] = min(1, stress)
        
        return pd.DataFrame([features])
    
    def train(self, X, y_phase, y_rate, test_size=0.2):
        """
        Train both phase classifier and rate regressor
        
        Args:
            X (pd.DataFrame): Feature matrix
            y_phase (pd.Series): Growth phases
            y_rate (pd.Series): Growth rates
            test_size (float): Proportion of test set
        
        Returns:
            dict: Training metrics
        """
        # Encode phases
        y_phase_encoded = self.label_encoder.fit_transform(y_phase)
        
        # Split data
        X_train, X_test, y_phase_train, y_phase_test, y_rate_train, y_rate_test = train_test_split(
            X, y_phase_encoded, y_rate, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("="*60)
        print("TRAINING PHASE CLASSIFIER")
        print("="*60)
        
        # Train phase classifier
        self.phase_classifier.fit(X_train_scaled, y_phase_train)
        y_phase_pred = self.phase_classifier.predict(X_test_scaled)
        
        # Decode predictions
        y_phase_test_decoded = self.label_encoder.inverse_transform(y_phase_test)
        y_phase_pred_decoded = self.label_encoder.inverse_transform(y_phase_pred)
        
        print("\nPhase Classification Report:")
        print(classification_report(y_phase_test_decoded, y_phase_pred_decoded))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_phase_test_decoded, y_phase_pred_decoded, labels=self.phases)
        print(cm)
        
        # Cross-validation
        cv_scores_phase = cross_val_score(self.phase_classifier, X_train_scaled, y_phase_train, cv=5)
        print(f"\nPhase Classification CV Accuracy: {cv_scores_phase.mean():.4f} (+/- {cv_scores_phase.std():.4f})")
        
        print("\n" + "="*60)
        print("TRAINING GROWTH RATE REGRESSOR")
        print("="*60)
        
        # Train rate regressor
        self.rate_regressor.fit(X_train_scaled, y_rate_train)
        y_rate_pred = self.rate_regressor.predict(X_test_scaled)
        
        # Evaluate rate predictions
        rmse = np.sqrt(mean_squared_error(y_rate_test, y_rate_pred))
        r2 = r2_score(y_rate_test, y_rate_pred)
        
        print(f"\nGrowth Rate R² Score: {r2:.4f}")
        print(f"Growth Rate RMSE: {rmse:.4f} h⁻¹")
        
        # Cross-validation
        cv_scores_rate = cross_val_score(
            self.rate_regressor, X_train_scaled, y_rate_train, 
            cv=5, scoring='r2'
        )
        print(f"Growth Rate CV R² Score: {cv_scores_rate.mean():.4f} (+/- {cv_scores_rate.std():.4f})")
        
        self.is_trained = True
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm)
        
        # Plot rate predictions
        self.plot_rate_predictions(y_rate_test, y_rate_pred)
        
        # Feature importance
        self.plot_feature_importance()
        
        return {
            'phase_accuracy': (y_phase_pred == y_phase_test).mean(),
            'phase_cv_mean': cv_scores_phase.mean(),
            'rate_r2': r2,
            'rate_rmse': rmse,
            'rate_cv_mean': cv_scores_rate.mean()
        }
    
    def predict_phase(self, X):
        """Predict growth phase"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        phase_encoded = self.phase_classifier.predict(X_scaled)
        return self.label_encoder.inverse_transform(phase_encoded)
    
    def predict_rate(self, X):
        """Predict growth rate"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.rate_regressor.predict(X_scaled)
    
    def predict(self, X):
        """Predict both phase and rate"""
        phases = self.predict_phase(X)
        rates = self.predict_rate(X)
        
        return pd.DataFrame({
            'growth_phase': phases,
            'growth_rate': rates
        })
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix for phase classification"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.phases,
                    yticklabels=self.phases)
        plt.title('Growth Phase Classification - Confusion Matrix')
        plt.ylabel('True Phase')
        plt.xlabel('Predicted Phase')
        plt.tight_layout()
        plt.savefig('growth_phase_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix saved as 'growth_phase_confusion_matrix.png'")
        plt.close()
    
    def plot_rate_predictions(self, y_true, y_pred):
        """Plot growth rate predictions vs actual"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Growth Rate (h⁻¹)')
        plt.ylabel('Predicted Growth Rate (h⁻¹)')
        plt.title('Growth Rate Predictions vs Actual')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('growth_rate_predictions.png', dpi=300, bbox_inches='tight')
        print("Growth rate predictions plot saved as 'growth_rate_predictions.png'")
        plt.close()
    
    def plot_feature_importance(self):
        """Plot feature importance for both models"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Phase classifier importance
        phase_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.phase_classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(data=phase_importance, x='importance', y='feature', 
                   palette='viridis', ax=axes[0])
        axes[0].set_title('Feature Importance - Phase Classification')
        axes[0].set_xlabel('Importance')
        
        # Rate regressor importance
        rate_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.rate_regressor.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(data=rate_importance, x='importance', y='feature',
                   palette='plasma', ax=axes[1])
        axes[1].set_title('Feature Importance - Rate Prediction')
        axes[1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('growth_feature_importance.png', dpi=300, bbox_inches='tight')
        print("Feature importance plot saved as 'growth_feature_importance.png'")
        plt.close()
    
    def save_model(self, filepath='growth_model.pkl'):
        """Save trained models"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'phase_classifier': self.phase_classifier,
            'rate_regressor': self.rate_regressor,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'phases': self.phases,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath='growth_model.pkl'):
        """Load trained models"""
        model_data = joblib.load(filepath)
        self.phase_classifier = model_data['phase_classifier']
        self.rate_regressor = model_data['rate_regressor']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.phases = model_data['phases']
        self.is_trained = True
        print(f"Model loaded from {filepath}")
        print(f"Model trained on: {model_data['timestamp']}")


def main():
    """Example usage of CellGrowthModel"""
    
    print("BioSensor Cell Growth Prediction Model")
    print("="*60)
    
    # Initialize model
    model = CellGrowthModel()
    
    # Generate synthetic training data
    print("\nGenerating synthetic growth data...")
    df = model.generate_synthetic_data(n_samples=2000)
    print(f"Generated {len(df)} samples")
    print(f"\nPhase distribution:")
    print(df['growth_phase'].value_counts())
    print(f"\nGrowth rate range: {df['growth_rate'].min():.3f} to {df['growth_rate'].max():.3f} h⁻¹")
    
    # Prepare data
    X = df[model.feature_names]
    y_phase = df['growth_phase']
    y_rate = df['growth_rate']
    
    # Train models
    metrics = model.train(X, y_phase, y_rate, test_size=0.2)
    
    # Save models
    model.save_model('growth_model.pkl')
    
    # Example predictions
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    
    # Exponential phase conditions
    exp_sample = pd.DataFrame([{
        'pH_mean': 7.0, 'DO_mean': 42, 'CO2_mean': 5.2, 'temp_mean': 37.0,
        'pH_variance': 0.02, 'DO_variance': 3.5, 'time_elapsed': 25,
        'DO_trend': -0.8, 'pH_stability': 0.95,
        'nutrient_depletion_indicator': 0.31,
        'metabolic_activity_score': 0.88, 'stress_indicator': 0.05
    }])
    exp_pred = model.predict(exp_sample)
    print("\nExponential Phase Sample (25h):")
    print(f"  pH: 7.0, DO: 42%, CO2: 5.2%, Temp: 37°C")
    print(f"  Predicted Phase: {exp_pred['growth_phase'].values[0]}")
    print(f"  Predicted Growth Rate: {exp_pred['growth_rate'].values[0]:.3f} h⁻¹")
    
    # Stressed conditions
    stress_sample = pd.DataFrame([{
        'pH_mean': 7.6, 'DO_mean': 25, 'CO2_mean': 8.8, 'temp_mean': 38.5,
        'pH_variance': 0.18, 'DO_variance': 8.2, 'time_elapsed': 35,
        'DO_trend': -1.5, 'pH_stability': 0.45,
        'nutrient_depletion_indicator': 0.44,
        'metabolic_activity_score': 0.32, 'stress_indicator': 0.85
    }])
    stress_pred = model.predict(stress_sample)
    print("\nStressed Conditions Sample (35h):")
    print(f"  pH: 7.6, DO: 25%, CO2: 8.8%, Temp: 38.5°C")
    print(f"  Predicted Phase: {stress_pred['growth_phase'].values[0]}")
    print(f"  Predicted Growth Rate: {stress_pred['growth_rate'].values[0]:.3f} h⁻¹")
    
    print("\n" + "="*60)
    print("Model training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
