"""
Fermentation Success Prediction Model
Predicts fermentation outcomes based on bioreactor conditions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class FermentationSuccessModel:
    """
    Machine Learning model for predicting fermentation success
    
    Predicts yield, productivity, and overall success score based on:
    - Process parameters (pH, DO, CO2, temperature)
    - Parameter stability metrics
    - Time-based features
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize fermentation success model
        
        Args:
            model_type (str): 'random_forest' or 'gradient_boosting'
        """
        self.model_type = model_type
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        
        self.scaler = StandardScaler()
        self.feature_names = [
            'pH_mean', 'DO_mean', 'CO2_mean', 'temp_mean',
            'pH_std', 'DO_std', 'CO2_std', 'temp_std',
            'pH_range', 'DO_range', 'time_in_optimal',
            'pH_stability_score', 'DO_stability_score',
            'temp_stability_score'
        ]
        self.is_trained = False
    
    def generate_synthetic_data(self, n_samples=1000):
        """
        Generate synthetic fermentation batch data
        
        Args:
            n_samples (int): Number of batch samples to generate
        
        Returns:
            pd.DataFrame: Synthetic fermentation data
        """
        np.random.seed(42)
        
        data = []
        for _ in range(n_samples):
            # Generate process parameters with some correlation to success
            pH_mean = np.random.normal(7.0, 0.3)
            DO_mean = np.random.normal(42, 8)
            CO2_mean = np.random.normal(5.5, 1.5)
            temp_mean = np.random.normal(37, 1.2)
            
            pH_std = np.random.uniform(0.05, 0.5)
            DO_std = np.random.uniform(1, 8)
            CO2_std = np.random.uniform(0.3, 2.5)
            temp_std = np.random.uniform(0.2, 1.5)
            
            pH_range = np.random.uniform(0.2, 1.5)
            DO_range = np.random.uniform(5, 20)
            
            time_in_optimal = np.random.uniform(0.4, 1.0)
            
            # Stability scores (lower std = higher stability)
            pH_stability = max(0, 1 - pH_std / 0.5)
            DO_stability = max(0, 1 - DO_std / 8)
            temp_stability = max(0, 1 - temp_std / 1.5)
            
            # Calculate success score (0-100)
            # Based on proximity to optimal conditions and stability
            success = 100
            
            # pH penalties
            if pH_mean < 6.8 or pH_mean > 7.4:
                success -= 15
            success -= pH_std * 20
            
            # DO penalties
            if DO_mean < 35 or DO_mean > 50:
                success -= 15
            success -= DO_std * 2
            
            # CO2 penalties
            if CO2_mean < 4 or CO2_mean > 7:
                success -= 10
            
            # Temperature penalties
            if temp_mean < 36 or temp_mean > 38:
                success -= 10
            success -= temp_std * 10
            
            # Stability bonuses
            success += time_in_optimal * 10
            
            # Add some noise
            success += np.random.normal(0, 5)
            success = max(0, min(100, success))
            
            data.append({
                'pH_mean': pH_mean,
                'DO_mean': DO_mean,
                'CO2_mean': CO2_mean,
                'temp_mean': temp_mean,
                'pH_std': pH_std,
                'DO_std': DO_std,
                'CO2_std': CO2_std,
                'temp_std': temp_std,
                'pH_range': pH_range,
                'DO_range': DO_range,
                'time_in_optimal': time_in_optimal,
                'pH_stability_score': pH_stability,
                'DO_stability_score': DO_stability,
                'temp_stability_score': temp_stability,
                'success_score': success
            })
        
        return pd.DataFrame(data)
    
    def calculate_batch_features(self, sensor_df):
        """
        Calculate batch-level features from time-series sensor data
        
        Args:
            sensor_df (pd.DataFrame): Time-series data with pH, DO, CO2, temperature
        
        Returns:
            pd.DataFrame: Batch-level features
        """
        # Define optimal ranges
        optimal_ranges = {
            'pH': (6.8, 7.4),
            'DO': (35, 50),
            'CO2': (4, 7),
            'temperature': (36, 38)
        }
        
        features = {}
        
        # Mean values
        for col in ['pH', 'DO', 'CO2', 'temperature']:
            features[f'{col}_mean'] = sensor_df[col].mean()
            features[f'{col}_std'] = sensor_df[col].std()
        
        # Range
        features['pH_range'] = sensor_df['pH'].max() - sensor_df['pH'].min()
        features['DO_range'] = sensor_df['DO'].max() - sensor_df['DO'].min()
        
        # Time in optimal range
        in_optimal = []
        for _, row in sensor_df.iterrows():
            all_optimal = all([
                optimal_ranges['pH'][0] <= row['pH'] <= optimal_ranges['pH'][1],
                optimal_ranges['DO'][0] <= row['DO'] <= optimal_ranges['DO'][1],
                optimal_ranges['CO2'][0] <= row['CO2'] <= optimal_ranges['CO2'][1],
                optimal_ranges['temperature'][0] <= row['temperature'] <= optimal_ranges['temperature'][1]
            ])
            in_optimal.append(all_optimal)
        
        features['time_in_optimal'] = sum(in_optimal) / len(in_optimal)
        
        # Stability scores (inverse of coefficient of variation)
        features['pH_stability_score'] = 1 / (1 + sensor_df['pH'].std() / sensor_df['pH'].mean())
        features['DO_stability_score'] = 1 / (1 + sensor_df['DO'].std() / sensor_df['DO'].mean())
        features['temp_stability_score'] = 1 / (1 + sensor_df['temperature'].std() / sensor_df['temperature'].mean())
        
        return pd.DataFrame([features])
    
    def train(self, X, y, test_size=0.2):
        """
        Train the fermentation success model
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target (success scores 0-100)
            test_size (float): Proportion of test set
        
        Returns:
            dict: Training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\n" + "="*60)
        print("FERMENTATION SUCCESS PREDICTION MODEL PERFORMANCE")
        print("="*60)
        print(f"\nR² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Mean Prediction Error: {mae:.2f} points (out of 100)")
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=5, scoring='r2'
        )
        print(f"\nCross-Validation R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 5 Most Important Features:")
            print(feature_importance.head())
            
            # Plot feature importance
            self.plot_feature_importance(feature_importance)
        
        self.is_trained = True
        
        # Plot predictions vs actual
        self.plot_predictions(y_test, y_pred)
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def predict(self, X):
        """
        Predict fermentation success score
        
        Args:
            X (pd.DataFrame): Feature matrix
        
        Returns:
            np.array: Success scores (0-100)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return np.clip(predictions, 0, 100)
    
    def predict_with_category(self, X):
        """
        Predict success with categorical labels
        
        Args:
            X (pd.DataFrame): Feature matrix
        
        Returns:
            pd.DataFrame: Predictions with categories
        """
        scores = self.predict(X)
        
        categories = []
        for score in scores:
            if score >= 85:
                categories.append('Excellent')
            elif score >= 70:
                categories.append('Good')
            elif score >= 50:
                categories.append('Fair')
            else:
                categories.append('Poor')
        
        return pd.DataFrame({
            'success_score': scores,
            'category': categories
        })
    
    def plot_feature_importance(self, feature_importance):
        """Plot feature importance"""
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature', palette='viridis')
        plt.title('Top 10 Feature Importance - Fermentation Success Model')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('feature_importance_fermentation.png', dpi=300, bbox_inches='tight')
        print("\nFeature importance plot saved as 'feature_importance_fermentation.png'")
        plt.close()
    
    def plot_predictions(self, y_true, y_pred):
        """Plot predictions vs actual values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
        plt.plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Success Score')
        plt.ylabel('Predicted Success Score')
        plt.title('Fermentation Success: Predictions vs Actual')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        print("Predictions plot saved as 'predictions_vs_actual.png'")
        plt.close()
    
    def save_model(self, filepath='fermentation_model.pkl'):
        """Save trained model"""
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
    
    def load_model(self, filepath='fermentation_model.pkl'):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = True
        print(f"Model loaded from {filepath}")
        print(f"Model trained on: {model_data['timestamp']}")


def main():
    """Example usage of FermentationSuccessModel"""
    
    print("BioSensor Fermentation Success Prediction Model")
    print("="*60)
    
    # Initialize model
    model = FermentationSuccessModel(model_type='random_forest')
    
    # Generate synthetic training data
    print("\nGenerating synthetic batch data...")
    df = model.generate_synthetic_data(n_samples=1500)
    print(f"Generated {len(df)} batch samples")
    print(f"Success score range: {df['success_score'].min():.1f} - {df['success_score'].max():.1f}")
    print(f"Mean success score: {df['success_score'].mean():.1f}")
    
    # Prepare features and target
    X = df[model.feature_names]
    y = df['success_score']
    
    # Train model
    metrics = model.train(X, y, test_size=0.2)
    
    # Save model
    model.save_model('fermentation_model.pkl')
    
    # Example predictions
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    
    # Excellent conditions
    excellent_batch = pd.DataFrame([{
        'pH_mean': 7.1, 'DO_mean': 43, 'CO2_mean': 5.2, 'temp_mean': 37,
        'pH_std': 0.08, 'DO_std': 2.5, 'CO2_std': 0.5, 'temp_std': 0.3,
        'pH_range': 0.25, 'DO_range': 8, 'time_in_optimal': 0.92,
        'pH_stability_score': 0.95, 'DO_stability_score': 0.90,
        'temp_stability_score': 0.95
    }])
    excellent_pred = model.predict_with_category(excellent_batch)
    print("\nExcellent Batch Conditions:")
    print(f"  pH: 7.1±0.08, DO: 43±2.5%, CO2: 5.2±0.5%, Temp: 37±0.3°C")
    print(f"  Time in Optimal: 92%")
    print(f"  Predicted Success Score: {excellent_pred['success_score'].values[0]:.1f}/100")
    print(f"  Category: {excellent_pred['category'].values[0]}")
    
    # Poor conditions
    poor_batch = pd.DataFrame([{
        'pH_mean': 7.6, 'DO_mean': 28, 'CO2_mean': 8.5, 'temp_mean': 38.8,
        'pH_std': 0.45, 'DO_std': 7.2, 'CO2_std': 2.1, 'temp_std': 1.3,
        'pH_range': 1.3, 'DO_range': 18, 'time_in_optimal': 0.35,
        'pH_stability_score': 0.45, 'DO_stability_score': 0.50,
        'temp_stability_score': 0.40
    }])
    poor_pred = model.predict_with_category(poor_batch)
    print("\nPoor Batch Conditions:")
    print(f"  pH: 7.6±0.45, DO: 28±7.2%, CO2: 8.5±2.1%, Temp: 38.8±1.3°C")
    print(f"  Time in Optimal: 35%")
    print(f"  Predicted Success Score: {poor_pred['success_score'].values[0]:.1f}/100")
    print(f"  Category: {poor_pred['category'].values[0]}")
    
    print("\n" + "="*60)
    print("Model training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
