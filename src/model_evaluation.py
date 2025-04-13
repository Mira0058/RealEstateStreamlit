"""
Evaluates model performance using metrics: calculating metrics, compaoring different model versions
logging evaluation results
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Evaluate a trained model on test data
def evaluate_model(model, X_test, y_test):
    """
       
    Parameters:
        model (object): Trained model
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test target
        
    Returns:
        dict: Performance metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

# Compare multiple models on test data.
def compare_models(models, X_test, y_test):
    """
    
    Parameters:
        models (dict): Dictionary of trained models (name: model)
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test target
        
    Returns:
        pandas.DataFrame: Comparison of model performance
    """
    results = []
    
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics['model'] = name
        results.append(metrics)
    
    return pd.DataFrame(results)

# Predict house price using a trained model.
def predict_price(model, features):
    """
      
    Parameters:
        model (object): Trained model
        features (dict or pandas.DataFrame): Input features
        
    Returns:
        float: Predicted price
    """
    # Convert dictionary to DataFrame if needed
    if isinstance(features, dict):
        features = pd.DataFrame([features])
    
    # Make prediction
    prediction = model.predict(features)
    
    return prediction[0]