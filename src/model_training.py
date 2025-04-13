"""
Model training module for real estate price prediction. Contains logic to train this model.
spliting dataset, initializing and training models; saving trained models
"""
import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    
    Parameters:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        
    Returns:
        object: Trained Linear Regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, max_depth=3, max_features=10, random_state=567):
    """
    Train a Decision Tree model.
    
    Parameters:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        max_depth (int): Maximum depth of the decision tree
        max_features (int): Maximum number of features to consider
        random_state (int): Random seed for reproducibility
        
    Returns:
        object: Trained Decision Tree model
    """
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        max_features=max_features,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=200, criterion='absolute_error', random_state=42):
    """
    Train a Random Forest model.
    
    Parameters:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        n_estimators (int): Number of trees in the forest
        criterion (str): Function to measure the quality of a split
        random_state (int): Random seed for reproducibility
        
    Returns:
        object: Trained Random Forest model
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        criterion=criterion,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, model_path):
    """
    Save a trained model to disk.
    
    Parameters:
        model (object): Trained model
        model_path (str): Path to save the model
        
    Returns:
        bool: True if successful
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return True


def load_model(model_path):
    """
    Load a trained model from disk.
    
    Parameters:
        model_path (str): Path to the saved model
        
    Returns:
        object: Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model