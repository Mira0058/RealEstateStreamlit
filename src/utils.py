"""
Utility functions for real estate price prediction.
"""
import pandas as pd
import numpy as np
import os

#  Create a dictionary of features with their default values from a sample.
def create_feature_dict(X_sample):
    """
       
    Parameters:
        X_sample (pandas.DataFrame): Sample DataFrame with feature columns
        
    Returns:
        dict: Dictionary of features with default values
    """
    return {col: X_sample[col].iloc[0] for col in X_sample.columns}

#   Format price as currency string.
def format_price(price):
    """
        
    Parameters:
        price (float): Price value
        
    Returns:
        str: Formatted price string
    """
    return f"${price:,.2f}"

#  Calculate property age based on year built.
def calculate_property_age(year_built, current_year=None):
    """
        
    Parameters:
        year_built (int): Year the property was built
        current_year (int, optional): Current year (defaults to current year)
        
    Returns:
        int: Property age in years
    """
    if current_year is None:
        import datetime
        current_year = datetime.datetime.now().year
        
    return current_year - year_built

#  Create input features from form data for prediction
def create_input_features(form_data):
    """
        
    Parameters:
        form_data (dict): User input form data
        
    Returns:
        dict: Formatted features for prediction
    """
    features = {}
    
    # Copy the form data
    for key, value in form_data.items():
        features[key] = value
    
    # Calculate derived features
    if 'year_built' in features and 'year_sold' in features:
        features['property_age'] = features['year_sold'] - features['year_built']
    
    # One-hot encode categorical variables
    if 'property_type' in features:
        property_type = features.pop('property_type')
        features['property_type_Bunglow'] = 1 if property_type == 'Bunglow' else 0
        features['property_type_Condo'] = 1 if property_type == 'Condo' else 0
    
    return features

#  Get min and max values for numerical features.
def get_feature_ranges(df):
    """
       
    Parameters:
        df (pandas.DataFrame): Dataset
        
    Returns:
        dict: Dictionary with min and max values for each feature
    """
    ranges = {}
    
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        ranges[col] = {
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    return ranges