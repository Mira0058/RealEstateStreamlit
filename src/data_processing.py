"""
Data preprocessing module for real estate price prediction. Responsible for loading and preparing dataset
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# load the data final.csv for Real State model training
def load_data(data_path='data/final.csv'):
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    return pd.read_csv(data_path)


def preprocess_data(df):
    """
    Necessary clenup/prepocessing operations.
    
    Parameters:
        df (pandas.DataFrame): Raw dataframe
        
    Returns:
        pandas.DataFrame: the cleaned and ready-to-use dataset
    """
    # a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Add any preprocessing steps here if needed
    # For example:
    # - Handle missing values
    # - Feature engineering
    # - Data normalization or scaling
    
    return processed_df


def split_data(df, target_col='price', test_size=0.2, stratify_col='property_type_Bunglow', random_state=42):
    """
    Split the data into training and testing sets.
    
    Args:
        df (pandas.DataFrame): Preprocessed dataframe
        target_col (str): Target column name
        test_size (float): Proportion of data to use for testing
        stratify_col (str): Column to use for stratified splitting
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=df[stratify_col],  
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def get_feature_names(X):
    """
    Get feature names from the dataset.
    
    Args:
        X (pandas.DataFrame): Feature dataframe
        
    Returns:
        list: List of feature names
    """
    return list(X.columns)