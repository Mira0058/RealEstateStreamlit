"""
Visualization module
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import tree

# Plot feature importance for tree-based models.
def plot_feature_importance(model, feature_names, top_n=10):
    """
        
    Parameters:
        model (object): Trained tree-based model
        feature_names (list): Names of features
        top_n (int): Number of top features to display
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame for visualization
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Limit to top N features
    if top_n:
        feature_imp = feature_imp.head(top_n)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_imp, ax=ax)
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    return fig

#  Plot the distribution of house prices.
def plot_price_distribution(df):
    """
     
    Parameters:
        df (pandas.DataFrame): Dataset with price column
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['price'], kde=True, ax=ax)
    ax.set_title('Distribution of House Prices')
    ax.set_xlabel('Price ($)')
    ax.set_ylabel('Frequency')
    
    return fig

#  Plot actual vs predicted values.
def plot_actual_vs_predicted(y_true, y_pred, model_name):
    """
        
    Parameters:
        y_true (array-like): Actual values
        y_pred (array-like): Predicted values
        model_name (str): Name of the model
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the points
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Plot the perfect prediction line
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax.set_title(f'Actual vs Predicted Prices ({model_name})')
    ax.set_xlabel('Actual Price ($)')
    ax.set_ylabel('Predicted Price ($)')
    
    return fig

#  Plot correlation matrix of features with target.
def plot_correlation_matrix(df, target_col='price'):
    """
      
    Parameters:
        df (pandas.DataFrame): Dataset
        target_col (str): Target column name
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Calculate correlation matrix
    corr = df.corr()
    
    # Sort features by correlation with target
    corr_with_target = corr[target_col].sort_values(ascending=False)
    
    # Select top correlated features
    top_features = list(corr_with_target.index)
    
    # Create correlation matrix for selected features
    corr_matrix = df[top_features].corr()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    
    return fig

# Plot decision tree structure.
def plot_decision_tree(model, feature_names, max_depth=3):
    """
      
    Parameters:
        model (DecisionTreeRegressor): Trained decision tree model
        feature_names (list): Names of features
        max_depth (int): Maximum depth to display
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, ax = plt.subplots(figsize=(20, 12))
    tree.plot_tree(
        model,
        feature_names=feature_names,
        filled=True,
        max_depth=max_depth,
        ax=ax
    )
    ax.set_title("Decision Tree Visualization")
    plt.tight_layout()
    
    return fig