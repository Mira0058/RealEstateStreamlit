"""
Streamlit app for real estate price prediction.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

# Import modules from src
from src.data_processing import load_data, preprocess_data, split_data, get_feature_names
from src.model_training import train_decision_tree, load_model, save_model
from src.model_evaluation import evaluate_model, predict_price
from src.visualization import (
    plot_feature_importance, 
    plot_price_distribution, 
    plot_actual_vs_predicted,
    plot_correlation_matrix,
    plot_decision_tree
)
from src.utils import format_price, create_input_features, get_feature_ranges

# Set page configuration
st.set_page_config(
    page_title="Real Estate Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Define paths
DATA_PATH = "data/final.csv"
MODEL_PATH = "models/decision_tree_model.pkl"

# Sidebar
st.sidebar.title("Real Estate Price Prediction")
st.sidebar.image("https://www.pngitem.com/pimgs/m/151-1517876_real-estate-house-png-transparent-png.png", width=200)

# Navigation
page = st.sidebar.radio("Navigation", ["Home", "Data Exploration", "Model Training", "Price Prediction"])

# Load data
@st.cache_data
def load_cached_data():
    try:
        df = load_data(DATA_PATH)
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at {DATA_PATH}. Please make sure the data file exists.")
        return None

df = load_cached_data()

if df is None:
    st.error("Failed to load data. Please check the data path and try again.")
else:
    if page == "Home":
        st.title("🏠 Real Estate Price Prediction")
        st.write("## Welcome to the Real Estate Price Prediction App")
        
        st.write("""
        This application helps predict real estate prices based on property features. 
        You can explore the data, train models, and predict house prices.
        
        ### Features:
        - **Data Exploration**: Visualize and analyze real estate data
        - **Model Training**: Train and evaluate machine learning models
        - **Price Prediction**: Predict the price of a house based on its features
        
        ### Dataset Overview
        """)
        
        # Show dataset info
        st.write(f"**Number of records:** {df.shape[0]}")
        st.write(f"**Number of features:** {df.shape[1] - 1}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Sample data:**")
            st.dataframe(df.head())
        
        with col2:
            st.write("**Summary statistics:**")
            st.dataframe(df.describe())
            
        # Show price distribution
        st.write("### House Price Distribution")
        fig = plot_price_distribution(df)
        st.pyplot(fig)
        
    elif page == "Data Exploration":
        st.title("📊 Data Exploration")
        
        # Show correlation analysis
        st.write("### Feature Correlation Matrix")
        fig = plot_correlation_matrix(df)
        st.pyplot(fig)
        
        # Feature analysis
        st.write("### Feature Analysis")
        
        feature_to_explore = st.selectbox(
            "Select a feature to explore its relationship with price:",
            options=[col for col in df.columns if col != 'price']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df[feature_to_explore], df['price'], alpha=0.5)
            ax.set_xlabel(feature_to_explore)
            ax.set_ylabel('Price ($)')
            ax.set_title(f'Price vs {feature_to_explore}')
            st.pyplot(fig)
        
        with col2:
            # Box plot for categorical features or histogram for numerical
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if df[feature_to_explore].nunique() < 10:  # Categorical
                df.boxplot(column='price', by=feature_to_explore, ax=ax)
                ax.set_title(f'Price by {feature_to_explore}')
            else:  # Numerical
                ax.hist(df[feature_to_explore], bins=30)
                ax.set_xlabel(feature_to_explore)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {feature_to_explore}')
            
            st.pyplot(fig)
        
        # Data statistics
        st.write("### Feature Statistics")
        st.dataframe(df.describe().T)
        
    elif page == "Model Training":
        st.title("🧠 Model Training")
        
        # Model training options
        st.write("### Train a Decision Tree Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_depth = st.slider("Maximum Tree Depth", min_value=1, max_value=10, value=3)
            max_features = st.slider("Maximum Features", min_value=5, max_value=14, value=10)
        
        with col2:
            test_size = st.slider("Test Size (%)", min_value=10, max_value=50, value=20) / 100
            random_state = st.number_input("Random State", min_value=1, max_value=1000, value=567)
        
        # Train model button
        if st.button("Train Model"):
            with st.spinner("Training the model..."):
                # Preprocess data
                processed_df = preprocess_data(df)
                
                # Split data
                X_train, X_test, y_train, y_test = split_data(
                    processed_df, 
                    test_size=test_size, 
                    random_state=random_state
                )
                
                # Train model
                model = train_decision_tree(
                    X_train, 
                    y_train, 
                    max_depth=max_depth, 
                    max_features=max_features, 
                    random_state=random_state
                )
                
                # Save model
                save_model(model, MODEL_PATH)
                
                # Evaluate model
                metrics = evaluate_model(model, X_test, y_test)
                
                # Display results
                st.success("Model trained successfully!")
                
                # Show metrics
                st.write("### Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", f"${metrics['mae']:.2f}")
                col2.metric("MSE", f"${metrics['mse']:.2f}")
                col3.metric("RMSE", f"${metrics['rmse']:.2f}")
                col4.metric("R²", f"{metrics['r2']:.4f}")
                
                # Visualize results
                st.write("### Feature Importance")
                feature_names = get_feature_names(X_train)
                fig = plot_feature_importance(model, feature_names)
                st.pyplot(fig)
                
                # Actual vs predicted plot
                st.write("### Actual vs Predicted Prices")
                y_pred = model.predict(X_test)
                fig = plot_actual_vs_predicted(y_test, y_pred, "Decision Tree")
                st.pyplot(fig)
                
                # Visualize tree
                st.write("### Decision Tree Visualization")
                fig = plot_decision_tree(model, feature_names, max_depth=3)
                st.pyplot(fig)
        
        else:
            # Check if model exists
            if os.path.exists(MODEL_PATH):
                st.info("A trained model already exists. You can retrain it with different parameters or use it for predictions.")
                
                # Load and evaluate existing model
                if st.button("Evaluate Existing Model"):
                    with st.spinner("Evaluating model..."):
                        # Preprocess data
                        processed_df = preprocess_data(df)
                        
                        # Split data
                        X_train, X_test, y_train, y_test = split_data(
                            processed_df, 
                            test_size=test_size, 
                            random_state=random_state
                        )
                        
                        # Load model
                        model = load_model(MODEL_PATH)
                        
                        # Evaluate model
                        metrics = evaluate_model(model, X_test, y_test)
                        
                        # Display results
                        st.write("### Model Performance")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("MAE", f"${metrics['mae']:.2f}")
                        col2.metric("MSE", f"${metrics['mse']:.2f}")
                        col3.metric("RMSE", f"${metrics['rmse']:.2f}")
                        col4.metric("R²", f"{metrics['r2']:.4f}")
            else:
                st.warning("No trained model found. Please train a model first.")
    
    elif page == "Price Prediction":
        st.title("💰 House Price Prediction")
        
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            st.error("No trained model found. Please go to the Model Training page and train a model first.")
        else:
            # Load model
            model = load_model(MODEL_PATH)
            
            # Get feature ranges
            feature_ranges = get_feature_ranges(df)
            
            st.write("### Enter Property Details")
            
            # Create form
            with st.form("prediction_form"):
                # Split form into columns
                col1, col2 = st.columns(2)
                
                with col1:
                    year_sold = st.number_input(
                        "Year Sold", 
                        min_value=int(feature_ranges['year_sold']['min']),
                        max_value=int(feature_ranges['year_sold']['max']),
                        value=2016
                       
                    )
                    
                    property_tax = st.number_input(
                        "Property Tax (annual)", 
                        min_value=int(feature_ranges['property_tax']['min']),
                        max_value=int(feature_ranges['property_tax']['max']),
                        value=500
                    )
                    
                    insurance = st.number_input(
                        "Insurance (annual)", 
                        min_value=int(feature_ranges['insurance']['min']),
                        max_value=int(feature_ranges['insurance']['max']),
                        value=150
                    )
                    
                    beds = st.slider(
                        "Bedrooms", 
                        min_value=int(feature_ranges['beds']['min']),
                        max_value=int(feature_ranges['beds']['max']),
                        value=3
                    )
                    
                    baths = st.slider(
                        "Bathrooms", 
                        min_value=int(feature_ranges['baths']['min']),
                        max_value=int(feature_ranges['baths']['max']),
                        value=2
                    )
                    
                    sqft = st.number_input(
                        "Square Footage", 
                        min_value=int(feature_ranges['sqft']['min']),
                        max_value=int(feature_ranges['sqft']['max']),
                        value=1500
                    )
                    
                with col2:
                    year_built = st.number_input(
                        "Year Built", 
                        min_value=int(feature_ranges['year_built']['min']),
                        max_value=int(feature_ranges['year_built']['max']),
                        value=2000
                    )
                    
                    lot_size = st.number_input(
                        "Lot Size (sq ft)", 
                        min_value=int(feature_ranges['lot_size']['min']),
                        max_value=int(feature_ranges['lot_size']['max']),
                        value=5000
                    )
                    
                    basement = st.selectbox(
                        "Basement", 
                        options=[0, 1],
                        format_func=lambda x: "Yes" if x == 1 else "No"
                    )
                    
                    popular = st.selectbox(
                        "Popular Area", 
                        options=[0, 1],
                        format_func=lambda x: "Yes" if x == 1 else "No"
                    )
                    
                    recession = st.selectbox(
                        "Recession Period", 
                        options=[0, 1],
                        format_func=lambda x: "Yes" if x == 1 else "No"
                    )
                    
                    property_type = st.selectbox(
                        "Property Type", 
                        options=["Bunglow", "Condo"]
                    )
                
                # Calculate property age automatically
                property_age = year_sold - year_built
                
                # Convert property type to one-hot encoding
                property_type_Bunglow = 1 if property_type == "Bunglow" else 0
                property_type_Condo = 1 if property_type == "Condo" else 0
                
                # Submit button
                submitted = st.form_submit_button("Predict Price")
            
            if submitted:
                # Create input features
                input_features = {
                    'year_sold': year_sold,
                    'property_tax': property_tax,
                    'insurance': insurance,
                    'beds': beds,
                    'baths': baths,
                    'sqft': sqft,
                    'year_built': year_built,
                    'lot_size': lot_size,
                    'basement': basement,
                    'popular': popular,
                    'recession': recession,
                    'property_age': property_age,
                    'property_type_Bunglow': property_type_Bunglow,
                    'property_type_Condo': property_type_Condo
                }
                
                # Convert to DataFrame
                input_df = pd.DataFrame([input_features])
                
                # Make prediction
                prediction = predict_price(model, input_df)
                
                # Display prediction
                st.success(f"### Predicted Price: {format_price(prediction)}")
                
                # Property details summary
                st.write("### Property Details Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Property Type:** {property_type}")
                    st.write(f"**Bedrooms:** {beds}")
                    st.write(f"**Bathrooms:** {baths}")
                    st.write(f"**Square Footage:** {sqft} sq ft")
                    st.write(f"**Lot Size:** {lot_size} sq ft")
                
                with col2:
                    st.write(f"**Year Built:** {year_built}")
                    st.write(f"**Year Sold:** {year_sold}")
                    st.write(f"**Property Age:** {property_age} years")
                    st.write(f"**Basement:** {'Yes' if basement == 1 else 'No'}")
                
                with col3:
                    st.write(f"**Property Tax:** ${property_tax}/year")
                    st.write(f"**Insurance:** ${insurance}/year")
                    st.write(f"**Popular Area:** {'Yes' if popular == 1 else 'No'}")
                    st.write(f"**Recession Period:** {'Yes' if recession == 1 else 'No'}")
                
                # Similar properties from the dataset
                st.write("### Similar Properties in the Dataset")
                
                # Find similar properties based on bedrooms, bathrooms, and square footage
                similar_props = df[
                    (df['beds'] == beds) &
                    (df['baths'] == baths) &
                    (df['sqft'] >= sqft * 0.8) &
                    (df['sqft'] <= sqft * 1.2)
                ].head(5)
                
                if not similar_props.empty:
                    st.dataframe(similar_props)
                else:
                    st.info("No similar properties found in the dataset.")
                
                # Explanation of factors affecting the price
                st.write("### Factors Influencing the Price Prediction")
                
                # Display feature importances
                feature_names = list(input_features.keys())
                fig = plot_feature_importance(model, feature_names)
                st.pyplot(fig)

if __name__ == "__main__":
    pass  # Streamlit already executes the script