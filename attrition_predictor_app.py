#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Load saved components
@st.cache_resource
def load_model():
    model = joblib.load('best_model.joblib')
    scaler = joblib.load('scaler.joblib')
    threshold = joblib.load('optimal_threshold.joblib')
    feature_names = joblib.load('feature_names.joblib')
    return model, scaler, threshold, feature_names

model, scaler, threshold, feature_names = load_model()

# Define core features and their types
core_features = {
    'Age': 'numeric',
    'DailyRate': 'numeric',
    'DistanceFromHome': 'numeric',
    'Education': 'numeric',
    'EnvironmentSatisfaction': 'numeric',
    'JobInvolvement': 'numeric',
    'JobLevel': 'numeric',
    'JobSatisfaction': 'numeric',
    'MonthlyIncome': 'numeric',
    'NumCompaniesWorked': 'numeric',
    'StockOptionLevel': 'numeric',
    'TotalWorkingYears': 'numeric',
    'YearsAtCompany': 'numeric',
    'YearsInCurrentRole': 'numeric',
    'YearsSinceLastPromotion': 'numeric',
    'YearsWithCurrManager': 'numeric',
    'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
    'Department': ['Sales', 'Research & Development', 'Human Resources'],
    'EducationField': ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'],
    'Gender': ['Male', 'Female'],
    'JobRole': ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'],
    'MaritalStatus': ['Single', 'Married', 'Divorced'],
    'OverTime': ['Yes', 'No']
}

def encode_categorical(input_data):
    return pd.get_dummies(pd.DataFrame([input_data]), columns=[k for k, v in core_features.items() if isinstance(v, list)])

st.title('Employee Attrition Predictor')

st.write('Enter the employee details to predict the likelihood of attrition.')

# Create input fields
user_input = {}
for feature, feature_type in core_features.items():
    if feature_type == 'numeric':
        user_input[feature] = st.number_input(f"Enter {feature}", value=0)
    elif isinstance(feature_type, list):
        user_input[feature] = st.selectbox(f"Select {feature}", options=feature_type)

if st.button('Predict Attrition'):
    # Prepare input data
    input_df = encode_categorical(user_input)
    
    # Ensure all necessary columns are present
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match the model's expected input
    input_df = input_df[feature_names]
    
    # Scale the input
    scaled_input = scaler.transform(input_df)
    
    # Make prediction
    probability = model.predict_proba(scaled_input)[0, 1]
    prediction = 1 if probability >= threshold else 0
    
    st.write(f"Attrition Probability: {probability:.2%}")
    st.write(f"Prediction: {'Attrition' if prediction == 1 else 'No Attrition'}")
    
    # Visualization
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Attrition Probability"},
        gauge = {'axis': {'range': [None, 1]},
                 'steps' : [
                     {'range': [0, 0.5], 'color': "lightgreen"},
                     {'range': [0.5, 0.75], 'color': "yellow"},
                     {'range': [0.75, 1], 'color': "red"}],
                 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold}}))
    
    st.plotly_chart(fig)

st.write("Note: This model provides predictions based on historical data and should be used as a tool to support decision-making, not as a sole determinant.")

