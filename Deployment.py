#!/usr/bin/env python
# coding: utf-8

# In[4]:


# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model artifacts
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Customer Cluster Predictor")
st.title("🧠 Customer Segmentation Dashboard")

st.markdown("Enter customer details to predict the cluster they belong to.")

# Sidebar inputs
st.sidebar.header("Customer Details")

age = st.sidebar.slider("Age", 18, 90, 35)
income = st.sidebar.number_input("Income", 1000, 200000, 40000, step=1000)
kidhome = st.sidebar.selectbox("Kids at Home", [0, 1, 2])
teenhome = st.sidebar.selectbox("Teens at Home", [0, 1, 2])
recency = st.sidebar.slider("Recency (days)", 0, 100, 30)
mntwines = st.sidebar.slider("Amount Spent on Wine", 0, 1000, 150)
mntmeat = st.sidebar.slider("Amount Spent on Meat", 0, 1000, 200)
mntgold = st.sidebar.slider("Amount Spent on Gold", 0, 1000, 50)
webvisits = st.sidebar.slider("Web Visits per Month", 0, 20, 5)

# Categorical features
education = st.sidebar.selectbox("Education Level", ['Graduation', 'PhD', 'Master', 'Basic'])
marital_status = st.sidebar.selectbox("Marital Status", ['Single', 'Married', 'Together', 'Divorced', 'Widow'])

# Base input dict
input_data = {
    'Year_Birth': [2024 - age],
    'Income': [income],
    'Kidhome': [kidhome],
    'Teenhome': [teenhome],
    'Recency': [recency],
    'MntWines': [mntwines],
    'MntMeatProducts': [mntmeat],
    'MntGoldProds': [mntgold],
    'NumWebVisitsMonth': [webvisits],
    'Education_' + education: [1],
    'Marital_Status_' + marital_status: [1],
}

# Create a DataFrame with all zeros for every feature in training
input_df = pd.DataFrame(columns=feature_names)
input_df.loc[0] = 0  # Fill first row with zeros

# Fill real values into appropriate columns
for key, value in input_data.items():
    if key in input_df.columns:
        input_df[key] = value

# Scale the input
scaled_input = scaler.transform(input_df)

# Predict cluster
if st.button("Predict Cluster"):
    cluster = model.predict(scaled_input)[0]
    st.success(f"🎯 This customer belongs to **Cluster {cluster}**")

    st.markdown(f"""
    ### 🔍 Cluster {cluster} Characteristics
    - Based on your inputs, this customer shares similar attributes with others in Cluster {cluster}.
    - Use this info for targeted marketing, offers, or personalization.
    """)


# In[ ]:




