Energy Demand Forecasting using XGBoost
Project Overview

This project implements a Machine Learning-based Energy Demand Forecasting system using Time-Series Analysis and XGBoost to predict future electricity demand from historical energy consumption data.

The system analyzes Italy’s 2016 hourly electricity usage dataset and learns temporal and seasonal energy consumption patterns using lag-based feature engineering and regression modeling techniques.

Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-learn
XGBoost
Machine Learning Concepts Used
Time-Series Forecasting
Feature Engineering
Lag Features
Regression Modeling
Predictive Analytics
Data Visualization
Model Evaluation
Dataset Information

The dataset contains hourly electricity consumption data for Italy during 2016.

Dataset Columns
Column	Description
utc_timestamp	Date and time
IT_load_new	Electricity demand/load
IT_solar_generation	Solar energy generation

Project Workflow

Historical Energy Dataset
        ↓
Load Dataset using Pandas
        ↓
Data Cleaning & Preprocessing
        ↓
Datetime Conversion
        ↓
Visualization of Energy Demand
        ↓
Feature Engineering
        ↓
Create Lag Features
        ↓
Train-Test Split
        ↓
Train XGBoost Model
        ↓
Predict Future Energy Demand
        ↓
Evaluate using MAE
        ↓
Visualize Actual vs Predicted Demand
