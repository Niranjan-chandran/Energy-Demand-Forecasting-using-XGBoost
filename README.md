# Energy Demand Forecasting using XGBoost

A Machine Learning-based Energy Demand Forecasting system using Time-Series Analysis and XGBoost to predict future electricity demand from historical energy consumption data.

---

# Features

- Time-series forecasting
- Energy demand prediction
- Lag-based feature engineering
- XGBoost regression model
- Actual vs predicted visualization
- MAE-based evaluation

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost

---

# Dataset

The dataset contains hourly electricity consumption data for Italy during 2016.

Columns:
- `utc_timestamp`
- `IT_load_new`
- `IT_solar_generation`

---

# Workflow

```text
Load Dataset
↓
Data Preprocessing
↓
Feature Engineering
↓
Train XGBoost Model
↓
Predict Future Energy Demand
↓
Evaluate Model
↓
Visualize Results
