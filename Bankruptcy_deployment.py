import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import streamlit as st

# Read the dataset
df = pd.read_csv("Bankruptcy Prediction (1).csv")

# Assuming x and y are your preprocessed features and target
x = df.drop("Bankrupt?", axis=1)
y = df["Bankrupt?"]

# Preprocess the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Oversample the minority class
ros = RandomOverSampler()
x_resample, y_resample = ros.fit_resample(x_scaled, y)

# Train the BaggingClassifier
dt = DecisionTreeClassifier()
bag_clf = BaggingClassifier(base_estimator=dt, n_estimators=100)
bag_clf.fit(x_resample, y_resample)

# Define a function to predict bankruptcy based on input features
def predict_bankruptcy(debt_ratio, net_income_to_assets, net_worth_to_assets):
    try:
        # Convert inputs to floats
        debt_ratio = float(debt_ratio)
        net_income_to_assets = float(net_income_to_assets)
        net_worth_to_assets = float(net_worth_to_assets)

        # Create a 2D array from the input values
        input_data = np.array([[debt_ratio, net_income_to_assets, net_worth_to_assets]])

        # Scale the input data
        scaler_refit = StandardScaler()
        input_data_scaled = scaler_refit.fit_transform(input_data)

        print("Input data shape:", input_data_scaled.shape)
        print("Input data dtype:", input_data_scaled.dtype)

        # Predict bankruptcy based on input features
        prediction = bag_clf.predict(input_data_scaled)
        return prediction[0]
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    except TypeError as te:
        print(f"TypeError: {te}")
        return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

# Streamlit UI
st.title("Bankruptcy Prediction")

# Input fields for user
debt_ratio = st.text_input("Enter Debt Ratio:")
net_income_to_assets = st.text_input("Enter Net Income to Total Assets:")
net_worth_to_assets = st.text_input("Enter Net Worth to Total Assets:")

# Predict button
if st.button("Predict"):
    try:
        # Validate input fields
        if debt_ratio.strip() and net_income_to_assets.strip() and net_worth_to_assets.strip():
            # Check if inputs are valid numerical values
            debt_ratio = float(debt_ratio)
            net_income_to_assets = float(net_income_to_assets)
            net_worth_to_assets = float(net_worth_to_assets)

            # Perform prediction
            prediction = predict_bankruptcy(debt_ratio, net_income_to_assets, net_worth_to_assets)
            if prediction is not None:
                st.write("Prediction:", "Bankrupt" if prediction == 1 else "Not Bankrupt")
            else:
                st.write("Unable to make a prediction.")
        else:
            st.write("Please fill in all input fields.")
    except ValueError:
        st.write("Please enter valid numerical values for all input fields.")
