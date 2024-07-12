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

# Fit and transform the data for preprocessing
scaler = StandardScaler()
pca = PCA(n_components=10)  # Adjust the number of components if needed
imputer = SimpleImputer(strategy='mean')
ros = RandomOverSampler()

# Fit on training data
x_scaled = scaler.fit_transform(x)
x_pca = pca.fit_transform(x_scaled)
x_imputed = imputer.fit_transform(x_pca)
x_resample_final, y_resample_final = ros.fit_resample(x_imputed, y)

# Train your BaggingClassifier
dt = DecisionTreeClassifier()
bag_clf = BaggingClassifier(base_estimator=dt,
                            n_estimators=100,
                            max_samples=0.6,
                            max_features=0.7)

# Train the bagging classifier on the final preprocessed data
bag_clf.fit(x_resample_final, y_resample_final)

# Define a function to predict bankruptcy based on input features
def predict_bankruptcy(debt_ratio, net_income_to_assets, net_worth_to_assets):
    try:
        # Convert inputs to floats
        debt_ratio = float(debt_ratio)
        net_income_to_assets = float(net_income_to_assets)
        net_worth_to_assets = float(net_worth_to_assets)
        
        # Transform input data using the complete preprocessing pipeline
        input_data = np.array([[debt_ratio, net_income_to_assets, net_worth_to_assets]])
        
        # Add padding to match the number of features if needed
        if input_data.shape[1] < 96:
            input_data = np.pad(input_data, ((0, 0), (0, 96 - input_data.shape[1])), 'constant')
        
        print("Input Data:", input_data)  # Debug log
        
        input_data_scaled = scaler.transform(input_data)  # Transform using trained scaler
        print("Scaled Input Data:", input_data_scaled)  # Debug log
        
        input_data_pca = pca.transform(input_data_scaled)  # Transform using trained PCA
        print("PCA Transformed Input Data:", input_data_pca)  # Debug log
        
        input_data_imputed = imputer.transform(input_data_pca)  # Transform using trained Imputer
        print("Imputed Input Data:", input_data_imputed)  # Debug log

        # Predict bankruptcy based on input features
        prediction = bag_clf.predict(input_data_imputed)
        print("Prediction:", prediction)  # Debug log
        
        return prediction[0]
    except Exception as e:
        # Print the error message for debugging
        print(f"An error occurred during prediction: {e}")
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
