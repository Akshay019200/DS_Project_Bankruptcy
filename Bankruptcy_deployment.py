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
pca = PCA(n_components=10)  # Use the same number of components as during training
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
    # Transform input data using the complete preprocessing pipeline
    input_data = np.array([[debt_ratio, net_income_to_assets, net_worth_to_assets] + [0] * 7])  # Pad with zeros to match 10 features
    input_data_scaled = scaler.transform(input_data)  # Transform using trained scaler
    input_data_pca = pca.transform(input_data_scaled)  # Transform using trained PCA
    input_data_imputed = imputer.transform(input_data_pca)  # Transform using trained Imputer

    # Predict bankruptcy based on input features
    prediction = bag_clf.predict(input_data_imputed)
    return prediction[0]

# Streamlit UI
st.title("Bankruptcy Prediction")

# Input fields for user
debt_ratio = st.text_input("Enter Debt Ratio:")
net_income_to_assets = st.text_input("Enter Net Income to Total Assets:")
net_worth_to_assets = st.text_input("Enter Net Worth to Total Assets:")

# Predict button
if st.button("Predict"):
    if debt_ratio and net_income_to_assets and net_worth_to_assets:
        try:
            debt_ratio = float(debt_ratio)
            net_income_to_assets = float(net_income_to_assets)
            net_worth_to_assets = float(net_worth_to_assets)
            prediction = predict_bankruptcy(debt_ratio, net_income_to_assets, net_worth_to_assets)
            st.write("Prediction:", "Bankrupt" if prediction == 1 else "Not Bankrupt")
        except ValueError:
            st.write("Please enter valid numerical values for all input fields.")
    else:
        st.write("Please fill in all input fields.")
