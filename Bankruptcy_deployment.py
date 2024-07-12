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

# Preprocessing
scaler = StandardScaler()
pca = PCA(n_components=10)
imputer = SimpleImputer(strategy='mean')
ros = RandomOverSampler()

# Fit on training data
x_scaled = scaler.fit_transform(x)
x_pca = pca.fit_transform(x_scaled)
x_imputed = imputer.fit_transform(x_pca)
x_resample_final, y_resample_final = ros.fit_resample(x_imputed, y)

# Train your BaggingClassifier
dt = DecisionTreeClassifier()
bag_clf = BaggingClassifier(estimator=dt,
                            n_estimators=100,
                            max_samples=0.6,
                            max_features=0.7)
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
        
        # Add padding to match the number of features expected by the scaler
        input_data = np.pad(input_data, ((0, 0), (0, x.shape[1] - input_data.shape[1])), 'constant')
        
        input_data_scaled = scaler.transform(input_data)
        input_data_pca = pca.transform(input_data_scaled)
        input_data_imputed = imputer.transform(input_data_pca)

        # Predicting bankruptcy based on input features
        prediction = bag_clf.predict(input_data_imputed)
        return prediction[0]
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

# Streamlit UI
st.title("Bankruptcy Prediction")

# Input fields for user
debt_ratio = st.text_input("Enter Debt Ratio:")
net_income_to_assets = st.text_input("Enter Net Income to Total Assets:")
net_worth_to_assets = st.text_input("Enter Net Worth to Total Assets:")

# Predict button
if st.button("Predict"):
    if debt_ratio.strip() and net_income_to_assets.strip() and net_worth_to_assets.strip():
        prediction = predict_bankruptcy(debt_ratio, net_income_to_assets, net_worth_to_assets)
        if prediction is not None:
            st.write("Prediction:", "Bankrupt" if prediction == 1 else "Not Bankrupt")
        else:
            st.write("Unable to make a prediction. Please check the input values.")
    else:
        st.write("Please fill in all input fields.")
