
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the retrained model and the model columns
model1 = pickle.load(open('best_rf_model.pkl', 'rb'))
model2 = pickle.load(open('best_xgb_model.pkl','rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))
feature_scaler = pickle.load(open('feature_scaler.pkl', 'rb'))  # Load the feature scaler
target_scaler = pickle.load(open('target_scaler.pkl', 'rb'))    # Load the target scaler

# Define the Streamlit app
st.title("Car Price Prediction App")

# Collecting user input for the features
st.header("Input the Details of the Car")

# Choose the model
option = st.sidebar.selectbox('Select the Model',['Linear Regressor','XGB Regressor'])

# Collecting user inputs
brand = st.selectbox("Select the Brand", [ "Chevrolet", "Ford","Honda","Hyundai", "Mahindra", "Maruti","Renault", "Tata", "Toyota", "Volkswagen", "Others"])
seller_type = st.selectbox("Seller Type", ["Individual", "Dealer or Trustmark Dealer"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
owner = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner & Above"])
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG or LPG"])
year = st.slider("Year of Manufacture", min_value=1995, max_value=2020, value=2005)
km_driven = st.slider("Kilometers Driven", min_value=0, max_value=175000, value=10000, step=100)

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'brand': [brand],
    'year': [year],
    'km_driven': [km_driven],
    'seller_type': [seller_type],
    'transmission': [transmission],
    'owner': [owner],
    'fuel': [fuel]
})

# Dummies for brand
input_data_encoded = pd.get_dummies(input_data, columns=['brand'])
input_data_encoded = input_data.reindex(columns=model_columns, fill_value=0)

# Manual Mapping
owner_mapping = {"First Owner": 0, "Second Owner": 1, "Third & Above Owner": 2}
seller_type_mapping = {"Dealer or Trustmark Dealer": 0, "Individual": 1}
fuel_mapping = {"Diesel": 0, "Petrol": 1, "CNG or LPG" : 2}
transmission_mapping = {"Automatic": 0, "Manual": 1}

# Apply the mappings
input_data_encoded['owner'] = input_data['owner'].map(owner_mapping)
input_data_encoded['seller_type'] = input_data['seller_type'].map(seller_type_mapping)
input_data_encoded['fuel'] = input_data['fuel'].map(fuel_mapping)
input_data_encoded['transmission'] = input_data['transmission'].map(transmission_mapping)


# Scale the feature data
input_data_encoded[['km_driven']] = feature_scaler.transform(input_data_encoded[['km_driven']])

# Predict the selling price using the model
prediction1_scaled = model1.predict(input_data_encoded)
prediction2_scaled = model2.predict(input_data_encoded)

# Reverse scaling of the predicted value
prediction1_unscaled = target_scaler.inverse_transform(prediction1_scaled.reshape(-1, 1))
prediction2_unscaled = target_scaler.inverse_transform(prediction2_scaled.reshape(-1, 1))

# Display the prediction
st.subheader("Predicted Selling Price")
if st.button('Predict'):
    if option=="Linear Regressor":
        st.write(f"The estimated selling price of the car is: ₹ {prediction1_unscaled[0, 0]:,.2f}")
    else:
        st.write(f"The estimated selling price of the car is: ₹ {prediction2_unscaled[0, 0]:,.2f}")



