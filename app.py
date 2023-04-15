# Import required libraries for deployment
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import numpy as np
from keras.models import load_model
import tensorflow

# Setting custom title and BG color
st.set_page_config(
	page_title = "Rainfall Prediction",
	page_icon = '🌧'
	)
st.markdown(
	"""
	<style>
	.main {
	background-color: #a4f2 ;
	}
	<style>
	""", unsafe_allow_html=True)

# Display banner image
st.image('resources/banner.png')

# load the saved scaler object
scaler = load('model/scaler_4.joblib')

# Load pre-built ANN model
ANN_model = load_model('model/ANN_rf_2_fe_t.h5')

# Collect data to predict
st.write("Fill in the input features to predict the rainfall.")
with st.form(key="values"):
    col1, col2, col3 = st.columns(3)
    with col1:
        input_temp = st.number_input("Temperature", min_value=0.0, max_value=30.0, step=0.1)
        input_season_summer = st.selectbox("Season", ["Summer", "Rainy", "Winter"])
        if input_season_summer == "Summer":
            input_season_summer = int(1)
        else:
            input_season_summer = int(0)
        input_pressure = st.slider("Pressure", min_value=995.0, max_value=1030.0, step=0.1)
    with col2:
        input_wind_sp = st.number_input("Wind Speed", min_value=0.0, max_value=12.0, step=0.1)
        input_soil_moisture = st.number_input("Soil Moisture", min_value=0.00, max_value=0.5, step=0.01)
        input_wind_dir = st.slider("Wind Direction", min_value=1.0, max_value=360.0, step=0.1)
    with col3:
        input_evapotrans = st.number_input("Evapotranspiration", min_value=0.0, max_value=10.0, step=0.1)
        input_previous_rain = st.number_input("Previous Day Rainfall", min_value=0.0, max_value=180.0, step=0.1)
        input_cloud = st.slider("Cloud Coverage", min_value=0.0, max_value=100.0, step=0.1)
    
    input_data = pd.DataFrame({'temperature' : [input_temp],
                               'cloud_covr' : [input_cloud], 
                               'pressure': [input_pressure], 
                               'wind_speed' : [input_wind_sp],
                               'season_summer' : [input_season_summer], 
                               'previous_rainfall' : [input_previous_rain], 
                               'wind_dir_sin' : [input_wind_dir],
                               'wind_dir_cos' : [input_wind_dir],
                               'soil_moisture' : [input_soil_moisture],
                               'evapotrans': [input_evapotrans]})
    
# Data pre-processing for input data 
    # Apply log +1 transformaion to previous rainfall data
    input_data['previous_rainfall'] = input_data['previous_rainfall'] + 1
    input_data['previous_rainfall'] = np.log(input_data['previous_rainfall'])

    # Compute sine and cosine of wind direction
    input_data['wind_dir_sin'] = np.sin(np.radians(input_data['wind_dir_sin']))
    input_data['wind_dir_cos'] = np.cos(np.radians(input_data['wind_dir_cos']))

    # use the loaded scaler object to transform new data
    scaled_data = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)

    predict_data = st.form_submit_button(label = 'Predict')
    if predict_data:
        rainfall = ANN_model.predict(scaled_data)
        rainfall = round(float(rainfall), 2)
        # Show 0 if the estimate is less than zero 
        if rainfall < 0: 
            rainfall = 0
        st.info(f"The estimated rainfall amount is :")
        st.subheader(f":green[{rainfall}]")

# Display disclamier 
st.write('Disclaimer: The rainfall prediction model presented here is still in the experimentalstage and is provided for informational purposes only. Please note that the developer of this model is not liable for any direct, indirect, incidental, consequential, or punitive damages arising from the use of the model or its results. Version 1.0, Last updated: April-2023. Developed by Mr.Sai Naing Lynn Oo (TP068393).')