import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import numpy as np
from keras.models import load_model

# Setting custom title and BG color
st.set_page_config(
	page_title = "Rainfall Prediction",
	page_icon = '🔎'
	)

st.markdown(
	"""
	<style>
	.main {
	background-color: #a4f2;
    background-image: "resources/gradient_1.png"
	}
	<style>
	""", unsafe_allow_html=True
)

# load the saved scaler object
scaler = load('model/scaler_4.joblib')

# Load pre-built ANN model
ANN_model = load_model('model/ANN_rf_2_fe_t.h5')

st.title("Rainfall prediction Model")

    # Define the sample data for button 2
sample_data_2 = {
    'temperature': 20.1,
    'cloud_covr': 85,
    'pressure': 1014,
    'wind_speed': 1.2,
    'season_summer': 0,
    'previous_rainfall': 2.0,
    'wind_dir_sin': 0.6427876096865393,
    'wind_dir_cos': 0.766044443118978,
    'soil_moisture': 0.25,
    'evapotrans': 1.2
}

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
        input_pressure = st.slider("Pressure", min_value=990, max_value=1030, step=1)
    with col2:
        input_wind_sp = st.number_input("Wind Speed", min_value=0.0, max_value=8.0, step=0.1)
        input_soil_moisture = st.number_input("Soil Moisture", min_value=0.01, max_value=0.5, step=0.01)
        input_wind_dir = st.slider("Wind Direction", min_value=1.0, max_value=360.0, step=1.0)
    with col3:
        input_evapotrans = st.number_input("Evapotranspiration", min_value=0.0, max_value=6.0, step=0.1)
        input_previous_rain = st.number_input("Previous Day Rainfall", min_value=0.0, max_value=161.0, step=0.1)
        input_cloud = st.slider("Cloud Coverage", min_value=0, max_value=100, step=1)
    
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
    
    # Data preparation before preidction

    # Add a constant of 1 to all values in the "previous_rainfall" variable
    input_data['previous_rainfall'] = input_data['previous_rainfall'] + 1

    # Apply a log transformation to the "previous_rainfall_transformed" variable
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
        if rainfall < 0: 
            rainfall = 0
        #st.subheader(f"The estimated rainfall amount is :green[{rainfall}]")
        #st.info(f"The estimated rainfall amount is :green[{rainfall}]")
        st.info(f"The estimated rainfall amount is,")
        space = "     "
        st.subheader(f":green[{rainfall}]")

    if st.form_submit_button('Reset'):
        st.experimental_rerun()

    if st.form_submit_button('Sample 1'):
        # Fill in input data fields with sample values
        input_temp = 25.0
        input_cloud = 50
        input_pressure = 1013.25
        input_wind_sp = 10.0
        input_wind_dir = 180
        input_soil_moisture = 0.2
        input_evapotrans = 5.0
        input_previous_rain = 0.0
        input_season_summer = 1

        # Update input fields with sample values
        st.session_state['input_temp'] = input_temp
        st.session_state['input_cloud'] = input_cloud
        st.session_state['input_pressure'] = input_pressure
        st.session_state['input_wind_sp'] = input_wind_sp
        st.session_state['input_wind_dir'] = input_wind_dir
        st.session_state['input_soil_moisture'] = input_soil_moisture
        st.session_state['input_evapotrans'] = input_evapotrans
        st.session_state['input_previous_rain'] = input_previous_rain
        st.session_state['input_season_summer'] = input_season_summer