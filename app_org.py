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
	background-color: #d2e4f1;
	}
	<style>
	""", unsafe_allow_html=True
)

# load the saved scaler object
scaler = load('model/scaler_4.joblib')

# Load pre-built ANN model
ANN_model = load_model('model/ANN_rf_2_fe_t.h5')

st.title("Rainfall prediction Model")

# Collect data to predict
with st.form(key='values'):   
    input_temp = st.number_input('Temperature')
    input_cloud = st.slider('Cloud Coverage')
    input_pressure = st.number_input('Pressure')
    input_wind_sp = st.number_input('Wind Speed')
    input_wind_dir = st.slider('Wind direction', min_value=1, max_value = 360)
    input_soil_moisture = st.number_input('Soil Moisture')
    input_evapotrans = st.number_input('Evapotranspiration')
    input_previous_rain = st.number_input('Previous day rainfall')
    input_season_summer = st.selectbox('Season', ['Summer', 'Rainy', 'Winter'])
    if input_season_summer == 'Summer':
        input_season_summer = int(1)
    else: 
       input_season_summer = int(0)

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
        st.success(rainfall)