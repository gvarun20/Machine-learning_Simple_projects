# -*- coding: utf-8 -*-
"""
Created on Sat Mar 9 19:46:33 2024

@author: varung
"""

import numpy as np
import pandas as pd
from decimal import Decimal
from sklearn.preprocessing import LabelEncoder
import pickle
import streamlit as st

st.set_page_config(
    page_title="Flight Price Prediction System",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the Saved Model
predictor = pickle.load(open('predictor_model_varun.sav', 'rb'))

# Function for Prediction
def flight_price(predictive_input):
    predictive_input_array = np.asarray(predictive_input).reshape(1, -1)
    prediction = predictor.predict(predictive_input_array)
    print('Prediction: ', prediction)
    return float(np.round(prediction, 2))

# Main Function
def main():
    # Title
    st.title('Flight Price Prediction System')

    df = pd.read_csv(r"C:\Users\Hi\Dataset1\flight_Dataset.csv")

    airline_options_list = df.airline.unique().tolist()
    le = LabelEncoder()
    df.airline = le.fit_transform(df.airline)
    airline_options_encoded = df.airline.unique().tolist()
    airline_options = {airline_options_list[i]: airline_options_encoded[i] for i in range(len(airline_options_list))}
    
    airline_text = st.selectbox("Airline", list(airline_options.keys()))
    airline = airline_options[airline_text]

    flight_options_list = df.flight.unique().tolist()
    le = LabelEncoder()
    df.flight = le.fit_transform(df.flight)
    flight_options_encoded = df.flight.unique().tolist()
    flight_options = {flight_options_list[i]: flight_options_encoded[i] for i in range(len(flight_options_list))}
    
    flight_text = st.selectbox("Flight", list(flight_options.keys()))
    flight = flight_options[flight_text]

    source_city_options_list = df.source_city.unique().tolist()
    le = LabelEncoder()
    df.source_city = le.fit_transform(df.source_city)
    source_city_options_encoded = df.source_city.unique().tolist()
    source_city_options = {source_city_options_list[i]: source_city_options_encoded[i] for i in range(len(source_city_options_list))}
    
    source_city_text = st.selectbox("Source", list(source_city_options.keys()))
    source_city = source_city_options[source_city_text]

    departure_time_options_list = df.departure_time.unique().tolist()
    le = LabelEncoder()
    df.departure_time = le.fit_transform(df.departure_time)
    departure_time_options_encoded = df.departure_time.unique().tolist()
    departure_time_options = {departure_time_options_list[i]: departure_time_options_encoded[i] for i in range(len(departure_time_options_list))}
    
    departure_time_text = st.selectbox("Departure Time", list(departure_time_options.keys()))
    departure_time = departure_time_options[departure_time_text]

    arrival_time_options_list = df.arrival_time.unique().tolist()
    le = LabelEncoder()
    df.arrival_time = le.fit_transform(df.arrival_time)
    arrival_time_options_encoded = df.arrival_time.unique().tolist()
    arrival_time_options = {arrival_time_options_list[i]: arrival_time_options_encoded[i] for i in range(len(arrival_time_options_list))}
    
    arrival_time_text = st.selectbox("Arrival Time", list(arrival_time_options.keys()))
    arrival_time = arrival_time_options[arrival_time_text]
    
    destination_city_options_list = df.destination_city.unique().tolist()
    le = LabelEncoder()
    df.destination_city = le.fit_transform(df.destination_city)
    destination_city_options_encoded = df.destination_city.unique().tolist()
    destination_city_options = {destination_city_options_list[i]: destination_city_options_encoded[i] for i in range(len(destination_city_options_list))}
    
    destination_city_text = st.selectbox("Destination", list(destination_city_options.keys()))
    destination_city = destination_city_options[destination_city_text]

    class_options_list = df['class'].unique().tolist()
    le = LabelEncoder()
    df['class'] = le.fit_transform(df['class'])
    class_options_encoded = df['class'].unique().tolist()
    class_options = {class_options_list[i]: class_options_encoded[i] for i in range(len(class_options_list))}
    
    class_text = st.selectbox("Class", list(class_options.keys()))
    class_f = class_options[class_text]

    duration = st.slider("Duration", 0, 30, 0)    

    # #Prediction
    prediction = None

    # Prediction Button
    if st.button('Price'):
        prediction = flight_price([airline, flight, source_city, departure_time, arrival_time, destination_city, class_f, duration])
    
    # Display prediction
    if prediction is not None:
        st.success(f'Predicted Price: ${float(prediction):.2f}')


if __name__ == '__main__':
    main()
