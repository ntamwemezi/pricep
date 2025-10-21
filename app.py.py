import streamlit as st
import numpy as np
import joblib

# Load the model
with open(r'C:\Users\pc\Desktop\Practical_cat\pricing.pkl', 'rb') as file:
    model = joblib.load(file)

st.title("✈️ Flight Price Predictor")

# Input fields
airline = st.number_input("Airline (encoded)", min_value=0)
source = st.number_input("Source (encoded)", min_value=0)
destination = st.number_input("Destination (encoded)", min_value=0)
total_stops = st.number_input("Total Stops", min_value=0)
date = st.number_input("Date", min_value=1, max_value=31)
month = st.number_input("Month", min_value=1, max_value=12)
year = st.number_input("Year", min_value=2000)
dep_hour = st.number_input("Departure Hour", min_value=0, max_value=23)
dep_min = st.number_input("Departure Minute", min_value=0, max_value=59)
arr_hour = st.number_input("Arrival Hour", min_value=0, max_value=23)
arr_min = st.number_input("Arrival Minute", min_value=0, max_value=59)
duration = st.number_input("Duration (minutes)", min_value=0)

# Predict button
if st.button("Predict Price"):
    input_data = np.array([airline, source, destination, total_stops, date, month, year,
                           dep_hour, dep_min, arr_hour, arr_min, duration])
    input_data = input_data.reshape(1, -1)
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: {prediction[0]:.2f} RWF")

# Run the app
# if __name__ == '__main__':
 #   main()
   
