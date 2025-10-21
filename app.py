import streamlit as st
import numpy as np
import pickle
import os

# Load the model safely
model_path = os.path.join(os.path.dirname(__file__), 'flight_price.pkl')
if not os.path.exists(model_path):
    st.error("❌ Model file 'flight_price.pkl' not found. Please check your GitHub repo.")
    st.stop()

try:
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
except ModuleNotFoundError as e:
    st.error(f"❌ Missing module: {e.name}. Add it to requirements.txt and redeploy.")
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

st.title("✈️ Flight Price Predictor")

# Use a form for cleaner layout
with st.form("flight_form"):
    st.subheader("📋 Enter Flight Details")

    airline = st.number_input("Airline (encoded)", min_value=0, help="Use the encoded value for the airline")
    source = st.number_input("Source (encoded)", min_value=0, help="Use the encoded value for the source airport")
    destination = st.number_input("Destination (encoded)", min_value=0, help="Use the encoded value for the destination airport")
    total_stops = st.number_input("Total Stops", min_value=0, help="Number of stops (0 = nonstop)")
    date = st.number_input("Date", min_value=1, max_value=31)
    month = st.number_input("Month", min_value=1, max_value=12)
    year = st.number_input("Year", min_value=2000)
    dep_hour = st.number_input("Departure Hour", min_value=0, max_value=23)
    dep_min = st.number_input("Departure Minute", min_value=0, max_value=59)
    arr_hour = st.number_input("Arrival Hour", min_value=0, max_value=23)
    arr_min = st.number_input("Arrival Minute", min_value=0, max_value=59)
    duration = st.number_input("Duration (minutes)", min_value=1, help="Total flight duration in minutes")

    submitted = st.form_submit_button("Predict Price")

    if submitted:
        input_data = np.array([
            airline, source, destination, total_stops, date, month, year,
            dep_hour, dep_min, arr_hour, arr_min, duration
        ]).reshape(1, -1)

        try:
            prediction = loaded_model.predict(input_data)
            st.success(f"💰 Predicted Price: {prediction[0]:,.2f} RWF")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Run the app
# if __name__ == '__main__':
 #   main()
   





