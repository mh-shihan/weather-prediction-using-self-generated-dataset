import pickle
import numpy as np
import pandas as pd

pipe = pickle.load(open('weather_prediction_pipeline.pkl', 'rb'))

FEATURES = [
    "PM2.5",
    "PM10",
    "temperature",
    "humidity",
    "wind_speed",
    "atmospheric_pressure",
    "historical_pollution",
]

raw = input("Enter PM2.5, PM10, temperature, humidity, wind_speed, atmospheric_pressure, historical_pollution: ")
values = list(map(float, raw.split(',')))

user_input = pd.DataFrame([values], columns=FEATURES)

prediction = pipe.predict(user_input)
print("Predicted weather:", prediction[0])
