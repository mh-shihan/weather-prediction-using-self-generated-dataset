import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Settings
# --------------------------------------------------

NUMBER_OF_ROWS = 10_000

np.random.seed(42)


# --------------------------------------------------
# 2. Generate basic weather variables
# --------------------------------------------------

temperature = np.random.normal(
    loc=28,
    scale=5,
    size=NUMBER_OF_ROWS
)

humidity = np.random.normal(
    loc=65,
    scale=15,
    size=NUMBER_OF_ROWS
)

wind_speed = np.random.gamma(
    shape=2,
    scale=2.5,
    size=NUMBER_OF_ROWS
)

atmospheric_pressure = np.random.normal(
    loc=1010,
    scale=7,
    size=NUMBER_OF_ROWS
)


# --------------------------------------------------
# 3. Generate pollution
# --------------------------------------------------

historical_pollution = np.random.normal(
    loc=60,
    scale=20,
    size=NUMBER_OF_ROWS
)

# Current pollution is somewhat related to historical pollution
pm25 = (
    historical_pollution * 0.6
    + np.random.normal(0, 15, NUMBER_OF_ROWS)
)

pm10 = (
    pm25 * 1.6
    + np.random.normal(0, 15, NUMBER_OF_ROWS)
)


# --------------------------------------------------
# 4. Keep variables within sensible ranges
# --------------------------------------------------

temperature = np.clip(temperature, 10, 45)
humidity = np.clip(humidity, 20, 100)

wind_speed = np.clip(wind_speed, 0, 30)

atmospheric_pressure = np.clip(
    atmospheric_pressure,
    980,
    1030
)

historical_pollution = np.clip(
    historical_pollution,
    5,
    150
)

pm25 = np.clip(pm25, 2, 200)
pm10 = np.clip(pm10, 5, 350)


# --------------------------------------------------
# 5. Calculate scores for each weather condition
# --------------------------------------------------

# Rain:
# Higher humidity and lower atmospheric pressure
# increase the likelihood of rain.

rain_score = (
    0.10 * (humidity - 65)
    - 0.8 * (atmospheric_pressure - 1010)
    - 0.5 * (wind_speed - 5)
)


# Sunny:
# Higher temperature, lower humidity and higher
# atmospheric pressure favor sunny conditions.

sunny_score = (
    0.12 * (temperature - 28)
    - 0.06 * (humidity - 65)
    + 0.7 * (atmospheric_pressure - 1010)
    - 0.01 * (pm25 - 50)
)


# Windy:
# Wind speed is the main factor.

windy_score = (
    1.2 * (wind_speed - 5)
    - 0.2 * (humidity - 65)
)


# Storm:
# Storms are associated with high wind, high humidity
# and low atmospheric pressure.

storm_score = (
    0.07 * (humidity - 65)
    + 1.1 * (wind_speed - 5)
    - 1.3 * (atmospheric_pressure - 1010)
)


# Normal:
# Conditions close to ordinary/average conditions
# receive a higher score.

normal_score = (
    8
    - 0.8 * abs(temperature - 28)
    - 0.12 * abs(humidity - 65)
    - 0.6 * abs(atmospheric_pressure - 1010)
    - 0.3 * abs(wind_speed - 4)
)


# --------------------------------------------------
# 6. Combine scores
# --------------------------------------------------

scores = np.column_stack([
    rain_score,
    sunny_score,
    windy_score,
    storm_score,
    normal_score
])


# Add random noise.
# This prevents the dataset from being perfectly
# deterministic.

scores += np.random.normal(
    0,
    2,
    scores.shape
)


# --------------------------------------------------
# 7. Choose the weather condition
# --------------------------------------------------

weather_conditions = [
    "Rainy",
    "Sunny",
    "Windy",
    "Storm",
    "Normal"
]

weather = np.array(weather_conditions)[
    np.argmax(scores, axis=1)
]


# --------------------------------------------------
# 8. Create the DataFrame
# --------------------------------------------------

df = pd.DataFrame({
    "PM2.5": pm25,
    "PM10": pm10,
    "temperature": temperature,
    "humidity": humidity,
    "wind_speed": wind_speed,
    "atmospheric_pressure": atmospheric_pressure,
    "historical_pollution": historical_pollution,
    "weather": weather
})


# --------------------------------------------------
# 9. Save CSV
# --------------------------------------------------

df.to_csv(
    "weather_dataset.csv",
    index=False
)


# --------------------------------------------------
# 10. Display some information
# --------------------------------------------------

print(df.head())
print()
print(df.shape)
print()
print(df["weather"].value_counts())
