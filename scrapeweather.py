# import requests
# import pandas as pd
# from datetime import datetime
#
# # OpenWeatherMap API key
# API_KEY = "4b3120747b9ef1f222316cf9cf47bd7a"
#
# # Function to fetch weather data based on latitude and longitude
# def fetch_weather(lat, lon):
#     url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = response.json()
#         return {
#             "Temperature (°C)": data["main"]["temp"],
#             "Weather Condition": data["weather"][0]["description"],
#             "Weather Timestamp": datetime.now().isoformat(),
#         }
#     else:
#         print(f"Failed to fetch weather for lat={lat}, lon={lon}. Status code: {response.status_code}")
#         return {"Temperature (°C)": None, "Weather Condition": None, "Weather Timestamp": None}
#
# # Input CSV path
# input_csv = "traffic_count_data.csv"  # Replace with your actual file path
# output_csv = "traffic_with_weather_data.csv"  # Path to save the updated data
#
# # Read the existing traffic data CSV
# traffic_df = pd.read_csv(input_csv)
#
# # List to store weather data for each row
# weather_data = []
#
# # Fetch weather data for each row in the traffic data
# for index, row in traffic_df.iterrows():
#     lat = row["Latitude"]
#     lon = row["Longitude"]
#     print(f"Fetching weather for {lat}, {lon} (Row {index})...")
#     weather = fetch_weather(lat, lon)
#     weather_data.append(weather)
#
# # Create a DataFrame for the fetched weather data
# weather_df = pd.DataFrame(weather_data)
#
# # Combine the traffic data with the weather data
# combined_df = pd.concat([traffic_df, weather_df], axis=1)
#
# # Save the combined data to a new CSV
# combined_df.to_csv(output_csv, index=False)
# print(f"Updated data with weather saved to {output_csv}")



import requests
import pandas as pd
from datetime import datetime

# OpenWeatherMap API key
API_KEY = "4b3120747b9ef1f222316cf9cf47bd7a"

# Function to fetch historical weather data based on latitude, longitude, and timestamp
def fetch_historical_weather(lat, lon, dt):
    url = f"http://api.openweathermap.org/data/3.0/onecall/timemachine"
    params = {
        "lat": lat,
        "lon": lon,
        "dt": int(dt.timestamp()),  # Convert datetime to Unix timestamp
        "appid": API_KEY,
        "units": "metric",
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        print(data)
        return {
            "Temperature (°C)": data["data"][0]["temp"],
            "Weather Condition": data["data"][0]["weather"][0]["description"],
            "Weather Timestamp": datetime.fromtimestamp(data["data"][0]["dt"]).isoformat(),
        }
    else:
        print(f"Failed to fetch weather for lat={lat}, lon={lon} at dt={dt}. Status code: {response.status_code}")
        return {"Temperature (°C)": None, "Weather Condition": None, "Weather Timestamp": None}

# Input CSV path
input_csv = "traffic_count_data.csv"  # Replace with your actual file path
output_csv = "traffic_with_historical_weather_data.csv"  # Path to save the updated data

# Read the existing traffic data CSV
traffic_df = pd.read_csv(input_csv)

# Ensure Date Time is parsed correctly
traffic_df["Date Time"] = pd.to_datetime(traffic_df["Date Time"])

# List to store weather data for each row
weather_data = []

# Fetch weather data for each row in the traffic data
for index, row in traffic_df.iterrows():
    lat = row["Latitude"]
    lon = row["Longitude"]
    date_time = row["Date Time"]
    print(f"Fetching historical weather for {lat}, {lon} at {date_time} (Row {index})...")
    weather = fetch_historical_weather(lat, lon, date_time)
    weather_data.append(weather)

# Create a DataFrame for the fetched weather data
weather_df = pd.DataFrame(weather_data)

# Combine the traffic data with the weather data
combined_df = pd.concat([traffic_df, weather_df], axis=1)

# Save the combined data to a new CSV
combined_df.to_csv(output_csv, index=False)
print(f"Updated data with historical weather saved to {output_csv}")
