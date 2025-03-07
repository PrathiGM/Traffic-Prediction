import requests
import csv
from datetime import datetime, timedelta

# Define TomTom API credentials
TOMTOM_API_KEY = "5QpZLwcTD1Gz8dJ0O7o1u9vGokfRF1Og"  # Replace with your API key
BASE_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

# Define traffic zones with coordinates (latitude, longitude) and road length (KM)
zones = {
    "Delhi": [
        ("Connaught Place", (28.6315, 77.2167), 4.5),
        ("Saket", (28.5245, 77.2103), 4),
        ("Karol Bagh", (28.6514, 77.1901), 3.5),
    ]
}


# Function to fetch traffic data from TomTom API
def fetch_traffic_data(location, date_time):
    lat, lon = location
    params = {
        "point": f"{lat},{lon}",
        "key": TOMTOM_API_KEY,
        "unit": "KMPH",
        "time": date_time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data for {location}: {response.text}")
        return None

# Function to calculate traffic count
def calculate_traffic_count(current_speed, free_flow_speed, road_length, road_capacity_per_km=200):
    if current_speed == 0 or free_flow_speed == 0:
        return 0  # No traffic if current speed is zero

    # Estimate vehicle density (vehicles per kilometer)
    vehicle_density = (current_speed / free_flow_speed) * road_capacity_per_km

    # Calculate traffic count (vehicles/hour)
    traffic_count = vehicle_density * road_length * current_speed
    return round(traffic_count)

# Prepare the output CSV file
output_file = "traffic_count_data3.csv"
header = ["City", "Zone", "Date Time", "Latitude", "Longitude", "Road Length (KM)", "Current Speed (KMPH)", "Free Flow Speed (KMPH)", "Traffic Count (Vehicles/Hour)"]

with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)

    # Loop through each city and its zones
    for city, city_zones in zones.items():
        for zone_name, coordinates, road_length in city_zones:
            # Fetch data for the past year
            for day_offset in range(0, 365, 7):  # Fetch data weekly for efficiency
                query_date = datetime.now() - timedelta(days=day_offset)

                # Fetch traffic data
                traffic_data = fetch_traffic_data(coordinates, query_date)

                if traffic_data:
                    # Extract relevant information
                    flow_segment_data = traffic_data.get("flowSegmentData", {})
                    current_speed = flow_segment_data.get("currentSpeed", 0)
                    free_flow_speed = flow_segment_data.get("freeFlowSpeed", 0)

                    # Calculate traffic count
                    traffic_count = calculate_traffic_count(current_speed, free_flow_speed, road_length)

                    # Write to CSV
                    writer.writerow([
                        city,
                        zone_name,
                        query_date.strftime("%Y-%m-%d %H:%M:%S"),
                        coordinates[0],
                        coordinates[1],
                        road_length,
                        current_speed,
                        free_flow_speed,
                        traffic_count,
                    ])

print(f"Traffic count data saved to {output_file}")
