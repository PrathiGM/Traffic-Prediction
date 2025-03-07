# import requests
# import pandas as pd
# from datetime import datetime, timedelta
#
# # Bing News Search API Key
# BING_API_KEY = "246ef77fe82f446a9505d09dd669500f"  # Replace with your Bing News API key
#
# # Input CSV path
# input_csv = "traffic_count_data.csv"  # Replace with your actual file path
# output_csv = "bing_news_with_traffic_data.csv"  # Path to save the updated data
#
# # Read the CSV file
# traffic_df = pd.read_csv(input_csv)
# traffic_df["Date Time"] = pd.to_datetime(traffic_df["Date Time"])  # Ensure correct date parsing
#
#
# # Function to fetch news using Bing News Search API
# def fetch_bing_news(city, date_time):
#     # Format date for Bing News API
#     from_date = (date_time - timedelta(days=1)).strftime("%Y-%m-%d")
#     to_date = (date_time + timedelta(days=1)).strftime("%Y-%m-%d")
#
#     # Bing News API endpoint
#     url = f"https://api.bing.microsoft.com/v7.0/news/search"
#     headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
#     params = {
#         "q": f"Traffic {city}",
#         "count": 10,  # Maximum number of results per request
#         "freshness": "Day",
#         "sortBy": "Date",
#         "mkt": "en-IN",  # Market for India
#     }
#
#     # Fetch news
#     response = requests.get(url, headers=headers, params=params)
#     if response.status_code == 200:
#         articles = response.json().get("value", [])
#         news_data = [
#             {
#                 "News Title": article["name"],
#                 "Published At": article["datePublished"],
#                 "Source": article["provider"][0]["name"],
#                 "URL": article["url"],
#             }
#             for article in articles
#         ]
#         return news_data
#     else:
#         print(f"Error fetching news: {response.status_code}, {response.text}")
#         return []
#
#
# # List to store news articles for each row
# all_news_data = []
#
# # Fetch news for each row in the traffic data
# for index, row in traffic_df.iterrows():
#     city = row["City"]
#     date_time = row["Date Time"]
#     print(f"Fetching news for {city} at {date_time} (Row {index})...")
#     news = fetch_bing_news(city, date_time)
#     for article in news:
#         article["City"] = city
#         article["Date Time"] = date_time
#     all_news_data.extend(news)
#
# # Create a DataFrame for the news articles
# news_df = pd.DataFrame(all_news_data)
#
# # Save news to a new CSV
# news_df.to_csv(output_csv, index=False)
# print(f"News articles saved to {output_csv}")


import requests
import pandas as pd
from datetime import datetime, timedelta

# Bing News Search API Key
BING_API_KEY = "246ef77fe82f446a9505d09dd669500f"  # Replace with your Bing News API key

# Input CSV path
input_csv = "traffic_count_data.csv"  # Replace with your actual file path
output_csv = "traffic_with_bing_news.csv"  # Path to save the updated data

# Read the CSV file
traffic_df = pd.read_csv(input_csv)
traffic_df["Date Time"] = pd.to_datetime(traffic_df["Date Time"])  # Ensure correct date parsing


# Function to fetch news using Bing News Search API
def fetch_bing_news(city, date_time):
    # Format date for Bing News API
    from_date = (date_time - timedelta(days=1)).strftime("%Y-%m-%d")
    to_date = (date_time + timedelta(days=1)).strftime("%Y-%m-%d")

    # Bing News API endpoint
    url = f"https://api.bing.microsoft.com/v7.0/news/search"
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {
        "q": f"Traffic {city}",
        "count": 10,  # Maximum number of results per request
        "freshness": "Day",
        "sortBy": "Date",
        "mkt": "en-IN",  # Market for India
    }

    # Fetch news
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        articles = response.json().get("value", [])
        news_data = [
            {
                "News Title": article["name"],
                "Published At": article["datePublished"],
                "Source": article["provider"][0]["name"],
                "URL": article["url"],
            }
            for article in articles
        ]
        return news_data
    else:
        print(f"Error fetching news: {response.status_code}, {response.text}")
        return []


# List to store news articles for each row
all_news_data = []

# Fetch news for each row in the traffic data
for index, row in traffic_df.iterrows():
    city = row["City"]
    date_time = row["Date Time"]
    print(f"Fetching news for {city} at {date_time} (Row {index})...")
    news = fetch_bing_news(city, date_time)
    for article in news:
        article["City"] = city
        article["Date Time"] = date_time
    all_news_data.extend(news)

# Create a DataFrame for the news articles
news_df = pd.DataFrame(all_news_data)

# Merge traffic data with the news data
combined_df = pd.merge(traffic_df, news_df, on=["City", "Date Time"], how="left")

# Save the updated data to a new CSV
combined_df.to_csv(output_csv, index=False)
print(f"Combined data with news saved to {output_csv}")
