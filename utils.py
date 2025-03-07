import pandas as pd
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("../scrapdata/traffic_with_bing_news.csv")  # Replace with your file path

# Sentiment analysis function with error handling
def analyze_sentiment(text):
    if not isinstance(text, str):
        return 0  # Neutral sentiment for missing or non-string data
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Returns sentiment score between -1 and 1

# Apply sentiment analysis safely
data['Sentiment Score'] = data['News Title'].apply(analyze_sentiment)

# Check for correlation
correlation_matrix = data[['Traffic Count (Vehicles/Hour)', 'Current Speed (KMPH)', 'Sentiment Score']].corr()

# Plot heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Save the enhanced dataset
data.to_csv("traffic_data_with_sentiment.csv", index=False)
print("Sentiment analysis complete. Updated file saved.")
