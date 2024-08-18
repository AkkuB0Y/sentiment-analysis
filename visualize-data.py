import json
import matplotlib.pyplot as plt

# Load the processed sentiment data from the JSON file
with open('sentiment-results/sentiment_results_1.json', 'r') as f:
    sentiment_data = json.load(f)

# Initialize lists to hold the sentiment scores
negative_scores = []
neutral_scores = []
positive_scores = []

# Extract the sentiment scores for each entry
for entry in sentiment_data:
    negative_scores.append(entry['Negative'])
    neutral_scores.append(entry['Neutral'])
    positive_scores.append(entry['Positive'])

# Calculate the average sentiment scores
avg_negative = sum(negative_scores) / len(negative_scores)
avg_neutral = sum(neutral_scores) / len(neutral_scores)
avg_positive = sum(positive_scores) / len(positive_scores)

# Data to plot
labels = ['Negative', 'Neutral', 'Positive']
avg_scores = [avg_negative, avg_neutral, avg_positive]
colors = ['red', 'yellow', 'green']

# Create a bar chart
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
bars = plt.bar(labels, avg_scores, color=colors)
plt.bar(labels, avg_scores, color=colors)
plt.title('Average Sentiment Scores')
plt.xlabel('Sentiment')
plt.ylabel('Average Score')

# Label each bar with the average value
for bar, score in zip(bars, avg_scores):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(score, 2), va='bottom', ha='center')


# Create a pie chart
plt.subplot(1, 2, 2)
plt.pie(avg_scores, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Distribution')

# Show the plot
plt.tight_layout()
plt.show()
