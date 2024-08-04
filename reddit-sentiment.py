import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# load JSON data from file
with open('scraped-datasets/dataset_reddit-api-scraper_2024-08-04_04-24-15-352.json', 'r') as f:
    data = json.load(f)

# the actual model from huggingface!
roberta = "cardiffnlp/twitter-roberta-base-sentiment-latest"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
labels = ['Negative', 'Neutral', 'Positive']

# functionsto pre-process text
def preprocess_text(text):
    text = text.replace("&lt;!-- SC_OFF --&gt;&lt;div class=\"md\"&gt;&lt;p&gt;", "")
    text = text.replace("&lt;/p&gt;\n\n&lt;p&gt;", "")
    text = text.replace("&lt;/p&gt;\n&lt;/div&gt;&lt;!-- SC_ON --&gt;", "")
    return text

# function using model to analyze sentiment
def analyze_sentiment(text):
    encoded_post = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    output = model(**encoded_post)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    sentiment_scores = {labels[i]: float(scores[i]) for i in range(len(scores))}
    return sentiment_scores

# Process each item in the JSON array
results = []
for item in data:
    if "selftext_html" in item:
        text = item["selftext_html"]
        preprocessed_text = preprocess_text(text)
        sentiment_scores = analyze_sentiment(preprocessed_text)
        results.append(sentiment_scores)

# Optionally save the results to a new JSON file
with open('sentiment_results.json', 'w') as f:
    json.dump(results, f, indent=4)