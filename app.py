import os
import logging
from dotenv import load_dotenv
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from preprocessing.text_cleaning import remove_special_characters
from preprocessing.tokenization import tokenize_and_remove_stopwords
from preprocessing.lemmatization import lemmatize_tokens
from transformers import pipeline 

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

def preprocess_text(text):
    text = remove_special_characters(text)
    tokens = tokenize_and_remove_stopwords(text)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)

def fetch_news(api_key, query, num_articles=10):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize={num_articles}&apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        return [{"title": article["title"], "description": article["description"], "url": article["url"]} 
                for article in articles if article["title"] and article["description"]]
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching news: {e}")
        return []

def analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']  

def analyze_sentiment_finbert(text):
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    result = sentiment_pipeline(text)
    if result:
        label = result[0]['label']
        if label == "positive":
            return 0.8
        elif label == "negative":
            return -0.8
        else:
            return 0.0

# Normalize Scores to 1–100
def normalize_score(score):
    normalized = (score + 1) * 50  # Map -1 to 1 into 0 to 100
    return max(1, min(100, round(normalized)))  # Ensure score is between 1 and 100

# Classify Ratings
def classify_rating(score):
    if 1 <= score <= 40:
        return "Bad"
    elif 41 <= score <= 60:
        return "Neutral"
    elif 61 <= score <= 100:
        return "Good"


if __name__ == "__main__":
    API_KEY = os.getenv("API_KEY")
    if not API_KEY:
        logging.error("API key not found. Please set it in the .env file.")
        exit(1)

    QUERY = input("Enter company name: ")
    articles = fetch_news(API_KEY, QUERY)
    scores = []
    for article in articles:
        original_text = f"{article['title']} {article['description']}"
        cleaned_text = preprocess_text(original_text)

       
        vader_score = analyze_sentiment_vader(cleaned_text)
        finbert_score = analyze_sentiment_finbert(cleaned_text)

        
        sentiment_score = (vader_score * 0.4) + (finbert_score * 0.6)

        normalized_score = normalize_score(sentiment_score)
        rating = classify_rating(normalized_score)
        scores.append(normalized_score)

        logging.info(f"Article: {article['title']}\nRating: {rating}\n")

 
    if scores:
        average_score = sum(scores) / len(scores)
        overall_rating = classify_rating(average_score)
        logging.info(f"Average Sentiment Score: {average_score:.2f}")
        logging.info(f"Overall Rating: {overall_rating}")
