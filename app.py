import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from preprocessing.text_cleaning import remove_special_characters
from preprocessing.tokenization import tokenize_and_remove_stopwords
from preprocessing.lemmatization import lemmatize_tokens
from transformers import pipeline 

def preprocess_text(text):
    text = remove_special_characters(text)
    tokens = tokenize_and_remove_stopwords(text)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)

def fetch_news(api_key, query, num_articles=10):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize={num_articles}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        return [{"title": article["title"], "description": article["description"], "url": article["url"]} 
                for article in articles if article["title"] and article["description"]]
    else:
        print("Error fetching news:", response.status_code, response.text)
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
    
    API_KEY = "d810af79c6324f16957d749f329b51f0"
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

        print(f"Original Article Title: {article['title']}")
        print(f"Summary: {article['description']}")
        print(f"Link: {article['url']}")
        print(f"Cleaned Text: {cleaned_text}")
        print(f"Sentiment Score: {sentiment_score:.2f}")
        print(f"Normalized Score: {normalized_score}")
        print(f"Rating: {rating}")
        print("-" * 50)

 
    if scores:
        average_score = sum(scores) / len(scores)
        overall_rating = classify_rating(average_score)
        print(f"Average Sentiment Score: {average_score:.2f}")
        print(f"Overall Rating: {overall_rating}")