#NewsSentimentAnalysisforStocks

This project fetches news articles related to a specific stock and analyzes their sentiment using both VADER and FinBERT sentiment analysis models. The sentiment scores are then normalized and classified into ratings.

Features
Fetches news articles using the NewsAPI.
Preprocesses text by removing special characters, tokenizing, removing stopwords, and lemmatizing.
Analyzes sentiment using VADER and FinBERT models.
Normalizes sentiment scores to a scale of 1 to 100.
Classifies sentiment scores into "Bad", "Neutral", or "Good" ratings.
