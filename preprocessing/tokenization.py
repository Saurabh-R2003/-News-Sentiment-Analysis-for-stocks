from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
nltk.download("punkt")

def tokenize_and_remove_stopwords(text):
    """
    Tokenizes the input text and removes stopwords.
    """
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word not in stop_words]
