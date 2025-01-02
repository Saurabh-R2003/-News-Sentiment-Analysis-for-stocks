from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("wordnet")

def lemmatize_tokens(tokens):
    """
    Lemmatizes the input tokens to their base forms.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]
