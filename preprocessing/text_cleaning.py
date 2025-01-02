import re

def remove_special_characters(text):
    """
    Removes special characters from the input text and converts it to lowercase.
    """
    return re.sub(r"[^a-zA-Z\s]", "", text.lower())
