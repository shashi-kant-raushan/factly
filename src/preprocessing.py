"""
Text Preprocessing Pipeline for Fake News Detection.

Provides specialized, dual preprocessing pathways:
1. Traditional ML Pipeline (for TF-IDF + Logistic Regression / Random Forest):
   - Lowercasing, HTML tag stripping, URL filtering, punctuation cleaning,
     stopword removal, and WordNet lemmatization.
2. Transformer Pipeline (for DistilBERT):
   - Minimalist cleaning preserving casing, punctuation, contextual markers,
     and natural sentence boundaries critical for attention mechanisms.
"""

import re
import html
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Union
import pandas as pd


# Download NLTK resources safely if not already present
try:
    _STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    _STOPWORDS = set(stopwords.words("english"))

try:
    _LEMMATIZER = WordNetLemmatizer()
    _ = _LEMMATIZER.lemmatize("testing")
except LookupError:
    nltk.download("wordnet", quiet=True)
    _LEMMATIZER = WordNetLemmatizer()


# Regex patterns precompiled for performance
RE_HTML = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_SPECIAL_CHARS = re.compile(r"[^a-zA-Z\s]")
RE_EXTRA_WHITESPACE = re.compile(r"\s+")


def clean_html_and_urls(text: str) -> str:
    """
    Strips raw HTML tags, unescapes HTML entities, and removes URLs.
    """
    if not isinstance(text, str):
        return ""
    # Unescape HTML entities (e.g., &amp; -> &)
    text = html.unescape(text)
    # Remove HTML tags
    text = RE_HTML.sub(" ", text)
    # Remove web URLs
    text = RE_URL.sub(" ", text)
    return text


def preprocess_traditional(text: str, remove_stopwords: bool = False, lemmatize: bool = True) -> str:
    """
    Traditional preprocessing tuned for news classification.
    It keeps informative words and negation terms instead of aggressively removing them,
    which helps TF-IDF models catch suspicious wording patterns in misinformation articles.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Strip HTML and URLs
    text = clean_html_and_urls(text)

    # Convert to lowercase
    text = text.lower()

    # Remove numbers, special symbols, and punctuation
    text = RE_SPECIAL_CHARS.sub(" ", text)

    # Tokenize by whitespace
    tokens = text.split()

    # Stopword removal
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

    # Lemmatization
    if lemmatize:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    cleaned_text = " ".join(tokens)
    return cleaned_text


def preprocess_transformer(text: str) -> str:
    """
    Minimalist preprocessing tailored for Pretrained Transformers (DistilBERT):
    Preserves case, punctuation, sentence structure, and vocabulary nuances.
    Only eliminates malformed HTML entities, URLs, and noisy whitespace.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Clean HTML and raw URLs
    text = clean_html_and_urls(text)

    # Normalize excessive newlines and whitespace into single spaces
    text = RE_EXTRA_WHITESPACE.sub(" ", text).strip()

    return text


def prepare_dataset_features(df: pd.DataFrame, text_col: str = "combined_text", for_transformer: bool = False) -> List[str]:
    """
    Vectorized batch preprocessing across a pandas DataFrame column.
    """
    if for_transformer:
        return df[text_col].apply(preprocess_transformer).tolist()
    else:
        return df[text_col].apply(preprocess_traditional).tolist()


if __name__ == "__main__":
    sample_text = (
        "<b>BREAKING NEWS:</b> Click here http://fakenews.com/shocking to see the secret report! "
        "Scientists were shocked by this unprecedented discovery in 2026."
    )
    print("Original:", sample_text)
    print("\n--- Traditional Preprocessing ---")
    print(preprocess_traditional(sample_text))
    print("\n--- Transformer Preprocessing ---")
    print(preprocess_transformer(sample_text))
