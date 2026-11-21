"""
NLP Pipeline for Skincare and Cosmetics Product Recommender
Student Name: Quách Tú Mỹ
Student ID: 29201153781 (CS467A)
Project Code: MP-A10
"""

import re
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Skincare domain stop words extending standard English stop words
STANDARD_STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are',
    'aren', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both',
    'but', 'by', 'can', 'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few',
    'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers',
    'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its',
    'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of',
    'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
    'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs',
    'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to',
    'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
    'while', 'who', 'whom', 'why', 'with', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves'
}

# Domain-specific sentiment lexicon for skincare and beauty reviews
POSITIVE_LEXICON = {
    'love': 1.0, 'loved': 1.0, 'holy grail': 1.5, 'great': 0.8, 'best': 1.0, 'amazing': 1.0,
    'incredible': 1.0, 'effective': 0.8, 'excellent': 1.0, 'favorite': 1.0, 'perfect': 1.0,
    'wonderful': 0.9, 'fantastic': 1.0, 'staple': 0.8, 'healthy': 0.7, 'good': 0.6,
    'hydrating': 0.8, 'moisturizing': 0.8, 'soothing': 0.9, 'calming': 0.9, 'gentle': 0.8,
    'soft': 0.7, 'clean': 0.6, 'relief': 0.9, 'plump': 0.8, 'glow': 0.8, 'glowing': 0.8,
    'dewy': 0.7, 'radiant': 0.8, 'smooth': 0.7, 'cleared': 0.9, 'lightweight': 0.7,
    'absorbs': 0.6, 'refreshed': 0.7, 'firming': 0.8, 'bouncy': 0.8, 'healed': 1.0,
    'repair': 0.8, 'repairing': 0.8, 'non-irritating': 0.9, 'comfortable': 0.7,
    'pleasant': 0.6, 'nourishing': 0.8, 'resilient': 0.7, 'supple': 0.8, 'vitality': 0.7
}

NEGATIVE_LEXICON = {
    'broke out': -1.2, 'breakout': -1.0, 'breakouts': -1.0, 'pimples': -0.8, 'irritated': -1.0,
    'irritation': -1.0, 'redness': -0.8, 'dry': -0.5, 'dryness': -0.6, 'burning': -1.2,
    'stinging': -1.0, 'greasy': -0.8, 'sticky': -0.5, 'oily': -0.4, 'rash': -1.2,
    'allergic': -1.3, 'heavy': -0.5, 'clogged': -0.9, 'clogging': -0.9, 'stripping': -0.9,
    'peeling': -0.8, 'tight': -0.6, 'tightness': -0.7, 'blotchiness': -0.8, 'bumps': -0.8,
    'flaking': -0.7, 'flaky': -0.7, 'chalky': -0.6, 'horrible': -1.2, 'terrible': -1.2,
    'bad': -0.8, 'awful': -1.2, 'stung': -1.0, 'sandpaper': -1.0, 'worse': -1.0,
    'sadly': -0.6, 'unfortunately': -0.5, 'burn': -1.0, 'reactive': -0.6
}


def clean_text(text: str) -> str:
    """
    Clean raw text:
    - Lowercase
    - Replace punctuation with space
    - Normalize extra whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Replace non-alphanumeric chars (except hyphens in terms like oil-free, anti-aging)
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
    # Replace isolated hyphens or multiple hyphens
    text = re.sub(r'\s+-\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize cleaned text into words."""
    cleaned = clean_text(text)
    return cleaned.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove standard and domain stopwords from token list."""
    return [t for t in tokens if t not in STANDARD_STOP_WORDS and len(t) > 1]


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline: clean, tokenize, remove stopwords, and rejoin."""
    tokens = tokenize(text)
    filtered = remove_stopwords(tokens)
    return " ".join(filtered)


def compute_sentiment_score(text: str) -> float:
    """
    Calculate a lexicon-based sentiment score in range [-1.0, 1.0].
    Takes phrase matches (e.g. 'holy grail', 'broke out') into account first,
    then individual word tokens.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return 0.0

    score = 0.0
    matched_phrases = 0

    # Multi-word phrase matching
    for phrase, val in POSITIVE_LEXICON.items():
        if ' ' in phrase and phrase in cleaned:
            score += val
            matched_phrases += 1
    for phrase, val in NEGATIVE_LEXICON.items():
        if ' ' in phrase and phrase in cleaned:
            score += val
            matched_phrases += 1

    # Single token matching
    tokens = cleaned.split()
    matched_words = 0
    for t in tokens:
        if t in POSITIVE_LEXICON and ' ' not in t:
            score += POSITIVE_LEXICON[t]
            matched_words += 1
        elif t in NEGATIVE_LEXICON and ' ' not in t:
            score += NEGATIVE_LEXICON[t]
            matched_words += 1

    total_matches = matched_phrases + matched_words
    if total_matches == 0:
        return 0.0

    # Normalize score between -1.0 and 1.0 with a soft tanh-like bounding
    normalized = np.tanh(score / 2.0)
    return float(np.round(normalized, 4))


class SkincareNLPPipeline:
    """
    Main NLP pipeline for skincare product catalog:
    - Text feature combination
    - TF-IDF Vectorization
    - Lexicon-based sentiment scoring for product reviews
    - Query transformation and keyword explanation
    """

    def __init__(self, ngram_range: Tuple[int, int] = (1, 2), min_df: int = 1):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words='english',
            sublinear_tf=True
        )
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        self.item_sentiment_map: Dict[str, float] = {}

    def prepare_item_text(self, df_items: pd.DataFrame) -> pd.Series:
        """
        Combine category, skin type tags, description, and key ingredients
        into a single rich textual representation for each product.
        """
        combined = (
            df_items['category'].fillna('') + " " +
            df_items['skin_type_tags'].fillna('') + " " +
            df_items['product_name'].fillna('') + " " +
            df_items['description'].fillna('') + " " +
            df_items['key_ingredients'].fillna('')
        )
        return combined.apply(preprocess_text)

    def fit_transform(self, df_items: pd.DataFrame) -> np.ndarray:
        """Fit TF-IDF on product catalog text and return feature matrix."""
        corpus = self.prepare_item_text(df_items)
        tfidf_matrix = self.vectorizer.fit_transform(corpus).toarray()
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()
        self.is_fitted = True
        return tfidf_matrix

    def transform_query(self, query: str) -> np.ndarray:
        """Vectorize a free-text user query or new product description using fitted vectorizer."""
        if not self.is_fitted:
            raise ValueError("Pipeline has not been fitted yet.")
        cleaned = preprocess_text(query)
        return self.vectorizer.transform([cleaned]).toarray()[0]

    def compute_product_sentiments(self, df_ratings: pd.DataFrame) -> Dict[str, float]:
        """
        Compute mean review sentiment for each item from user reviews.
        Returns a dict: {item_id: normalized_sentiment_in_[0, 1]}.
        """
        item_sentiments: Dict[str, List[float]] = {}
        for _, row in df_ratings.iterrows():
            item_id = str(row['item_id'])
            review = str(row.get('review_text', ''))
            score = compute_sentiment_score(review)
            if item_id not in item_sentiments:
                item_sentiments[item_id] = []
            item_sentiments[item_id].append(score)

        # Average sentiment per item, scaled from [-1, 1] to [0, 1]
        self.item_sentiment_map = {}
        for item_id, scores in item_sentiments.items():
            mean_polarity = np.mean(scores)
            # Map [-1, 1] to [0, 1]
            scaled_sentiment = float((mean_polarity + 1.0) / 2.0)
            self.item_sentiment_map[item_id] = np.round(scaled_sentiment, 4)

        return self.item_sentiment_map

    def get_sentiment(self, item_id: str, default: float = 0.5) -> float:
        """Retrieve precomputed sentiment score for an item (default 0.5 for cold items)."""
        return self.item_sentiment_map.get(str(item_id), default)

    def explain_match(self, query_vec: np.ndarray, item_vec: np.ndarray, top_k: int = 4) -> List[str]:
        """
        Identify top contributing keyword tokens that overlap between
        the query vector and item vector.
        """
        overlap = query_vec * item_vec
        nonzero_indices = np.where(overlap > 0)[0]
        if len(nonzero_indices) == 0:
            # Fallback to top terms of item
            top_indices = np.argsort(item_vec)[::-1][:top_k]
            return [self.feature_names[i] for i in top_indices if item_vec[i] > 0]

        ranked_indices = nonzero_indices[np.argsort(overlap[nonzero_indices])[::-1]]
        top_terms = [self.feature_names[i] for i in ranked_indices[:top_k]]
        return top_terms
