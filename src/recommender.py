"""
Recommendation Engine for Skincare and Cosmetics Products
Student Name: Quách Tú Mỹ
Student ID: 29201153781 (CS467A)
Project Code: MP-A10

Implements three distinct approaches:
1. PopularityBaselineRecommender (Bayesian / IMDB weighted rating)
2. NLPContentBasedRecommender (Cosine similarity over TF-IDF item representations)
3. SentimentHybridRecommender (Content similarity + review sentiment + rating quality)
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.nlp_pipeline import SkincareNLPPipeline


class PopularityBaselineRecommender:
    """
    Popularity Baseline Recommender:
    Ranks items using a Bayesian weighted rating formula:
        WR = (v / (v + m)) * R + (m / (v + m)) * C
    where:
        v: number of ratings for the item
        m: minimum rating count threshold (default 2)
        R: average rating of the item
        C: global mean rating across all items
    Filters out items the target user has already rated.
    """

    def __init__(self, min_ratings: int = 2):
        self.min_ratings = min_ratings
        self.item_stats: Dict[str, Dict[str, float]] = {}
        self.global_mean: float = 3.5
        self.user_history: Dict[str, set] = {}
        self.df_items: Optional[pd.DataFrame] = None

    def fit(self, df_ratings: pd.DataFrame, df_items: pd.DataFrame):
        self.df_items = df_items.copy().set_index('item_id')
        self.global_mean = float(df_ratings['rating'].mean())

        # Track user rating history to filter out seen items
        self.user_history = {}
        for uid, group in df_ratings.groupby('user_id'):
            self.user_history[str(uid)] = set(group['item_id'].astype(str))

        # Compute popularity stats per item
        item_groups = df_ratings.groupby('item_id')['rating'].agg(['count', 'mean'])
        m = self.min_ratings
        C = self.global_mean

        self.item_stats = {}
        for item_id, row in item_groups.iterrows():
            v = row['count']
            R = row['mean']
            wr = (v / (v + m)) * R + (m / (v + m)) * C
            self.item_stats[str(item_id)] = {
                'count': float(v),
                'mean_rating': float(R),
                'score': float(wr)
            }

        # For items in catalog without ratings
        for item_id in self.df_items.index:
            s_id = str(item_id)
            if s_id not in self.item_stats:
                self.item_stats[s_id] = {
                    'count': 0.0,
                    'mean_rating': self.global_mean,
                    'score': float(self.global_mean * 0.8)
                }

    def recommend(self, user_id: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Recommend top-k popular items, filtering out user's already rated items."""
        seen_items = self.user_history.get(str(user_id), set()) if user_id else set()

        candidates = [
            (item_id, stats['score'], stats['count'], stats['mean_rating'])
            for item_id, stats in self.item_stats.items()
            if item_id not in seen_items
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        results = []
        for item_id, score, count, mean_r in candidates[:top_k]:
            item_info = self.df_items.loc[item_id] if item_id in self.df_items.index else {}
            results.append({
                'item_id': item_id,
                'product_name': item_info.get('product_name', 'Unknown'),
                'category': item_info.get('category', ''),
                'score': round(score, 4),
                'rating_count': int(count),
                'avg_rating': round(mean_r, 2),
                'reason': f"Popularity score: {score:.2f} ({int(count)} ratings, avg {mean_r:.1f}★)"
            })
        return results

    def predict(self, user_id: str, item_id: str) -> float:
        """Predict numeric rating based on item's weighted popularity."""
        s_id = str(item_id)
        if s_id in self.item_stats:
            return float(np.clip(self.item_stats[s_id]['mean_rating'], 1.0, 5.0))
        return self.global_mean


class NLPContentBasedRecommender:
    """
    NLP Content-Based Recommender:
    Uses Cosine Similarity over TF-IDF feature representations extracted from
    product category, skin types, descriptions, and ingredients.
    Supports:
      (a) Free-text query matching
      (b) Item-to-item similarity recommendations
      (c) User-profile based recommendations from rated items
    """

    def __init__(self, nlp_pipeline: SkincareNLPPipeline):
        self.nlp = nlp_pipeline
        self.df_items: Optional[pd.DataFrame] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.item_id_to_idx: Dict[str, int] = {}
        self.idx_to_item_id: Dict[int, str] = {}
        self.user_profiles: Dict[str, np.ndarray] = {}
        self.user_history: Dict[str, set] = {}
        self.user_means: Dict[str, float] = {}
        self.global_mean: float = 3.5

    def fit(self, df_items: pd.DataFrame, df_ratings: Optional[pd.DataFrame] = None):
        self.df_items = df_items.copy().reset_index(drop=True)
        self.tfidf_matrix = self.nlp.fit_transform(self.df_items)

        self.item_id_to_idx = {str(row['item_id']): i for i, row in self.df_items.iterrows()}
        self.idx_to_item_id = {i: str(row['item_id']) for i, row in self.df_items.iterrows()}

        if df_ratings is not None:
            self._build_user_profiles(df_ratings)

    def _build_user_profiles(self, df_ratings: pd.DataFrame):
        """Construct user profile vectors by weighting TF-IDF vectors of liked items."""
        self.user_history = {}
        self.user_means = {}
        self.global_mean = float(df_ratings['rating'].mean())

        for uid, group in df_ratings.groupby('user_id'):
            s_uid = str(uid)
            self.user_history[s_uid] = set(group['item_id'].astype(str))
            self.user_means[s_uid] = float(group['rating'].mean())

            # Weighted sum of item vectors based on normalized rating centered at 2.0
            profile_vec = np.zeros(self.tfidf_matrix.shape[1], dtype=np.float64)
            weight_sum = 0.0

            for _, row in group.iterrows():
                i_id = str(row['item_id'])
                if i_id in self.item_id_to_idx:
                    idx = self.item_id_to_idx[i_id]
                    rating_weight = max(0.1, float(row['rating']) - 2.0)
                    profile_vec += rating_weight * self.tfidf_matrix[idx]
                    weight_sum += rating_weight

            if weight_sum > 0:
                profile_norm = np.linalg.norm(profile_vec)
                if profile_norm > 0:
                    profile_vec = profile_vec / profile_norm
                self.user_profiles[s_uid] = profile_vec

    def add_cold_start_item(self, item_dict: Dict[str, str]) -> str:
        """
        Dynamically add a newly launched item with text description only.
        Demonstrates cold-start support without needing user interaction.
        """
        new_row = pd.DataFrame([item_dict])
        self.df_items = pd.concat([self.df_items, new_row], ignore_index=True)

        # Refit TF-IDF matrix with the new item included
        self.tfidf_matrix = self.nlp.fit_transform(self.df_items)
        self.item_id_to_idx = {str(row['item_id']): i for i, row in self.df_items.iterrows()}
        self.idx_to_item_id = {i: str(row['item_id']) for i, row in self.df_items.iterrows()}
        return str(item_dict['item_id'])

    def recommend_by_query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Recommend products matching a free-text need query."""
        query_vec = self.nlp.transform_query(query).reshape(1, -1)
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            item_row = self.df_items.iloc[idx]
            item_vec = self.tfidf_matrix[idx]
            matching_terms = self.nlp.explain_match(query_vec[0], item_vec, top_k=3)
            results.append({
                'item_id': str(item_row['item_id']),
                'product_name': item_row['product_name'],
                'category': item_row['category'],
                'skin_type_tags': item_row.get('skin_type_tags', ''),
                'score': round(float(sims[idx]), 4),
                'top_keywords': matching_terms,
                'reason': f"TF-IDF cosine similarity: {sims[idx]:.3f} (matches: {', '.join(matching_terms)})"
            })
        return results

    def recommend_similar_items(self, item_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Recommend items most similar to a given product (item-to-item)."""
        s_id = str(item_id)
        if s_id not in self.item_id_to_idx:
            raise KeyError(f"Item ID {s_id} not found in catalog.")

        target_idx = self.item_id_to_idx[s_id]
        target_vec = self.tfidf_matrix[target_idx].reshape(1, -1)
        sims = cosine_similarity(target_vec, self.tfidf_matrix)[0].copy()

        # Exclude self
        sims[target_idx] = -1.0
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            item_row = self.df_items.iloc[idx]
            item_vec = self.tfidf_matrix[idx]
            shared_keywords = self.nlp.explain_match(target_vec[0], item_vec, top_k=3)
            results.append({
                'item_id': str(item_row['item_id']),
                'product_name': item_row['product_name'],
                'category': item_row['category'],
                'skin_type_tags': item_row.get('skin_type_tags', ''),
                'score': round(float(sims[idx]), 4),
                'top_keywords': shared_keywords,
                'reason': f"Ingredient & feature similarity: {sims[idx]:.3f} (shared: {', '.join(shared_keywords)})"
            })
        return results

    def recommend_for_user(self, user_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Recommend items for a user based on their NLP profile vector."""
        s_uid = str(user_id)
        if s_uid not in self.user_profiles:
            return []

        user_vec = self.user_profiles[s_uid].reshape(1, -1)
        sims = cosine_similarity(user_vec, self.tfidf_matrix)[0].copy()

        # Filter out items already rated
        seen_items = self.user_history.get(s_uid, set())
        for s_iid in seen_items:
            if s_iid in self.item_id_to_idx:
                sims[self.item_id_to_idx[s_iid]] = -1.0

        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if sims[idx] <= 0:
                continue
            item_row = self.df_items.iloc[idx]
            item_vec = self.tfidf_matrix[idx]
            keywords = self.nlp.explain_match(user_vec[0], item_vec, top_k=3)
            results.append({
                'item_id': str(item_row['item_id']),
                'product_name': item_row['product_name'],
                'category': item_row['category'],
                'score': round(float(sims[idx]), 4),
                'top_keywords': keywords,
                'reason': f"Profile match: {sims[idx]:.3f} (interests: {', '.join(keywords)})"
            })
        return results

    def predict(self, user_id: str, item_id: str) -> float:
        """Predict numeric rating based on user baseline and profile similarity."""
        s_uid = str(user_id)
        s_iid = str(item_id)
        u_mean = self.user_means.get(s_uid, self.global_mean)

        if s_uid not in self.user_profiles or s_iid not in self.item_id_to_idx:
            return u_mean

        user_vec = self.user_profiles[s_uid].reshape(1, -1)
        item_idx = self.item_id_to_idx[s_iid]
        item_vec = self.tfidf_matrix[item_idx].reshape(1, -1)
        sim = float(cosine_similarity(user_vec, item_vec)[0][0])

        # Centered prediction: user average offset by similarity deviation
        pred = u_mean + (sim - 0.15) * 1.5
        return float(np.clip(pred, 1.0, 5.0))


class SentimentHybridRecommender:
    """
    Project-Specific Extension: Sentiment-Boosted Content-Based Hybrid Recommender
    Combines:
      1. TF-IDF Content Similarity (Sim_content in [0, 1])
      2. Review Sentiment Score (S_sentiment in [0, 1]) from customer reviews
      3. Quality / Average Rating (S_rating in [0, 1])

    For query recommendations:
      Score = w_content * Sim_content + w_rating * S_rating + w_sentiment * S_sentiment
    For personalized user recommendations:
      Score = Sim_content * (0.60 + 0.20 * S_rating + 0.20 * S_sentiment)
    """

    def __init__(
        self,
        content_recommender: NLPContentBasedRecommender,
        nlp_pipeline: SkincareNLPPipeline,
        weights: Tuple[float, float, float] = (0.60, 0.20, 0.20)
    ):
        self.content_rec = content_recommender
        self.nlp = nlp_pipeline
        self.w_content, self.w_rating, self.w_sentiment = weights
        self.item_avg_ratings: Dict[str, float] = {}
        self.item_sentiments: Dict[str, float] = {}
        self.user_means: Dict[str, float] = {}
        self.global_mean_rating: float = 3.5
        self.df_items: Optional[pd.DataFrame] = None
        self.user_history: Dict[str, set] = {}

    def fit(self, df_ratings: pd.DataFrame, df_items: pd.DataFrame):
        self.df_items = df_items.copy().set_index('item_id')
        self.global_mean_rating = float(df_ratings['rating'].mean())

        # Average rating per item
        avg_series = df_ratings.groupby('item_id')['rating'].mean()
        self.item_avg_ratings = {str(k): float(v) for k, v in avg_series.items()}

        # User mean ratings
        u_series = df_ratings.groupby('user_id')['rating'].mean()
        self.user_means = {str(k): float(v) for k, v in u_series.items()}

        # Sentiment scores per product calculated by NLP pipeline
        self.item_sentiments = self.nlp.compute_product_sentiments(df_ratings)

        # Track history
        self.user_history = {}
        for uid, group in df_ratings.groupby('user_id'):
            self.user_history[str(uid)] = set(group['item_id'].astype(str))

    def _get_item_rating_score(self, item_id: str) -> float:
        """Normalize rating to [0, 1] range (default 0.6 for unrated/cold items)."""
        raw = self.item_avg_ratings.get(str(item_id), self.global_mean_rating)
        return float(np.clip((raw - 1.0) / 4.0, 0.0, 1.0))

    def _get_item_sentiment_score(self, item_id: str) -> float:
        """Get precomputed review sentiment in [0, 1] (default 0.5 for neutral/cold items)."""
        return self.item_sentiments.get(str(item_id), 0.5)

    def recommend_by_query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend products for a free-text need query boosted by review sentiment
        and customer ratings.
        """
        query_vec = self.nlp.transform_query(query).reshape(1, -1)
        sims = cosine_similarity(query_vec, self.content_rec.tfidf_matrix)[0]

        scored_items = []
        for idx in range(len(self.content_rec.df_items)):
            item_row = self.content_rec.df_items.iloc[idx]
            item_id = str(item_row['item_id'])
            sim_content = float(sims[idx])

            norm_rating = self._get_item_rating_score(item_id)
            sentiment_score = self._get_item_sentiment_score(item_id)

            composite_score = (
                self.w_content * sim_content +
                self.w_rating * norm_rating +
                self.w_sentiment * sentiment_score
            )

            item_vec = self.content_rec.tfidf_matrix[idx]
            keywords = self.nlp.explain_match(query_vec[0], item_vec, top_k=3)

            scored_items.append({
                'item_id': item_id,
                'product_name': item_row['product_name'],
                'category': item_row['category'],
                'skin_type_tags': item_row.get('skin_type_tags', ''),
                'score': round(composite_score, 4),
                'content_similarity': round(sim_content, 3),
                'sentiment_score': round(sentiment_score, 3),
                'avg_rating': round(self.item_avg_ratings.get(item_id, self.global_mean_rating), 2),
                'top_keywords': keywords,
                'reason': (
                    f"Hybrid Score: {composite_score:.3f} "
                    f"[Content: {sim_content:.2f}, Rating: {norm_rating:.2f}, Sentiment: {sentiment_score:.2f}]"
                )
            })

        scored_items.sort(key=lambda x: x['score'], reverse=True)
        return scored_items[:top_k]

    def recommend_for_user(self, user_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend items for a user combining profile similarity with sentiment boost.
        Uses multiplicative sentiment-quality modulation to preserve topic relevance.
        Filters out items the user has already rated.
        """
        s_uid = str(user_id)
        if s_uid not in self.content_rec.user_profiles:
            return []

        user_vec = self.content_rec.user_profiles[s_uid].reshape(1, -1)
        sims = cosine_similarity(user_vec, self.content_rec.tfidf_matrix)[0]

        seen_items = self.user_history.get(s_uid, set())
        scored_items = []

        for idx in range(len(self.content_rec.df_items)):
            item_row = self.content_rec.df_items.iloc[idx]
            item_id = str(item_row['item_id'])
            if item_id in seen_items:
                continue

            sim_content = float(sims[idx])
            norm_rating = self._get_item_rating_score(item_id)
            sentiment_score = self._get_item_sentiment_score(item_id)

            # Modulate content similarity with review sentiment and rating quality
            composite_score = sim_content * (0.60 + 0.20 * norm_rating + 0.20 * sentiment_score)

            item_vec = self.content_rec.tfidf_matrix[idx]
            keywords = self.nlp.explain_match(user_vec[0], item_vec, top_k=3)

            scored_items.append({
                'item_id': item_id,
                'product_name': item_row['product_name'],
                'category': item_row['category'],
                'score': round(composite_score, 4),
                'content_similarity': round(sim_content, 3),
                'sentiment_score': round(sentiment_score, 3),
                'avg_rating': round(self.item_avg_ratings.get(item_id, self.global_mean_rating), 2),
                'top_keywords': keywords,
                'reason': (
                    f"Hybrid match: {composite_score:.3f} "
                    f"[Content: {sim_content:.2f}, Rating: {norm_rating:.2f}, Sentiment: {sentiment_score:.2f}]"
                )
            })

        scored_items.sort(key=lambda x: x['score'], reverse=True)
        return scored_items[:top_k]

    def predict(self, user_id: str, item_id: str) -> float:
        """
        Predict numeric rating combining user baseline, item mean,
        profile similarity, and review sentiment.
        """
        s_uid = str(user_id)
        s_iid = str(item_id)

        u_mean = self.user_means.get(s_uid, self.global_mean_rating)
        i_mean = self.item_avg_ratings.get(s_iid, self.global_mean_rating)

        sim_content = 0.20
        if s_uid in self.content_rec.user_profiles and s_iid in self.content_rec.item_id_to_idx:
            user_vec = self.content_rec.user_profiles[s_uid].reshape(1, -1)
            item_idx = self.content_rec.item_id_to_idx[s_iid]
            item_vec = self.content_rec.tfidf_matrix[item_idx].reshape(1, -1)
            sim_content = float(cosine_similarity(user_vec, item_vec)[0][0])

        s_score = self._get_item_sentiment_score(s_iid)

        # Blends user baseline, item mean, content match offset, and sentiment quality
        pred = 0.40 * u_mean + 0.40 * i_mean + 0.20 * (1.0 + 4.0 * s_score) + (sim_content - 0.15) * 0.50
        return float(np.clip(pred, 1.0, 5.0))
