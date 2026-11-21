"""
Evaluation and Visualization Module
Student Name: Quách Tú Mỹ
Student ID: 29201153781 (CS467A)
Project Code: MP-A10

Performs:
1. Train/Test Holdout Split (20% ratings holdout per user)
2. Evaluation of 3 Recommender Approaches:
   - Popularity Baseline
   - NLP Content-Based
   - Sentiment-Boosted Hybrid
   Metrics: Precision@5, Recall@5, HitRate@5, RMSE, MAE
3. Generation and saving of at least 2 visualization charts in results/
4. Execution of the 3 required test cases with outputs saved to results/sample_outputs.txt
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.nlp_pipeline import SkincareNLPPipeline
from src.recommender import (
    PopularityBaselineRecommender,
    NLPContentBasedRecommender,
    SentimentHybridRecommender
)


def create_train_test_split(
    df_ratings: pd.DataFrame,
    test_ratio: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings into train and test sets by holding out ~20% of ratings
    per user so every user has ratings in both train and test.
    """
    np.random.seed(random_state)
    train_rows = []
    test_rows = []

    for user_id, group in df_ratings.groupby('user_id'):
        shuffled = group.sample(frac=1.0, random_state=random_state)
        n_test = max(1, int(round(len(shuffled) * test_ratio)))
        test_rows.append(shuffled.iloc[:n_test])
        train_rows.append(shuffled.iloc[n_test:])

    df_train = pd.concat(train_rows, ignore_index=True)
    df_test = pd.concat(test_rows, ignore_index=True)
    return df_train, df_test


def evaluate_recommenders(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_items: pd.DataFrame,
    top_k: int = 5,
    relevance_threshold: float = 4.0
) -> Dict[str, Dict[str, float]]:
    """
    Compute Precision@K, Recall@K, HitRate@K, RMSE, and MAE across test users.
    Relevant items are defined as ground-truth test items rated >= relevance_threshold (4.0).
    """
    # 1. Initialize pipelines and models
    nlp = SkincareNLPPipeline()
    nlp.fit_transform(df_items)

    pop_rec = PopularityBaselineRecommender()
    pop_rec.fit(df_train, df_items)

    content_rec = NLPContentBasedRecommender(nlp)
    content_rec.fit(df_items, df_train)

    hybrid_rec = SentimentHybridRecommender(content_rec, nlp)
    hybrid_rec.fit(df_train, df_items)

    models = {
        'Popularity Baseline': pop_rec,
        'NLP Content-Based': content_rec,
        'Sentiment Hybrid': hybrid_rec
    }

    # Group test items by user
    test_user_items: Dict[str, set] = {}
    test_user_ratings: Dict[str, Dict[str, float]] = {}

    for _, row in df_test.iterrows():
        u = str(row['user_id'])
        i = str(row['item_id'])
        r = float(row['rating'])
        if u not in test_user_items:
            test_user_items[u] = set()
            test_user_ratings[u] = {}
        if r >= relevance_threshold:
            test_user_items[u].add(i)
        test_user_ratings[u][i] = r

    results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        precisions = []
        recalls = []
        hits = []
        sq_errors = []
        abs_errors = []

        for u, true_relevant in test_user_items.items():
            # Get top-k recommendations
            if name == 'Popularity Baseline':
                recs = model.recommend(user_id=u, top_k=top_k)
            elif name == 'NLP Content-Based':
                recs = model.recommend_for_user(user_id=u, top_k=top_k)
            else:
                recs = model.recommend_for_user(user_id=u, top_k=top_k)

            rec_item_ids = [str(r['item_id']) for r in recs]
            relevant_retrieved = len(set(rec_item_ids) & true_relevant)

            # Precision@K
            p_k = relevant_retrieved / float(top_k)
            precisions.append(p_k)

            # Recall@K
            if len(true_relevant) > 0:
                r_k = relevant_retrieved / float(len(true_relevant))
                recalls.append(r_k)
                hits.append(1.0 if relevant_retrieved > 0 else 0.0)

            # Rating prediction metrics (RMSE & MAE)
            for item_id, actual_rating in test_user_ratings[u].items():
                pred = model.predict(u, item_id)
                err = pred - actual_rating
                sq_errors.append(err ** 2)
                abs_errors.append(abs(err))

        results[name] = {
            f'Precision@{top_k}': float(np.round(np.mean(precisions), 4)),
            f'Recall@{top_k}': float(np.round(np.mean(recalls), 4)),
            f'HitRate@{top_k}': float(np.round(np.mean(hits), 4)),
            'RMSE': float(np.round(np.sqrt(np.mean(sq_errors)), 4)),
            'MAE': float(np.round(np.mean(abs_errors), 4))
        }

    return results


def plot_dataset_distributions(df_items: pd.DataFrame, df_ratings: pd.DataFrame, save_path: str):
    """Plot rating score frequency and item distribution by category."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subplot 1: Rating distribution
    rating_counts = df_ratings['rating'].value_counts().sort_index()
    colors_ratings = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
    bars1 = axes[0].bar(
        [str(r) + " ★" for r in rating_counts.index],
        rating_counts.values,
        color=colors_ratings[:len(rating_counts)],
        edgecolor='black',
        alpha=0.85
    )
    axes[0].set_title("User Rating Distribution (1 - 5 Stars)", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Rating Score", fontsize=11)
    axes[0].set_ylabel("Interaction Count", fontsize=11)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, int(yval), ha='center', va='bottom', fontsize=10)

    # Subplot 2: Category distribution
    cat_counts = df_items['category'].value_counts(ascending=True)
    palette = plt.cm.viridis(np.linspace(0.2, 0.85, len(cat_counts)))
    bars2 = axes[1].barh(cat_counts.index, cat_counts.values, color=palette, edgecolor='black', alpha=0.85)
    axes[1].set_title("Product Catalog by Category (40 Products)", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Number of Products", fontsize=11)
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)

    for bar in bars2:
        xval = bar.get_width()
        axes[1].text(xval + 0.15, bar.get_y() + bar.get_height()/2.0, int(xval), ha='left', va='center', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Chart Saved] Dataset distributions: {save_path}")


def plot_evaluation_comparison(eval_results: Dict[str, Dict[str, float]], save_path: str):
    """Plot bar chart comparing recommendation accuracy metrics across models."""
    models = list(eval_results.keys())
    metrics = ['Precision@5', 'Recall@5', 'HitRate@5']

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#3498db', '#e67e22', '#2ecc71']

    for i, metric in enumerate(metrics):
        vals = [eval_results[m][metric] for m in models]
        offset = (i - 1) * width
        rects = ax.bar(x + offset, vals, width, label=metric, color=colors[i], edgecolor='black', alpha=0.85)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f'{height:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )

    ax.set_title("Offline Recommender Performance Comparison (Top-5)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper left', frameon=True, fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Chart Saved] Recommender comparison: {save_path}")


def run_all_test_cases(
    content_rec: NLPContentBasedRecommender,
    hybrid_rec: SentimentHybridRecommender,
    output_path: str
) -> str:
    """
    Execute the 3 required test cases and write clear structured results
    to results/sample_outputs.txt.
    """
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("CS467 MINI-PROJECT MP-A10: SKINCARE AND COSMETICS PRODUCT RECOMMENDER")
    output_lines.append("Student Name: Quách Tú Mỹ  |  Student ID: 29201153781  |  Section: CS467A (No. 10)")
    output_lines.append("=" * 80)
    output_lines.append("")

    # -------------------------------------------------------------
    # TEST CASE 1: Need Query Test
    # -------------------------------------------------------------
    query_1 = "gentle hydrating moisturizer for dry sensitive skin"
    output_lines.append("--------------------------------------------------------------------------------")
    output_lines.append(f"TEST CASE 1: Need Query Recommendation")
    output_lines.append(f"User Query: '{query_1}'")
    output_lines.append("Goal: Recommend top-5 products matching user needs using Sentiment-Boosted Hybrid.")
    output_lines.append("--------------------------------------------------------------------------------")

    recs_1 = hybrid_rec.recommend_by_query(query_1, top_k=5)
    for rank, item in enumerate(recs_1, 1):
        output_lines.append(
            f"  {rank}. [{item['item_id']}] {item['product_name']} ({item['category']})\n"
            f"     Score: {item['score']} | Content Sim: {item['content_similarity']} | Sentiment: {item['sentiment_score']} | Avg Rating: {item['avg_rating']}★\n"
            f"     Top Matching Keywords: {', '.join(item['top_keywords'])}\n"
            f"     Reason: {item['reason']}\n"
        )
    output_lines.append("")

    # -------------------------------------------------------------
    # TEST CASE 2: Similar Product Test
    # -------------------------------------------------------------
    target_sunscreen_id = "P026"
    target_name = "Aqua UV Daily Sun Water Gel SPF50+ PA++++"
    output_lines.append("--------------------------------------------------------------------------------")
    output_lines.append(f"TEST CASE 2: Similar Product Recommendation (Item-to-Item)")
    output_lines.append(f"Selected Product: [{target_sunscreen_id}] {target_name}")
    output_lines.append("Goal: Find top-5 alternatives based on ingredient profile and skincare benefits.")
    output_lines.append("--------------------------------------------------------------------------------")

    recs_2 = content_rec.recommend_similar_items(target_sunscreen_id, top_k=5)
    for rank, item in enumerate(recs_2, 1):
        output_lines.append(
            f"  {rank}. [{item['item_id']}] {item['product_name']} ({item['category']})\n"
            f"     Similarity Score: {item['score']}\n"
            f"     Skin Type Tags: {item['skin_type_tags']}\n"
            f"     Shared Key Features: {', '.join(item['top_keywords'])}\n"
            f"     Reason: {item['reason']}\n"
        )
    output_lines.append("")

    # -------------------------------------------------------------
    # TEST CASE 3: Cold-Start Test
    # -------------------------------------------------------------
    new_product = {
        'item_id': 'P041',
        'product_name': 'Phyto-Glow Brightening Barrier Serum',
        'category': 'Serum',
        'skin_type_tags': 'Dull, Uneven Skin Tone, Sensitive, Dehydrated',
        'description': 'Advanced antioxidant radiance serum with 10% Ascorbic Acid, Ferulic Acid, and Hyaluronic Acid for brightening hyperpigmentation and smoothing rough skin texture.',
        'key_ingredients': 'Ascorbic Acid, Ferulic Acid, Hyaluronic Acid, Vitamin E, Alpha-Arbutin'
    }
    output_lines.append("--------------------------------------------------------------------------------")
    output_lines.append("TEST CASE 3: Cold-Start Test (Newly Launched Product with No User Ratings)")
    output_lines.append(f"Adding New Product: [{new_product['item_id']}] {new_product['product_name']}")
    output_lines.append(f"Description: {new_product['description']}")
    output_lines.append(f"Key Ingredients: {new_product['key_ingredients']}")
    output_lines.append("--------------------------------------------------------------------------------")

    # Add cold start item to recommender
    content_rec.add_cold_start_item(new_product)

    cold_query = "antioxidant brightening serum with ascorbic acid and ferulic acid"
    output_lines.append(f"Test Query: '{cold_query}'")
    recs_3 = content_rec.recommend_by_query(cold_query, top_k=5)

    found_in_top5 = False
    for rank, item in enumerate(recs_3, 1):
        is_new = "(★ NEW COLD-START ITEM ★)" if item['item_id'] == 'P041' else ""
        if item['item_id'] == 'P041':
            found_in_top5 = True
        output_lines.append(
            f"  {rank}. [{item['item_id']}] {item['product_name']} {is_new}\n"
            f"     Cosine Similarity: {item['score']} | Category: {item['category']}\n"
            f"     Top Matching Keywords: {', '.join(item['top_keywords'])}\n"
            f"     Reason: {item['reason']}\n"
        )

    output_lines.append(f"Verification Result: Newly added item P041 in Top-5? -> {'SUCCESS' if found_in_top5 else 'FAILED'}")
    output_lines.append("=" * 80)

    full_output = "\n".join(output_lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_output)

    print(f"[Results Saved] 3 test cases saved to: {output_path}")
    return full_output


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    items_path = os.path.join(base_dir, 'data', 'items.csv')
    ratings_path = os.path.join(base_dir, 'data', 'ratings.csv')
    results_dir = os.path.join(base_dir, 'results')

    df_items = pd.read_csv(items_path)
    df_ratings = pd.read_csv(ratings_path)

    # 1. Dataset distribution plot
    chart1_path = os.path.join(results_dir, 'rating_category_distribution.png')
    plot_dataset_distributions(df_items, df_ratings, chart1_path)

    # 2. Train-test evaluation
    df_train, df_test = create_train_test_split(df_ratings, test_ratio=0.20, random_state=42)
    metrics = evaluate_recommenders(df_train, df_test, df_items, top_k=5)

    print("\n--- Evaluation Metrics (Test Split) ---")
    for model, m_dict in metrics.items():
        print(f"\n[{model}]")
        for k, v in m_dict.items():
            print(f"  {k}: {v}")

    # 3. Model comparison plot
    chart2_path = os.path.join(results_dir, 'recommender_evaluation_comparison.png')
    plot_evaluation_comparison(metrics, chart2_path)

    # 4. Run test cases on full dataset
    nlp = SkincareNLPPipeline()
    nlp.fit_transform(df_items)
    content_rec = NLPContentBasedRecommender(nlp)
    content_rec.fit(df_items, df_ratings)
    hybrid_rec = SentimentHybridRecommender(content_rec, nlp)
    hybrid_rec.fit(df_ratings, df_items)

    test_case_out_path = os.path.join(results_dir, 'sample_outputs.txt')
    run_all_test_cases(content_rec, hybrid_rec, test_case_out_path)
