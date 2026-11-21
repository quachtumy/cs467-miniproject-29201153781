"""
Main Entry Point & Interactive CLI Demo
Student Name: Quách Tú Mỹ
Student ID: 29201153781 (CS467A)
Project Code: MP-A10
Project Title: Skincare and Cosmetics Product Recommender
"""

import os
import sys
import argparse
import pandas as pd

from src.nlp_pipeline import SkincareNLPPipeline
from src.recommender import (
    PopularityBaselineRecommender,
    NLPContentBasedRecommender,
    SentimentHybridRecommender
)
from src.evaluate import (
    create_train_test_split,
    evaluate_recommenders,
    plot_dataset_distributions,
    plot_evaluation_comparison,
    run_all_test_cases
)


def load_data(base_dir: str):
    items_path = os.path.join(base_dir, 'data', 'items.csv')
    ratings_path = os.path.join(base_dir, 'data', 'ratings.csv')
    df_items = pd.read_csv(items_path)
    df_ratings = pd.read_csv(ratings_path)
    return df_items, df_ratings


def print_banner():
    print("=" * 80)
    print(" DUY TAN UNIVERSITY | CS467 ADVANCED ARTIFICIAL INTELLIGENCE")
    print(" Skincare and Cosmetics Product Recommender (MP-A10)")
    print(" Student: Quách Tú Mỹ  |  ID: 29201153781  |  Section: CS467A")
    print("=" * 80)


def display_recommendations(recs, title="Top-5 Recommended Products"):
    print("\n" + "-" * 75)
    print(f" {title}")
    print("-" * 75)
    if not recs:
        print("  No recommendations found.")
        return

    for rank, item in enumerate(recs, 1):
        name = item.get('product_name', 'Unknown')
        cat = item.get('category', '')
        score = item.get('score', 0.0)
        reasons = item.get('reason', '')
        keywords = item.get('top_keywords', [])
        kw_str = f" | Keywords: [{', '.join(keywords)}]" if keywords else ""
        print(f"  {rank}. [{item['item_id']}] {name} ({cat})")
        print(f"     Score: {score}{kw_str}")
        print(f"     Details: {reasons}")
    print("-" * 75 + "\n")


def run_interactive_mode(nlp, content_rec, hybrid_rec, pop_rec, df_items, df_ratings, results_dir):
    while True:
        print_banner()
        print("Select an option:")
        print(" 1. Run Test Case 1: Free-text need query ('gentle hydrating moisturizer for dry sensitive skin')")
        print(" 2. Run Test Case 2: Similar product recommendation (Find alternatives to Sunscreen P026)")
        print(" 3. Run Test Case 3: Cold-start test (Add newly launched serum and verify recommendation)")
        print(" 4. Enter a custom free-text skincare query")
        print(" 5. Get personalized recommendations for a User ID (e.g., U001 - U014)")
        print(" 6. Run full offline evaluation and update charts in results/")
        print(" 7. Run all 3 test cases and save to results/sample_outputs.txt")
        print(" 0. Exit")
        print("=" * 80)

        choice = input("Enter choice (0-7): ").strip()

        if choice == '1':
            q = "gentle hydrating moisturizer for dry sensitive skin"
            print(f"\n[Running Test Case 1] Query: '{q}'")
            recs = hybrid_rec.recommend_by_query(q, top_k=5)
            display_recommendations(recs, f"Test Case 1: Sentiment-Boosted Recommendations for '{q}'")

        elif choice == '2':
            target_id = "P026"
            print(f"\n[Running Test Case 2] Target: [P026] Aqua UV Daily Sun Water Gel SPF50+ PA++++")
            recs = content_rec.recommend_similar_items(target_id, top_k=5)
            display_recommendations(recs, "Test Case 2: Similar Sunscreen Alternatives (Content-Based)")

        elif choice == '3':
            print("\n[Running Test Case 3] Cold-Start Newly Launched Product")
            new_item = {
                'item_id': 'P041',
                'product_name': 'Phyto-Glow Brightening Barrier Serum',
                'category': 'Serum',
                'skin_type_tags': 'Dull, Uneven Skin Tone, Sensitive, Dehydrated',
                'description': 'Advanced antioxidant radiance serum with 10% Ascorbic Acid, Ferulic Acid, and Hyaluronic Acid for brightening hyperpigmentation and smoothing rough skin texture.',
                'key_ingredients': 'Ascorbic Acid, Ferulic Acid, Hyaluronic Acid, Vitamin E, Alpha-Arbutin'
            }
            content_rec.add_cold_start_item(new_item)
            test_q = "antioxidant brightening serum with ascorbic acid and ferulic acid"
            print(f"Added: [{new_item['item_id']}] {new_item['product_name']}")
            print(f"Querying: '{test_q}'")
            recs = content_rec.recommend_by_query(test_q, top_k=5)
            display_recommendations(recs, "Test Case 3: Cold-Start Item Search Results")
            found = any(r['item_id'] == 'P041' for r in recs)
            print(f"Result: New item P041 successfully retrieved in Top-5: {found}\n")

        elif choice == '4':
            custom_q = input("\nEnter your skincare need (e.g., 'calming cica cream for red irritated skin'): ").strip()
            if custom_q:
                recs = hybrid_rec.recommend_by_query(custom_q, top_k=5)
                display_recommendations(recs, f"Sentiment-Boosted Results for '{custom_q}'")

        elif choice == '5':
            uid = input("\nEnter User ID (available: U001 to U014): ").strip()
            if uid in content_rec.user_profiles:
                recs = hybrid_rec.recommend_for_user(uid, top_k=5)
                display_recommendations(recs, f"Personalized Recommendations for User {uid}")
            else:
                print(f"User ID '{uid}' not recognized. Showing Popularity Baseline:")
                recs = pop_rec.recommend(user_id=uid, top_k=5)
                display_recommendations(recs, "Popularity Baseline Recommendations")

        elif choice == '6':
            print("\n[Running Train/Test Holdout Evaluation]...")
            df_train, df_test = create_train_test_split(df_ratings, test_ratio=0.20, random_state=42)
            metrics = evaluate_recommenders(df_train, df_test, df_items, top_k=5)

            print("\n" + "=" * 60)
            print(f"{'Model':<25} | {'P@5':<7} | {'R@5':<7} | {'Hit@5':<7} | {'RMSE':<7}")
            print("-" * 60)
            for m_name, m_dict in metrics.items():
                print(f"{m_name:<25} | {m_dict['Precision@5']:<7.4f} | {m_dict['Recall@5']:<7.4f} | {m_dict['HitRate@5']:<7.4f} | {m_dict['RMSE']:<7.4f}")
            print("=" * 60)

            c1 = os.path.join(results_dir, 'rating_category_distribution.png')
            c2 = os.path.join(results_dir, 'recommender_evaluation_comparison.png')
            plot_dataset_distributions(df_items, df_ratings, c1)
            plot_evaluation_comparison(metrics, c2)
            print("Evaluation completed and charts updated!\n")

        elif choice == '7':
            out_path = os.path.join(results_dir, 'sample_outputs.txt')
            run_all_test_cases(content_rec, hybrid_rec, out_path)
            print(f"All 3 test cases executed and saved to {out_path}\n")

        elif choice == '0':
            print("\nExiting. Thank you!")
            break
        else:
            print("Invalid choice, please try again.\n")


def main():
    parser = argparse.ArgumentParser(description="Skincare and Cosmetics Product Recommender (MP-A10)")
    parser.add_argument("--all", action="store_true", help="Run all 3 test cases and save outputs")
    parser.add_argument("--evaluate", action="store_true", help="Run offline evaluation and save charts")
    parser.add_argument("--query", type=str, help="Run a custom free-text search query")
    parser.add_argument("--user", type=str, help="Get recommendations for a user ID (e.g., U001)")
    parser.add_argument("--item", type=str, help="Find similar items for an item ID (e.g., P026)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    df_items, df_ratings = load_data(base_dir)

    # Initialize models
    nlp = SkincareNLPPipeline()
    nlp.fit_transform(df_items)

    pop_rec = PopularityBaselineRecommender()
    pop_rec.fit(df_ratings, df_items)

    content_rec = NLPContentBasedRecommender(nlp)
    content_rec.fit(df_items, df_ratings)

    hybrid_rec = SentimentHybridRecommender(content_rec, nlp)
    hybrid_rec.fit(df_ratings, df_items)

    if args.all:
        print_banner()
        out_path = os.path.join(results_dir, 'sample_outputs.txt')
        run_all_test_cases(content_rec, hybrid_rec, out_path)
        return

    if args.evaluate:
        print_banner()
        print("[Evaluating Models & Generating Charts]...")
        c1 = os.path.join(results_dir, 'rating_category_distribution.png')
        plot_dataset_distributions(df_items, df_ratings, c1)

        df_train, df_test = create_train_test_split(df_ratings, test_ratio=0.20, random_state=42)
        metrics = evaluate_recommenders(df_train, df_test, df_items, top_k=5)

        print("\n" + "=" * 65)
        print(f"{'Model':<25} | {'P@5':<7} | {'R@5':<7} | {'Hit@5':<7} | {'RMSE':<7}")
        print("-" * 65)
        for m_name, m_dict in metrics.items():
            print(f"{m_name:<25} | {m_dict['Precision@5']:<7.4f} | {m_dict['Recall@5']:<7.4f} | {m_dict['HitRate@5']:<7.4f} | {m_dict['RMSE']:<7.4f}")
        print("=" * 65)

        c2 = os.path.join(results_dir, 'recommender_evaluation_comparison.png')
        plot_evaluation_comparison(metrics, c2)
        return

    if args.query:
        print_banner()
        recs = hybrid_rec.recommend_by_query(args.query, top_k=5)
        display_recommendations(recs, f"Sentiment-Boosted Recommendations for '{args.query}'")
        return

    if args.user:
        print_banner()
        if args.user in content_rec.user_profiles:
            recs = hybrid_rec.recommend_for_user(args.user, top_k=5)
            display_recommendations(recs, f"Personalized Recommendations for User {args.user}")
        else:
            recs = pop_rec.recommend(args.user, top_k=5)
            display_recommendations(recs, f"Popularity Baseline Recommendations (User {args.user})")
        return

    if args.item:
        print_banner()
        recs = content_rec.recommend_similar_items(args.item, top_k=5)
        display_recommendations(recs, f"Similar Items to [{args.item}]")
        return

    # If no flags passed and running in a terminal, launch interactive menu
    run_interactive_mode(nlp, content_rec, hybrid_rec, pop_rec, df_items, df_ratings, results_dir)


if __name__ == "__main__":
    main()
