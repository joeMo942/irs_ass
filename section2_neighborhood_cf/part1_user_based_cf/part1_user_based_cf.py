import sys
import os
import pandas as pd
import numpy as np

# Add the project root to sys.path to allow importing from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(_file_), '..', '..')))

from utils import data_loader
from utils import similarity


def get_user_ratings_dict(df):
    """
    Converts dataframe to a dictionary of {user_id: {item_id: rating}}
    """
    user_ratings = {}
    for user, group in df.groupby('user'):
        user_ratings[user] = dict(zip(group['item'], group['rating']))
    return user_ratings

def get_item_users_dict(df):
    """
    Converts dataframe to a dictionary of {item_id: set(user_ids)}
    """
    item_users = {}
    for item, group in df.groupby('item'):
        item_users[item] = set(group['user'])
    return item_users

def get_user_means(all_users_ratings):
    """
    Calculates the mean rating for each user.
    """
    means = {}
    for user, ratings in all_users_ratings.items():
        if ratings:
            means[user] = sum(ratings.values()) / len(ratings)
        else:
            means[user] = 0.0
    return means

def get_top_neighbors(target_user_id, target_user_ratings, all_users_ratings, item_users, similarity_func, n_percent=0.2, similarity_args={}, only_positive=False):
    """
    Identifies top n% most similar users using inverted index for efficiency.
    Returns tuple: (all_similarities, top_n_similarities)
    """
    candidate_users = set()
    target_items = target_user_ratings.keys()
    
    for item in target_items:
        if item in item_users:
            candidate_users.update(item_users[item])
            
    if target_user_id in candidate_users:
        candidate_users.remove(target_user_id)
        
    print(f"  Comparing against {len(candidate_users)} candidate users.")
    
    similarities = []
    
    for user_id in candidate_users:
        ratings = all_users_ratings[user_id]
        try:
            score = similarity_func(target_user_ratings, ratings, **similarity_args)
            
            if only_positive:
                if score > 0:
                    similarities.append((user_id, score))
            else:
                if score != 0:
                    similarities.append((user_id, score))

        except Exception:
            continue
            
    # Sort descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Identify Top n%
    top_n_count = int(len(similarities) * n_percent)
    
    return similarities, similarities[:top_n_count]

# ==========================================
# Similarity Wrappers
# ==========================================

def calculate_discounted_similarity_cosine(user1_ratings, user2_ratings, beta):
    """
    Calculates Discounted Similarity (DS) using Raw Cosine.
    """
    raw_sim = similarity.calculate_user_raw_cosine(user1_ratings, user2_ratings)
    
    if raw_sim == 0:
        return 0.0
        
    common_items = set(user1_ratings.keys()) & set(user2_ratings.keys())
    overlap = len(common_items)
    
    df = min(overlap, beta) / beta
    return raw_sim * df

def calculate_discounted_similarity_pearson(user1_ratings, user2_ratings, beta):
    """
    Calculates Discounted Similarity (DS) using Pearson.
    """
    pearson = similarity.calculate_user_pearson(user1_ratings, user2_ratings)
    
    if pearson == 0:
        return 0.0
        
    common_items = set(user1_ratings.keys()) & set(user2_ratings.keys())
    overlap = len(common_items)
    
    df = min(overlap, beta) / beta
    return pearson * df

# ==========================================
# Prediction Functions
# ==========================================

def predict_ratings_weighted_avg(target_user_id, neighbors, all_users_ratings, all_items):
    """
    Predicts ratings using Weighted Average (Case Study 1).
    """
    target_rated_items = set(all_users_ratings[target_user_id].keys())
    
    candidate_items = set()
    for neighbor_id, _ in neighbors:
        candidate_items.update(all_users_ratings[neighbor_id].keys())
        
    items_to_predict = candidate_items - target_rated_items
    
    predictions = []
    
    for item_id in items_to_predict:
        numerator = 0.0
        denominator = 0.0
        
        for neighbor_id, score in neighbors:
            neighbor_ratings = all_users_ratings[neighbor_id]
            if item_id in neighbor_ratings:
                numerator += score * neighbor_ratings[item_id]
                denominator += abs(score)
                
        if denominator > 0:
            pred_rating = numerator / denominator
            predictions.append((item_id, pred_rating))
            
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

def predict_ratings_mean_centered(target_user_id, neighbors, all_users_ratings, user_means):
    """
    Predicts ratings using Mean-Centered formula (Case Study 2/3).
    """
    target_mean = user_means[target_user_id]
    target_rated_items = set(all_users_ratings[target_user_id].keys())
    
    candidate_items = set()
    for neighbor_id, score in neighbors:
        if score > 0: 
            candidate_items.update(all_users_ratings[neighbor_id].keys())
        
    items_to_predict = candidate_items - target_rated_items
    
    predictions = []
    
    for item_id in items_to_predict:
        numerator = 0.0
        denominator = 0.0
        
        for neighbor_id, score in neighbors:
            if score <= 0: continue

            neighbor_ratings = all_users_ratings[neighbor_id]
            if item_id in neighbor_ratings:
                neighbor_mean = user_means[neighbor_id]
                # Formula: mean + sum(sim * (r - mean)) / sum(|sim|)
                rating_diff = neighbor_ratings[item_id] - neighbor_mean
                
                numerator += score * rating_diff
                denominator += abs(score)
                
        if denominator > 0:
            pred_rating = target_mean + (numerator / denominator)
            pred_rating = max(1.0, min(5.0, pred_rating))
            predictions.append((item_id, pred_rating))
            
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

# ==========================================
# Analysis Functions
# ==========================================

def analyze_negative_sim_positive_cosine(target_user, target_ratings, all_users_ratings):
    print("\n  Analysis: Negative Pearson but Positive Cosine")
    count = 0
    for user_id, ratings in all_users_ratings.items():
        if user_id == target_user: continue
        
        common = set(target_ratings) & set(ratings)
        if not common: continue
        
        p = similarity.calculate_user_pearson(target_ratings, ratings)
        if p < -0.5: 
            c = similarity.calculate_user_raw_cosine(target_ratings, ratings)
            if c > 0.1:
                print(f"    User {user_id}: Pearson={p:.3f}, Cosine={c:.3f}, Common={len(common)}")
                count += 1
                if count >= 3: break

def analyze_rating_scales(target_user, target_ratings, neighbors, user_means):
    print("\n  Analysis: Different Rating Scales (Generous vs Strict)")
    target_mean = user_means[target_user]
    max_diff = 0
    best_ex = None
    neighbor_dict = dict(neighbors)
    
    for nid, score in neighbors:
        if score < 0.8: continue
        nm = user_means[nid]
        diff = abs(target_mean - nm)
        if diff > max_diff:
            max_diff = diff
            best_ex = nid
            
    if best_ex:
        print(f"    Target Mean: {target_mean:.2f}. Neighbor {best_ex} Mean: {user_means[best_ex]:.2f}. Diff: {max_diff:.2f}. Sim: {neighbor_dict[best_ex]:.3f}")
    else:
        print("    No high-sim neighbor with large mean difference found.")


def run_case_study_1(target_user, target_ratings, all_users_ratings, item_users, all_items):
    print(f"\n--- Case Study 1: User-Based CF (Raw Cosine) for {target_user} ---")
    
    # 1 & 2: Raw Cosine & Top 20%
    print("  Calculating Raw Cosine Neighbors...")
    _, neighbors_raw = get_top_neighbors(
        target_user, 
        target_ratings, 
        all_users_ratings,
        item_users,
        similarity.calculate_user_raw_cosine,
        only_positive=True
    )
    print(f"  Found {len(neighbors_raw)} neighbors (Raw Cosine). Top 5: {neighbors_raw[:5]}")
    
    # 3: Predict Raw
    print("  Predicting ratings (Raw)...")
    predictions_raw = predict_ratings_weighted_avg(target_user, neighbors_raw, all_users_ratings, all_items)
    print(f"  Predicted {len(predictions_raw)} items. Top 5: {predictions_raw[:5]}")
    
    # 4: Beta
    beta = 0.30 * len(target_ratings)
    print(f"  Beta: {beta}")
    
    # 5: DS
    print("  Calculating Discounted Similarity (Cosine) Neighbors...")
    _, neighbors_ds = get_top_neighbors(
        target_user,
        target_ratings,
        all_users_ratings,
        item_users,
        calculate_discounted_similarity_cosine,
        similarity_args={'beta': beta},
        only_positive=True
    )
    print(f"  Found {len(neighbors_ds)} neighbors (DS). Top 5: {neighbors_ds[:5]}")
    
    # 6: Predict DS
    print("  Predicting ratings (DS)...")
    predictions_ds = predict_ratings_weighted_avg(target_user, neighbors_ds, all_users_ratings, all_items)
    print(f"  Predicted {len(predictions_ds)} items. Top 5: {predictions_ds[:5]}")
    
    # Comparison
    set_raw = set(u for u, s in neighbors_raw)
    set_ds = set(u for u, s in neighbors_ds)
    print(f"  Comparison: Common={len(set_raw & set_ds)}, Only Raw={len(set_raw - set_ds)}, Only DS={len(set_ds - set_raw)}")

def run_case_study_2(target_user, target_ratings, all_users_ratings, item_users, user_means):
    print(f"\n--- Case Study 2: User-Based CF (Pearson) for {target_user} ---")
    
    # 1 & 2: Pearson & Top 20%
    print("  Calculating Pearson Neighbors...")
    all_sims_pearson, neighbors_pearson = get_top_neighbors(
        target_user, 
        target_ratings, 
        all_users_ratings,
        item_users,
        similarity.calculate_user_pearson,
        only_positive=False # CS2 allows negative for distribution analysis, though top 20% likely positive
    )
    print(f"  Found {len(neighbors_pearson)} neighbors (Pearson). Top 5: {neighbors_pearson[:5]}")
    
    # 3: Predict Pearson
    print("  Predicting ratings (Pearson)...")
    predictions_pearson = predict_ratings_mean_centered(target_user, neighbors_pearson, all_users_ratings, user_means)
    print(f"  Predicted {len(predictions_pearson)} items. Top 5: {predictions_pearson[:5]}")
    
    # 4: Beta
    beta = 0.30 * len(target_ratings)
    
    # 5: DS Pearson
    print("  Calculating Discounted Similarity (Pearson) Neighbors...")
    _, neighbors_ds = get_top_neighbors(
        target_user,
        target_ratings,
        all_users_ratings,
        item_users,
        calculate_discounted_similarity_pearson,
        similarity_args={'beta': beta},
        only_positive=False
    )
    
    # 6: Predict DS
    print("  Predicting ratings (DS)...")
    predictions_ds = predict_ratings_mean_centered(target_user, neighbors_ds, all_users_ratings, user_means)
    print(f"  Predicted {len(predictions_ds)} items. Top 5: {predictions_ds[:5]}")

    return all_sims_pearson # For CS3 or deeper analysis if needed

def run_case_study_3(target_user, target_ratings, all_users_ratings, item_users, user_means, all_items):
    print(f"\n--- Case Study 3: Analysis & Comparisons for {target_user} ---")
    
    # We need neighbors for analysis.
    all_sims_p, neighbors_p = get_top_neighbors(target_user, target_ratings, all_users_ratings, item_users, similarity.calculate_user_pearson, only_positive=False)
    
    # Analysis 1: Negative Pearson but Positive Cosine
    analyze_negative_sim_positive_cosine(target_user, target_ratings, all_users_ratings)
    
    # Analysis 2: Rating Scales
    analyze_rating_scales(target_user, target_ratings, neighbors_p, user_means)


def main():
    print("Loading data...")
    try:
        df = data_loader.get_preprocessed_dataset()
        target_users = data_loader.get_target_users()
    except Exception as e:
        print(f"Error: {e}")
        return

    print("Data loaded. Converting to dictionaries...")
    all_users_ratings = get_user_ratings_dict(df)
    item_users = get_item_users_dict(df)
    user_means = get_user_means(all_users_ratings)
    all_items = set(df['item'].unique())
    
    for target_user in target_users:
        print(f"\n{'='*60}\nPROCESSING TARGET USER: {target_user}\n{'='*60}")
        
        # Run Case Study 1
        run_case_study_1(target_user, all_users_ratings[target_user], all_users_ratings, item_users, all_items)
        
        # Run Case Study 2
        run_case_study_2(target_user, all_users_ratings[target_user], all_users_ratings, item_users, user_means)
        
        # Run Case Study 3
        run_case_study_3(target_user, all_users_ratings[target_user], all_users_ratings, item_users, user_means, all_items)
        
    print("\nAll Case Studies Completed.")

if _name_ == "_main_":
    main()