import sys
import os
import pandas as pd
import numpy as np

# Add the project root to sys.path to allow importing from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils import data_loader
from utils import similarity

def get_user_ratings_dict(df):
    """
    Converts dataframe to a dictionary of {user_id: {item_id: rating}}
    """
    user_ratings = {}
    # It's faster to iterate if we group by user first or just zip
    # Given the size, let's try a robust way
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

def get_top_neighbors(target_user_id, target_user_ratings, all_users_ratings, item_users, similarity_func, n_percent=0.2, similarity_args={}):
    """
    Identifies top n% most similar users using inverted index for efficiency.
    """
    # Find candidate users: those who rated at least one item that the target user rated
    candidate_users = set()
    target_items = target_user_ratings.keys()
    
    for item in target_items:
        if item in item_users:
            candidate_users.update(item_users[item])
            
    if target_user_id in candidate_users:
        candidate_users.remove(target_user_id)
        
    print(f"  Comparing against {len(candidate_users)} candidate users (with overlap) out of {len(all_users_ratings)} total.")
    
    similarities = []
    count = 0
    
    for user_id in candidate_users:
        ratings = all_users_ratings[user_id]
        try:
            score = similarity_func(target_user_ratings, ratings, **similarity_args)
            if score > 0: 
                similarities.append((user_id, score))
        except Exception:
            continue
        
        count += 1
        if count % 10000 == 0:
            print(f"  Processed {count} candidates...")
            
    # Sort descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Top 20%
    # Note: "Top 20% most similar users". 
    # Does this mean top 20% of ALL users or top 20% of CANDIDATES?
    # Usually it means top K neighbors, but here it says percentage.
    # Assuming top 20% of the *similar* users found (i.e. those with score > 0).
    top_n_count = int(len(similarities) * n_percent)
    
    return similarities[:top_n_count]

def predict_ratings(target_user_id, neighbors, all_users_ratings, all_items):
    """
    Predicts ratings for unrated items using neighbors.
    Formula: Weighted Average
    """
    # Get items rated by target user
    target_rated_items = set(all_users_ratings[target_user_id].keys())
    
    # Items to predict: All items - Rated items
    # For efficiency, we can only predict items that neighbors have rated
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

def calculate_discounted_similarity(user1_ratings, user2_ratings, beta):
    """
    Calculates Discounted Similarity (DS).
    DS = Sim * min(|I_u n I_v|, beta) / beta
    """
    # First calculate raw cosine
    raw_sim = similarity.calculate_user_raw_cosine(user1_ratings, user2_ratings)
    
    if raw_sim == 0:
        return 0.0
        
    common_items = set(user1_ratings.keys()) & set(user2_ratings.keys())
    overlap = len(common_items)
    
    df = min(overlap, beta) / beta
    return raw_sim * df

def main():
    print("Loading data...")
    try:
        df = data_loader.get_preprocessed_dataset()
        target_users = data_loader.get_target_users()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset is present and preprocessing has been run.")
        return

    print(f"Data loaded. {len(df)} ratings.")
    print(f"Target users: {target_users}")
    
    # Convert to dictionary for fast access
    print("Converting data to dictionary format...")
    all_users_ratings = get_user_ratings_dict(df)
    item_users = get_item_users_dict(df)
    all_items = set(df['item'].unique())
    
    results = {}

    for target_user in target_users:
        print(f"\nProcessing Target User: {target_user}")
        target_ratings = all_users_ratings[target_user]
        
        # --- Step 1 & 2: Raw Cosine & Top 20% ---
        print("  Calculating Raw Cosine Neighbors...")
        neighbors_raw = get_top_neighbors(
            target_user, 
            target_ratings, 
            all_users_ratings,
            item_users,
            similarity.calculate_user_raw_cosine
        )
        print(f"  Found {len(neighbors_raw)} neighbors (Raw Cosine). Top 5: {neighbors_raw[:5]}")
        
        # --- Step 3: Predict Ratings (Raw) ---
        print("  Predicting ratings (Raw)...")
        predictions_raw = predict_ratings(target_user, neighbors_raw, all_users_ratings, all_items)
        print(f"  Predicted {len(predictions_raw)} items. Top 5: {predictions_raw[:5]}")
        
        # --- Step 4: Calculate Beta & DS ---
        beta = 0.30 * len(target_ratings)
        print(f"  Beta (30% of {len(target_ratings)}): {beta}")
        
        # --- Step 5: Top 20% Neighbors (DS) ---
        print("  Calculating Discounted Similarity Neighbors...")
        # We need a lambda or wrapper for DS because it takes extra arg 'beta'
        neighbors_ds = get_top_neighbors(
            target_user,
            target_ratings,
            all_users_ratings,
            item_users,
            calculate_discounted_similarity,
            similarity_args={'beta': beta}
        )
        print(f"  Found {len(neighbors_ds)} neighbors (DS). Top 5: {neighbors_ds[:5]}")
        
        # --- Step 6: Predict Ratings (DS) ---
        print("  Predicting ratings (DS)...")
        predictions_ds = predict_ratings(target_user, neighbors_ds, all_users_ratings, all_items)
        print(f"  Predicted {len(predictions_ds)} items. Top 5: {predictions_ds[:5]}")
        
        # --- Step 7: Compare Neighbors ---
        set_raw = set(u for u, s in neighbors_raw)
        set_ds = set(u for u, s in neighbors_ds)
        
        common = set_raw & set_ds
        only_raw = set_raw - set_ds
        only_ds = set_ds - set_raw
        
        print(f"  Neighbor Comparison: Common={len(common)}, Only Raw={len(only_raw)}, Only DS={len(only_ds)}")
        
        # --- Step 8: Compare Predictions ---
        # Compare top 10 predictions maybe?
        top_raw_items = [i for i, r in predictions_raw[:10]]
        top_ds_items = [i for i, r in predictions_ds[:10]]
        print(f"  Top 10 Predictions Raw: {top_raw_items}")
        print(f"  Top 10 Predictions DS:  {top_ds_items}")
        
        results[target_user] = {
            'neighbors_raw': neighbors_raw,
            'neighbors_ds': neighbors_ds,
            'predictions_raw': predictions_raw,
            'predictions_ds': predictions_ds
        }

    # --- Analysis Steps (9-12) ---
    print("\n--- Analysis ---")
    
    # 9. Perfect 1.0 cosine with different item counts
    print("\n9. Perfect 1.0 Cosine Analysis:")
    # We can look at one target user and their raw neighbors
    u_ex = target_users[0]
    perfect_neighbors = [n for n in results[u_ex]['neighbors_raw'] if abs(n[1] - 1.0) < 1e-5]
    print(f"Target User {u_ex} has {len(perfect_neighbors)} perfect neighbors.")
    if perfect_neighbors:
        # Check their item counts
        for pid, score in perfect_neighbors[:5]:
            p_items = len(all_users_ratings[pid])
            common = len(set(all_users_ratings[u_ex]) & set(all_users_ratings[pid]))
            print(f"  User {pid}: Rating Count={p_items}, Common={common}, Score={score}")
            
    # 10. Trust: Common items vs More items
    print("\n10. Trust Analysis (Common vs More Items):")
    # This is a discussion point, but we can find examples.
    # Find a neighbor with high overlap but low total items vs low overlap but high total items?
    # Or rather: "rated the common items" (subset) vs "rated more items than the common items" (superset)
    # If User A has items {1,2} and User B has {1,2,3,4}. If they agree on {1,2}, sim is 1.0.
    # If User C has {1,2} and agrees.
    # We can just print some stats for the perfect neighbors above.
    
    # 11. Low ratings but high cosine
    print("\n11. Low Ratings High Cosine Analysis:")
    # Check if any perfect neighbor has low average rating but high similarity?
    # Cosine is scale-independent? No, raw cosine is NOT scale independent. 
    # Raw cosine of (1,1) and (5,5) -> (1*5 + 1*5) / (sqrt(2)*sqrt(50)) = 10 / (1.414 * 7.07) = 10 / 10 = 1.0
    # So yes, if one user rates everything 1 and another rates everything 5, raw cosine is 1.0.
    # Let's verify this with data.
    for pid, score in perfect_neighbors[:5]:
        p_ratings = list(all_users_ratings[pid].values())
        p_avg = sum(p_ratings)/len(p_ratings)
        t_ratings = list(all_users_ratings[u_ex].values())
        t_avg = sum(t_ratings)/len(t_ratings)
        print(f"  User {pid}: Avg Rating={p_avg:.2f} vs Target Avg={t_avg:.2f} (Sim={score:.2f})")

    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()
