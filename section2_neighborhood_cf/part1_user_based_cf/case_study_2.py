import sys
import os
import pandas as pd
import numpy as np

# Add the project root to sys.path to allow importing from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils import data_loader
from utils import similarity

# --- Helper Functions ---

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

def get_top_neighbors(target_user_id, target_user_ratings, all_users_ratings, item_users, similarity_func, n_percent=0.2, similarity_args={}):
    """
    Identifies top n% most similar users using inverted index for efficiency.
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
            # We want correlation, so valid range is -1 to 1.
            # But "Most similar" usually implies positive correlation for neighbors?
            # Standard CF typically uses positive neighbors, or uses all and handles negative.
            # The prompt asks for "Top 20% most similar".
            score = similarity_func(target_user_ratings, ratings, **similarity_args)
            
            # Note: For Pearson, score can be negative. 
            # Case Study 1 filtered score > 0.
            # Usually for prediction we keep K neighbors. If we sort by similarity value descending,
            # positive ones come first.
            if score > 0: # Keeping only positive correlations for "Neighbors" is standard in standard KNN
                 similarities.append((user_id, score))
            # If we want to analyze -1.0 later, maybe we should store them separately or check later?
            # Step 9 asks "Find users whose mean-centered similarity became -1.0".
            # If we filter > 0 here, we miss them.
            # But for *Prediction* (Steps 3, 6), we usually use positive neighbors.
            # Let's modify this to return ALL computed similarities for Analysis, 
            # or handle the filtering outside?
            # To be safe for the "Top 20% most similar" part (which implies high sim),
            # we will grab the top ones from the sorted list.
            # However, for Step 9 analysis, we need to know who got -1.0.
            # So I will NOT filter > 0 here, but filter later for prediction?
            # Wait, "Top 20% similar" definitely implies the highest values.
            # -1.0 is "Dissimilar".
            # So I will append ALL scores != 0 (or even 0) and sort.
            
            # Efficiency: If I keep all, it might be large.
            # But candidate_users is subset.
            if score != 0:
                 similarities.append((user_id, score))

        except Exception:
            continue
            
    # Sort descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Identify Top 20%
    # Top 20% of what? The candidates found.
    top_n_count = int(len(similarities) * n_percent)
    
    # For prediction, we typically only use the Top N.
    return similarities, similarities[:top_n_count]

def predict_ratings_mean_centered(target_user_id, neighbors, all_users_ratings, user_means):
    """
    Predicts ratings using Mean-Centered formula:
    pred = mean_u + sum(sim * (r_v - mean_v)) / sum(|sim|)
    """
    target_mean = user_means[target_user_id]
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
                neighbor_mean = user_means[neighbor_id]
                rating_diff = neighbor_ratings[item_id] - neighbor_mean
                
                numerator += score * rating_diff
                denominator += abs(score)
                
        if denominator > 0:
            pred_rating = target_mean + (numerator / denominator)
            # Clip to valid range? Usually 1-5 or 0-5. Assuming 1-5.
            pred_rating = max(1.0, min(5.0, pred_rating))
            predictions.append((item_id, pred_rating))
            
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

def calculate_discounted_similarity_pearson(user1_ratings, user2_ratings, beta):
    """
    Calculates Discounted Similarity (DS) using Pearson.
    DS = Pearson * min(|I_u n I_v|, beta) / beta
    """
    pearson = similarity.calculate_user_pearson(user1_ratings, user2_ratings)
    
    if pearson == 0:
        return 0.0
        
    common_items = set(user1_ratings.keys()) & set(user2_ratings.keys())
    overlap = len(common_items)
    
    df = min(overlap, beta) / beta
    return pearson * df

def main():
    print("Loading data...")
    try:
        df = data_loader.get_preprocessed_dataset()
        target_users = data_loader.get_target_users()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Data loaded. {len(df)} ratings.")
    
    all_users_ratings = get_user_ratings_dict(df)
    item_users = get_item_users_dict(df)
    user_means = get_user_means(all_users_ratings)
    
    results = {}
    
    for target_user in target_users:
        print(f"\nProcessing Target User: {target_user}")
        target_ratings = all_users_ratings[target_user]
        
        # --- Step 1 & 2: Mean-Centered Cosine (Pearson) & Top 20% ---
        print("  Step 1-2: Computing Mean-Centered Cosine and identifying neighbors...")
        # passing Calculate Pearson. 
        all_sims_pearson, neighbors_pearson = get_top_neighbors(
            target_user, 
            target_ratings, 
            all_users_ratings,
            item_users,
            similarity.calculate_user_pearson
        )
        print(f"  Found {len(neighbors_pearson)} neighbors (Pearson Top 20%). Top 5: {neighbors_pearson[:5]}")
        
        # --- Step 3: Predict (Pearson) ---
        print("  Step 3: Predicting ratings (Pearson)...")
        predictions_pearson = predict_ratings_mean_centered(target_user, neighbors_pearson, all_users_ratings, user_means)
        print(f"  Predicted {len(predictions_pearson)} items.")
        if predictions_pearson:
             print(f"  Top 5 Predictions: {predictions_pearson[:5]}")
        
        # --- Step 4: Calculate Beta & DS ---
        beta = 0.30 * len(target_ratings) # Using integer cast handled by similarity if needed, but min returns float/int
        # beta should not be int if len is small? "Threshold beta" usually implies check count.
        # "beta = 30% of user rated items count"? Assuming implied from Case Study 1 context or previous code.
        # Previous code: beta = 0.30 * len(target_ratings)
        print(f"  Step 4: Beta (30% of {len(target_ratings)}) = {beta}")
        
        # --- Step 5: Top 20% Neighbors (DS) ---
        print("  Step 5: Calculating Discounted Similarity Neighbors...")
        all_sims_ds, neighbors_ds = get_top_neighbors(
            target_user,
            target_ratings,
            all_users_ratings,
            item_users,
            calculate_discounted_similarity_pearson,
            similarity_args={'beta': beta}
        )
        print(f"  Found {len(neighbors_ds)} neighbors (DS Top 20%). Top 5: {neighbors_ds[:5]}")
        
        # --- Step 6: Predict Ratings (DS) ---
        print("  Step 6: Predicting ratings (DS)...")
        predictions_ds = predict_ratings_mean_centered(target_user, neighbors_ds, all_users_ratings, user_means)
        print(f"  Predicted {len(predictions_ds)} items.")
        if predictions_ds:
             print(f"  Top 5 Predictions: {predictions_ds[:5]}")
        
        # --- Step 7: Compare Top-K Lists ---
        print("\n  Step 7: Comparison of Top-K Neighbors")
        s_pearson = set(u for u, s in neighbors_pearson)
        s_ds = set(u for u, s in neighbors_ds)
        
        common = s_pearson & s_ds
        only_p = s_pearson - s_ds
        only_ds_set = s_ds - s_pearson
        
        print(f"    Common: {len(common)}")
        print(f"    Only Pearson: {len(only_p)}")
        print(f"    Only DS: {len(only_ds_set)}")
        
        # --- Step 8: Compare Predictions ---
        print("\n  Step 8: Comparison of Predictions")
        # Compare top 10 item IDs
        top_k = 10
        top_items_p = [i for i, r in predictions_pearson[:top_k]]
        top_items_ds = [i for i, r in predictions_ds[:top_k]]
        
        common_preds = set(top_items_p) & set(top_items_ds)
        print(f"    Top {top_k} Overlap: {len(common_preds)} items.")
        print(f"    Pearson Top {top_k}: {top_items_p}")
        print(f"    DS Top {top_k}:      {top_items_ds}")
        
        # --- Step 9: Analyze -1.0 Correlations ---
        print("\n  Step 9: Analysis of -1.0 Correlations")
        # Find users with Pearson -1.0 (approx)
        neg_one_users = [u for u, s in all_sims_pearson if abs(s + 1.0) < 0.01]
        print(f"    Found {len(neg_one_users)} users with approx -1.0 correlation.")
        
        if neg_one_users:
            # Check their Raw Cosine
            print("    Checking Raw Cosine for sample of -1.0 Pearson users:")
            sample = neg_one_users[:5]
            for u_neg in sample:
                raw_sim = similarity.calculate_user_raw_cosine(target_ratings, all_users_ratings[u_neg])
                pearson_sim = similarity.calculate_user_pearson(target_ratings, all_users_ratings[u_neg])
                common_cnt = len(set(target_ratings) & set(all_users_ratings[u_neg]))
                print(f"      User {u_neg}: Pearson={pearson_sim:.4f}, Raw Cosine={raw_sim:.4f}, Common Items={common_cnt}")
                
        # --- Step 10: Fairness Analysis (Favorites vs Full List) ---
        print("\n  Step 10: Fairness Analysis (Favorites vs Full List)")
        # This asks: "If one some users rate only their favorite items, and other users rate a full list... do they appear unfairly close or far?"
        # We can simulate or look for examples.
        # "Favorites only" -> High Mean. "Full list" -> Lower Mean (contains low ratings).
        # In mean-centered, high mean users (favorites only) will have their ratings centered to near 0 (if variance is small) or centered around 0.
        # If User A (Favorites): {Item1: 5, Item2: 5}. Mean=5. Centered: {I1: 0, I2: 0}. Variance=0. Pearson=Undefined/0.
        # If User A (Favorites): {Item1: 4, Item2: 5}. Mean=4.5. Centered: {I1: -0.5, I2: 0.5}.
        # If User B (Full): {I1: 1, I2: 5, I3: 2...}. Mean=3. Centered: {I1: -2, I2: 2}.
        # Sim(A,B) on {I1, I2}: (-0.5*-2 + 0.5*2) = 1 + 1 = 2.
        # Analysis: Mean-centering helps remove the "bias" of high rater vs low rater.
        # Raw Cosine: (4*1 + 5*5) / (...) = 29 / (...). High.
        # We should comment on this.
        
        # Let's find a "High Mean" neighbor and a "Low Mean" neighbor and compare.
        # Or simply output stats for report.
        print("    Gathering stats for top neighbors...")
        for nid, score in neighbors_pearson[:5]:
            n_mean = user_means[nid]
            n_std = np.std(list(all_users_ratings[nid].values()))
            print(f"      Neighbor {nid}: Mean Rating={n_mean:.2f}, Std Dev={n_std:.2f}, Sim={score:.4f}")

if __name__ == "__main__":
    main()
