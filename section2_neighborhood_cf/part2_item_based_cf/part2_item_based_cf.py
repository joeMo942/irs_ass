import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to sys.path to allow importing utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils import data_loader
from utils import similarity
from utils import prediction

def main():
    # =========================================================================
    # Part 2: Item-Based Collaborative Filtering
    # =========================================================================
    
    print("\n" + "="*80)
    print("SECTION 2 - PART 2: Item-Based Collaborative Filtering")
    print("="*80)

    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
    print("\n--- Data Loading ---")
    try:
        df = data_loader.get_preprocessed_dataset()
        target_items = data_loader.get_target_items()
    except Exception as e:
        print(f"  [ERROR] {e}")
        return

    print(f"  {'Dataset shape:':<40} {str(df.shape):>15}")
    print(f"  {'Number of target items:':<40} {len(target_items):>15}")

    # Build lookup dictionaries
    print("\n--- Building Lookup Dictionaries ---")
    item_user_ratings = defaultdict(dict)
    user_item_ratings = defaultdict(dict)
    
    # We also need item means for prediction fallback
    item_sums = defaultdict(float)
    item_counts = defaultdict(int)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing"):
        user = row['user']
        item = row['item']
        rating = row['rating']
        
        item_user_ratings[item][user] = rating
        user_item_ratings[user][item] = rating
        
        item_sums[item] += rating
        item_counts[item] += 1
        
    item_means = {k: v / item_counts[k] for k, v in item_sums.items()}
    all_items = list(item_user_ratings.keys())
    print(f"  {'Total items in dataset:':<40} {len(all_items):>15,}")
    
    # Compute user means for Adjusted Cosine similarity
    # Load user means from pre-computed file using data_loader
    print("\n--- Loading User Means ---")
    user_avg_df = data_loader.get_user_avg_ratings()
    user_means = dict(zip(user_avg_df['user'], user_avg_df['r_u_bar']))

    # -------------------------------------------------------------------------
    # CASE STUDY 1: Cosine Similarity with Mean-Centering (Adjusted Cosine)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("CASE STUDY 1: Adjusted Cosine Similarity")
    print("="*80)
    
    # Step 1: Apply item-based collaborative filtering using Cosine similarity
    #         with mean-centering. This normalizes ratings by subtracting each
    #         user's mean rating, accounting for different rating scales.
    print("\n--- Step 1: Calculating Adjusted Cosine Similarities ---")
    
    target_similarities_case1 = {}
    
    for target_item in tqdm(target_items, desc="Target Items (Case 1 - Adjusted Cosine)"):
        if target_item not in item_user_ratings:
            print(f"Warning: Target item {target_item} not found in dataset.")
            continue
            
        sims = {}
        target_ratings = item_user_ratings[target_item]
        
        for other_item in all_items:
            if target_item == other_item:
                sims[other_item] = 1.0
                continue
                
            other_ratings = item_user_ratings[other_item]
            
            # Use Adjusted Cosine (subtracts user means)
            sim = similarity.calculate_item_mean_centered_cosine(target_ratings, other_ratings, user_means)
            
            if sim != 0:
                sims[other_item] = sim
                
        target_similarities_case1[target_item] = sims

    # -------------------------------------------------------------------------
    # Step 2: Identify the top 20% of similar items for each target item
    # -------------------------------------------------------------------------
    print("\n--- Step 2: Selecting Top 20% Neighbors ---")
    
    top_k_percent = int(len(all_items) * 0.2)
    top_k_neighbors_sim = {}
    
    for target_item in target_similarities_case1:
        sims = target_similarities_case1[target_item]
        # Sort desc
        sorted_items = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        # Filter > 0 and take top K
        top_items = [(item, sim) for item, sim in sorted_items if sim > 0][:top_k_percent]
        
        top_k_neighbors_sim[target_item] = top_items
        print(f"  {'Target ' + str(target_item) + ':':<40} {len(top_items):>15} neighbors")

    # -------------------------------------------------------------------------
    # Step 3: Predict the missing ratings using the top 20% similar items
    # -------------------------------------------------------------------------
    print("\n--- Step 3: Predicting Missing Ratings ---")

    print("  Creating validation set (100 unrated items, 50 per target)...")
    validation_set = []
    actuals = []
    
    all_users_list = set(user_item_ratings.keys())
    
    for target_item in target_items:
        rated_users = set(item_user_ratings.get(target_item, {}).keys())
        
        unrated_users = list(all_users_list - rated_users)
        
        # Sample 50
        if len(unrated_users) >= 50:
            sample_indices = np.random.choice(len(unrated_users), 50, replace=False)
            for idx in sample_indices:
                u = unrated_users[idx]
                validation_set.append((u, target_item, np.nan))
                actuals.append(np.nan)
        else:
            print(f"Warning: Not enough unrated users for item {target_item}. Taking all {len(unrated_users)}.")
            for u in unrated_users:
                validation_set.append((u, target_item, np.nan))
                actuals.append(np.nan)

    actuals = np.array(actuals)
    predictions_sim = []
    
    
    # Perform predictions for Step 3
    for u, i, r in validation_set:
        neighbors = top_k_neighbors_sim.get(i, [])
        pred = prediction.predict_mean_centered(u, i, neighbors, item_means, user_item_ratings)
        predictions_sim.append(pred)
        
    predictions_sim = np.array(predictions_sim)
    

    # -------------------------------------------------------------------------
    # Step 4: Compute DF (Discount Factor) and DS (Discounted Similarity)
    #         using threshold β (beta = 30% of target item's rating count)
    #         DS = min(β, co_rated_users) / β
    # -------------------------------------------------------------------------
    print("\n--- Step 4: Computing DF and DS ---")
    
    target_ds_scores = {}
    
    for target_item in target_items:
        if target_item not in item_user_ratings: continue
        
        ds_scores = {}
        target_users = set(item_user_ratings[target_item].keys())
        num_ratings_target = len(target_users)
        
        # Beta = 30% of number of ratings for the target item
        beta = 0.3 * num_ratings_target
        if beta == 0: beta = 1e-9 # Prevent div/0
        
        for other_item in all_items:
            if other_item == target_item:
                ds_scores[other_item] = 1.0
                continue
                
            other_users = set(item_user_ratings[other_item].keys())
            
            co_rated_count = len(target_users.intersection(other_users))
            
            # DS = min(beta, corated_users) / beta
            ds = min(beta, co_rated_count) / beta
            ds_scores[other_item] = ds
            
        target_ds_scores[target_item] = ds_scores
        print(f"  {'Target ' + str(target_item) + ':':<40} Beta = {beta:>10.2f}")

    # -------------------------------------------------------------------------
    # Step 5: Select top 20% items based on discounted similarity (DS)
    #         Weighted similarity = original_similarity * DS
    # -------------------------------------------------------------------------
    print("\n--- Step 5: Selecting DS-Weighted Neighbors ---")
    
    top_k_neighbors_ds = {}
    
    for target_item in target_similarities_case1:
        sims = target_similarities_case1[target_item]
        ds_vals = target_ds_scores[target_item]
        
        weighted_sims = {}
        for other_item, sim in sims.items():
            if sim > 0:
                ds = ds_vals.get(other_item, 0.0)
                weighted_sims[other_item] = sim * ds
                
        sorted_items = sorted(weighted_sims.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:top_k_percent]
        top_items = [(item, score) for item, score in top_items if score > 0]
        
        top_k_neighbors_ds[target_item] = top_items
        print(f"  {'Target ' + str(target_item) + ':':<40} {len(top_items):>15} DS-neighbors")

    # -------------------------------------------------------------------------
    # Step 6: Use the DS-weighted selection for updated rating predictions
    # -------------------------------------------------------------------------
    print("\n--- Step 6: Predicting with DS-Weighted Neighbors ---")
    
    predictions_ds = []
    
    for u, i, r in validation_set:
        neighbors = top_k_neighbors_ds.get(i, [])
        pred = prediction.predict_mean_centered(u, i, neighbors, item_means, user_item_ratings)
        predictions_ds.append(pred)
        
    predictions_ds = np.array(predictions_ds)
    
    
    # -------------------------------------------------------------------------
    # Step 7: Compare similarity lists from steps 2 and 5
    #         Provide commentary on the differences
    # -------------------------------------------------------------------------
    print("\n--- Step 7: Comparing Neighborhoods ---")
    
    sim_comp_path = os.path.join(project_root, 'results', 'Sec2_part2_case1_similarity_comparison.txt')
    with open(sim_comp_path, 'w') as f_sim:
        for target_item in target_items:
            if target_item not in top_k_neighbors_sim: continue
            
            set_sim = set([x[0] for x in top_k_neighbors_sim[target_item]])
            set_ds = set([x[0] for x in top_k_neighbors_ds[target_item]])
            
            header = f"Target {target_item}: Comparison of Top 20 Neighbors"
            col_headers = f"{'Rank':<5} | {'Sim-Item':<10} | {'Sim-Score':<10} | {'DS-Item':<10} | {'DS-Score':<10}"
            separator = "-" * 60
            
            # Print to console
            print(header)
            print(col_headers)
            print(separator)
            
            # Write to file
            f_sim.write(header + "\n")
            f_sim.write(col_headers + "\n")
            f_sim.write(separator + "\n")
            
            # Get top 20 for both
            top_sim = top_k_neighbors_sim.get(target_item, [])[:20]
            top_ds = top_k_neighbors_ds.get(target_item, [])[:20]
            
            max_len = max(len(top_sim), len(top_ds))
            
            for i in range(max_len):
                rank = i + 1
                
                if i < len(top_sim):
                    item_s, score_s = top_sim[i]
                    s_str = f"{item_s:<10} | {score_s:<10.4f}"
                else:
                    s_str = f"{'-':<10} | {'-':<10}"
                    
                if i < len(top_ds):
                    item_d, score_d = top_ds[i]
                    d_str = f"{item_d:<10} | {score_d:<10.4f}"
                else:
                    d_str = f"{'-':<10} | {'-':<10}"
                    
                row_str = f"{rank:<5} | {s_str} | {d_str}"
                print(row_str)
                f_sim.write(row_str + "\n")
            
            print("\n")
            f_sim.write("\n\n")
            
    print(f"  [SAVED] Sec2_part2_case1_similarity_comparison.txt")
        
    # -------------------------------------------------------------------------
    # Steps 8 & 9: Compare predicted ratings from steps 3 and 6
    #              Discuss insights and provide commentary
    # -------------------------------------------------------------------------
    print("\n--- Steps 8-9: Prediction Comparison ---")
    
    pred_comp_path = os.path.join(project_root, 'results', 'Sec2_part2_case1_prediction_comparison.txt')
    
    with open(pred_comp_path, 'w') as f_pred:
        header_title = "Comparison of Predictions (First 20 Samples):"
        col_headers = f"{'User':<15} | {'Item':<12} | {'Sim-Pred':<8} | {'DS-Pred':<8}"
        separator = "-" * 55
        
        # Print
        print(f"\n{header_title}")
        print(col_headers)
        print(separator)
        
        # Write
        f_pred.write(header_title + "\n")
        f_pred.write(col_headers + "\n")
        f_pred.write(separator + "\n")
        
        for k in range(min(20, len(actuals))):
            u, i, r = validation_set[k]
            p_sim = predictions_sim[k]
            p_ds = predictions_ds[k]
            
            row_str = f"{u:<15} | {i:<12} | {p_sim:<8.2f} | {p_ds:<8.2f}"
            print(row_str)
            f_pred.write(row_str + "\n")
            
    print(f"  [SAVED] Sec2_part2_case1_prediction_comparison.txt")
    
    # =========================================================================
    # CASE STUDY 2: PCC (Pearson Correlation Coefficient) Similarity
    # =========================================================================
    # This case study uses PCC to compute similarity between target items.
    # Unlike Adjusted Cosine (Case 1), PCC normalizes by ITEM means computed
    # over co-rated users, rather than user means.
    # =========================================================================
    print("\n" + "="*80)
    print("CASE STUDY 2: Pearson Correlation Coefficient (PCC)")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # Step 1: Use PCC to compute similarity between target items
    # -------------------------------------------------------------------------
    print("\n--- Step 1: Calculating PCC Similarities ---")
    
    target_similarities_case2 = {}
    
    for target_item in tqdm(target_items, desc="Target Items (Case 2 - PCC)"):
        if target_item not in item_user_ratings:
            continue
            
        sims = {}
        target_ratings = item_user_ratings[target_item]
        
        for other_item in all_items:
            if target_item == other_item:
                sims[other_item] = 1.0
                continue
                
            other_ratings = item_user_ratings[other_item]
            
            # Use Pearson Correlation Coefficient
            sim = similarity.calculate_item_pearson(target_ratings, other_ratings)
            
            if sim != 0:
                sims[other_item] = sim
                
        target_similarities_case2[target_item] = sims
    
    # -------------------------------------------------------------------------
    # Step 2: Identify the top 20% most similar items using PCC
    # -------------------------------------------------------------------------
    print("\n--- Step 2: Selecting Top 20% PCC Neighbors ---")
    
    top_k_neighbors_sim_case2 = {}
    
    for target_item in target_similarities_case2:
        sims = target_similarities_case2[target_item]
        sorted_items = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        top_items = [(item, sim) for item, sim in sorted_items if sim > 0][:top_k_percent]
        
        top_k_neighbors_sim_case2[target_item] = top_items
        print(f"  {'Target ' + str(target_item) + ':':<40} {len(top_items):>15} PCC neighbors")
    
    # -------------------------------------------------------------------------
    # Step 3: Predict the missing ratings using PCC-based neighbors
    # -------------------------------------------------------------------------
    print("\n--- Step 3: Predicting with PCC Neighbors ---")
    
    predictions_sim_case2 = []
    
    for u, i, r in validation_set:
        neighbors = top_k_neighbors_sim_case2.get(i, [])
        pred = prediction.predict_pearson(u, i, neighbors, item_means, user_item_ratings)
        predictions_sim_case2.append(pred)
        
    predictions_sim_case2 = np.array(predictions_sim_case2)
    
    # -------------------------------------------------------------------------
    # Case 2 - Step 4: Compute DF and DS with threshold beta
    # (Reusing target_ds_scores from Case 1 - same beta computation)
    # -------------------------------------------------------------------------
    print("\n--- Step 4: Using DS Scores (from Case 1) ---")
    
    # -------------------------------------------------------------------------
    # Step 5: Select top 20% items based on discounted PCC similarity
    #         DS weights are applied to PCC similarities
    # -------------------------------------------------------------------------
    print("\n--- Step 5: Selecting DS-Weighted PCC Neighbors ---")
    
    top_k_neighbors_ds_case2 = {}
    
    for target_item in target_similarities_case2:
        sims = target_similarities_case2[target_item]
        ds_vals = target_ds_scores[target_item]
        
        weighted_sims = {}
        for other_item, sim in sims.items():
            if sim > 0:
                ds = ds_vals.get(other_item, 0.0)
                weighted_sims[other_item] = sim * ds
                
        sorted_items = sorted(weighted_sims.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:top_k_percent]
        top_items = [(item, score) for item, score in top_items if score > 0]
        
        top_k_neighbors_ds_case2[target_item] = top_items
        print(f"  {'Target ' + str(target_item) + ':':<40} {len(top_items):>15} DS-PCC neighbors")
    
    # -------------------------------------------------------------------------
    # Step 6: Use DS-weighted PCC selection for updated rating predictions
    # -------------------------------------------------------------------------
    print("\n--- Step 6: Predicting with DS-Weighted PCC ---")
    
    predictions_ds_case2 = []
    
    for u, i, r in validation_set:
        neighbors = top_k_neighbors_ds_case2.get(i, [])
        pred = prediction.predict_pearson(u, i, neighbors, item_means, user_item_ratings)
        predictions_ds_case2.append(pred)
        
    predictions_ds_case2 = np.array(predictions_ds_case2)
    
    # -------------------------------------------------------------------------
    # Step 7: Compare item lists from steps 2 and 5, provide analysis
    # -------------------------------------------------------------------------
    print("\n--- Step 7: Comparing PCC Neighborhoods ---")
    
    c2_sim_path = os.path.join(project_root, 'results', 'Sec2_part2_case2_similarity_comparison.txt')
    with open(c2_sim_path, 'w') as f_sim:
        f_sim.write("Case Study 2: PCC Similarity Analysis\n")
        f_sim.write("=" * 60 + "\n\n")
        
        for target_item in target_items:
            if target_item not in top_k_neighbors_sim_case2: continue
            
            header = f"Target {target_item}: Comparison of Top 20 PCC Neighbors"
            col_headers = f"{'Rank':<5} | {'PCC-Item':<10} | {'PCC-Score':<10} | {'DS-Item':<10} | {'DS-Score':<10}"
            separator = "-" * 60
            
            print(header)
            print(col_headers)
            print(separator)
            
            f_sim.write(header + "\n")
            f_sim.write(col_headers + "\n")
            f_sim.write(separator + "\n")
            
            top_sim = top_k_neighbors_sim_case2.get(target_item, [])[:20]
            top_ds = top_k_neighbors_ds_case2.get(target_item, [])[:20]
            max_len = max(len(top_sim), len(top_ds))
            
            for i in range(max_len):
                rank = i + 1
                if i < len(top_sim):
                    item_s, score_s = top_sim[i]
                    s_str = f"{item_s:<10} | {score_s:<10.4f}"
                else: 
                    s_str = f"{'-':<10} | {'-':<10}"
                    
                if i < len(top_ds):
                    item_d, score_d = top_ds[i]
                    d_str = f"{item_d:<10} | {score_d:<10.4f}"
                else: 
                    d_str = f"{'-':<10} | {'-':<10}"
                    
                row = f"{rank:<5} | {s_str} | {d_str}"
                print(row)
                f_sim.write(row + "\n")
                
            print("\n")
            f_sim.write("\n\n")
        
    print(f"  [SAVED] Sec2_part2_case2_similarity_comparison.txt")

    # -------------------------------------------------------------------------
    # Steps 8 & 9: Compare predictions from steps 3 and 6, share insights
    #              Give comments in a separate section
    # -------------------------------------------------------------------------
    print("\n--- Steps 8-9: PCC Prediction Comparison ---")
    
    c2_pred_path = os.path.join(project_root, 'results', 'Sec2_part2_case2_prediction_comparison.txt')
    with open(c2_pred_path, 'w') as f_pred:
        f_pred.write("Case Study 2: PCC Prediction Comparison\n")
        f_pred.write("=" * 55 + "\n\n")
        
        header = "Comparison of Predictions (First 20 Samples):"
        cols = f"{'User':<15} | {'Item':<12} | {'PCC-Pred':<8} | {'DS-Pred':<8}"
        sep = "-" * 55
        
        f_pred.write(header + "\n")
        f_pred.write(cols + "\n")
        f_pred.write(sep + "\n")
        print("\n" + header)
        print(cols)
        print(sep)
        
        for k in range(min(20, len(predictions_sim_case2))):
            u, i, _ = validation_set[k]
            p_s = predictions_sim_case2[k]
            p_d = predictions_ds_case2[k]
            
            row = f"{u:<15} | {i:<12} | {p_s:<8.2f} | {p_d:<8.2f}"
            f_pred.write(row + "\n")
            print(row)
            
    print(f"  [SAVED] Sec2_part2_case2_prediction_comparison.txt")
    
    # =========================================================================
    # COMPLETION
    # =========================================================================
    print("\n" + "="*80)
    print("[DONE] Section 2 - Part 2: Item-Based CF completed successfully.")
    print("="*80)

if __name__ == "__main__":
    main()
