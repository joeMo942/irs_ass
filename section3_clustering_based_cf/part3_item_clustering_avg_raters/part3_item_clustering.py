import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Add project root to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import data_loader
from utils.similarity import calculate_item_mean_centered_cosine
from utils.prediction import predict_item_based


def main():
    # --- Configuration ---
    RANDOM_STATE = 42
    K_VALUES = [5, 10, 15, 20, 30, 50]
    OPTIMAL_K = 10
    SAMPLE_SIZE_SILHOUETTE = 40000
    RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    print("Loading data...")
    df = data_loader.get_preprocessed_dataset()
    r_i = data_loader.get_item_avg_ratings()

    # ---------------------------------------------------------
    # 2. Feature Engineering
    # ---------------------------------------------------------
    print("Computing item statistics...")
    
    # Task 1.1 & 1.2: Calculate Num Raters and Std Dev per Item
    item_stats = df.groupby('item')['rating'].agg(['count', 'std']).reset_index()
    item_stats.rename(columns={'count': 'num_raters', 'std': 'std_rating'}, inplace=True)
    item_stats['std_rating'] = item_stats['std_rating'].fillna(0)

    # Task 1.3: Create Feature Vector [num_raters, avg_rating, std_rating]
    # 2.2 Merge with Average Rating (r_i)
    # r_i likely has columns ['item', 'r_i_bar'] or similar
    feature_df = pd.merge(item_stats, r_i, on='item', how='inner')
    
    # Identify the average rating column (column that is not 'item')
    avg_col = [c for c in r_i.columns if c != 'item'][0]
    
    feature_cols = ['num_raters', avg_col, 'std_rating']
    X = feature_df[feature_cols].copy()
    
    print(f"Feature vector shape: {X.shape}")
    print(f"Features: {feature_cols}")

    # Task 2: Normalize Features (Z-score standardization)
    print("Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Mean after scaling: {np.mean(X_scaled, axis=0)}")
    print(f"Std after scaling: {np.std(X_scaled, axis=0)}")

    # ---------------------------------------------------------
    # 3. Clustering Analysis (Elbow & Silhouette)
    # ---------------------------------------------------------
    print("Starting clustering loop...")
    wcss_list = []
    silhouette_scores = []
    
    # Task 3: Apply K-means clustering (K=5, 10, 15, 20, 30, 50)
    for k in K_VALUES:
        print(f"  Clustering with K={k}...")
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Task 3.3: Calculate WCSS
        wcss_list.append(kmeans.inertia_)
        
        # Calculate Silhouette Score (sample if dataset is large)
        if len(X_scaled) > SAMPLE_SIZE_SILHOUETTE:
            score = silhouette_score(X_scaled, labels, sample_size=SAMPLE_SIZE_SILHOUETTE, random_state=RANDOM_STATE)
        else:
            score = silhouette_score(X_scaled, labels)
            
        silhouette_scores.append(score)
        print(f"    WCSS: {kmeans.inertia_:.4f}, Silhouette: {score:.4f}")

    # Plotting Metrics
    print("Generating metric plots...")
    plt.figure(figsize=(15, 6))

    # Elbow Curve
    plt.subplot(1, 2, 1)
    plt.plot(K_VALUES, wcss_list, 'bx-')
    plt.xlabel('k')
    plt.ylabel('WCSS')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True)

    # Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(K_VALUES, silhouette_scores, 'rx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for different k')
    plt.grid(True)

    # Task 4.1: Plot the elbow curve and silhouette scores
    output_plot_path = os.path.join(RESULTS_DIR, 'sec3_part3_clustering_metrics.png')
    plt.tight_layout()
    plt.savefig(output_plot_path)
    print(f"Plots saved to {output_plot_path}")

    # ---------------------------------------------------------
    # 4. Cluster Analysis with Optimal K
    # ---------------------------------------------------------
    # Task 4.2: Select optimal K (Hardcoded based on analysis)
    # Task 5: Analyze characteristics of each item cluster
    print(f"\nStarting Cluster Analysis (K={OPTIMAL_K})...")
    kmeans_opt = KMeans(n_clusters=OPTIMAL_K, random_state=RANDOM_STATE, n_init=10)
    feature_df['cluster'] = kmeans_opt.fit_predict(X_scaled)
    
    # 4.1 Cluster Statistics
    cluster_stats = feature_df.groupby('cluster')['num_raters'].agg(['mean', 'count']).reset_index()
    cluster_stats = cluster_stats.sort_values('mean', ascending=False)
    
    print("\nCluster Statistics (Sorted by Avg Raters):")
    print(cluster_stats)
    
    # 4.2 Head vs Tail Analysis
    # Top 20% by popularity = Head
    popularity_threshold = feature_df['num_raters'].quantile(0.8)
    feature_df['type'] = feature_df['num_raters'].apply(lambda x: 'Head' if x >= popularity_threshold else 'Tail')
    
    print(f"\nPopularity Threshold (top 20%): {popularity_threshold:.2f} raters")
    
    head_tail_dist = feature_df.groupby(['cluster', 'type']).size().unstack(fill_value=0)
    
    # 4.3 Visualizing Characteristics
    print("Generating analysis plots...")
    plt.figure(figsize=(18, 6))

    # Item Distribution
    plt.subplot(1, 3, 1)
    sns.countplot(data=feature_df, x='cluster', palette='viridis', hue='cluster', legend=False)
    plt.title('Item Distribution across Clusters')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Items')

    # Rater Distribution (Boxplot)
    plt.subplot(1, 3, 2)
    sns.boxplot(data=feature_df, x='cluster', y='num_raters', palette='viridis', hue='cluster', legend=False)
    plt.yscale('log')
    plt.title('Distribution of # Raters per Cluster (Log Scale)')
    plt.xlabel('Cluster ID') # Fixed typo from original
    plt.ylabel('Number of Raters (Log)')

    # Head vs Tail
    plt.subplot(1, 3, 3)
    head_tail_dist.plot(kind='bar', stacked=True, ax=plt.gca(), color=['red', 'blue'])
    plt.title('Head (Popular) vs Tail (Niche) Composition')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Items')
    plt.legend(title='Item Type')

    # Task 5.5 & 6.1: Visualize distribution
    analysis_plot_path = os.path.join(RESULTS_DIR, 'sec3_part3_cluster_analysis.png')
    plt.tight_layout()
    plt.savefig(analysis_plot_path)
    print(f"Analysis plots saved to {analysis_plot_path}")

    # ---------------------------------------------------------
    # Task 7 & 8: Prediction Comparison (Baseline vs Clustered)
    # ---------------------------------------------------------
    print("\n--- Tasks 7 & 8: Prediction Comparison ---")
    
    # Load targets
    target_users = data_loader.get_target_users()
    target_items = data_loader.get_target_items()
    
    print(f"Target Users: {target_users}")
    print(f"Target Items: {target_items}")
    
    # Prepare Helper Dictionaries
    # User Means
    user_means_df = data_loader.get_user_avg_ratings()
    if 'user' in user_means_df.columns:
        user_means = dict(zip(user_means_df['user'], user_means_df['r_u_bar']))
    else:
        user_means = dict(zip(user_means_df.iloc[:,0], user_means_df.iloc[:,1]))

    # Item Means
    item_means_df = data_loader.get_item_avg_ratings()
    item_avg_col = [c for c in item_means_df.columns if c != 'item'][0]
    item_means = dict(zip(item_means_df['item'], item_means_df[item_avg_col]))

    # Rating Lookups
    print("Building rating dictionaries...")
    # Item -> {User: Rating}
    item_user_ratings = df.groupby('item').apply(lambda x: dict(zip(x['user'], x['rating']))).to_dict()
    # User -> {Item: Rating}
    user_item_ratings = df.groupby('user').apply(lambda x: dict(zip(x['item'], x['rating']))).to_dict()

    # Cluster Lookup
    cluster_items_map = feature_df.groupby('cluster')['item'].apply(list).to_dict()
    item_to_cluster = dict(zip(feature_df['item'], feature_df['cluster']))

    # Baseline Candidate Set (Global)
    all_items = list(item_user_ratings.keys())

    print(f"\n{'User':<10} | {'Item':<10} | {'Actual':<6} | {'BasePred':<8} | {'ClusPred':<8} | {'ErrBase':<7} | {'ErrClus':<7}")
    print("-" * 80)

    mae_base_list = []
    mae_clus_list = []

    # Task 9 Variables
    item_types = dict(zip(feature_df['item'], feature_df['type']))
    tail_errors_base = []
    tail_errors_clus = []
    tail_candidates_base = []
    tail_candidates_clus = []

    # Task 11 Variables
    cluster_errors = {c: [] for c in range(OPTIMAL_K)}

    for u in target_users:
        for i in target_items:
            # Common data for this pair
            t_item_ratings = item_user_ratings.get(i, {})
            
            # --- 5.1 Baseline Prediction (Global Neighbors) ---
            base_similarities = []
            for cand_item in all_items:
                if cand_item == i:
                    continue
                cand_ratings = item_user_ratings.get(cand_item, {})
                try:
                    sim = calculate_item_mean_centered_cosine(t_item_ratings, cand_ratings, user_means)
                    if sim > 0:
                        base_similarities.append((cand_item, sim))
                except Exception:
                    continue
            
            # Top 20%
            base_similarities.sort(key=lambda x: x[1], reverse=True)
            k_base = max(1, int(len(base_similarities) * 0.20))
            top_neighbors_base = base_similarities[:k_base]
            
            base_pred = predict_item_based(u, i, top_neighbors_base, user_item_ratings, user_means, item_means)

            # --- Task 7.2 & 7.3 Clustered Prediction (Local Neighbors) ---
            clus_similarities = []
            
            # Identify cluster for item i
            i_cluster = item_to_cluster.get(i)
            if i_cluster is not None:
                cluster_candidates = cluster_items_map.get(i_cluster, [])
                
                for cand_item in cluster_candidates:
                    if cand_item == i:
                        continue
                    cand_ratings = item_user_ratings.get(cand_item, {})
                    try:
                        sim = calculate_item_mean_centered_cosine(t_item_ratings, cand_ratings, user_means)
                        if sim > 0:
                            clus_similarities.append((cand_item, sim))
                    except Exception:
                        continue

            # Top 20%
            clus_similarities.sort(key=lambda x: x[1], reverse=True)
            k_clus = max(1, int(len(clus_similarities) * 0.20))
            top_neighbors_clus = clus_similarities[:k_clus]
            
            clus_pred = predict_item_based(u, i, top_neighbors_clus, user_item_ratings, user_means, item_means)

            # --- Result ---

            # 8.2 Calculate prediction error
            # Actual rating: if missing, use user's average
            actual_rating = t_item_ratings.get(u)
            if actual_rating is None:
                actual_rating = user_means.get(u, 3.0) # Fallback to 3.0 if user mean missing
            
            err_base = abs(actual_rating - base_pred)
            err_clus = abs(actual_rating - clus_pred)
            # Task 8.2: Calculate prediction error
            # Actual rating: if missing, use user's average
            actual_rating = t_item_ratings.get(u)
            if actual_rating is None:
                actual_rating = user_means.get(u, 3.0) # Fallback to 3.0 if user mean missing
            
            err_base = abs(actual_rating - base_pred)
            err_clus = abs(actual_rating - clus_pred)
            
            mae_base_list.append(err_base)
            mae_clus_list.append(err_clus)

            print(f"{u:<10} | {i:<10} | {actual_rating:<6.2f} | {base_pred:<8.2f} | {clus_pred:<8.2f} | {err_base:<7.2f} | {err_clus:<7.2f}")

            # --- Task 11 Collection (Cluster Size) ---
            if i_cluster is not None:
                cluster_errors[i_cluster].append(err_clus)

    # Task 8.3 Comparison Summary
    avg_mae_base = np.mean(mae_base_list) if mae_base_list else 0
    avg_mae_clus = np.mean(mae_clus_list) if mae_clus_list else 0
    
    print("\n" + "="*50)
    print("SECTION 8.2 & 8.3: PREDICTION ERROR ANALYSIS")
    print("="*50)
    print(f"Overall MAE (Baseline - Global):   {avg_mae_base:.4f}")
    print(f"Overall MAE (Clustering - Local):  {avg_mae_clus:.4f}")
    print("-" * 50)
    
    if avg_mae_clus < avg_mae_base:
        print("CONCLUSION: Clustering-based approach produces more reliable predictions (Lower Error).")
    else:
        print("CONCLUSION: Baseline approach produces more reliable predictions (Lower Error).")
    print("="*50 + "\n")


    # ---------------------------------------------------------
    # Task 9: Long-Tail Analysis (Evaluate Random Sample of Tail Items)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("TASK 9: LONG-TAIL ANALYSIS")
    print("="*50)

    # 1. Select Random Tail Items
    tail_items_list = feature_df[feature_df['type'] == 'Tail']['item'].tolist()
    
    # Sample 50 tail items for evaluation (to ensure we have data)
    np.random.seed(RANDOM_STATE)
    if len(tail_items_list) > 50:
        sample_tail_items = np.random.choice(tail_items_list, 50, replace=False)
    else:
        sample_tail_items = tail_items_list

    print(f"Sampling {len(sample_tail_items)} Tail Items for detailed analysis...")

    tail_errors_base = []
    tail_errors_clus = []
    tail_candidates_base = []
    tail_candidates_clus = []
    
    # Re-use prediction logic for these items
    # We need a user who rated these items to check accuracy. 
    # Strategy: For each sampled tail item, find a user who rated it, predict, and compare.
    
    for i in sample_tail_items:
        # Find users who rated this item
        users_who_rated = list(item_user_ratings.get(i, {}).keys())
        if not users_who_rated:
            continue
            
        # Pick one random user
        u = np.random.choice(users_who_rated)
        actual_rating = item_user_ratings[i][u] # We know this exists
        
        # Hide this rating for prediction? 
        # Ideally yes, but for simplicity in this analysis we can use the existing function 
        # which checks `user_item_ratings`. If we don't remove it, it might just return the rating?
        # The `predict_item_based` logic sums sim * (r - mean). 
        # If the target item is in the neighbor list (which it won't be, because `cand != i`), we are fine.
        # But `predict_item_based` uses `user_item_ratings` to find rating of *neighbor* items.
        # The target item `i` is what we are predicting. `user_item_ratings` contains `i` for `u`.
        # Standard LOOCV: We should simulate `i` not being rated by `u`.
        # However, `predict_item_based` calculates prediction based on *other* items `j` that `u` rated.
        # So the presence of `i` in `user_item_ratings[u]` doesn't affect the calculation, 
        # unless `i` ends up in the neighbor list (which we explicitly exclude `cand_item == i`).
        # So we can proceed without modifying the data structure.
        
        # --- Baseline ---
        base_similarities = []
        for cand_item in all_items:
            if cand_item == i: continue
            cand_ratings = item_user_ratings.get(cand_item, {})
            try:
                sim = calculate_item_mean_centered_cosine(item_user_ratings[i], cand_ratings, user_means)
                if sim > 0: base_similarities.append((cand_item, sim))
            except: continue
        
        base_similarities.sort(key=lambda x: x[1], reverse=True)
        k_base = max(1, int(len(base_similarities) * 0.20))
        top_neighbors_base = base_similarities[:k_base]
        base_pred = predict_item_based(u, i, top_neighbors_base, user_item_ratings, user_means, item_means)
        
        # --- Clustering ---
        clus_similarities = []
        i_cluster = item_to_cluster.get(i)
        if i_cluster is not None:
            cluster_candidates = cluster_items_map.get(i_cluster, [])
            for cand_item in cluster_candidates:
                if cand_item == i: continue
                cand_ratings = item_user_ratings.get(cand_item, {})
                try:
                    sim = calculate_item_mean_centered_cosine(item_user_ratings[i], cand_ratings, user_means)
                    if sim > 0: clus_similarities.append((cand_item, sim))
                except: continue
        
        clus_similarities.sort(key=lambda x: x[1], reverse=True)
        k_clus = max(1, int(len(clus_similarities) * 0.20))
        top_neighbors_clus = clus_similarities[:k_clus]
        clus_pred = predict_item_based(u, i, top_neighbors_clus, user_item_ratings, user_means, item_means)
        
        # Store results
        err_base = abs(actual_rating - base_pred)
        err_clus = abs(actual_rating - clus_pred)
        
        tail_errors_base.append(err_base)
        tail_errors_clus.append(err_clus)
        tail_candidates_base.append(len(base_similarities))
        tail_candidates_clus.append(len(clus_similarities))

    
    avg_tail_err_base = np.mean(tail_errors_base) if tail_errors_base else 0
    avg_tail_err_clus = np.mean(tail_errors_clus) if tail_errors_clus else 0
    
    avg_tail_cand_base = np.mean(tail_candidates_base) if tail_candidates_base else 0
    avg_tail_cand_clus = np.mean(tail_candidates_clus) if tail_candidates_clus else 0
    
    print(f"Tail Items Evaluated: {len(tail_errors_base)}")
    print(f"Avg Error (Tail) - Baseline:   {avg_tail_err_base:.4f}")
    print(f"Avg Error (Tail) - Clustering: {avg_tail_err_clus:.4f}")
    print(f"Avg Neighbor Candidates (Tail) - Baseline:   {avg_tail_cand_base:.1f}")
    print(f"Avg Neighbor Candidates (Tail) - Clustering: {avg_tail_cand_clus:.1f}")
    
    if avg_tail_err_clus < avg_tail_err_base:
        print("Insight: Clustering improves reliability for long-tail items.")
    else:
        print("Insight: Clustering does NOT improve reliability for long-tail items.")

    # ---------------------------------------------------------
    # Task 10: Computational Efficiency
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("TASK 10: COMPUTATIONAL EFFICIENCY")
    print("="*50)
    
    n_items = float(len(all_items))
    ops_base = n_items * n_items # Compares every item with every other item
    
    # Calculate ops for clustering: sum of (cluster_size^2)
    cluster_sizes = feature_df['cluster'].value_counts().to_dict()
    ops_clus = sum([size**2 for size in cluster_sizes.values()])
    
    reduction_pct = ((ops_base - ops_clus) / ops_base) * 100
    speedup = ops_base / ops_clus if ops_clus > 0 else 0
    
    print(f"Total Items: {int(n_items)}")
    print(f"Baseline Comparisons (Global):  {int(ops_base):,}")
    print(f"Clustering Comparisons (Local): {int(ops_clus):,}")
    print(f"Reduction in Computations:      {reduction_pct:.2f}%")
    print(f"Speedup Factor:                 {speedup:.2f}x")

    # ---------------------------------------------------------
    # Task 11: Cluster Size vs Prediction Quality
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("TASK 11: CLUSTER SIZE VS PREDICTION QUALITY")
    print("="*50)
    
    cluster_stats_11 = []
    for c_id, errs in cluster_errors.items():
        size = cluster_sizes.get(c_id, 0)
        avg_err = np.mean(errs) if errs else 0
        cluster_stats_11.append({'cluster': c_id, 'size': size, 'avg_error': avg_err})
        
    df_c_stats = pd.DataFrame(cluster_stats_11).sort_values('size', ascending=False)
    print(df_c_stats.to_string(index=False))
    
    # Correlation
    corr = df_c_stats['size'].corr(df_c_stats['avg_error'])
    print(f"\nCorrelation between Cluster Size and Error: {corr:.4f}")
    if corr < -0.3:
        print("Trend: Larger clusters tend to have LOWER error (Better).")
    elif corr > 0.3:
        print("Trend: Larger clusters tend to have HIGHER error (Worse).")
    else:
        print("Trend: Weak or no correlation.")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
