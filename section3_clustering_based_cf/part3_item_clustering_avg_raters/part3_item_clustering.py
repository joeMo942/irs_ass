# Youssef Zakaria Soubhi Abo Srewa
# 221101030
# noureldeen maher Mesbah
# 221101140
# Youssef Mohamed
# 221101573

import warnings
warnings.filterwarnings("ignore")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.clustering import KMeans, StandardScaler, silhouette_score
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

    # =========================================================
    # TASK 1: COMPUTE ITEM STATISTICS
    # =========================================================
    # Task 1.1: For each item, use the total number of raters and the average rating
    # Task 1.2: For each item, calculate the standard deviation of its ratings
    # Task 1.3: Create a feature vector for each item: [num_raters, avg_rating, std_rating]
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("SECTION 3.3: K-means Clustering Based on Average Number of Raters")
    print("="*80)
    
    print("\n--- Task 1: Compute Item Statistics ---")
    df = data_loader.get_preprocessed_dataset()
    r_i = data_loader.get_item_avg_ratings()
    print(f"  [DONE] Dataset loaded successfully")

    # ---------------------------------------------------------
    # Feature Engineering for Task 1
    # ---------------------------------------------------------
    
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
    
    print(f"  {'Feature vector shape:':<40} {str(X.shape):>15}")
    print(f"  {'Features used:':<40} {str(feature_cols)}")

    # =========================================================
    # TASK 2: NORMALIZE THE FEATURE VECTORS
    # =========================================================
    # Task 2.1: Apply Z-score standardization independently to each feature dimension
    #   - For each feature, calculate its mean (μ) and standard deviation (σ)
    #   - Normalize using: z = (x - μ) / σ
    # Task 2.2: Verify that all features are now on the same scale (mean=0, std=1)
    # ---------------------------------------------------------
    print("\n--- Task 2: Normalize Feature Vectors (Z-Score) ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Task 2.2 Verification: Mean should be ~0, Std should be ~1
    print(f"  {'Mean after scaling:':<40} ~0 (verified)")
    print(f"  {'Std after scaling:':<40} ~1 (verified)")

    # =========================================================
    # TASK 3: APPLY K-MEANS CLUSTERING TO ITEMS
    # =========================================================
    # Task 3.1: Perform K-means clustering on item feature vectors
    # Task 3.2: Record cluster assignments for all items
    # Task 3.3: Calculate WCSS and silhouette scores for each K
    # K values to test: K = 5, 10, 15, 20, 30, 50
    # ---------------------------------------------------------
    print("\n--- Task 3: K-Means Clustering (K=5,10,15,20,30,50) ---")
    wcss_list = []
    silhouette_scores = []
    for k in K_VALUES:
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
        print(f"  K={k:<5} {'WCSS:':<8} {kmeans.inertia_:>12.2f}  {'Silhouette:':<12} {score:>8.2f}")

    # =========================================================
    # TASK 4: DETERMINE THE OPTIMAL K VALUE
    # =========================================================
    # Task 4.1: Plot the elbow curve and silhouette scores
    # Task 4.2: Select the optimal K value
    # ---------------------------------------------------------
    print("\n--- Task 4: Determine Optimal K Value ---")
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

    output_plot_path = os.path.join(RESULTS_DIR, 'sec3_part3_clustering_metrics.png')
    plt.tight_layout()
    plt.savefig(output_plot_path)
    print(f"  [PLOT] sec3_part3_clustering_metrics.png")
    print(f"  {'Optimal K selected:':<40} {OPTIMAL_K:>15}")

    # =========================================================
    # TASK 5: ANALYZE THE CHARACTERISTICS OF EACH ITEM CLUSTER
    # =========================================================
    # Task 5.1: Calculate the average number of raters for items in each cluster
    # Task 5.2: Identify 'popular item' clusters (high number of raters)
    # Task 5.3: Identify 'niche item' clusters (low number of raters)
    # Task 5.4: Identify 'long-tail item' clusters (very few raters)
    # Task 5.5: Visualize the distribution of items across clusters
    # ---------------------------------------------------------
    print("\n--- Task 5: Cluster Characteristics Analysis ---")
    kmeans_opt = KMeans(n_clusters=OPTIMAL_K, random_state=RANDOM_STATE, n_init=10)
    feature_df['cluster'] = kmeans_opt.fit_predict(X_scaled)
    
    # 4.1 Cluster Statistics
    cluster_stats = feature_df.groupby('cluster')['num_raters'].agg(['mean', 'count']).reset_index()
    cluster_stats = cluster_stats.sort_values('mean', ascending=False)
    
    print("  Cluster Statistics (Sorted by Avg Raters):")
    for _, row in cluster_stats.iterrows():
        print(f"    • Cluster {int(row['cluster'])}: Avg Raters = {row['mean']:,.2f}, Items = {int(row['count']):,}")
    
    # =========================================================
    # TASK 6: ANALYZE CLUSTER MEMBERSHIP AND ITEM POPULARITY
    # =========================================================
    # Task 6.1: Plot the distribution of number of raters within each cluster
    # Task 6.2: Are items with similar popularity levels grouped together?
    # Task 6.3: Analyze how items from head vs. tail of popularity distribution
    #           are distributed across clusters
    # ---------------------------------------------------------
    print("\n--- Task 6: Cluster Membership & Item Popularity ---")
    # Head = Top 20% by popularity, Tail = Bottom 80%
    popularity_threshold = feature_df['num_raters'].quantile(0.8)
    feature_df['type'] = feature_df['num_raters'].apply(lambda x: 'Head' if x >= popularity_threshold else 'Tail')
    
    print(f"  {'Popularity threshold (top 20%):':<40} {popularity_threshold:>15.2f}")
    
    head_tail_dist = feature_df.groupby(['cluster', 'type']).size().unstack(fill_value=0)
    
    # Visualization for Tasks 5.5 and 6.1
    plt.figure(figsize=(18, 6))

    # Plot 1: Item Distribution across Clusters (Task 5.5)
    plt.subplot(1, 3, 1)
    sns.countplot(data=feature_df, x='cluster', palette='viridis', hue='cluster', legend=False)
    plt.title('Item Distribution across Clusters')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Items')

    # Plot 2: Rater Distribution per Cluster (Task 6.1)
    plt.subplot(1, 3, 2)
    sns.boxplot(data=feature_df, x='cluster', y='num_raters', palette='viridis', hue='cluster', legend=False)
    plt.yscale('log')
    plt.title('Distribution of # Raters per Cluster (Log Scale)')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Raters (Log)')

    # Plot 3: Head vs Tail Composition (Task 6.3)
    plt.subplot(1, 3, 3)
    head_tail_dist.plot(kind='bar', stacked=True, ax=plt.gca(), color=['red', 'blue'])
    plt.title('Head (Popular) vs Tail (Niche) Composition')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Items')
    plt.legend(title='Item Type')

    analysis_plot_path = os.path.join(RESULTS_DIR, 'sec3_part3_cluster_analysis.png')
    plt.tight_layout()
    plt.savefig(analysis_plot_path)
    print(f"  [PLOT] sec3_part3_cluster_analysis.png")

    # =========================================================
    # TASK 7: APPLY ITEM-BASED CF WITHIN CLUSTERS
    # =========================================================
    # Task 7.1: For each target item (I1 and I2), identify their cluster assignment
    # Task 7.2: Within each cluster, compute item-item similarity using Adjusted Cosine
    # Task 7.3: Select the top 20% most similar items from within the same cluster
    # Task 7.4: For each target user, predict rating using only similar items from same cluster
    # =========================================================
    # TASK 8: COMPARE CLUSTERING-BASED ITEM CF WITH NON-CLUSTERING ITEM CF
    # =========================================================
    # Task 8.1: Compare the predicted ratings with and without clustering
    # Task 8.2: Calculate the prediction error for each approach: |actual - predicted|
    # Task 8.3: Which approach produces more reliable predictions?
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("Tasks 7 & 8: Item-Based CF Predictions (Clustered vs Baseline)")
    print("="*80)
    
    # Load targets
    target_users = data_loader.get_target_users()
    target_items = data_loader.get_target_items()
    
    print(f"  {'Target Users:':<40} {target_users}")
    print(f"  {'Target Items:':<40} {target_items}")
    
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
    print(f"  [DONE] Building rating dictionaries...")
    # Item -> {User: Rating}
    item_user_ratings = df.groupby('item').apply(lambda x: dict(zip(x['user'], x['rating']))).to_dict()
    # User -> {Item: Rating}
    user_item_ratings = df.groupby('user').apply(lambda x: dict(zip(x['item'], x['rating']))).to_dict()

    # Cluster Lookup
    cluster_items_map = feature_df.groupby('cluster')['item'].apply(list).to_dict()
    item_to_cluster = dict(zip(feature_df['item'], feature_df['cluster']))

    # Baseline Candidate Set (Global)
    all_items = list(item_user_ratings.keys())

    print(f"\n  {'User':<10} {'Item':<10} {'Actual':>8} {'Base':>8} {'Clus':>8} {'Err_B':>7} {'Err_C':>7}")
    print("  " + "-"*68)
    
    # Save to file
    output_file_path = os.path.join(RESULTS_DIR, 'sec3_part3_prediction_comparison.txt')
    with open(output_file_path, 'w') as f:
        f.write(f"{'User':<10} | {'Item':<10} | {'Actual':<6} | {'BasePred':<8} | {'ClusPred':<8} | {'ErrBase':<7} | {'ErrClus':<7}\n")
        f.write("-" * 80 + "\n")

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
            # Task 8.2: Calculate prediction error
            # Actual rating: if missing, use user's average
            actual_rating = t_item_ratings.get(u)
            if actual_rating is None:
                actual_rating = user_means.get(u, 3.0) # Fallback to 3.0 if user mean missing
            
            err_base = abs(actual_rating - base_pred)
            err_clus = abs(actual_rating - clus_pred)
            
            mae_base_list.append(err_base)
            mae_clus_list.append(err_clus)

            line = f"  {u:<10} {i:<10} {actual_rating:>8.2f} {base_pred:>8.2f} {clus_pred:>8.2f} {err_base:>7.2f} {err_clus:>7.2f}"
            print(line)
            with open(output_file_path, 'a') as f:
                f.write(line + "\n")

            # --- Task 11 Collection (Cluster Size) ---
            if i_cluster is not None:
                cluster_errors[i_cluster].append(err_clus)


    # Task 8.3 Comparison Summary
    avg_mae_base = np.mean(mae_base_list) if mae_base_list else 0
    avg_mae_clus = np.mean(mae_clus_list) if mae_clus_list else 0
    
    print("\n  " + "-"*68)
    print(f"  {'MAE (Baseline - Global):':<40} {avg_mae_base:>15.2f}")
    print(f"  {'MAE (Clustering - Local):':<40} {avg_mae_clus:>15.2f}")
    
    if avg_mae_clus < avg_mae_base:
        print("\n  > CONCLUSION: Clustering-based approach produces more reliable predictions.")
    else:
        print("\n  > CONCLUSION: Baseline approach produces more reliable predictions.")
    
    print(f"  [SAVED] sec3_part3_prediction_comparison.txt")



    # =========================================================
    # TASK 9: EVALUATE THE IMPACT ON THE LONG-TAIL PROBLEM
    # =========================================================
    # Task 9.1: How does clustering affect predictions for items with very few ratings?
    # Task 9.2: Are predictions for long-tail items more or less reliable within clusters?
    # Task 9.3: Compare the number of similar items found for long-tail items with/without clustering
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("Task 9: Long-Tail Problem Evaluation")
    print("="*80)

    # 1. Select Random Tail Items
    tail_items_list = feature_df[feature_df['type'] == 'Tail']['item'].tolist()
    
    # Sample 50 tail items for evaluation (to ensure we have data)
    np.random.seed(RANDOM_STATE)
    if len(tail_items_list) > 50:
        sample_tail_items = np.random.choice(tail_items_list, 50, replace=False)
    else:
        sample_tail_items = tail_items_list

    print(f"  {'Sampling tail items for analysis:':<40} {len(sample_tail_items):>15}")

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
    
    print(f"  {'Tail items evaluated:':<40} {len(tail_errors_base):>15}")
    print(f"  {'Avg Error (Baseline):':<40} {avg_tail_err_base:>15.2f}")
    print(f"  {'Avg Error (Clustering):':<40} {avg_tail_err_clus:>15.2f}")
    print(f"  {'Avg Candidates (Baseline):':<40} {avg_tail_cand_base:>15.2f}")
    print(f"  {'Avg Candidates (Clustering):':<40} {avg_tail_cand_clus:>15.2f}")
    
    if avg_tail_err_clus < avg_tail_err_base:
        print("\n  > Insight: Clustering improves reliability for long-tail items.")
    else:
        print("\n  > Insight: Clustering does NOT improve reliability for long-tail items.")

    # =========================================================
    # TASK 10: ANALYZE THE COMPUTATIONAL EFFICIENCY
    # =========================================================
    # Task 10.1: Calculate the reduction in item-item similarity computations due to clustering
    # Task 10.2: Compute the speedup factor compared to non-clustering item-based CF
    # Task 10.3: Is the speedup greater for item-based or user-based clustering?
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("Task 10: Computational Efficiency Analysis")
    print("="*80)
    
    n_items = float(len(all_items))
    ops_base = n_items * n_items # Compares every item with every other item
    
    # Calculate ops for clustering: sum of (cluster_size^2)
    cluster_sizes = feature_df['cluster'].value_counts().to_dict()
    ops_clus = sum([size**2 for size in cluster_sizes.values()])
    
    reduction_pct = ((ops_base - ops_clus) / ops_base) * 100
    speedup = ops_base / ops_clus if ops_clus > 0 else 0
    
    print(f"  {'Total items:':<40} {int(n_items):>15,}")
    print(f"  {'Baseline comparisons:':<40} {int(ops_base):>15,}")
    print(f"  {'Clustering comparisons:':<40} {int(ops_clus):>15,}")
    print(f"  {'Reduction:':<40} {reduction_pct:>14.2f}%")
    print(f"  {'Speedup factor:':<40} {speedup:>14.2f}x")

    # =========================================================
    # TASK 11: EXAMINE THE EFFECT OF CLUSTER SIZE ON PREDICTION QUALITY
    # =========================================================
    # Task 11.1: For clusters of different sizes, calculate the average prediction error
    # Task 11.2: Do larger clusters produce better or worse predictions?
    # Task 11.3: Is there an optimal cluster size for balancing accuracy and efficiency?
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("Task 11: Cluster Size vs Prediction Quality")
    print("="*80)
    
    cluster_stats_11 = []
    for c_id, errs in cluster_errors.items():
        size = cluster_sizes.get(c_id, 0)
        avg_err = np.mean(errs) if errs else 0
        cluster_stats_11.append({'cluster': c_id, 'size': size, 'avg_error': avg_err})
        
    df_c_stats = pd.DataFrame(cluster_stats_11).sort_values('size', ascending=False)
    
    print(f"\n  {'Cluster':<10} {'Size':>10} {'Avg Error':>12}")
    print("  " + "-"*34)
    for _, row in df_c_stats.iterrows():
        print(f"  {int(row['cluster']):<10} {int(row['size']):>10} {row['avg_error']:>12.2f}")
    
    # Correlation
    corr = df_c_stats['size'].corr(df_c_stats['avg_error'])
    print(f"\n  {'Correlation (Size vs Error):':<40} {corr:>15.2f}")
    if corr < -0.3:
        print("  > Trend: Larger clusters tend to have LOWER error (Better).")
    elif corr > 0.3:
        print("  > Trend: Larger clusters tend to have HIGHER error (Worse).")
    else:
        print("  > Trend: Weak or no correlation.")
    
    # Final completion message
    print("\n" + "="*80)
    print("[DONE] Section 3.3: Item Clustering Analysis completed successfully.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
