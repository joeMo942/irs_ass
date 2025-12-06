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
    
    # 2.1 Calculate Num Raters and Std Dev per Item
    item_stats = df.groupby('item')['rating'].agg(['count', 'std']).reset_index()
    item_stats.rename(columns={'count': 'num_raters', 'std': 'std_rating'}, inplace=True)
    item_stats['std_rating'] = item_stats['std_rating'].fillna(0)

    # 2.2 Merge with Average Rating (r_i)
    # r_i likely has columns ['item', 'r_i_bar'] or similar
    feature_df = pd.merge(item_stats, r_i, on='item', how='inner')
    
    # Identify the average rating column (column that is not 'item')
    avg_col = [c for c in r_i.columns if c != 'item'][0]
    
    feature_cols = ['num_raters', avg_col, 'std_rating']
    X = feature_df[feature_cols].copy()
    
    print(f"Feature vector shape: {X.shape}")
    print(f"Features: {feature_cols}")

    # 2.3 Normalize Features
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
    
    for k in K_VALUES:
        print(f"  Clustering with K={k}...")
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
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

    output_plot_path = os.path.join(RESULTS_DIR, 'clustering_metrics.png')
    plt.tight_layout()
    plt.savefig(output_plot_path)
    print(f"Plots saved to {output_plot_path}")

    # ---------------------------------------------------------
    # 4. Cluster Analysis with Optimal K
    # ---------------------------------------------------------
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

    analysis_plot_path = os.path.join(RESULTS_DIR, 'cluster_analysis.png')
    plt.tight_layout()
    plt.savefig(analysis_plot_path)
    print(f"Analysis plots saved to {analysis_plot_path}")

    # ---------------------------------------------------------
    # 5. Prediction Comparison (Baseline vs Clustered)
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

    print(f"\n{'User':<10} | {'Item':<10}  | {'BasePred':<8} | {'ClusPred':<8} | {'ErrBase':<7} | {'ErrClus':<7}")
    print("-" * 80)

    mae_base_list = []
    mae_clus_list = []

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

            # --- 5.2 Clustered Prediction (Local Neighbors) ---
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
            
            mae_base_list.append(err_base)
            mae_clus_list.append(err_clus)

            print(f"{u:<10} | {i:<10}  | {base_pred:<8.2f} | {clus_pred:<8.2f} | {err_base:<7.2f} | {err_clus:<7.2f}")

    # 8.3 Comparison Summary
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


if __name__ == "__main__":
    main()
