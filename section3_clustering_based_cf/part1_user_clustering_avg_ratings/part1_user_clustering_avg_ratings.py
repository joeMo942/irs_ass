import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import sys
import os

# Add the project root to sys.path to enable importing from utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from utils.data_loader import get_user_avg_ratings, get_target_users, get_target_items, get_preprocessed_dataset
from utils.similarity import calculate_user_mean_centered_cosine
from utils.prediction import predict_user_based

def get_user_item_ratings(df):
    """
    Converts dataframe to dictionary of {user_id: {item_id: rating}}
    """
    user_item_ratings = {}
    for _, row in df.iterrows():
        user_id = int(row['user'])
        item_id = int(row['item'])
        rating = float(row['rating'])
        
        if user_id not in user_item_ratings:
            user_item_ratings[user_id] = {}
        user_item_ratings[user_id][item_id] = rating
    return user_item_ratings


def main():
    print("Loading user average ratings...")
    df = get_user_avg_ratings()
    
    # ------------------------------------------------------------------------------------------------------------------
    # 2. Create a 1-dimensional feature vector for each user u, consisting of their average rating r_u_bar.
    # ------------------------------------------------------------------------------------------------------------------
    X = df[['r_u_bar']].values
    
    # ------------------------------------------------------------------------------------------------------------------
    # 3. Calculate the mean (mu) of the feature values.
    # ------------------------------------------------------------------------------------------------------------------
    mu = df['r_u_bar'].mean()
    print(f"Mean of users' average ratings (mu): {mu:.4f}")
    
    # ------------------------------------------------------------------------------------------------------------------
    # 4. Compute the standard deviation (sigma) of the feature values.
    # ------------------------------------------------------------------------------------------------------------------
    sigma = df['r_u_bar'].std()
    print(f"Standard deviation (sigma): {sigma:.4f}")
    
    # ------------------------------------------------------------------------------------------------------------------
    # 5. Normalize the feature values using Z-score standardization.
    # ------------------------------------------------------------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ------------------------------------------------------------------------------------------------------------------
    # 6. Apply K-means clustering with the following values of K: 5, 10, 15, 20, 30, and 50.
    # ------------------------------------------------------------------------------------------------------------------
    k_values = [5, 10, 15, 20, 30, 50]
    wcss = []
    silhouette_scores = []
    
    print("\nStarting K-means clustering analysis...")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # --------------------------------------------------------------------------------------------------------------
        # 6.1 Record the WCSS (inertia) for each K.
        # --------------------------------------------------------------------------------------------------------------
        wcss.append(kmeans.inertia_)
        
        # --------------------------------------------------------------------------------------------------------------
        # 6.2 Calculate the Silhouette Score for each clustering result.
        # --------------------------------------------------------------------------------------------------------------
        # using a large sample of 40,000 to balance accuracy and speed
        #eldata kbera lazem na5od sample 
        if len(X_scaled) > 1000:
            score = silhouette_score(X_scaled, labels, sample_size=1000, random_state=42)
        else:
            score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        
        print(f"  K={k}: WCSS={kmeans.inertia_:.4f}, Silhouette={score:.4f}")

    # ------------------------------------------------------------------------------------------------------------------
    # 7. Plot the Elbow Curve (WCSS vs K) and Silhouette Score vs K.
    # ------------------------------------------------------------------------------------------------------------------
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('WCSS', color=color)
    ax1.plot(k_values, wcss, color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(k_values, silhouette_scores, color=color, marker='s')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Elbow Method and Silhouette Score per K')
    plt.grid(True)
    plot_path = os.path.join(results_dir, 'clustering_metrics.png')
    plt.savefig(plot_path)
    print(f"\nPlots saved to {plot_path}")

    # ------------------------------------------------------------------------------------------------------------------
    # 8. Determine the optimal K based on the plots (e.g., using the Elbow Method or max Silhouette Score).
    # ------------------------------------------------------------------------------------------------------------------
    # optimal k manually selected based on elbow method plot
    optimal_k = 10
    print(f"\nOptimal K manually selected based on elbow method plot: {optimal_k}")

    # Rerun for Optimal K
    best_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['cluster'] = best_kmeans.fit_predict(X_scaled)
    centroids_scaled = best_kmeans.cluster_centers_
    
    # Inverse transform centroids to get original rating scale
    centroids_original = scaler.inverse_transform(centroids_scaled).flatten()
    
    # ------------------------------------------------------------------------------------------------------------------
    # 9. Perform detailed analysis on the clusters formed with the optimal K.
    # 9.1 Visualize the distribution of users across clusters.
    # ------------------------------------------------------------------------------------------------------------------
    cluster_counts = df['cluster'].value_counts().sort_index()
    print("\nUser distribution per cluster:")
    print(cluster_counts)
    
    # Bar chart for distribution
    plt.figure(figsize=(8, 5))
    cluster_counts.plot(kind='bar', color='skyblue')
    plt.title(f'User Distribution for K={optimal_k}')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Users')
    dist_plot_path = os.path.join(results_dir, f'user_distribution_k{optimal_k}.png')
    plt.savefig(dist_plot_path)
    print(f"Distribution plot saved to {dist_plot_path}")

    # ------------------------------------------------------------------------------------------------------------------
    # 9.2 Calculate the centroid (average rating) for each cluster.
    # 9.3 Interpret the nature of each cluster (e.g., "Generous" or "Strict" raters).
    # ------------------------------------------------------------------------------------------------------------------
    print("\nCluster Centroids (Average Ratings) and Interpretation:")
    cluster_info = []
    for cluster_id in range(optimal_k):
        rating = centroids_original[cluster_id]
        # Determine strict or generous relative to global mean
        nature = "Generous" if rating > mu else "Strict"
        cluster_info.append({
            'Cluster': cluster_id,
            'Avg Rating': rating,
            'Nature': nature,
            'Count': cluster_counts[cluster_id]
        })
    
    info_df = pd.DataFrame(cluster_info).sort_values(by='Avg Rating')
    print(info_df.to_string(index=False))

    # =========================================================
    # Part 2: Clustering-Based Collaborative Filtering (K=50)
    # =========================================================
    print("\n" + "="*50)
    print("Starting Clustering-Based CF (using Optimal K=50)")
    print("="*50)

    # 1. Load Data
    print("Loading full dataset for CF...")
    full_df = get_preprocessed_dataset()
    user_item_ratings = get_user_item_ratings(full_df)
    target_users = get_target_users()
    target_items = get_target_items()
    
    # Pre-compute user means for similarity calculation
    user_means = df.set_index('user')['r_u_bar'].to_dict()
    
    # 2. Assign Users to Clusters
    # We already have 'cluster' column in `df` from the best_kmeans (K=50)
    user_cluster_map = df.set_index('user')['cluster'].to_dict()
    
    # Group users by cluster for fast retrieval
    cluster_users = {}
    for user, cluster in user_cluster_map.items():
        if cluster not in cluster_users:
            cluster_users[cluster] = []
        cluster_users[cluster].append(user)

    # 3. Prediction Loop
    print("\nPredicting ratings for Target Users...")
    
    total_users_N = len(df)
    total_similarity_computations_clustering = 0
    
    # Store predictions for comparison: {user_id: {item_id: prediction}}
    clustering_predictions = {}
    
    for u_id in target_users:
        if u_id not in user_cluster_map:
            print(f"Target User {u_id} not found in clusters (maybe filtered out?). Skipping.")
            continue
            
        u_cluster = user_cluster_map[u_id]
        neighbors_in_cluster = cluster_users[u_cluster]
        
        # Filter out the user themselves
        neighbors_in_cluster = [n for n in neighbors_in_cluster if n != u_id]
        
        cluster_size = len(neighbors_in_cluster)
        total_similarity_computations_clustering += cluster_size
        
        print(f"\nTarget User: {u_id} | Cluster: {u_cluster} | Potential Neighbors: {cluster_size}")
        
        # Compute Similarity with users in the SAME cluster only
        similarities = []
        u_ratings = user_item_ratings.get(u_id, {})
        u_mean = user_means.get(u_id, 3.0)
        
        for v_id in neighbors_in_cluster:
            v_ratings = user_item_ratings.get(v_id, {})
            v_mean = user_means.get(v_id, 3.0)
            
            sim = calculate_user_mean_centered_cosine(u_ratings, v_ratings, u_mean, v_mean)
            if sim > 0:
                similarities.append((v_id, sim))
        
        # Sort and Select Top 20%
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k_count = max(1, int(len(similarities) * 0.20))
        top_neighbors = similarities[:top_k_count]
        
        print(f"  Valid Neighbors (Sim > 0): {len(similarities)}")
        print(f"  Selected Top 20% Neighbors: {len(top_neighbors)}")
        if top_neighbors:
             print(f"  Top Neighbor ID: {top_neighbors[0][0]}, Sim: {top_neighbors[0][1]:.4f}")

        # Predict for Target Items
        if u_id not in clustering_predictions:
            clustering_predictions[u_id] = {}
            
        for i_id in target_items:
            prediction = predict_user_based(u_id, i_id, top_neighbors, user_item_ratings, user_means)
            clustering_predictions[u_id][i_id] = prediction
            print(f"  -> Prediction for Item {i_id}: {prediction:.4f}")

    # =========================================================
    # Part 3: Baseline CF (No Clustering)
    # =========================================================
    print("\n" + "="*50)
    print("Starting Baseline CF (No Clustering) - ALL Users")
    print("="*50)
    
    baseline_predictions = {}
    total_similarity_computations_baseline = 0
    all_users = list(user_cluster_map.keys()) # All valid users
    
    for u_id in target_users:
        if u_id not in user_means:
            continue
            
        print(f"\nTarget User: {u_id} | Searching ALL {len(all_users)-1} users...")
        
        similarities = []
        u_ratings = user_item_ratings.get(u_id, {})
        u_mean = user_means.get(u_id, 3.0)
        
        # Search ALL users
        for v_id in all_users:
            if v_id == u_id:
                continue
            
            total_similarity_computations_baseline += 1
            
            v_ratings = user_item_ratings.get(v_id, {})
            v_mean = user_means.get(v_id, 3.0)
            
            sim = calculate_user_mean_centered_cosine(u_ratings, v_ratings, u_mean, v_mean)
            if sim > 0:
                similarities.append((v_id, sim))
        
        # Sort and Select Top 20%
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k_count = max(1, int(len(similarities) * 0.20))
        top_neighbors = similarities[:top_k_count]
        
        print(f"  Valid Neighbors (Sim > 0): {len(similarities)}")
        print(f"  Selected Top 20% Neighbors: {len(top_neighbors)}")
        if top_neighbors:
             print(f"  Top Neighbor ID: {top_neighbors[0][0]}, Sim: {top_neighbors[0][1]:.4f}")
             
        if u_id not in baseline_predictions:
            baseline_predictions[u_id] = {}
            
        for i_id in target_items:
            prediction = predict_user_based(u_id, i_id, top_neighbors, user_item_ratings, user_means)
            baseline_predictions[u_id][i_id] = prediction
            print(f"  -> Baseline Pred for Item {i_id}: {prediction:.4f}")

    # =========================================================
    # Part 4: Comparison & Efficiency Analysis
    # =========================================================
    print("\n" + "="*50)
    # =========================================================
    # 10. Compare clustering-based predictions with non-clustering predictions from Section TWO
    # =========================================================
    print("\n" + "="*50)
    print("10. Comparison of Clustering-Based vs Baseline CF")
    print("="*50)
    
    print(f"{'User':<10} | {'Item':<10} | {'Cluster Pred':<15} | {'Baseline Pred':<15} | {'Diff':<10}")
    print("-" * 75)
    
    comparison_lines = []
    
    for u_id in target_users:
        if u_id not in clustering_predictions or u_id not in baseline_predictions:
             continue
        for i_id in target_items:
            p_cluster = clustering_predictions[u_id].get(i_id, 0.0)
            p_baseline = baseline_predictions[u_id].get(i_id, 0.0)
            diff = abs(p_cluster - p_baseline)
            line = f"{u_id:<10} | {i_id:<10} | {p_cluster:<15.4f} | {p_baseline:<15.4f} | {diff:<10.4f}"
            print(line)
            comparison_lines.append(line)

    # Save comparison to file
    comparison_file_path = os.path.join(results_dir, 'comparison_results.txt')
    with open(comparison_file_path, 'w') as f:
        f.write("Comparison of Clustering-Based vs Baseline CF\n")
        f.write("="*75 + "\n")
        f.write(f"{'User':<10} | {'Item':<10} | {'Cluster Pred':<15} | {'Baseline Pred':<15} | {'Diff':<10}\n")
        f.write("-" * 75 + "\n")
        for line in comparison_lines:
            f.write(line + "\n")
    print(f"\nComparison results saved to {comparison_file_path}")

    print("\n" + "="*50)
    # =========================================================
    # 11. Compare computational efficiency
    # ------------------------------------------------------------------------------------------------------------------
    # 11.1 Calculate the theoretical speedup factor
    # 11.2 Express the efficiency gain as a percentage reduction
    # =========================================================
    print("\n" + "="*50)
    print("11. Efficiency Analysis (Computations)")
    print("="*50)
    
    comps_no_clustering = total_similarity_computations_baseline
    comps_with_clustering = total_similarity_computations_clustering
    
    speedup = comps_no_clustering / comps_with_clustering if comps_with_clustering > 0 else 0
    percent_reduction = (1 - (comps_with_clustering / comps_no_clustering)) * 100 if comps_no_clustering > 0 else 0
    
    print(f"Total Users: {len(all_users)}")
    print(f"Computations (Baseline - Actual): {comps_no_clustering:,}")
    print(f"Computations (With Clustering - Actual): {comps_with_clustering:,}")
    print(f"Speedup Factor: {speedup:.2f}x")
    print(f"Efficiency Gain: {percent_reduction:.2f}%")

    # =========================================================
    # Part 5: Evaluate Impact of Cluster Imbalance
    # =========================================================
    print("\n" + "="*50)
    # =========================================================
    # 12. Evaluate the impact of cluster imbalance
    # 12.1 Identify if any clusters are significantly larger or smaller
    # =========================================================
    print("\n" + "="*50)
    print("12. Cluster Imbalance Evaluation")
    print("="*50)
    
    # cluster_counts is already calculated for optimal_k
    sizes = cluster_counts.values
    max_size = sizes.max()
    min_size = sizes.min()
    mean_size = sizes.mean()
    std_size = sizes.std()
    
    print(f"Cluster Sizes Stats (K={optimal_k}):")
    print(f"  Max Size: {max_size}")
    print(f"  Min Size: {min_size}")
    print(f"  Mean Size: {mean_size:.2f}")
    print(f"  Std Dev: {std_size:.2f}")
    
    imbalance_ratio = max_size / min_size if min_size > 0 else float('inf')
    print(f"  Imbalance Ratio (Max/Min): {imbalance_ratio:.2f}")
    
    print("\nDiscussion on Imbalance:")
    if std_size > mean_size * 0.5:
        print("  - Significant imbalance observed.")
        print("  - Large clusters -> Slower neighbor search (bottleneck).")
        print("  - Small clusters -> Potential lack of neighbors (coverage issue).")
    else:
        print("  - Clusters are relatively balanced.")
    
    # =========================================================
    # Part 6: Test Robustness (Re-run K-means)
    # =========================================================
    print("\n" + "="*50)
    # =========================================================
    # 13. Test the robustness of the clustering approach
    # 13.1 Re-run K-means with different random initializations (at least 3 times)
    # =========================================================
    print("\n" + "="*50)
    print("13. Robustness Test (Re-run K-means 3 times)")
    print("="*50)
    
    seeds = [42, 100, 2023]
    robustness_results = []
    
    for seed in seeds:
        print(f"  Running K-means with seed={seed}...")
        km = KMeans(n_clusters=optimal_k, random_state=seed, n_init=10)
        labels_seed = km.fit_predict(X_scaled)
        inertia = km.inertia_
        
        # Calculate cluster sizes std dev for this run
        counts = pd.Series(labels_seed).value_counts()
        std_dev = counts.std()
        
        robustness_results.append({
            'Seed': seed,
            'Inertia': inertia,
            'Cluster Size Std': std_dev
        })
        print(f"    Inertia: {inertia:.4f} | Size Std: {std_dev:.2f}")
    '''
    Results
    Imbalance: There is significant imbalance (Ratio: 10.93), 
    with cluster sizes ranging from ~3.4k to ~37.2k. This suggests that while clustering speeds up the 
    process on average, the "Generous" clusters (which are larger) will still be slower to process than the 
    "Strict" clusters.
    Robustness: The K-means clustering is STABLE. Re-running with different random 
    seeds produced nearly identical Inertia values (Difference < 0.2%).
    '''

    print("\nRobustness Summary:")
    res_df = pd.DataFrame(robustness_results)
    print(res_df.to_string(index=False))
    
    inertia_std = res_df['Inertia'].std()
    if inertia_std < res_df['Inertia'].mean() * 0.05:
        print("\n  -> Clustering is STABLE (Inertia varies < 5%).")
    else:
        print("\n  -> Clustering shows VARIANCE (Inertia varies > 5%).")

    print("\nAnalysis completed.")

if __name__ == "__main__":
    main()
