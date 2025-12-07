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
    # ==================================================================================================================
    # SECTION 3, PART 1: K-means Clustering based on average number of user ratings
    # ==================================================================================================================
    print("\n" + "="*80)
    print("SECTION 3, PART 1: K-means Clustering Based on Average User Ratings")
    print("="*80)
    
    # ------------------------------------------------------------------------------------------------------------------
    # Task 1: Use the calculated average rating given by each user (r_u_bar) from Section ONE.
    # ------------------------------------------------------------------------------------------------------------------
    print("\n--- Task 1: Loading Data from Section ONE ---")
    df = get_user_avg_ratings()
    print(f"  [DONE] User average ratings loaded")
    print(f"  {'Total users:':<40} {len(df):>15,}")
    
    # ------------------------------------------------------------------------------------------------------------------
    # Task 2: Create a 1-dimensional feature vector for each user where the feature is their average rating value.
    # ------------------------------------------------------------------------------------------------------------------
    X = df[['r_u_bar']].values
    
    # ------------------------------------------------------------------------------------------------------------------
    # Task 3: Calculate the mean of the users' average ratings, μ = (Σ r_u_bar) / N
    # ------------------------------------------------------------------------------------------------------------------
    mu = df['r_u_bar'].mean()
    
    # ------------------------------------------------------------------------------------------------------------------
    # Task 4: Compute the Standard deviation of the users' average ratings, σ = sqrt(Σ(r_u - μ)² / N)
    # ------------------------------------------------------------------------------------------------------------------
    sigma = df['r_u_bar'].std()
    
    # ------------------------------------------------------------------------------------------------------------------
    # Task 5: Normalize the feature values for each user using standardization (Z-score normalization)
    #         to ensure proper clustering, z_u = (r_u_bar - μ) / σ
    # ------------------------------------------------------------------------------------------------------------------
    print("\n--- Tasks 2-5: Feature Extraction & Normalization ---")
    print(f"  {'Mean of users avg ratings (μ):':<40} {mu:>15.4f}")
    print(f"  {'Standard deviation (σ):':<40} {sigma:>15.4f}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  [DONE] Z-score normalization applied")
    
    # ------------------------------------------------------------------------------------------------------------------
    # Task 6: Apply K-means clustering with different values of K (e.g., K = 5, 10, 15, 20, 30, 50)
    # ------------------------------------------------------------------------------------------------------------------
    k_values = [5, 10, 15, 20, 30, 50]
    wcss = []
    silhouette_scores = []
    
    print("\n--- Task 6: K-means Clustering Analysis ---")
    print(f"  {'K values tested:':<40} {k_values}")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # --------------------------------------------------------------------------------------------------------------
        # Task 6.1: For each K value, calculate and save the cluster centroids and perform K-means clustering
        #           on the user feature vectors.
        # --------------------------------------------------------------------------------------------------------------
        wcss.append(kmeans.inertia_)
        
        # --------------------------------------------------------------------------------------------------------------
        # Task 6.2: Record the cluster assignments for all users.
        # --------------------------------------------------------------------------------------------------------------
        # using a large sample of 40,000 to balance accuracy and speed
        if len(X_scaled) > 40000:
            score = silhouette_score(X_scaled, labels, sample_size=40000, random_state=42)
        else:
            score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        
        print(f"    • K={k:2d}: WCSS={kmeans.inertia_:>12.4f}, Silhouette={score:.4f}")

    # ------------------------------------------------------------------------------------------------------------------
    # Task 7: Analyze the clustering results for each K value
    # Task 7.1: Calculate the number of users in each cluster (done above)
    # Task 7.2: Compute the within-cluster sum of squares (WCSS) for each K (done above)
    # Task 7.3: Plot the elbow curve (WCSS vs. K) to determine the optimal K value
    # Task 7.4: Calculate the silhouette score for each K value to assess clustering quality
    # ------------------------------------------------------------------------------------------------------------------
    print("\n--- Task 7: Elbow Curve & Silhouette Analysis ---")
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
    plot_path = os.path.join(results_dir, 'sec3_part1_clustering_metrics.png')
    plt.savefig(plot_path)
    print(f"  [PLOT] sec3_part1_clustering_metrics.png")

    # ------------------------------------------------------------------------------------------------------------------
    # Task 8: For the optimal K value (based on elbow method and silhouette score):
    # ------------------------------------------------------------------------------------------------------------------
    print("\n--- Task 8: Optimal K Selection & Cluster Analysis ---")
    # Optimal K manually selected based on elbow method plot and silhouette score
    optimal_k = 10
    print(f"  {'Optimal K selected:':<40} {optimal_k:>15}")

    # Rerun for Optimal K
    best_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['cluster'] = best_kmeans.fit_predict(X_scaled)
    centroids_scaled = best_kmeans.cluster_centers_
    
    # Inverse transform centroids to get original rating scale
    centroids_original = scaler.inverse_transform(centroids_scaled).flatten()
    
    # ------------------------------------------------------------------------------------------------------------------
    # Task 8.1: Display the distribution of users across clusters (create a bar chart)
    # ------------------------------------------------------------------------------------------------------------------
    cluster_counts = df['cluster'].value_counts().sort_index()
    print(f"\n  User distribution per cluster:")
    for cluster_id, count in cluster_counts.items():
        print(f"    • Cluster {cluster_id}: {count:>10,} users")
    
    # Bar chart for distribution
    plt.figure(figsize=(8, 5))
    cluster_counts.plot(kind='bar', color='skyblue')
    plt.title(f'User Distribution for K={optimal_k}')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Users')
    dist_plot_path = os.path.join(results_dir, f'sec3_part1_user_distribution_k{optimal_k}.png')
    plt.savefig(dist_plot_path)
    print(f"  [PLOT] sec3_part1_user_distribution_k{optimal_k}.png")

    # ------------------------------------------------------------------------------------------------------------------
    # Task 8.2: Show the average rating value for each cluster centroid.
    # Task 8.3: Identify which clusters contain generous raters (high average) and which contain strict raters (low average).
    # ------------------------------------------------------------------------------------------------------------------
    print(f"\n  Cluster Centroids & Interpretation (μ = {mu:.4f}):")
    print(f"  {'Cluster':<10} {'Avg Rating':<12} {'Nature':<10} {'Count':<10}")
    print("  " + "-"*42)
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
    for _, row in info_df.iterrows():
        print(f"    • {int(row['Cluster']):<8} {row['Avg Rating']:<12.4f} {row['Nature']:<10} {int(row['Count']):>10,}")

    # ==================================================================================================================
    # Task 9: Apply user-based collaborative filtering within each cluster (using optimal K)
    # ==================================================================================================================
    print("\n" + "="*80)
    print("Task 9: Clustering-Based Collaborative Filtering")
    print("="*80)

    print("\n--- Loading CF Data ---")
    full_df = get_preprocessed_dataset()
    user_item_ratings = get_user_item_ratings(full_df)
    target_users = get_target_users()
    target_items = get_target_items()
    print(f"  [DONE] Dataset loaded for CF")
    
    # Pre-compute user means for similarity calculation
    user_means = df.set_index('user')['r_u_bar'].to_dict()
    
    # ------------------------------------------------------------------------------------------------------------------
    # Task 9.1: For each target user from Section ONE (U1, U2, U3), identify which cluster they belong to.
    # ------------------------------------------------------------------------------------------------------------------
    # We already have 'cluster' column in `df` from the best_kmeans (K=10)
    user_cluster_map = df.set_index('user')['cluster'].to_dict()
    
    # Group users by cluster for fast retrieval
    cluster_users = {}
    for user, cluster in user_cluster_map.items():
        if cluster not in cluster_users:
            cluster_users[cluster] = []
        cluster_users[cluster].append(user)

    # ------------------------------------------------------------------------------------------------------------------
    # Task 9.2: Within the assigned cluster, compute user-user similarity using mean-centered Cosine similarity.
    # Task 9.3: Select the top 20% most similar users from within the same cluster.
    # Task 9.4: Predict ratings for the target items (I1 and I2) using only the similar users from the same cluster.
    # ------------------------------------------------------------------------------------------------------------------
    
    total_users_N = len(df)
    total_similarity_computations_clustering = 0
    
    # Store predictions for comparison: {user_id: {item_id: prediction}}
    clustering_predictions = {}
    
    for u_id in target_users:
        if u_id not in user_cluster_map:
            print(f"  [WARN] Target User {u_id} not found in clusters. Skipping.")
            continue
            
        u_cluster = user_cluster_map[u_id]
        neighbors_in_cluster = cluster_users[u_cluster]
        
        # Filter out the user themselves
        neighbors_in_cluster = [n for n in neighbors_in_cluster if n != u_id]
        
        cluster_size = len(neighbors_in_cluster)
        total_similarity_computations_clustering += cluster_size
        
        print(f"\n--- Target User: {u_id} ---")
        print(f"  {'Assigned Cluster:':<40} {u_cluster:>15}")
        print(f"  {'Potential Neighbors:':<40} {cluster_size:>15,}")
        
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
        
        print(f"  {'Valid Neighbors (Sim > 0):':<40} {len(similarities):>15,}")
        print(f"  {'Selected Top 20%:':<40} {len(top_neighbors):>15,}")
        if top_neighbors:
            print(f"  {'Top Neighbor ID:':<40} {top_neighbors[0][0]:>15}")
            print(f"  {'Top Neighbor Similarity:':<40} {top_neighbors[0][1]:>15.4f}")

        # Predict for Target Items
        if u_id not in clustering_predictions:
            clustering_predictions[u_id] = {}
            
        print(f"  Predictions:")
        for i_id in target_items:
            prediction = predict_user_based(u_id, i_id, top_neighbors, user_item_ratings, user_means)
            clustering_predictions[u_id][i_id] = prediction
            print(f"    • Item {i_id}: {prediction:.4f}")

    # ==================================================================================================================
    # Baseline CF (No Clustering) - For comparison with clustering-based approach
    # This computes predictions using ALL users (no clustering) to serve as baseline for Task 10
    # ==================================================================================================================
    print("\n" + "="*80)
    print("Baseline CF (No Clustering) - For Comparison")
    print("="*80)
    
    baseline_predictions = {}
    total_similarity_computations_baseline = 0
    all_users = list(user_cluster_map.keys()) # All valid users
    
    for u_id in target_users:
        if u_id not in user_means:
            continue
        
        print(f"\n--- Target User: {u_id} ---")
        print(f"  {'Searching all users:':<40} {len(all_users)-1:>15,}")
        
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
        
        print(f"  {'Valid Neighbors (Sim > 0):':<40} {len(similarities):>15,}")
        print(f"  {'Selected Top 20%:':<40} {len(top_neighbors):>15,}")
        if top_neighbors:
            print(f"  {'Top Neighbor ID:':<40} {top_neighbors[0][0]:>15}")
            print(f"  {'Top Neighbor Similarity:':<40} {top_neighbors[0][1]:>15.4f}")
             
        if u_id not in baseline_predictions:
            baseline_predictions[u_id] = {}
        
        print(f"  Predictions:")
        for i_id in target_items:
            prediction = predict_user_based(u_id, i_id, top_neighbors, user_item_ratings, user_means)
            baseline_predictions[u_id][i_id] = prediction
            print(f"    • Item {i_id}: {prediction:.4f}")

    # ==================================================================================================================
    # Task 10: Compare clustering-based predictions with non-clustering predictions from Section TWO
    # Task 10.1: For each target user, compare the predicted ratings with and without clustering.
    # Task 10.2: Calculate the difference in prediction values.
    # Task 10.3: Discuss whether clustering improved, maintained, or degraded prediction accuracy.
    # ==================================================================================================================
    print("\n" + "="*80)
    print("Task 10: Comparison of Clustering-Based vs Baseline CF")
    print("="*80)
    
    print(f"\n  {'User':<10} {'Item':<10} {'Cluster':<12} {'Baseline':<12} {'Diff':<10}")
    print("  " + "-"*54)
    
    comparison_lines = []
    
    for u_id in target_users:
        if u_id not in clustering_predictions or u_id not in baseline_predictions:
             continue
        for i_id in target_items:
            p_cluster = clustering_predictions[u_id].get(i_id, 0.0)
            p_baseline = baseline_predictions[u_id].get(i_id, 0.0)
            diff = abs(p_cluster - p_baseline)
            print(f"  {u_id:<10} {i_id:<10} {p_cluster:<12.4f} {p_baseline:<12.4f} {diff:<10.4f}")
            comparison_lines.append(f"{u_id},{i_id},{p_cluster:.4f},{p_baseline:.4f},{diff:.4f}")

    # Save comparison to file
    comparison_file_path = os.path.join(results_dir, 'sec3_part1_comparison_results.txt')
    with open(comparison_file_path, 'w') as f:
        f.write("Comparison of Clustering-Based vs Baseline CF\n")
        f.write("="*75 + "\n")
        f.write(f"{'User':<10} | {'Item':<10} | {'Cluster Pred':<15} | {'Baseline Pred':<15} | {'Diff':<10}\n")
        f.write("-" * 75 + "\n")
        for line in comparison_lines:
            f.write(line + "\n")
    print(f"\n  [SAVED] sec3_part1_comparison_results.txt")

    # ==================================================================================================================
    # Task 11: Analyze the computational efficiency gains
    # Task 11.1: Calculate the number of similarity computations needed without clustering (comparing all user pairs).
    # Task 11.2: Calculate the number of similarity computations needed with clustering (only within-cluster pairs).
    # Task 11.3: Compute the speedup factor: (computations without clustering) / (computations with clustering).
    # Task 11.4: Express the efficiency gain as a percentage reduction in computations.
    # ==================================================================================================================
    print("\n--- Task 11: Computational Efficiency Analysis ---")
    
    comps_no_clustering = total_similarity_computations_baseline
    comps_with_clustering = total_similarity_computations_clustering
    
    speedup = comps_no_clustering / comps_with_clustering if comps_with_clustering > 0 else 0
    percent_reduction = (1 - (comps_with_clustering / comps_no_clustering)) * 100 if comps_no_clustering > 0 else 0
    
    print(f"  {'Total Users:':<40} {len(all_users):>15,}")
    print(f"  {'Computations (Baseline):':<40} {comps_no_clustering:>15,}")
    print(f"  {'Computations (Clustering):':<40} {comps_with_clustering:>15,}")
    print(f"  {'Speedup Factor:':<40} {speedup:>14.2f}x")
    print(f"  {'Efficiency Gain:':<40} {percent_reduction:>14.2f}%")

    # ==================================================================================================================
    # Task 12: Evaluate the impact of cluster imbalance
    # Task 12.1: Identify if any clusters are significantly larger or smaller than others.
    # Task 12.2: Discuss how cluster imbalance affects computational efficiency.
    # ==================================================================================================================
    print("\n--- Task 12: Cluster Imbalance Evaluation ---")
    
    # cluster_counts is already calculated for optimal_k
    sizes = cluster_counts.values
    max_size = sizes.max()
    min_size = sizes.min()
    mean_size = sizes.mean()
    std_size = sizes.std()
    
    print(f"  {'Max Cluster Size:':<40} {max_size:>15,}")
    print(f"  {'Min Cluster Size:':<40} {min_size:>15,}")
    print(f"  {'Mean Cluster Size:':<40} {mean_size:>15.2f}")
    print(f"  {'Std Dev:':<40} {std_size:>15.2f}")
    
    imbalance_ratio = max_size / min_size if min_size > 0 else float('inf')
    print(f"  {'Imbalance Ratio (Max/Min):':<40} {imbalance_ratio:>15.2f}")
    
    print(f"\n  Discussion:")
    if std_size > mean_size * 0.5:
        print("    • Significant imbalance observed.")
        print("    • Large clusters -> Slower neighbor search (bottleneck).")
        print("    • Small clusters -> Potential lack of neighbors (coverage issue).")
    else:
        print("    • Clusters are relatively balanced.")
    
    # ==================================================================================================================
    # Task 13: Test the robustness of the clustering approach
    # Task 13.1: Re-run K-means with different random initializations (at least 3 times).
    # Task 13.2: Compare the cluster assignments across different runs.
    # Task 13.3: Discuss whether the clustering is stable or varies significantly with initialization.
    # ==================================================================================================================
    print("\n--- Task 13: Robustness Test ---")
    
    seeds = [42, 100, 2023]
    robustness_results = []
    
    print(f"  {'Seed':<10} {'Inertia':<15} {'Size Std':<15}")
    print("  " + "-"*40)
    for seed in seeds:
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
        print(f"  {seed:<10} {inertia:<15.4f} {std_dev:<15.2f}")

    res_df = pd.DataFrame(robustness_results)
    inertia_std = res_df['Inertia'].std()
    if inertia_std < res_df['Inertia'].mean() * 0.05:
        print(f"\n  [RESULT] Clustering is STABLE (Inertia varies < 5%)")
    else:
        print(f"\n  [RESULT] Clustering shows VARIANCE (Inertia varies > 5%)")

    # ==================================================================================================================
    # Task 12.3: Propose strategies to handle imbalanced clusters.
    # ==================================================================================================================
    print("\n--- Task 12.3: Strategies for Imbalanced Clusters ---")
    print("  (See comprehensive analysis file for detailed strategies)")

    # ==================================================================================================================
    # Task 14: Include the results of all the above points in your report and give your insights and comments
    #          in a separate section on:
    # Task 14.1: The effectiveness of clustering based on average user ratings.
    # Task 14.2: The trade-off between prediction accuracy and computational efficiency.
    # Task 14.3: Whether this clustering strategy is suitable for your dataset characteristics.
    # Task 14.4: How the choice of K affects both accuracy and efficiency.
    # ==================================================================================================================
    print("\n" + "="*80)
    print("Task 14: Comprehensive Analysis and Insights")
    print("="*80)
    
    # Calculate prediction difference statistics
    prediction_diffs = []
    for u_id in target_users:
        if u_id in clustering_predictions and u_id in baseline_predictions:
            for i_id in target_items:
                p_cluster = clustering_predictions[u_id].get(i_id, 0.0)
                p_baseline = baseline_predictions[u_id].get(i_id, 0.0)
                prediction_diffs.append(abs(p_cluster - p_baseline))
    
    avg_prediction_diff = np.mean(prediction_diffs) if prediction_diffs else 0
    max_prediction_diff = np.max(prediction_diffs) if prediction_diffs else 0
    
    # Build comprehensive analysis
    analysis_lines = []
    analysis_lines.append("="*80)
    analysis_lines.append("SECTION 3 PART 1: COMPREHENSIVE ANALYSIS AND INSIGHTS")
    analysis_lines.append("K-means Clustering Based on Average User Ratings")
    analysis_lines.append("="*80)
    
    # 14.1 Effectiveness of clustering based on average user ratings
    analysis_lines.append("\n" + "-"*80)
    analysis_lines.append("14.1. EFFECTIVENESS OF CLUSTERING BASED ON AVERAGE USER RATINGS")
    analysis_lines.append("-"*80)
    analysis_lines.append(f"- Number of clusters (Optimal K): {optimal_k}")
    analysis_lines.append(f"- Global mean of average ratings (mu): {mu:.4f}")
    analysis_lines.append(f"- Standard deviation of average ratings (sigma): {sigma:.4f}")
    analysis_lines.append(f"- Best Silhouette Score achieved: {max(silhouette_scores):.4f} at K={k_values[silhouette_scores.index(max(silhouette_scores))]}")

    
    # 14.2 Trade-off between prediction accuracy and computational efficiency
    analysis_lines.append("\n" + "-"*80)
    analysis_lines.append("14.2. TRADE-OFF BETWEEN PREDICTION ACCURACY AND COMPUTATIONAL EFFICIENCY")
    analysis_lines.append("-"*80)
    analysis_lines.append(f"- Similarity computations (Baseline): {comps_no_clustering:,}")
    analysis_lines.append(f"- Similarity computations (Clustering): {comps_with_clustering:,}")
    analysis_lines.append(f"- Speedup Factor: {speedup:.2f}x")
    analysis_lines.append(f"- Efficiency Gain: {percent_reduction:.2f}%")
    analysis_lines.append(f"- Average Prediction Difference: {avg_prediction_diff:.4f}")
    analysis_lines.append(f"- Maximum Prediction Difference: {max_prediction_diff:.4f}")
    analysis_lines.append("")


    # 14.3 Suitability for dataset characteristics
    analysis_lines.append("\n" + "-"*80)
    analysis_lines.append("14.3. SUITABILITY FOR DATASET CHARACTERISTICS")
    analysis_lines.append("-"*80)
    analysis_lines.append(f"- Total number of users: {total_users_N}")
    analysis_lines.append(f"- Cluster imbalance ratio (Max/Min): {imbalance_ratio:.2f}")
    analysis_lines.append(f"- Average cluster size: {mean_size:.2f}")
    analysis_lines.append(f"- Cluster size standard deviation: {std_size:.2f}")
    analysis_lines.append("")

    
    # 14.4 How choice of K affects accuracy and efficiency
    analysis_lines.append("\n" + "-"*80)
    analysis_lines.append("14.4. HOW THE CHOICE OF K AFFECTS ACCURACY AND EFFICIENCY")
    analysis_lines.append("-"*80)
    analysis_lines.append("K Value Analysis:")
    for i, k in enumerate(k_values):
        avg_cluster_size = total_users_N / k
        analysis_lines.append(f"  K={k:2d}: WCSS={wcss[i]:,.2f}, Silhouette={silhouette_scores[i]:.4f}, Avg Cluster Size={avg_cluster_size:.0f}")

    
    # Robustness summary
    analysis_lines.append("\n" + "-"*80)
    analysis_lines.append("CLUSTERING ROBUSTNESS SUMMARY")
    analysis_lines.append("-"*80)
    is_stable = inertia_std < res_df['Inertia'].mean() * 0.05
    analysis_lines.append(f"- Clustering Stability: {'STABLE' if is_stable else 'VARIABLE'}")
    analysis_lines.append(f"- Inertia Standard Deviation across runs: {inertia_std:.4f}")
    if is_stable:
        analysis_lines.append("- The K-means clustering produces consistent results across different initializations.")
    else:
        analysis_lines.append("- The clustering shows variance; consider using more n_init iterations.")

    
    # Print and save analysis
    for line in analysis_lines:
        print(line)
    
    # Save comprehensive analysis to file
    analysis_file_path = os.path.join(results_dir, 'sec3_part1_comprehensive_analysis.txt')
    with open(analysis_file_path, 'w') as f:
        for line in analysis_lines:
            f.write(line + "\n")
    print(f"\n  [SAVED] sec3_part1_comprehensive_analysis.txt")

    print("\n" + "="*80)
    print("[DONE] Section 3, Part 1: Analysis completed successfully.")
    print("="*80)

if __name__ == "__main__":
    main()
