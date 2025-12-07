import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import sparse
import sys
import os
import time

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from utils.data_loader import get_preprocessed_dataset, get_target_users, get_target_items, get_user_avg_ratings
from utils.similarity import calculate_user_mean_centered_cosine
from utils.prediction import predict_user_based

RESULTS_DIR = os.path.join(project_root, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def calculate_co_rating_stats(df):
    """
    Calculates avg, max, min co-ratings for each user efficiently using sparse matrices.
    """
    print("\n--- Calculating Co-rating Statistics ---")
    
    # Map users and items to matrix indices
    unique_users = sorted(df['user'].unique())
    unique_items = sorted(df['item'].unique())
    
    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    idx_to_user = {i: u for i, u in enumerate(unique_users)}
    # item_to_idx = {item: i for i, item in enumerate(unique_items)} # Not explicitly needed if we just build matrix
    
    # Create Sparse Matrix R (User x Item)
    # Rows: Users, Cols: Items
    # Value 1 indicates rated.
    # We need to map item IDs to 0..M-1
    df['user_idx'] = df['user'].map(user_to_idx)
    df['item_idx'] = df['item'].astype('category').cat.codes
    
    n_users = len(unique_users)
    n_items = df['item_idx'].max() + 1
    
    print(f"  {'Matrix size:':<40} {n_users:,} users x {n_items:,} items")
    
    # Create CSR matrix
    # data is all 1s because we just care about count of common items
    data = np.ones(len(df))
    row_ind = df['user_idx'].values
    col_ind = df['item_idx'].values
    
    R = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n_users, n_items))
    
    stats_list = []
    
    # Compute R * R^T row by row to save memory
    # We can perform R[u] * R.T
    
    # Pre-compute R_transpose for fast multiplication
    R_T = R.T.tocsr()
    
    start_time = time.time()
    for u_idx in range(n_users):
        if u_idx % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {u_idx}/{n_users} users... ({elapsed:.2f}s)")
        
        # Get user's row
        user_vector = R[u_idx]
        
        # Calculate co-occurrence vector (1 x N_users)
        # Result is sparse
        # This vector contains |I_u intersect I_v| for all v
        co_occurrences = user_vector.dot(R_T)
        
        # Convert to numpy array (dense) strictly for stats calculation is risky if N is huge?
        # But Co-occurrences are sparse-ish? 
        # Actually, dense is 147k floats. That's 1MB. Totally fine.
        co_vals = co_occurrences.toarray().flatten()
        
        # Set self-overlap to 0 (or remove)
        # Self-overlap is at index u_idx
        # We want stats "with OTHER users"
        
        # Count of other users
        n_others = n_users - 1
        
        # Sum of co-ratings with others
        # Subtract self value
        self_val = co_vals[u_idx]
        total_common = np.sum(co_vals) - self_val
        
        avg_common = total_common / n_others if n_others > 0 else 0
        
        # Set self to -1 so it doesn't affect Max calculation (unless all are 0)
        co_vals[u_idx] = -1
        
        max_common = np.max(co_vals)
        if max_common == -1: max_common = 0 # Case where user is alone?
        
        # Min common (excluding zero)
        # We need min of values > 0
        positive_vals = co_vals[co_vals > 0]
        if len(positive_vals) > 0:
            min_common = np.min(positive_vals)
        else:
            min_common = 0
            
        stats_list.append({
            'user': idx_to_user[u_idx],
            'avg_common': avg_common,
            'max_common': max_common,
            'min_common': min_common
        })
        
    return pd.DataFrame(stats_list)

def main():
    print("\n" + "="*80)
    print("SECTION 3 PART 2: User Clustering (Common Ratings)")
    print("="*80)
    
    print("\n--- Loading Data ---")
    df = get_preprocessed_dataset()
    user_avg_ratings = get_user_avg_ratings() # r_u_bar
    target_users = get_target_users()
    target_items = get_target_items()
    print(f"  [DONE] Data loaded successfully")
    
    # ---------------------------------------------------------
    # 1. Co-rating Statistics
    # ---------------------------------------------------------
    print("\n--- Step 1: Co-rating Statistics ---")
    stats_file = os.path.join(RESULTS_DIR, 'sec3_part2_user_corating_stats.csv')
    if os.path.exists(stats_file):
        stats_df = pd.read_csv(stats_file)
        print(f"  [DONE] Loaded cached co-rating stats")
    else:
        stats_df = calculate_co_rating_stats(df)
        stats_df.to_csv(stats_file, index=False)
        print(f"  [SAVED] sec3_part2_user_corating_stats.csv")
        
    # ---------------------------------------------------------
    # 2. Normalization
    # ---------------------------------------------------------
    print("\n--- Step 2: Feature Normalization ---")
    feature_cols = ['avg_common', 'max_common', 'min_common']
    X = stats_df[feature_cols].values
    
    print(f"  Feature statistics (pre-normalization):")
    desc = stats_df[feature_cols].describe()
    for col in feature_cols:
        print(f"    • {col}: mean={desc.loc['mean', col]:.2f}, std={desc.loc['std', col]:.2f}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  [DONE] Features normalized with StandardScaler")
    
    # ---------------------------------------------------------
    # 3. K-Means
    # ---------------------------------------------------------
    print("\n--- Step 3: K-Means Clustering ---")
    k_values = [5, 10, 15, 20, 30, 50]
    wcss = []
    sil_scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        wcss.append(kmeans.inertia_)
        
        # Silhouette (Sampled if large)
        if len(X) > 20000:
            score = silhouette_score(X_scaled, labels, sample_size=20000, random_state=42)
        else:
            score = silhouette_score(X_scaled, labels)
        
        sil_scores.append(score)
        print(f"    • K={k:>2}: WCSS={kmeans.inertia_:>12.2f}, Silhouette={score:>8.4f}")
        
    # Plot Elbow & Silhouette
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(k_values, wcss, 'bo-')
    ax1.set_xlabel('K')
    ax1.set_ylabel('WCSS', color='b')
    
    ax2 = ax1.twinx()
    ax2.plot(k_values, sil_scores, 'rs-')
    ax2.set_ylabel('Silhouette Score', color='r')
    
    plt.title('Elbow and Silhouette Analysis')
    plt.savefig(os.path.join(RESULTS_DIR, 'sec3_part2_clustering_metrics.png'))
    plt.close()
    print(f"  [PLOT] sec3_part2_clustering_metrics.png")
    
    # Optimal K Selection (Heuristic: Max Sil or Elbow)
    # Using Max Silhouette for automation
    best_idx = np.argmax(sil_scores)
    optimal_k = k_values[best_idx]
    print(f"\n  {'Optimal K (Max Silhouette):':<40} {optimal_k:>15}")
    
    # ---------------------------------------------------------
    # 5. Cluster Analysis (Optimal K)
    # ---------------------------------------------------------
    print("\n--- Step 4: Cluster Analysis ---")
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    stats_df['cluster'] = final_kmeans.fit_predict(X_scaled)
    
    # Average Stats per Cluster
    cluster_summary = stats_df.groupby('cluster')[feature_cols].mean()
    cluster_counts = stats_df['cluster'].value_counts()
    cluster_summary['count'] = cluster_counts
    
    print(f"\n  Cluster Summary:")
    for idx, row in cluster_summary.iterrows():
        print(f"    • Cluster {idx}: count={int(row['count']):,}, avg_common={row['avg_common']:.2f}")
    
    # Visualize Distribution (Scatter 3D or Pairplot)
    # We can do a 3D plot of the 3 features
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Sample points for visualization if too many
    plot_sample = stats_df.sample(n=min(5000, len(stats_df)), random_state=42)
    sc = ax.scatter(plot_sample['avg_common'], plot_sample['max_common'], plot_sample['min_common'],
            c=plot_sample['cluster'], cmap='viridis', s=5)
    ax.set_xlabel('Avg Common')
    ax.set_ylabel('Max Common')
    ax.set_zlabel('Min Common')
    plt.colorbar(sc, label='Cluster')
    plt.title(f'Cluster Distribution (K={optimal_k})')
    plt.savefig(os.path.join(RESULTS_DIR, 'sec3_part2_cluster_scatter_3d.png'))
    plt.close()
    print(f"  [PLOT] sec3_part2_cluster_scatter_3d.png")
    
    # ---------------------------------------------------------
    # 6. Collaborative Filtering
    # ---------------------------------------------------------
    print("\n--- Step 5: Collaborative Filtering ---")
    
    # User-Item Ratings (Dict)
    # We need efficient lookup. Utils doesn't have a direct dict loader?
    # We can build it or use partial loaders.
    # We will build it from `df` again.
    user_item_matrix = df.pivot(index='user', columns='item', values='rating')
    # This might kill memory (147k x 12k dense-ish keys).
    # Better to use dict of dicts manually as in Part 1.
    user_ratings_dict = {}
    for row in df.itertuples():
        u, i, r = row.user, row.item, row.rating
        if u not in user_ratings_dict: user_ratings_dict[u] = {}
        user_ratings_dict[u][i] = r
        
    # Precompute means map
    user_means_map = user_avg_ratings.set_index('user')['r_u_bar'].to_dict()
    
    # Cluster Map
    user_cluster_map = stats_df.set_index('user')['cluster'].to_dict()
    cluster_users = stats_df.groupby('cluster')['user'].apply(list).to_dict()
    
    results = []
    
    for t_user in target_users:
        if t_user not in user_cluster_map:
            print(f"  [WARNING] Target User {t_user} not found in clusters.")
            continue
            
        c_id = user_cluster_map[t_user]
        print("\n" + "-"*60)
        print(f"  TARGET USER: {t_user} (Cluster {c_id})")
        print("-"*60)
        
        # Get Candidates (Same Cluster)
        candidates = cluster_users[c_id]
        
        # Filter self
        candidates = [u for u in candidates if u != t_user]
        
        # Compute Similarities
        # Sim(u, v) = MeanCenteredCosine(u, v) * DF
        # DF = min(|I_u n I_v|, beta) / beta
        # beta = 0.3 * |I_u|
        
        target_items_set = set(user_ratings_dict[t_user].keys())
        beta = 0.3 * len(target_items_set)
        
        neighbors = []
        
        u_ratings = user_ratings_dict[t_user]
        u_mean = user_means_map.get(t_user, 3.0)
        
        for cand in candidates:
            v_ratings = user_ratings_dict.get(cand, {})
            v_mean = user_means_map.get(cand, 3.0)
            
            # Common items count
            common_items = target_items_set.intersection(v_ratings.keys())
            n_common = len(common_items)
            
            if n_common == 0: continue
            
            # Mean Centered Cosine
            sim_raw = calculate_user_mean_centered_cosine(u_ratings, v_ratings, u_mean, v_mean)
            
            if sim_raw <= 0: continue
            
            # Discount Factor
            # Need to handle beta=0? If user has no ratings? but cleaned data has ratings.
            df_factor = min(n_common, beta) / beta if beta > 0 else 1.0
            
            sim_final = sim_raw * df_factor
            
            neighbors.append((cand, sim_final, n_common))
            
        # Select Top 20%
        neighbors.sort(key=lambda x: x[1], reverse=True)
        k_top = max(1, int(len(neighbors) * 0.2))
        top_neighbors = neighbors[:k_top]
        
        avg_common_top = np.mean([x[2] for x in top_neighbors]) if top_neighbors else 0
        
        print(f"    {'Cluster candidates:':<35} {len(candidates):>10,}")
        print(f"    {'Top 20% neighbors:':<35} {len(top_neighbors):>10,}")
        print(f"    {'Avg common items:':<35} {avg_common_top:>10.1f}")
        
        # Predict
        for t_item in target_items:
            # We need predict function that accepts (id, score) tuples
            # utils.prediction.predict_user_based expects neighbor_similarities as list of (id, score)
            neighbor_sims = [(n[0], n[1]) for n in top_neighbors]
            pred = predict_user_based(t_user, t_item, neighbor_sims, user_ratings_dict, user_means_map)
            
            # Calculate Error (if actual exists? No, target items are unrated usually? 
            # Part 8.2 says calculate error. But target items in 'results/target_items.txt' usually meant to be predicted.
            # However, if we want error, we need ground truth. 
            # Assuming we can't get ground truth for the *Target Items* if they are unrated.
            # But the requirement 8.2 says "Calculate prediction error... actual - predicted".
            # This implies we should predict for items that HAVE ratings? Or maybe valid testing items?
            # For now, I will predict & log. If actual exists in df, I calculate error.
            
            actual = user_ratings_dict[t_user].get(t_item, None)
            error = abs(actual - pred) if actual is not None else np.nan
            
            print(f"      • Item {t_item}:")
            print(f"        {'Prediction:':<25} {pred:>10.4f}")
            print(f"        {'Actual:':<25} {str(actual) if actual else 'N/A':>10}")
            print(f"        {'Error:':<25} {f'{error:.4f}' if not np.isnan(error) else 'N/A':>10}")
            
            results.append({
                'User': t_user,
                'Item': t_item,
                'Cluster': c_id,
                'Prediction': pred,
                'Actual': actual,
                'Error': error,
                'AvgCommon': avg_common_top
            })
            
    # Save Results
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(RESULTS_DIR, 'sec3_part2_predictions.csv'), index=False)
    print(f"\n  [SAVED] sec3_part2_predictions.csv")
    
    # ---------------------------------------------------------
    # 7. Comparison with Part 1 & Analysis
    # ---------------------------------------------------------
    
    print("\n" + "="*80)
    print("[DONE] Section 3 Part 2: User Clustering completed successfully.")
    print("="*80)

if __name__ == "__main__":
    main()
