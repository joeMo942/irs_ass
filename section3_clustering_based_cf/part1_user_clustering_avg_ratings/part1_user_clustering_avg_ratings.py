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

from utils.data_loader import get_user_avg_ratings

def main():
    print("Loading user average ratings...")
    df = get_user_avg_ratings()
    
    # 2. Create a 1-dimensional feature vector for each user
    X = df[['r_u_bar']].values
    
    # 3. Calculate mean
    mu = df['r_u_bar'].mean()
    print(f"Mean of users' average ratings (mu): {mu:.4f}")
    
    # 4. Compute Standard Deviation
    sigma = df['r_u_bar'].std()
    print(f"Standard deviation (sigma): {sigma:.4f}")
    
    # 5. Normalize feature values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 6. Apply K-means with different K
    k_values = [5, 10, 15, 20, 30, 50]
    wcss = []
    silhouette_scores = []
    
    print("\nStarting K-means clustering analysis...")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # 7.2 WCSS
        wcss.append(kmeans.inertia_)
        
        # 7.4 Silhouette Score (Full Data)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        
        print(f"  K={k}: WCSS={kmeans.inertia_:.4f}, Silhouette={score:.4f}")

    # 7.3 Plot Elbow Curve and Silhouette
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

    # 8. Optimal K selection (Heuristic: Max Silhouette Score)
    optimal_k_index = np.argmax(silhouette_scores)
    optimal_k = k_values[optimal_k_index]
    print(f"\nOptimal K based on max Silhouette Score: {optimal_k}")

    # Rerun for Optimal K
    best_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['cluster'] = best_kmeans.fit_predict(X_scaled)
    centroids_scaled = best_kmeans.cluster_centers_
    
    # Inverse transform centroids to get original rating scale
    centroids_original = scaler.inverse_transform(centroids_scaled).flatten()
    
    # 8.1 Distribution of users
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

    # 8.2 & 8.3 Analyze Centroids
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

    print("\nAnalysis completed.")

if __name__ == "__main__":
    main()
