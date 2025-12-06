# Part 2: K-means Clustering based on Average Number of Common Ratings

## 1. Methodology

### 1.1 Feature Engineering
We constructed a feature vector for each user to capture the density of their co-rating network:
1.  **Average Common Ratings**: The average number of co-rated items with all other users. This proxies the user's "centrality" or "reach" in the dataset.
2.  **Max Common Ratings**: The maximum overlap with any single user. This indicates the potential for finding at least one "strong" neighbor.
3.  **Min Common Ratings (excluding zero)**: The minimum nonzero overlap.

We calculated these statistics using efficient Sparse Matrix operations ($C = R \cdot R^T$) to handle the 147k user dataset.

### 1.2 Clustering
- **Algorithm**: K-Means
- **K Range**: [5, 10, 15, 20, 30, 50]
- **Selection Metric**: Silhouette Score and Elbow Method.
- **Optimal K**: **5** (Selected based on Max Silhouette Score).

## 2. Clustering Results

### 2.1 Cluster Characteristics (K=5)

| Cluster ID | Avg Common | Max Common | Count | Interpretation |
|:---:|:---:|:---:|:---:|:---|
| **0** | ~0.03 | ~2.9 | 114,319 | **Sparse / Periphery**: Users with very little overlap. The vast majority of the dataset (77%). |
| **4** | ~0.21 | ~12.5 | 26,141 | **Low Overlap**: Users with some connections, but limited. |
| **2** | ~0.57 | ~31.0 | 6,565 | **Medium Overlap**: Users with moderate connectivity. |
| **3** | ~1.35 | ~81.1 | 886 | **High Overlap (Core)**: Highly connected users. Ideal for significance weighting. |
| **1** | ~0.00 | ~0.0 | 3 | **Outliers**: Likely disconnected or data anomalies. |

**Observation**: The clusters clearly follow a power-law distribution. Most users are in the "Sparse" cluster, while a small "Core" (Cluster 3) contains highly connected users.

### 2.2 Visualization
(See `results_media/cluster_scatter_3d.png` and `results_media/clustering_metrics.png`)

## 3. Collaborative Filtering Results

We applied User-Based CF within each cluster using **Mean-Centered Cosine Similarity** and a **Discount Factor** ($\beta = 0.3 \times |I_{target}|$).

### 3.1 Prediction comparison (Part 1 vs Part 2)

| User | Item | Part 1 Pred (Avg Rating Cluster) | Part 2 Pred (Common Rating Cluster) | Difference | Context |
|---|---|---|---|---|---|
| 134471 | 1333 | 3.60 | **3.43** | 0.17 | User 134471 is in **Sparse Cluster (0)**. |
| 134471 | 1162 | 3.31 | **3.55** | 0.24 | |
| 27768 | 1333 | 3.29 | **3.38** | 0.09 | User 27768 is in **Core Cluster (3)**. |
| 27768 | 1162 | 3.24 | **3.34** | 0.10 | |

### 3.2 Analysis

#### Accuracy & Significance Weighting
- **Core Users (Cluster 3)**: User 27768 found neighbors with an average overlap of **~40 items**. This indicates extremely high confidence (Significance Weighting is high). The predictions are robust.
- **Sparse Users (Cluster 0)**: User 134471 found neighbors with an average overlap of **~1 item**. This is a **Cold Start / Sparse** problem. The predictions are likely less reliable despite the clustering.
- **Comparison with Part 1**: Part 2 predictions for the Sparse user (134471) were closer to the global baseline (calculated in Part 1 logs) than Part 1's predictions were. This suggests Part 2 might be safer for sparse users by keeping them in a large "General" pool rather than forcing them into artificial "Strict/Generous" groups.

#### Computational Efficiency
- **Part 1 (Avg Rating)**: Likely created more balanced clusters. Speedup is roughly proportional to $K$.
- **Part 2 (Common Rating)**: Created **highly unbalanced** clusters.
    - Cluster 0 has 114k users. Searching for neighbors in Cluster 0 is almost as slow as the global search (Speedup $\approx 1.3x$).
    - Cluster 3 has 886 users. Searching in Cluster 3 is instantaneous (Speedup $\approx 160x$).
- **Conclusion**: Part 2 is **less efficient** for the majority of users (the sparse ones) but highly efficient for the power users.

#### Significance Weighting Impact
- **Part 2 directly addresses Significance Weighting** by grouping users with similar "data quality".
- Users in Cluster 3 use high-quality neighbors (high overlap).
- Users in Cluster 0 are forced to use low-quality neighbors, but at least they are not mixed with high-overlap users who might dominate (or be irrelevant).
- **Effectiveness**: Highly effective for identifying and treating "Power Users" differently from "Casual Users".

## 4. Recommendations
- Use **Part 2 (Common Rating Clustering)** when: 
    - You want to provide **high-confidence recommendations** to power users.
    - You want to identify and handle **Cold Start** users separately (e.g., give them non-personalized recommendations instead of weak CF predictions).
- Use **Part 1 (Avg Rating Clustering)** when:
    - You need **consistent speedup** across all users (balanced clusters).
    - User behavior (Strictness/Generosity) is the primary driver of preference difference.
