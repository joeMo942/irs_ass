# Section 3 Part 3: Item-Based Clustering Analysis Report

**Course:** IE425 Intelligent Recommender Systems  
**Date:** Fall 2025-2026

---

## 1. Feature Engineering & Normalization (Tasks 1 & 2)

**1.1 & 1.2 Item Statistics:**
We calculated the following statistics for each of the $N=11,123$ items:
*   **Number of Raters ($N_u(i)$):** Total users who rated item $i$.
*   **Average Rating ($\bar{r}_i$):** Mean rating score.
*   **Standard Deviation ($\sigma_i$):** Variability of ratings.

**1.3 Feature Vector:**
$v_i = [N_u(i), \bar{r}_i, \sigma_i]$

**2.1 Normalization:**
To ensure equal contribution to distance calculations, we applied Z-score standardization:
$z_{i,f} = \frac{x_{i,f} - \mu_f}{\sigma_f}$
After normalization, all features have $\mu \approx 0$ and $\sigma = 1$.

---

## 2. Clustering Analysis (Tasks 3, 4, 5, 6)

**3. & 4. Optimal K Selection:**
We evaluated K values $[5, 10, 15, 20, 30, 50]$.
*   **Elbow Method:** The WCSS decreased significantly up to $K=10$ and started leveling off (diminishing returns) around $K=15-20$.
*   **Silhouette Score:** The score peaked around $K=10$ (Score $\approx 0.306$).
*   **Selected Optimal K:** **10**

**5. Cluster Characteristics (K=10):**
Clusters were defined by *popularity* (average number of raters).
*   **Popular (Head) Clusters:** e.g., Cluster 2 (Avg ~3260 raters), Cluster 7 (Avg ~1542 raters). These contain "Blockbuster" items.
*   **Niche (Tail) Clusters:** e.g., Cluster 5 (Avg ~2.5 raters), Cluster 8 (Avg ~3.0 raters). These contain the vast majority of items which are rarely rated.

**6. Distribution:**
*   **Head Items (Top 20%):** Grouped tightly into small numbers of clusters with high rater counts.
*   **Tail Items (Bottom 80%):** Spread across multiple "low-density" clusters.
*   *Observation:* The clustering basis strongly separates items by popularity.

---

## 3. Prediction & Error Analysis (Tasks 7 & 8)

We compared two approaches:
1.  **Baseline Item-Based CF:** Global search (k-NN prediction using all items).
2.  **Clustering-Based Item-Based CF:** Local search (Prediction using only neighbors within the same cluster).

**8.2 Prediction Error Results:**
For the target users and items, we calculated the Mean Absolute Error (MAE):

| Approach | Overall MAE |
| :--- | :--- |
| Baseline (Global) | **0.1906** |
| Clustering (Local) | **0.1233** |

**8.3 Conclusion:**
The **Clustering-based approach** produced *more reliable predictions* (lower error) for the target "Head/Popular" items. By filtering out the noise of thousands of irrelevant/dissimilar items, the local neighborhood provided a stronger signal for these popular items.

---

## 4. Long-Tail Problem Impact (Task 9)

We sampled 50 Random "Tail" items (unpopular items) to evaluate performance on the long tail.

**9.1 & 9.2 Reliability:**
*   **Avg Error (Baseline):** 0.3659
*   **Avg Error (Clustering):** 0.4502
*   **Result:** Clustering **increased** the prediction error for long-tail items.

**9.3 Neighbor Analysis:**
*   **Avg Neighbors Found (Baseline):** 1188.4
*   **Avg Neighbors Found (Clustering):** 244.0
*   **Insight:** Long-tail items suffer from data sparsity. In the global baseline, they might find ~1000 items with *some* similarity. In clustering, they are confined to a "Niche Cluster" (e.g., Cluster 8 with avg 3 raters). This drastically reduces the pool of potential neighbors ($\sim 80\%$ reduction in candidates), making it likely that *none* of the neighbors have been rated by the active user, leading to poor predictions.

---

## 5. Computational Efficiency (Task 10)

**Mathematical Calculations:**

**10.1 Steps:**
1.  **Baseline Complexity:** $N_{items}^2$ comparisons.
    $OPS_{base} = 11,123^2 = 123,721,129$ comparisons.
2.  **Clustering Complexity:** Sum of squared cluster sizes $\sum |C_k|^2$.
    $OPS_{clus} = \sum_{k=1}^{10} size_k^2 \approx 22,954,151$ comparisons.
3.  **Reduction Formula:**
    $Reduction = \frac{OPS_{base} - OPS_{clus}}{OPS_{base}} \times 100$
    $Reduction = \frac{123,721,129 - 22,954,151}{123,721,129} \approx 0.8145$

**10.2 Speedup Factor:**
$Speedup = \frac{Time_{base}}{Time_{clus}} \approx \frac{OPS_{base}}{OPS_{clus}}$
$Speedup = \frac{123,721,129}{22,954,151} \approx \mathbf{5.39\times}$

**10.3 Comparison:**
This speedup (5.39x) is significant. It is typically **greater** than User-Based clustering speedups for this dataset because the item distribution (Head/Tail) allows for extremely unbalanced clusters where the "Tail" clusters are large in number but sparse in computation, whereas users might be more uniformly distributed.

---

## 6. Cluster Size Analysis (Task 11)

**11.1 Avg Error by Cluster:**

| Cluster | Size | Avg Error | Type |
| :--- | :--- | :--- | :--- |
| 1 | 3291 | 0.00 | Head |
| 4 | 2058 | 0.00 | Head |
| 7 | 275 | 0.12 | Mid |
| ... | ... | ... | ... |
| Tail Sample | ~300 | ~0.45 | Tail |

**11.2 & 11.3 Analysis:**
*   **Correlation:** -0.2713 (Negative)
*   **Relationship:** Larger clusters $\rightarrow$ Lower Error.
*   **Reasoning:** Larger clusters provide a richer "pool of neighbors". This confirms that `Size` is a proxy for `Information Availability`.
*   **Optimal Size:** Medium-to-Large clusters (> 500 items). Small clusters (< 100 items) are too sparse for reliable CF.

---

## 7. Comparative Analysis (Task 12)

**12.1 Effectiveness:**
*   **Item-Based Clustering:** Better for **Scalability** (5.4x speedup) and **Accuracy on Popular Items** (MAE 0.12).
*   **User-Based Clustering:** Often suffers from higher dimensionality and shifting user preferences.
*   **Verdict:** **Item-Based Clustering** is more effective for this dataset due to the stability of item attributes and the clear popularity-based segmentation.

**12.2 Recommendations by Scenario:**
*   **Use User-Based:** For social networks or "serendipity" discovery where item content matters less than peer groups.
*   **Use Item-Based:** For E-commerce (Amazon, Netflix) where inventory is large/stable and "Item-Item" relationships (People who bought X also bought Y) are strong predictive signals.

**12.3 Combination (Biclustering):**
*   **Feasibility:** Yes.
*   **Benefit:** Identify "User Communities" + "Item Genres" blocks.
*   **Risk:** Extreme sparsity. The intersection of a niche user cluster and a niche item cluster might have zero ratings.

---

## 8. Final Conclusion & Recommendations (Task 13)

**13.1 Long-Tail Strategy:**
Item-based clustering **fails** the long tail (MAE increased).
*Recommendation:* Do NOT use clustering for the bottom 80% of items. Use a global search or content-based filtering for these items.

**13.2 Popularity Impact:**
Clustering quality is driven by item popularity. Popular items form dense, reliable clusters. Unpopular items form sparse, unreliable clusters.

**13.3 Deployment Recommendation (Practical):**
Implement a **Hybrid Hybrid System**:
1.  **Tier 1 (Head Items):** Use **Clustered Item-Based CF**. It is 5x faster and more accurate for the top 20% of traffic.
2.  **Tier 2 (Tail Items):** Use **Global Item-Based CF** (or Content-Based). Do not limit the search space for niche items; they need every potential neighbor they can find.
3.  **Optimization:** Pre-compute the similarities for the Head Clusters (offline) since they are stable. Compute Tail similarities on-demand or using an approximated nearest neighbor (ANN) index to mitigate the computational cost.
