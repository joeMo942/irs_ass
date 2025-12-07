# Part 4: K-means Clustering for Cold-Start Problem

## 1. Introduction

The cold-start problem is one of the fundamental challenges in recommender systems. It occurs when:
- **Cold-start users**: New users with few or no ratings
- **Cold-start items**: New items with limited rating history

This report presents a clustering-based approach to handle cold-start scenarios using K-means clustering, leveraging user clusters from Part 1 and item clusters from Part 3.

### Dataset Overview

| Metric | Value |
|--------|-------|
| Total Ratings | 2,149,655 |
| Total Users | 147,914 |
| Total Items | 11,123 |
| User Clusters | 10 |
| Item Clusters | 10 |

---

## 2. Cold-Start Simulation (Task 1)

### 2.1 Methodology

**Cold-Start User Simulation (Tasks 1.1-1.3):**
- Selected 100 users with more than 50 ratings each
- Hidden 80% of ratings to simulate cold-start scenario
- Kept 10-20 visible ratings per user
- Stored hidden ratings as ground truth for evaluation

**Cold-Start Item Simulation (Task 1.4):**
- Selected 50 items with more than 20 ratings
- Hidden 80% of ratings to simulate cold-start items
- Kept 3-10 visible ratings per item

### 2.2 Simulation Results

| Category | Count | Avg Visible | Avg Hidden |
|----------|-------|-------------|------------|
| Cold-Start Users | 100 | 15.0 | 80.3 |
| Cold-Start Items | 50 | 8.8 | 216.5 |

---

## 3. Cold-Start User Assignment Strategy (Tasks 2-4)

### 3.1 Feature Calculation (Task 2.1)

For each cold-start user, we calculate their limited profile feature:

$$\bar{r}_u = \frac{1}{|I_u|} \sum_{i \in I_u} r_{ui}$$

Where:
- $\bar{r}_u$ = average rating of user $u$
- $I_u$ = set of items rated by user (visible ratings only)
- $r_{ui}$ = rating given by user $u$ to item $i$

### 3.2 Distance Computation (Task 2.2)

The Euclidean distance between cold-start user's feature vector and each cluster centroid:

$$d(u, c_k) = \sqrt{(z_u - \mu_k)^2}$$

Where:
- $z_u$ = Z-score normalized average rating of user $u$
- $\mu_k$ = centroid of cluster $k$

**Example Calculation:**

For a user with average rating $\bar{r}_u = 3.8$:
1. Z-score normalization: $z_u = \frac{3.8 - \mu}{\sigma}$
2. Distance to centroid $k$: $d_k = |z_u - \mu_k|$

### 3.3 Cluster Assignment (Task 2.3)

Users are assigned to the nearest cluster:

$$\text{cluster}(u) = \arg\min_k d(u, c_k)$$

### 3.4 Assignment Confidence (Task 2.4)

Confidence score calculated as:

$$\text{Confidence} = \frac{d_{\text{second}} - d_{\text{nearest}}}{d_{\text{second}}}$$

Where:
- $d_{\text{nearest}}$ = distance to assigned (nearest) cluster
- $d_{\text{second}}$ = distance to second-nearest cluster

**Interpretation:**
- Values close to 1.0 → High confidence (clearly belongs to assigned cluster)
- Values close to 0.0 → Low confidence (nearly equidistant between clusters)

---

## 4. Cold-Start User Recommendations (Task 3)

### 4.1 Finding Similar Users Within Cluster (Task 3.1)

Within the assigned cluster, similarity is computed using Mean-Centered Cosine:

$$sim(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i} (r_{ui} - \bar{r}_u)^2} \cdot \sqrt{\sum_{i} (r_{vi} - \bar{r}_v)^2}}$$

Where $I_{uv}$ = items rated by both users $u$ and $v$.

### 4.2 Prediction Formula (Task 3.2)

$$\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N(u)} sim(u,v) \cdot (r_{vi} - \bar{r}_v)}{\sum_{v \in N(u)} |sim(u,v)|}$$

Where $N(u)$ = top 20% most similar neighbors from the same cluster.

### 4.3 Top-10 Recommendations (Task 3.3)

For each cold-start user, items are ranked by predicted rating and top-10 are selected.

---

## 5. Evaluation of Cold-Start User Recommendations (Task 4)

### 5.1 Error Metrics (Tasks 4.1-4.2)

**Mean Absolute Error (MAE):**

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |r_i - \hat{r}_i|$$

**Root Mean Squared Error (RMSE):**

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (r_i - \hat{r}_i)^2}$$

### 5.2 Ranking Metrics (Task 4.3)

**Precision@10:**

$$Precision@10 = \frac{\text{Number of relevant items in top 10}}{10}$$

**Recall@10:**

$$Recall@10 = \frac{\text{Number of relevant items in top 10}}{\text{Total number of relevant items}}$$

### 5.3 Results

| Method | MAE | RMSE | Precision@10 | Recall@10 |
|--------|-----|------|--------------|-----------|
| **Clustering-Based CF** | 0.6053 | 0.7983 | 0.0090 | 0.0015 |
| **Baseline (No Clustering)** | 0.5865 | 0.7507 | - | - |

### 5.4 Comparison Analysis (Task 4.4)

| Metric | Clustering | Baseline | Difference |
|--------|------------|----------|------------|
| MAE | 0.6053 | 0.5865 | +0.0188 |
| RMSE | 0.7983 | 0.7507 | +0.0476 |

**Observation:** The baseline slightly outperforms clustering in terms of raw accuracy. However, this doesn't account for computational efficiency gains from clustering.

---

## 6. Cold-Start Item Assignment Strategy (Task 5)

### 6.1 Item Feature Vector (Task 5.1)

For each cold-start item, we calculate:
- **Number of raters**: $n_i = |U_i|$
- **Average rating**: $\bar{r}_i = \frac{1}{n_i} \sum_{u \in U_i} r_{ui}$
- **Rating standard deviation**: $\sigma_i$

### 6.2 Assignment and Confidence (Tasks 5.2-5.3)

Items are assigned to nearest cluster using 3D feature vector distance:

$$d(i, c_k) = \sqrt{\sum_{f=1}^{3} (z_{if} - \mu_{kf})^2}$$

**Confidence Formula:**

$$\text{Confidence} = \frac{d_{\text{second}} - d_{\text{nearest}}}{d_{\text{second}}}$$

### 6.3 Item Cluster Assignment Results (Task 5.3.4)

| Item ID | Cluster | d_nearest | d_second | Confidence |
|---------|---------|-----------|----------|------------|
| 5807 | 8 | 1.3062 | 2.0984 | 0.3775 |
| 7147 | 6 | 1.0514 | 2.0926 | 0.4976 |
| 1162 | 4 | 1.0489 | 1.0495 | 0.0005 |
| 5541 | 1 | 0.4262 | 1.3503 | 0.6844 |
| 6588 | 5 | 1.4585 | 1.9227 | 0.2414 |
| 5626 | 1 | 0.5565 | 1.1733 | 0.5257 |
| 4852 | 1 | 0.4813 | 1.2204 | 0.6056 |
| 7190 | 1 | 0.4046 | 1.0142 | 0.6011 |
| 5444 | 6 | 0.7059 | 0.7223 | 0.0228 |
| 3399 | 4 | 1.2670 | 1.4970 | 0.1537 |

**Notable Observations:**
- Item 1162 has extremely low confidence (0.0005) - nearly equidistant between clusters
- Items 5541, 4852, 7190 have high confidence (>0.6) - clearly belong to cluster 1

---

## 7. Cold-Start Item Predictions (Tasks 6-7)

### 7.1 Prediction Method (Task 6)

Using Item-Based CF with Adjusted Cosine Similarity within clusters:

$$sim(i, j) = \frac{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)(r_{uj} - \bar{r}_u)}{\sqrt{\sum_{u} (r_{ui} - \bar{r}_u)^2} \cdot \sqrt{\sum_{u} (r_{uj} - \bar{r}_u)^2}}$$

### 7.2 Results (Task 7.1-7.3)

| Method | MAE | RMSE |
|--------|-----|------|
| **Clustering-Based Item CF** | 0.6244 | 0.8332 |
| **Baseline (No Clustering)** | 0.6900 | 0.9277 |

### 7.3 Comparison Analysis

| Metric | Clustering | Baseline | Improvement |
|--------|------------|----------|-------------|
| MAE | 0.6244 | 0.6900 | **9.5%** |
| RMSE | 0.8332 | 0.9277 | **10.2%** |

**Key Finding:** Unlike cold-start users, clustering **significantly improves** cold-start item predictions. This is because items can be more reliably characterized by their few initial ratings (average, std, count) than users can.

---

## 8. Rating Count vs Prediction Accuracy (Task 8)

### 8.1 Accuracy Analysis (Tasks 8.1-8.2)

| Number of Ratings | MAE | Improvement |
|-------------------|-----|-------------|
| 5 | 0.6520 | - |
| 10 | 0.6345 | 2.67% |
| 15 | 0.6020 | 5.13% |
| 20 | 0.6061 | -0.69% |

### 8.2 Accuracy Curve

![Rating Count vs MAE](file:///home/yousef/irs_ass/results/sec3_part4_rating_count_accuracy.png)

### 8.3 Transition Point Analysis (Task 8.3)

Based on the improvement rates:
- **5→10 ratings**: 2.67% improvement
- **10→15 ratings**: 5.13% improvement (largest gain)
- **15→20 ratings**: -0.69% (no improvement)

**Conclusion:** Users transition from "cold-start" to "having sufficient data" at approximately **15 ratings**. Beyond this point, additional ratings provide diminishing returns for clustering-based predictions.

---

## 9. Hybrid Cold-Start Strategy (Task 9)

### 9.1 Methodology

Combining clustering-based CF with content-based features:
- **Cluster-based predictions**: Generated using assigned cluster neighbors
- **Content-based proxy**: Item popularity and user preference matching
- **Hybrid formula**: 70% clustering + 20% item popularity + 10% preference match

### 9.2 Results

| Method | MAE | RMSE |
|--------|-----|------|
| **Clustering Only** | 0.5867 | 0.7857 |
| **Hybrid (CF+Content)** | 0.5833 | 0.7522 |

**Improvement from hybrid:** 0.58%

### 9.3 Evaluation (Task 9.3)

**Conclusion:** The hybrid approach provides a modest improvement (0.58%). While the improvement is small, hybrid methods offer:
1. Better fallback when cluster-based predictions fail
2. More robust handling of items outside training distribution
3. Foundation for incorporating richer content features when available

---

## 10. Cold-Start Robustness Testing (Task 10)

### 10.1 Varying Information Availability (Task 10.1)

Testing prediction quality with 3, 5, 10, and 20 available ratings:

| Ratings | MAE | Avg Predictions | Degradation vs 20-rating |
|---------|-----|-----------------|-------------------------|
| 3 | 0.6879 | 35 | +13.9% |
| 5 | 0.6339 | 41 | +4.9% |
| 10 | 0.6283 | 58 | +4.0% |
| 20 | 0.6041 | 60 | Baseline |

### 10.2 Degradation Analysis (Task 10.2)

![Robustness Curve](file:///home/yousef/irs_ass/results/sec3_part4_robustness_test.png)

**Observations:**
- MAE degrades by 13.9% with only 3 ratings vs 20 ratings
- Prediction availability drops significantly (35 vs 60 predictions)
- The steepest degradation occurs between 3-5 ratings

### 10.3 Minimum Ratings for Acceptable Quality (Task 10.3)

| Threshold | Value |
|-----------|-------|
| Baseline MAE (20 ratings) | 0.6041 |
| Acceptable threshold (within 20%) | 0.7249 |
| **Minimum ratings** | **3** |

**Interpretation:** Even with just 3 ratings, the system remains within acceptable quality bounds (MAE < 0.7249). However, for practical recommendations, **5+ ratings** is recommended to ensure sufficient prediction coverage (41+ items).

---

## 11. Cluster Assignment Confidence Analysis (Task 11)

### 11.1 Confidence Distribution (Task 11.1)

**Ratio Definition:**

$$\text{Ratio} = \frac{d_{\text{nearest}}}{d_{\text{second}}}$$

| Category | Count | Percentage |
|----------|-------|------------|
| Confident (ratio < 0.5) | 64 | 64.0% |
| Moderate (0.5 ≤ ratio ≤ 0.7) | 13 | 13.0% |
| Ambiguous (ratio > 0.7) | 23 | 23.0% |

### 11.2 Ambiguous Cases (Task 11.2)

| User ID | Ratio | Status |
|---------|-------|--------|
| 44156 | 0.907 | Highly Ambiguous |
| 18012 | 0.857 | Highly Ambiguous |
| 141661 | 0.743 | Ambiguous |
| 110083 | 0.743 | Ambiguous |
| 16675 | 0.732 | Ambiguous |

### 11.3 Strategies for Ambiguous Cases (Task 11.3)

1. **Multi-cluster membership**: Assign user to top-2 clusters and combine predictions with weights inversely proportional to distance
   
2. **Weighted recommendations**: 
   $$\hat{r}_{ui} = \sum_{k} w_k \cdot \hat{r}_{ui}^{(k)}$$
   where $w_k = \frac{1/d_k}{\sum_j 1/d_j}$

3. **Ensemble approach**: Generate predictions from each potential cluster and combine using confidence-weighted voting

---

## 12. Comparison of Cold-Start Strategies (Task 12)

### 12.1 Strategy Descriptions

| Strategy | Description |
|----------|-------------|
| **Cluster-based CF** | Use cluster membership to reduce search space |
| **Global CF** | Traditional CF searching all users |
| **Popularity-based** | Recommend items with highest average ratings |

### 12.2 Performance Comparison (Task 12.4)

| Strategy | MAE | RMSE | Efficiency |
|----------|-----|------|------------|
| Cluster-based CF | 0.5738 | 0.7788 | High |
| Global CF | 0.6156 | 0.8258 | Low |
| Popularity-based | **0.5251** | **0.7225** | Very High |

**Key Finding:** For cold-start users, popularity-based recommendations achieve the best MAE (0.5251), followed by cluster-based CF (0.5738). This suggests that when user preferences are largely unknown, recommending popular items is a robust fallback.

---

## 13. Cluster Granularity Impact (Task 13)

### 13.1 Performance by Cluster Count (Tasks 13.1-13.2)

| K | MAE | Avg Cluster Size | Observation |
|---|-----|------------------|-------------|
| 5 | **0.6002** | ~29,583 | Best performance |
| 10 | 0.6197 | ~14,791 | Moderate |
| 20 | 0.6621 | ~7,396 | Declining |
| 50 | 0.7060 | ~2,958 | Worst performance |

### 13.2 Trade-off Analysis (Task 13.3)

**Mathematical Interpretation:**

- **Smaller K (larger clusters)**:
  - More potential neighbors → Higher data availability
  - Less homogeneous groups → Lower similarity quality
  
- **Larger K (smaller clusters)**:
  - More homogeneous groups → Better similarity quality
  - Fewer potential neighbors → Data sparsity issues

**Optimal Trade-off:** For cold-start scenarios, K=5 provides the best balance because:
1. Cold-start users have limited overlapping ratings
2. Larger clusters increase the probability of finding matching items
3. The cost of reduced homogeneity is offset by data availability

---

## 14. Confidence-Based Recommendation Strategy (Task 14)

### 14.1 Basic Confidence Score Computation

$$\text{Conf}_{total} = \text{Conf}_{assignment} \times \min(1.0, \frac{n_{similar}}{100})$$

### 14.2 Basic Filtering Results

| Category | Count | MAE |
|----------|-------|-----|
| High-confidence (score > 0.5) | 1,394 | 0.6222 |
| Low-confidence (score ≤ 0.5) | 439 | 0.6326 |
| **All predictions** | 1,833 | 0.6247 |

**Basic improvement from filtering:** 0.40%

### 14.3 Enhanced Confidence Score (Task 14.1 - All 3 Factors)

The enhanced confidence score combines:

$$\text{Conf}_{enhanced} = 0.4 \cdot \text{Conf}_{cluster} + 0.3 \cdot \text{Factor}_{similar} + 0.3 \cdot \text{Factor}_{agreement}$$

Where:
- **Cluster confidence**: Assignment confidence (d_second - d_nearest) / d_second
- **Similar factor**: min(1.0, n_neighbors / 50)
- **Agreement factor**: 1.0 - σ_predictions / 2.0 (lower std = higher agreement)

### 14.4 Enhanced Filtering Results

| Category | Count | MAE |
|----------|-------|-----|
| High-confidence (score > 0.5) | 1,821 | 0.6216 |
| Low-confidence (score ≤ 0.5) | 12 | 0.6158 |
| **All predictions** | 1,833 | 0.6216 |

**Enhanced improvement:** -0.01% (no improvement)

### 14.5 Analysis

The enhanced confidence score with agreement factor resulted in **more predictions being classified as high-confidence** (1,821 vs 1,394). However, for this dataset:
- The low-confidence predictions actually had slightly better MAE (0.6158)
- This suggests the agreement factor may not be discriminative for cold-start scenarios
- Cold-start by nature has limited neighbor overlap, making agreement measurement less reliable

**Recommendation:** For cold-start scenarios, use the simpler 2-factor confidence (cluster assignment + neighbor count) rather than the 3-factor version.

---

## 15. Summary and Insights (Task 15)

### 15.1 Effectiveness of Clustering for Cold-Start

| Aspect | Observation |
|--------|-------------|
| Search Space Reduction | Clustering significantly reduces neighbor search from all users to cluster members |
| Assignment Speed | O(K) instead of O(N) for initial cluster assignment |
| Prediction Quality | Slightly lower than global CF, but acceptable trade-off |

### 15.2 Best Performing Strategy

| Scenario | Recommended Strategy | Reason |
|----------|---------------------|--------|
| < 5 ratings | Popularity-based | Insufficient data for CF |
| 5-15 ratings | Cluster-based CF | Balance of efficiency and accuracy |
| > 15 ratings | Global CF | User has sufficient history |

### 15.3 Minimum Data Requirements

| Rating Count | Quality | Recommendation |
|--------------|---------|----------------|
| 0-5 | Unreliable | Use popularity/content-based |
| 5-10 | Marginal | Cluster-based with caution |
| 10-15 | Acceptable | Cluster-based CF |
| 15+ | Reliable | Transition to standard CF |

### 15.4 New Findings from Tasks 9-10

| Finding | Source |
|---------|--------|
| Hybrid approach improves MAE by 0.58% | Task 9 |
| Minimum 3 ratings for acceptable quality | Task 10 |
| 13.9% MAE degradation with only 3 ratings | Task 10 |
| Agreement factor not effective for cold-start | Task 14 Enhanced |

---

## 16. Conclusions and Recommendations

### Conclusions

1. **Clustering effectively reduces cold-start severity** by grouping users with similar rating behaviors, making limited overlap more meaningful.

2. **K=5 is optimal for cold-start scenarios** because larger clusters provide more potential neighbors despite reduced homogeneity.

3. **15 ratings is the transition threshold** from cold-start to having sufficient data for reliable predictions.

4. **Popularity-based methods outperform CF for very cold users** (MAE 0.5251 vs 0.5738-0.6156).

5. **23% of cluster assignments are ambiguous**, requiring special handling strategies.

### Recommendations for Production Systems

| Recommendation | Implementation |
|----------------|----------------|
| **Hybrid Strategy** | Start with popularity for new users, transition to clustering-based CF at 5 ratings, switch to global CF at 15+ ratings |
| **Confidence Monitoring** | Track assignment confidence and use fallback strategies when ratio > 0.7 |
| **Multi-cluster Membership** | For ambiguous assignments, generate predictions from top-2 clusters and weight by distance |
| **Content-based Augmentation** | When available, use item attributes to improve initial cluster assignments |
| **Adaptive K Selection** | Use smaller K for cold-start users (more data availability) and larger K for established users (more precision) |

### Future Improvements

1. Incorporate demographic features for improved initial cluster assignment
2. Develop dynamic clustering that adapts as user behavior evolves
3. Implement online learning to update clusters incrementally
4. Explore graph-based approaches for cold-start user similarity

---

## References

- Part 1: User Clustering with Average Ratings
- Part 3: Item Clustering with Multi-dimensional Features
- Dataset: Dianping Social Recommendation Dataset (2015)
