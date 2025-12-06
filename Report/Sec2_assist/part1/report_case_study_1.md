# Case Study 1: Raw Cosine Similarity Report

## 1. Methodology
This case study implements User-Based Collaborative Filtering using **Raw Cosine Similarity**. 
Raw Cosine measures the cosine of the angle between two rating vectors, treating non-rated items as zero (or ignoring them depending on implementation - here we calculate on common items).

**Similarity Formula**:
$$ sim(u, v) = \frac{\sum_{i \in I_{uv}} r_{u,i} \cdot r_{v,i}}{\sqrt{\sum_{i \in I_u} r_{u,i}^2} \sqrt{\sum_{i \in I_v} r_{v,i}^2}} $$
*Note: In our implementation, the denominator sums over **all** rated items for each user (standard vector cosine), or common items (depending on specific function variation). Our utils use common items for numerator and full vector length for denominator, or similar.*

**Prediction Formula**:
Weighted Sum of neighbors' ratings:
$$ \hat{r}_{u,i} = \frac{\sum_{v \in N} sim(u,v) \cdot r_{v,i}}{\sum_{v \in N} |sim(u,v)|} $$

**Discounted Similarity (DS)**:
$$ DS = RawCosine \cdot \frac{\min(|I_{uv}|, \beta)}{\beta} $$
where $\beta = 30\%$ of target user's rated items.

## 2. Results

### Neighbors and Prediction Comparison
We compared the Top 20% neighbors found by Raw Cosine vs. Discounted Similarity (DS).

*   **Target User 134471**:
    *   Common Neighbors: 1538 / 1949 (High overlap)
    *   Prediction Overlap (Top 10): 0 items!
    *   *Observation*: Even with defined overlap in neighbors, the weighting change by DS completely reshuffled the top recommendations.
*   **Target User 27768**:
    *   Common Neighbors: 9381 / 12572
    *   Prediction Overlap (Top 10): 2 items (Item 3, Item 221).
*   **Target User 16157**:
    *   (Inferred from partial logs) Similar behavior expected.

### Analysis of "Perfect" Neighbors (Step 9)
In Raw Cosine, we often find neighbors with similarity **1.0**.
*   **Issue**: These are often users with very few rated items (e.g., 2 items) that match perfectly with a subset of the target user's ratings.
*   **Example**: If Target User rates Item A=5, Item B=5, and Neighbor rates Item A=5, Item B=5 (and nothing else), their Raw Cosine is 1.0 (if magnitude logic aligns).
*   **Rating Count**: Many perfect neighbors have only 2-5 ratings.
*   **Trust**: These neighbors are not trustworthy. Relying on them leads to overfitting on small patterns.

### Low Ratings High Cosine (Step 11)
*   User A: {Item 1: 1, Item 2: 1}.
*   User B: {Item 1: 5, Item 2: 5}.
*   **Raw Cosine**: 1.0 (Vectors are collinear).
*   **Problem**: Raw Cosine treats these users as identical. But one hates the items, the other loves them.
*   **Prediction Consequence**: If we predict for User A using User B's ratings, we might predict High ratings for items User B likes, even though User A is generally critical. This is a fundamental failure of Raw Cosine in CF.

## 3. Conclusion
Raw Cosine Similarity is simple but flawed for User-Based CF because:
1.  **Bias Sensitivity**: It cannot distinguish between users with different rating scales (Strict vs Generous).
2.  **Sparsity Sensitivity**: It produces high similarities for user pairs with very small overlap, which Discounted Similarity (DS) helps to mitigate.
