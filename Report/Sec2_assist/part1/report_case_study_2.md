# Case Study 2: Mean-Centered Cosine Similarity Report

## 1. Methodology
This case study implements User-Based Collaborative Filtering using **Mean-Centered Cosine Similarity** (Pearson Correlation). This metric accounts for user rating bias (some users consistently rate higher or lower than others) by centering ratings around each user's mean.

### Formulas
**Similarity (Pearson Correlation):**
$$ sim(u, v) = \frac{\sum_{i \in I_{uv}} (r_{u,i} - \bar{r}_u)(r_{v,i} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{u,i} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{v,i} - \bar{r}_v)^2}} $$

**Prediction (Mean-Centered):**
$$ \hat{r}_{u,i} = \bar{r}_u + \frac{\sum_{v \in N} sim(u,v) \cdot (r_{v,i} - \bar{r}_v)}{\sum_{v \in N} |sim(u,v)|} $$

**Discounted Similarity (DS):**
$$ DS(u,v) = sim(u,v) \cdot \frac{\min(|I_{uv}|, \beta)}{\beta} $$
Where $\beta$ is a threshold parameter (set to 30% of target user's rated item count).

## 2. Results and Comparison

### Comparison of Neighbors (Step 7)
We compared the Top 20% neighbors identified by standard Pearson Similarity vs. Discounted Similarity (DS).
*   **Observation**: There is a significant divergence between the two lists. For example, for Target User 27768, out of thousands of neighbors, only ~500 were common to both lists, while >2800 were unique to each method.
*   **Interpretation**: The "Standard" Pearson list is dominated by users with high correlation (often 1.0 or -1.0) calculated on very few common items (e.g., 2 items). DS penalizes these low-overlap users, promoting neighbors with more substantial shared history, even if their correlation is slightly lower.

### Comparison of Predictions (Step 8)
*   **Observation**: The top-10 recommended items changed drastically. In some cases (e.g., User 16157), there was **0 overlap** between the top 10 items predicted by Pearson vs. DS.
*   **Interpretation**: Since the neighborhood changed from "lucky matches" (low overlap, high sim) to "reliable matches" (high overlap), the recommendations shifted to items liked by the more reliable group. This suggests DS is crucial for stabilizing recommendations.

## 3. Analysis & Discussion

### Reliability of -1.0 Correlation (Step 9)
We identified many users with a mean-centered similarity of **-1.0**.
*   **Constraint**: These users often had a **Raw Cosine Similarity** that was positive (e.g., ~0.02 - 0.08).
*   **Cause**: A correlation of -1.0 usually occurs when two users have very few common items (e.g., 2 items) and their ratings move in opposite directions (User A: Low, High; User B: High, Low).
*   **Reliability**: Neither the highly positive Raw Cosine (if it were 1.0 on 2 items) nor the -1.0 Pearson is reliable. Small sample sizes ($|I_{uv}| = 2$) yield volatile correlations. However, a -1.0 Pearson explicitly indicates a "disagreement" pattern, whereas Raw Cosine simply indicates "presence" of ratings. Raw Cosine is misleadingly positive here, while Pearson correctly flags the opposition, though the magnitude is exaggerated by the small sample size. **Conclusion**: The -1.0 users are likely noise. DS effectively filters them out by penalizing the low overlap.

### Fairness: Favorites Only vs. Full List (Step 10)
**Scenario**: User A rates only favorites (e.g., all 5s). User B rates a full list (1s to 5s).
*   **Pearson Behavior**: User A has a variance of 0. Their ratings are all equal to their mean. The Pearson correlation is undefined (or 0). User A is effectively **excluded** from being a neighbor to anyone.
*   **Raw Cosine Behavior**: User A (5, 5, 5) would have high Raw Cosine similarity with other users who gave 5s.
*   **Fairness**:
    *   It can be considered **unfair** in Pearson that "positivity-only" raters are ignored, as they do share preferences with others who like those items.
    *   However, it is **fair** in the sense that without variance, we cannot determine if User A's *relative* preference tracks with User B.
    *   **Distance**: They appear "unfairly far" (Sim=0) in Mean-Centered Cosine/Pearson compared to Raw Cosine. This is a known limitation of Pearson Correlation: it requires variance to measure similarity.

## 4. Conclusion
Mean-Centered Cosine (Pearson) offers better accuracy for prediction by removing user bias, but it suffers from volatility with low-overlap users and inability to handle zero-variance users. Extending it with **Discounted Similarity (DS)** is essential to filter out unreliable neighbors (those with high correlation but low support), leading to more robust recommendations.
