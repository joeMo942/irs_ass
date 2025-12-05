# Case Study 3: Pearson Correlation Coefficient (PCC) Report

## 1. Methodology
This case study uses **Pearson Correlation Coefficient (PCC)** to determine user similarity. PCC measures the linear correlation between two users' ratings, effectively handling differences in rating scales (mean-centering).

**Prediction Logic**:
$$ \hat{r}_{u,i} = \bar{r}_u + \frac{\sum_{v \in N} sim(u,v) \cdot (r_{v,i} - \bar{r}_v)}{\sum_{v \in N} |sim(u,v)|} $$
We filter neighbors to use only those with **positive correlation** ($sim > 0$) for prediction, as is standard practice to align predictions with supporting evidence.

**Discounted Similarity (DS)**:
$$ DS = PCC \cdot \frac{\min(|I_{uv}|, \beta)}{\beta} $$
We confirmed that it is **correct** to apply DS to Pearson. Pearson is notoriously unreliable when $n$ (common items) is small, producing extreme values (-1.0 or 1.0) by chance. DS penalizes these low-confidence correlations.

## 2. Results Comparison (Steps 7 & 8)

### Similarity Lists
Comparing the top 20% neighbors from pure PCC vs. DS (PCC based):
*   **Target User 27768**: 972 common neighbors out of ~6700. This low overlap indicates that pure PCC picks many "noisy" neighbors (low $n$, high correlation), while DS favors those with more shared history.
*   **Target User 16157**: 2029 common neighbors out of ~8200.

### Prediction Results
*   **Target User 27768**: Only **1** item in common between the top-10 lists.
*   **Target User 16157**: **0** items in common.
*   **Conclusion**: DS significantly alters the recommendations. Since pure PCC is susceptible to noise (neighbors with 1.0 correlation on 2 items), the resulting predictions can be skewed by these "lucky" matches. DS provides more stable and reliable recommendations by trusting users with sustained agreement.

## 3. Detailed Analysis

### Negative Pearson vs. Positive Cosine (Step 9)
We observed users with **Negative Pearson** (e.g., -1.0 or -0.5) but **Positive Cosine** (e.g., ~0.12).
*   **Example**: User 134471 vs User 4561 (Pearson -1.0, Cosine 0.127).
*   **Why?**:
    *   **Cosine**: Ratings are non-negative (1-5), so the dot product is always positive. Cosine only checks angle in the first quadrant.
    *   **Pearson**: Checks for correlation relative to the mean. If User A rates Item 1 above their mean and User B rates it below their mean, they disagree.
    *   **Scenario**: Common Items {A, B}. User 1: {A: 2, B: 4} (Mean 3). User 2: {A: 4, B: 3} (Mean 3.5).
        *   User 1 goes Up (+1), User 2 goes Down (-0.5). Direction is opposite $\rightarrow$ Negative Pearson.
        *   Magnitudes are positive $\rightarrow$ Positive Cosine.
    *   **Implication**: Pearson is more informative about *preference alignment* than Raw Cosine, which just measures "co-occurrence" of ratings.

### Small Sample Size (Step 10)
**Does Pearson give meaningful output when users rate $\le 20\%$ of items?**
*   **No**, not necessarily. "$\le 20\%$ of items" might still be a large number (e.g., 20% of 1000 items = 200), which is fine.
*   However, if they rate **very few common items** (e.g. 2 or 3 items), Pearson is **unstable**.
*   We saw many correlations of $\pm 1.0$ based on just 2 common items. This is statistically meaningless.
*   **Solution**: Use Discounted Similarity (DS) or Significance Weighting (filtering $n < threshold$) to mitigate this.

### Different Rating Scales (Step 11)
**Generous vs Strict Users**:
*   **Finding**: Found Target User 27768 (Mean 3.26) and Neighbor 139483 (Mean 1.50) with **Similarity = 1.0**.
*   **Discussion**: This is the core strength of Pearson. It is **invariant to location and scale**.
    *   If User A rates {3, 4, 5} and User B rates {1, 2, 3}, they have perfect correlation (1.0).
    *   Pearson correctly identifies that they agree on the *relative* quality of items, despite the absolute difference in strictness.
    *   Raw Cosine would penalize this pair due to the magnitude difference.

### Opposite Patterns on Small Data (Step 12)
**Trustworthiness?**
*   We found cases where Pearson detected opposite patterns (Neg Sim) compared to Positive Cosine on small samples ($n < 5$).
*   **Do we trust it?**: **No**. With $n < 5$, the "pattern" is likely noise. A correlation of -1.0 on 2 items just means "lines crossed". It doesn't predict future disagreement reliably.
*   **Recommendation**: Do not trust Pearson (positive or negative) without a minimum support threshold (e.g., $n \ge 10$ or $n \ge \beta$).

## 4. Final Comments
Pearson CF is theoretically superior to Raw Cosine for handling user bias (strictness/generosity). However, it introduces a reliance on variance and overlap size. Pure Pearson is dangerous in sparse datasets due to high variance on low-overlap pairs. The application of Discounted Similarity (DS) is essential to harness the benefits of Pearson (bias removal) while mitigating its weakness (sensitivity to sparsity).
