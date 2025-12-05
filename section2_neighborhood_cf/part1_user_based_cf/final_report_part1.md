# Part 1: User-Based CF - Final Comparison Report

## Overview
This report synthesizes the findings from three case studies exploring User-Based Collaborative Filtering using different similarity metrics:
1.  **Case Study 1**: Raw Cosine Similarity.
2.  **Case Study 2**: Mean-Centered Cosine Similarity (Pearson logic, centered prediction).
3.  **Case Study 3**: Pearson Correlation Coefficient (PCC) with specific focus on bias and correlation properties.

Across all studies, we also evaluated the impact of **Discounted Similarity (DS)**, which penalizes similarities based on low common item counts.

## Comparison of Similarity Metrics

| Metric | Handling User Bias | Reliability with Sparse Data | Key Characteristic |
| :--- | :--- | :--- | :--- |
| **Raw Cosine** | **Poor**. Treats users with ratings (1,1) and (5,5) as identical (Sim=1.0). Ignores strictness/generosity. | **Moderate**. Less likely to find negative correlations, but prone to high similarity on small subsets if ratings are non-negative. | Measures "Co-occurrence" and vector angle. Scale-independent but magnitude-ignorant in a way that hurts CF (treats Dislike as Like if both are low values?). Actually, Cosine treats 1 and 5 as positive vectors, so it struggles to distinguish "dislike" from "like" effectively without normalization. |
| **Mean-Centered / Pearson** | **Excellent**. Centers ratings around user mean. (1,1) becomes negative/zero deviation, (5,5) becomes positive. Identifies true preference alignment. | **Poor**. Extremely sensitive to low overlap ($n<3$). Can produce perfect $\pm 1.0$ correlations by chance, leading to noisy neighborhoods. | Measures "Correlation" of deviations. Invariant to shift and scale of rating distributions. |

### Impact of Bias Adjustment (Mean-Centering)
*   **Observation**: In Case Study 1 (Raw), users who rated everything '1' were found to be perfect neighbors of users who rated everything '5'. This is a fatal flaw for prediction, as predicting a '5' based on a neighbor who gave a '1' (but is considered "similar") leads to inaccurate absolute values.
*   **Correction**: Case Study 2 & 3 (Pearson) fixed this. A widespread low-rater (avg 1.0) and high-rater (avg 5.0) would likely have 0 variance or undefined correlation, or if they had variance, the centering ensures we predict *deviations* from the target's mean. Prediction formula $\hat{r} = \bar{r}_u + \Delta$ correctly scales the result to the target user's baseline.

## Impact of Discounted Similarity (DS)

Across all three case studies, **Discounted Similarity** proved to be the most critical improvement, regardless of the base metric.

*   **Problem**: "Top 20%" neighbors in sparse datasets are dominated by pairs with very few common items (e.g., 2 items) who happen to agree perfectly.
    *   **Raw Cosine**: Found many "1.0" neighbors with only 2 items.
    *   **Pearson**: Found many "1.0" (and "-1.0") neighbors with only 2 items.
*   **Solution**: DS applies a penalty factor $\frac{\min(|I_{uv}|, \beta)}{\beta}$.
*   **Outcome**:
    *   **Neighborhood Stability**: The "Top 20%" list shifted dramatically (often <10% overlap between Pure vs DS lists). The DS list favored users with sustained agreement (e.g., 50+ common items) even if correlation was slightly lower (e.g., 0.8 vs 1.0).
    *   **Prediction Quality**: Recommendations changed completely. DS predictions are based on "trusted" neighbors rather than "lucky" ones. This is expected to significantly improve offline accuracy (RMSE) and user trust.

## Key Reflections

1.  **Pearson vs Cosine**: Pearson is theoretically the correct metric for CF because it captures *preference* (up/down) rather than just *magnitude*. The analysis in Case Study 3 showed cases where Pearson was negative (disagreement) while Cosine was positive (co-occurrence), proving Pearson's superior semantic value.
2.  **The "Sample Size" Trap**: Pearson's theoretical superiority is completely undermined in sparse data without significance weighting (DS). A correlation of 1.0 based on 2 items is statistically meaningless and introduces high noise.
3.  **Conclusion**: The optimal approach for User-Based CF in this dataset is **Mean-Centered Cosine (Pearson) combined with Discounted Similarity (DS)**. This combination handles user bias (via centering) and sparsity noise (via discounting), providing the most robust foundation for recommendations.
