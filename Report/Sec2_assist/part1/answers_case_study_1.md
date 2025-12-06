# Analysis of User-Based CF Results (Case Study 1)

## Question 7: Compare the lists of top users from steps 2 and 5. Discuss any differences or patterns observed.

**comparison:**
Comparing the neighbors results from the output:
- **For Target User 134471:** The top 5 neighbors were completely replaced. The raw cosine neighbors (Step 2) all had identical high scores (~0.334), likely due to sparse overlap (e.g., matching on a single item). In Step 5 (DS), these were replaced by users with lower raw scores but presumably higher overlap, resulting in DS scores around 0.16–0.20.
- **For Target User 16157:** Similarly, none of the top 5 raw neighbors appeared in the top 5 DS neighbors. The similarity scores dropped significantly (from ~0.19 to ~0.12), indicating the raw neighbors were heavily penalized for low support.
- **For Target User 27768:** There was more stability, but reordering occurred. For example, neighbor `16611` dropped from rank 2 to 4, and neighbor `41956` dropped out of the top 5 entirely.

**Discussion of Patterns:**
1.  **Penalizarion of Sparse Neighbors:** The primary pattern is that Step 2 (Raw Cosine) favors users who have very few rated items but happen to match the target user perfectly on those few items. These "accidental" high similarities are often not robust.
2.  **Emergence of Trustworthy Neighbors:** Step 5 (Discounted Similarity) introduces the `beta` threshold. Neighbors with fewer than `beta` co-rated items are penalized. This causes the list to shift towards users who might have a slightly lower raw cosine score (e.g., 0.8 instead of 1.0) but have rated many more items in common with the target.
3.  **Reliability Shift:** The shift from Step 2 to Step 5 represents a move from "highest mathematical similarity" to "highest reliable similarity."

## Question 8: Compare the rating predictions from steps 3 and 6. Provide comments on the impact of using DS.

**Comparison:**
- **Divergence in Recommendations:** The sets of top predicted items are vastly different.
    - User 134471: Raw top 10 `[1216, 1408...]` vs DS top 10 `[3247, 4971...]`.
    - User 16157: Raw top 10 `[2574, 6496...]` vs DS top 10 `[11085, 1555...]`.
- **Score Stability:** In some cases, items predicted in Step 3 might disappear completely from the top list in Step 6 because the neighbors who rated those items high were deemed "unreliable" and removed from the neighborhood.

**Impact of Using Discounted Similarity (DS):**
1.  **Noise Reduction:** DS filters out recommendations that are based on "flukes." If a neighbor matches the target user on just 1 item and then recommends a random other item, Step 3 would take that recommendation seriously. Step 6 (DS) likely discards it because that neighbor's weight is heavily discounted.
2.  **Increased Confidence:** The items predicted in Step 6 come from a neighborhood of users who have a substantial history of agreement (overlap) with the target user. This makes the predicted ratings more statistically significant and trustworthy.
3.  **Correction of Bias:** Raw cosine on sparse data often results in trivial similarities (e.g., 1.0 similarity based on 1 item). DS corrects this bias, ensuring that high influence is reserved for users who have demonstrated consistent similarity over a larger set of items (approx 30% of the target's profile size).

## Question 10: Find users who rated the common items with any of the target users

We analyzed the neighbors of the target users to distinguish between "Subset Neighbors" (users who have rated *only* the specific items that overlap with the target user) and "Superset Neighbors" (users who have rated the common items *plus* many others).

**Findings for User 134471 (Small Profile):**
- **Subset Neighbors found:** Yes.
- **Example:** User `131078` (and others like `131084`).
- **Data:** This neighbor rated exactly **1 item** total, and that 1 item is also rated by the target user (Overlap = 1).
- **Trust Analysis:** This neighbor is a "Subset Neighbor." Their high similarity (Raw Cosine) is based on the absolute minimum evidence possible. They provide no additional information or breadth of taste outside the target's own narrow scope, yet they would strongly influence standard CF recommendations. This highlights the risk of relying on raw similarity for sparse users.

**Findings for User 27768 (Medium Profile):**
- **Subset Neighbors found:** No (among top neighbors).
- **Example:** User `8629`.
- **Data:** This neighbor rated **480 items**, with **82** in common with the target.
- **Trust Analysis:** This is a "Superset Neighbor." They have a vast rating history outside the overlap region (398 additional items). This user is highly trustworthy because the similarity is established over a large sample (82 items), and their recommendations come from a rich verified history.

**Findings for User 16157 (Large Profile):**
- **Subset Neighbors found:** No (among top neighbors).
- **Example:** User `5285`.
- **Data:** This neighbor rated **91 items**, with **47** in common with the target.
- **Trust Analysis:** Also a "Superset Neighbor." Even though they have fewer total ratings than the target user ( who has 626), they are not a "Subset" neighbor because they have rated items that the target user has *not* rated (44 additional items), which allows them to make valid recommendations.
