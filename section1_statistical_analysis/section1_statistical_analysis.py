import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')


DATASET_PATH= './dataset/Dianping_SocialRec_2015/rating.txt'
DATASET_SAVE_PATH= './dataset/Dianping_SocialRec_2015/'
RESULTS_DIR = './results'

# =============================================================================
print("\n" + "="*80)
print("SECTION 1: Statistical Analysis")
print("="*80)

# =============================================================================
# 1. Prepare a dataset of at least 100,000 users, >= 1000 products, and >= 1 million ratings.
# =============================================================================
print("\n--- Step 1: Loading Dataset ---")
column_names = ['user', 'item', 'rating', 'date']

# Read the rating.txt file into a DataFrame
df = pd.read_csv(DATASET_PATH, sep='|', names=column_names)
print(f"  [DONE] Dataset loaded successfully")

# =============================================================================
# 2. Preprocess the dataset to adjust the rating on a 1-to-5 scale.
# =============================================================================
print("\n--- Step 2: Preprocessing Dataset ---")
# Drop timestamp
df = df.drop(columns=['date'])
df = df[df['rating'] != 0]

df.to_csv(os.path.join(DATASET_SAVE_PATH, 'preprocessed_data.csv'), index=False)
print(f"  [SAVED] preprocessed_data.csv")

# Verify that all ratings are within the 1-5 scale
unique_ratings = df['rating'].unique()
min_rating = df['rating'].min()
max_rating = df['rating'].max()
valid_ratings = df['rating'].between(1, 5, inclusive='both')
num_valid = valid_ratings.sum()
num_invalid = (~valid_ratings).sum()

print(f"  {'Unique ratings found:':<40} {sorted(unique_ratings)}")
print(f"  {'Rating range:':<40} {min_rating} - {max_rating}")
print(f"  {'Total ratings:':<40} {len(df):>15,}")
print(f"  {'Valid ratings (1-5):':<40} {num_valid:>15,}")
print(f"  {'Invalid ratings:':<40} {num_invalid:>15,}")

# =============================================================================
# 3. Calculate the number of ratings for each user (n_u) and save it.
# =============================================================================
print("\n--- Step 3: Calculating Ratings per User (n_u) ---")
user_counts = df.groupby('user')['rating'].count().rename('n_u')
user_counts.to_csv(os.path.join(RESULTS_DIR, 'Sec1_n_u.csv'), header=True)
print(f"  [SAVED] Sec1_n_u.csv")

# =============================================================================
# 4. Calculate the number of ratings for each item (n_i) and save it.
# =============================================================================
print("\n--- Step 4: Calculating Ratings per Item (n_i) ---")
item_counts = df.groupby('item')['rating'].count().rename('n_i')
item_counts.to_csv(os.path.join(RESULTS_DIR, 'Sec1_n_i.csv'), header=True)
print(f"  [SAVED] Sec1_n_i.csv")

# =============================================================================
# 5. Compute the average ratings per user (r̄_u) in your dataset and save it.
# =============================================================================
print("\n--- Step 5: Calculating Average Rating per User (r̄_u) ---")
user_means = df.groupby('user')['rating'].mean().rename('r_u_bar')
user_means.to_csv(os.path.join(RESULTS_DIR, 'Sec1_r_u.csv'), header=True)
print(f"  [SAVED] Sec1_r_u.csv")

# =============================================================================
# 6. Compute the average ratings per item (r̄_i) in your dataset and save it.
# =============================================================================
print("\n--- Step 6: Calculating Average Rating per Item (r̄_i) ---")
item_means = df.groupby('item')['rating'].mean().rename('r_i_bar')
item_means.to_csv(os.path.join(RESULTS_DIR, 'Sec1_r_i.csv'), header=True)
print(f"  [SAVED] Sec1_r_i.csv")

# =============================================================================
# 7. Ascendingly order the total number of ratings per item and plot the distribution per item.
# =============================================================================
print("\n--- Step 7: Plotting Long-Tail Distribution ---")
sorted_item_counts = item_counts.sort_values(ascending=True)

plt.figure(figsize=(10, 6))
plt.plot(range(len(sorted_item_counts)), sorted_item_counts.values)
plt.xlabel('Items (sorted by popularity)')
plt.ylabel('Number of Ratings')
plt.title('Distribution of Ratings per Item (Ascending)')
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'Sec1_long_tail_plot.png'))
plt.close()
print(f"  [PLOT] Sec1_long_tail_plot.png")

# =============================================================================
# 8. Compute the number of products based on their average ratings such that:
#    G1 <= 1%, 1% < G2 <= 5%, 5% < G3 <= 10%, 10% < G4 <= 20%, 20% < G5 <= 30%,
#    30% < G6 <= 40%, 40% < G7 <= 50%, 50% < G8 <= 60%, 60% < G9 <= 70%, 70% < G10 <= 100%
# =============================================================================
print("\n--- Step 8: Grouping Products by Average Rating ---")
max_rating = 5
bin_percentages = [0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 1.00]
bins = [p * max_rating for p in bin_percentages]
labels = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10']

item_groups = pd.cut(item_means, bins=bins, labels=labels, include_lowest=True)
group_counts = item_groups.value_counts().sort_index()

print(f"  Products per group:")
for group, count in group_counts.items():
    print(f"    • {group}: {count:,}")
group_counts.to_csv(os.path.join(RESULTS_DIR, 'Sec1_group_counts.csv'), header=True)
print(f"  [SAVED] Sec1_group_counts.csv")

# =============================================================================
# 9. Compute the total number of ratings in each group and order them ascendingly.
# =============================================================================
print("\n--- Step 9: Total Ratings per Group ---")
item_data = pd.DataFrame({'group': item_groups, 'n_i': item_counts})
group_ratings = item_data.groupby('group')['n_i'].sum()
sorted_group_ratings = group_ratings.sort_values(ascending=True)

print(f"  Total ratings per group (sorted):")
for group, count in sorted_group_ratings.items():
    print(f"    • {group}: {count:,}")
sorted_group_ratings.to_csv(os.path.join(RESULTS_DIR, 'Sec1_group_ratings_sorted.csv'), header=True)
print(f"  [SAVED] Sec1_group_ratings_sorted.csv")

# =============================================================================
# 10. Plot the distribution of the number of ratings in each group before and after ordering.
# =============================================================================
print("\n--- Step 10: Plotting Ratings Distribution per Group ---")
plt.figure(figsize=(10, 6))
sorted_group_ratings.plot(kind='bar')
plt.xlabel('Groups (sorted by total ratings)')
plt.ylabel('Total Number of Ratings')
plt.title('Distribution of Ratings per Group (Ascending)')
plt.grid(axis='y')
plt.savefig(os.path.join(RESULTS_DIR, 'Sec1_ratings_per_group.png'))
plt.close()
print(f"  [PLOT] Sec1_ratings_per_group.png")

plt.figure(figsize=(10, 6))
group_ratings.plot(kind='bar')
plt.xlabel('Groups (unsorted)')
plt.ylabel('Total Number of Ratings')
plt.title('Distribution of Ratings per Group (unsorted)')
plt.grid(axis='y')
plt.savefig(os.path.join(RESULTS_DIR, 'Sec1_ratings_per_group_unsorted.png'))
plt.close()
print(f"  [PLOT] Sec1_ratings_per_group_unsorted.png")

# =============================================================================
# 11. Select three target users:
#     - U1 with <= 2% ratings
#     - U2 with ratings > 2% and <= 5%
#     - U3 with ratings > 5% and <= 10%
# =============================================================================
print("\n--- Step 11: Selecting Target Users ---")
n_items = df['item'].nunique()
t1 = 0.02 * n_items
t2 = 0.05 * n_items
t3 = 0.10 * n_items

print(f"  {'Total items:':<40} {n_items:>15,}")
print(f"  {'Threshold T1 (2%):':<40} {t1:>15.2f}")
print(f"  {'Threshold T2 (5%):':<40} {t2:>15.2f}")
print(f"  {'Threshold T3 (10%):':<40} {t3:>15.2f}")

u1_candidates = user_counts[user_counts <= t1]
u2_candidates = user_counts[(user_counts > t1) & (user_counts <= t2)]
u3_candidates = user_counts[(user_counts > t2) & (user_counts <= t3)]

np.random.seed(42)

def get_random_user(candidates, label):
    if not candidates.empty:
        user = candidates.sample(n=1, random_state=42).index[0]
        print(f"  Selected {label}: {user} (Ratings: {candidates[user]})")
        return user
    else:
        print(f"  No candidates for {label}")
        return None

u1 = get_random_user(u1_candidates, "U1")
u2 = get_random_user(u2_candidates, "U2")
u3 = get_random_user(u3_candidates, "U3")

# =============================================================================
# 12. Select two target items:
#     - Select the two lowest rated items (I1 and I2) as target items.
# =============================================================================
print("\n--- Step 12: Selecting Target Items ---")
n_users = df['user'].nunique()
it1 = 0.01 * n_users
it2 = 0.02 * n_users

print(f"  {'Total users:':<40} {n_users:>15,}")
print(f"  {'Item Threshold IT1 (1%):':<40} {it1:>15.2f}")
print(f"  {'Item Threshold IT2 (2%):':<40} {it2:>15.2f}")

target_item_candidates = item_counts[(item_counts > it1) & (item_counts <= it2)]

if not target_item_candidates.empty:
    if len(target_item_candidates) >= 2:
        selected_items = target_item_candidates.sample(n=2, random_state=42)
        i1 = selected_items.index[0]
        i2 = selected_items.index[1]
        print(f"  Selected I1: {i1} (Ratings: {selected_items[i1]})")
        print(f"  Selected I2: {i2} (Ratings: {selected_items[i2]})")
    else:
        i1 = target_item_candidates.index[0]
        i2 = None
        print(f"  Selected I1: {i1}")
        print(f"  Not enough items for I2")
else:
    print("  No items found in the 1-2% popularity range.")
    i1 = None
    i2 = None

# =============================================================================
# 15. Save all intermediate results for use in later parts.
# =============================================================================
print("\n--- Step 15: Saving Target Users and Items ---")
if not os.path.exists(os.path.join(RESULTS_DIR, 'Sec1_target_users.txt')):
    with open(os.path.join(RESULTS_DIR, 'Sec1_target_users.txt'), 'w') as f:
        f.write(f"{u1}\n")
        f.write(f"{u2}\n")
        f.write(f"{u3}\n")
    print(f"  [SAVED] Sec1_target_users.txt")
else:
    print(f"  [EXISTS] Sec1_target_users.txt (skipped)")

if not os.path.exists(os.path.join(RESULTS_DIR, 'Sec1_target_items.txt')):
    with open(os.path.join(RESULTS_DIR, 'Sec1_target_items.txt'), 'w') as f:
        f.write(f"{i1}\n")
        f.write(f"{i2}\n")
    print(f"  [SAVED] Sec1_target_items.txt")
else:
    print(f"  [EXISTS] Sec1_target_items.txt (skipped)")

# =============================================================================
# 13. Count the number of co-rating users between each target user and other users
#     (No_common_users), and the number of co-rated items between each target item
#     and other items (No_coRated_items).
# 14. Determine the threshold β: maximum number of users who have co-rated at least
#     30% of items with each target user.
# =============================================================================
print("\n--- Steps 13-14: Co-Rating Analysis ---")

user_items = df.groupby('user')['item'].apply(set).to_dict()
item_users = df.groupby('item')['user'].apply(set).to_dict()

target_users = [u for u in [u1, u2, u3] if u is not None]
target_items_list = [i for i in [i1, i2] if i is not None]

results_13_14 = []

print(f"\n  Target Users Analysis:")
for u_target in target_users:
    target_items_set = user_items.get(u_target, set())
    n_target_ratings = len(target_items_set)
    
    no_common_users = 0
    beta_count = 0
    threshold_30 = 0.30 * n_target_ratings
    
    for u_other, other_items_set in user_items.items():
        if u_other == u_target:
            continue
        intersection_size = len(target_items_set.intersection(other_items_set))
        if intersection_size > 0:
            no_common_users += 1
        if intersection_size >= threshold_30:
            beta_count += 1
            
    print(f"    • User {u_target}: Ratings={n_target_ratings}, No_common_users={no_common_users:,}, Beta={beta_count}")
    results_13_14.append({'Type': 'User', 'ID': u_target, 'Count': no_common_users, 'Beta': beta_count})

print(f"\n  Target Items Analysis:")
for i_target in target_items_list:
    target_users_set = item_users.get(i_target, set())
    
    no_corated_items = 0
    
    for i_other, other_users_set in item_users.items():
        if i_other == i_target:
            continue
        intersection_size = len(target_users_set.intersection(other_users_set))
        if intersection_size > 0:
            no_corated_items += 1
            
    print(f"    • Item {i_target}: No_coRated_items={no_corated_items:,}")
    results_13_14.append({'Type': 'Item', 'ID': i_target, 'Count': no_corated_items, 'Beta': 'N/A'})

pd.DataFrame(results_13_14).to_csv(os.path.join(RESULTS_DIR, 'Sec1_results_13_14.csv'), index=False)
print(f"\n  [SAVED] Sec1_results_13_14.csv")

# =============================================================================
# 16. Compare the results from point 13 & 14 and give your insights into the dataset
#     by evaluating and discussing the matrix sparsity, rating bias and long-tail problems.
# =============================================================================
print("\n" + "="*80)
print("Step 16: Dataset Analysis and Insights")
print("="*80)

n_users = df['user'].nunique()
n_items = df['item'].nunique()
n_ratings = len(df)
max_possible_ratings = n_users * n_items

# --- 1. Matrix Sparsity Analysis ---
sparsity = 1 - (n_ratings / max_possible_ratings)
density = n_ratings / max_possible_ratings

print("\n--- 16.1 Matrix Sparsity Analysis ---")
print(f"  {'Total Users:':<40} {n_users:>15,}")
print(f"  {'Total Items:':<40} {n_items:>15,}")
print(f"  {'Total Ratings:':<40} {n_ratings:>15,}")
print(f"  {'Max Possible Ratings:':<40} {max_possible_ratings:>15,}")
print(f"  {'Matrix Density:':<40} {density*100:>14.4f}%")
print(f"  {'Matrix Sparsity:':<40} {sparsity*100:>14.4f}%")

# --- 2. Rating Bias Analysis ---
print("\n--- 16.2 Rating Bias Analysis ---")
overall_mean = df['rating'].mean()
rating_distribution = df['rating'].value_counts().sort_index()
rating_percentages = (rating_distribution / n_ratings * 100).round(2)

print(f"  {'Overall Average Rating:':<40} {overall_mean:>15.2f}")
print(f"\n  Rating Distribution:")
for rating, count in rating_distribution.items():
    print(f"    • Rating {rating}: {count:>12,} ({rating_percentages[rating]:>6.2f}%)")

high_ratings = df[df['rating'] >= 4]['rating'].count()
low_ratings = df[df['rating'] <= 2]['rating'].count()
high_rating_percentage = high_ratings / n_ratings * 100
low_rating_percentage = low_ratings / n_ratings * 100

print(f"\n  {'High Ratings (4-5):':<40} {high_ratings:>12,} ({high_rating_percentage:>6.2f}%)")
print(f"  {'Low Ratings (1-2):':<40} {low_ratings:>12,} ({low_rating_percentage:>6.2f}%)")

# --- 3. Long-Tail Problem Analysis ---
print("\n--- 16.3 Long-Tail Problem Analysis ---")

top_10_percent_items = int(0.1 * n_items)
top_20_percent_items = int(0.2 * n_items)
sorted_items = item_counts.sort_values(ascending=False)

ratings_by_top_10 = sorted_items.head(top_10_percent_items).sum()
ratings_by_top_20 = sorted_items.head(top_20_percent_items).sum()
ratings_by_top_10_pct = ratings_by_top_10 / n_ratings * 100
ratings_by_top_20_pct = ratings_by_top_20 / n_ratings * 100

items_with_less_than_10_ratings = (item_counts < 10).sum()
items_with_less_than_5_ratings = (item_counts < 5).sum()
tail_item_percentage = items_with_less_than_10_ratings / n_items * 100

print(f"  {'Top 10% items ratings share:':<40} {ratings_by_top_10_pct:>14.2f}%")
print(f"  {'Top 20% items ratings share:':<40} {ratings_by_top_20_pct:>14.2f}%")
print(f"  {'Items with < 10 ratings:':<40} {items_with_less_than_10_ratings:>12,} ({tail_item_percentage:>6.2f}%)")
print(f"  {'Items with < 5 ratings:':<40} {items_with_less_than_5_ratings:>12,} ({items_with_less_than_5_ratings/n_items*100:>6.2f}%)")

top_10_percent_users = int(0.1 * n_users)
sorted_users = user_counts.sort_values(ascending=False)
ratings_by_top_10_users = sorted_users.head(top_10_percent_users).sum()
ratings_by_top_10_users_pct = ratings_by_top_10_users / n_ratings * 100

print(f"  {'Top 10% users ratings share:':<40} {ratings_by_top_10_users_pct:>14.2f}%")

# --- 4. Co-rating Analysis Insights ---
print("\n--- 16.4 Co-Rating Analysis Insights ---")

for result in results_13_14:
    if result['Type'] == 'User':
        user_id = result['ID']
        no_common = result['Count']
        beta = result['Beta']
        common_pct = no_common / (n_users - 1) * 100
        print(f"\n  User {user_id}:")
        print(f"    • Co-ratings with: {no_common:,} users ({common_pct:.2f}%)")
        print(f"    • Beta (>=30% overlap): {beta}")
    else:
        item_id = result['ID']
        no_corated = result['Count']
        corated_pct = no_corated / (n_items - 1) * 100
        print(f"\n  Item {item_id}:")
        print(f"    • Co-rated with: {no_corated:,} items ({corated_pct:.2f}%)")

print("\n" + "="*80)
print("[DONE] Section 1: Statistical Analysis completed successfully.")
print("="*80)
