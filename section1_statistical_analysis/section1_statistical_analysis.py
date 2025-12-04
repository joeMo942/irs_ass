import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os



DATASET_PATH= './dataset/Dianping_SocialRec_2015/rating.txt'
DATASET_SAVE_PATH= './dataset/Dianping_SocialRec_2015/'
RESULTS_DIR = './results'


column_names = ['user', 'item', 'rating', 'date']

# Read the rating.txt file into a DataFrame
df = pd.read_csv(DATASET_PATH, sep='|', names=column_names)
df.head()


print("Preprocessing...")
# Drop timestamp
df = df.drop(columns=['date'])
df = df[df['rating'] != 0]

df.to_csv(os.path.join(DATASET_SAVE_PATH, 'preprocessed_data.csv'), index=False)


# 3. Calculate number of ratings for each user (n_u)
user_counts = df.groupby('user')['rating'].count().rename('n_u')
user_counts.to_csv(os.path.join(RESULTS_DIR, 'n_u.csv'), header=True)

# 4. Calculate number of ratings for each item (n_i)
item_counts = df.groupby('item')['rating'].count().rename('n_i')
item_counts.to_csv(os.path.join(RESULTS_DIR, 'n_i.csv'), header=True)

# 5. Compute average ratings per user (r_u_bar)
user_means = df.groupby('user')['rating'].mean().rename('r_u_bar')
user_means.to_csv(os.path.join(RESULTS_DIR, 'r_u.csv'), header=True)

# 6. Compute average ratings per item (r_i_bar)
item_means = df.groupby('item')['rating'].mean().rename('r_i_bar')
item_means.to_csv(os.path.join(RESULTS_DIR, 'r_i.csv'), header=True)

# 7. Ascendingly order the total number of ratings per item and plot the distribution per item.
sorted_item_counts = item_counts.sort_values(ascending=True)
# Correct (Ascending)

plt.figure(figsize=(10, 6))
# Reset index to get a range for x-axis (0 to n_items) representing the items
plt.plot(range(len(sorted_item_counts)), sorted_item_counts.values)
plt.xlabel('Items (sorted by popularity)')
plt.ylabel('Number of Ratings')
plt.title('Distribution of Ratings per Item (Ascending)')
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'long_tail_plot.png'))
plt.close()
print("Plot saved to results/long_tail_plot.png")

# 8. Compute the number of products based on their average ratings
# Groups: G1 <= 1%, 1% < G2 <= 5%, ..., 70% < G10 <= 100%
# Max rating is 5
max_rating = 5
# Define bin edges as percentages of max_rating
bin_percentages = [0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 1.00]
bins = [p * max_rating for p in bin_percentages]
labels = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10']

# Use pd.cut to bin the item means
# include_lowest=True ensures that the first bin includes the left edge (0)
item_groups = pd.cut(item_means, bins=bins, labels=labels, include_lowest=True)

# Count number of products in each group
group_counts = item_groups.value_counts().sort_index()

print("\nNumber of products per group:")
print(group_counts)
group_counts.to_csv(os.path.join(RESULTS_DIR, 'group_counts.csv'), header=True)

# 9. Compute the total number of ratings in each group and order them ascendingly
# item_counts has the number of ratings per item, item_groups has the group assignment
item_data = pd.DataFrame({'group': item_groups, 'n_i': item_counts})
# Sum n_i for each group
group_ratings = item_data.groupby('group')['n_i'].sum()

# Order them ascendingly (by total ratings)
sorted_group_ratings = group_ratings.sort_values(ascending=True)

print("\nTotal ratings per group (sorted):")
print(sorted_group_ratings)
sorted_group_ratings.to_csv(os.path.join(RESULTS_DIR, 'group_ratings_sorted.csv'), header=True)

# Plot distribution
plt.figure(figsize=(10, 6))
sorted_group_ratings.plot(kind='bar')
plt.xlabel('Groups (sorted by total ratings)')
plt.ylabel('Total Number of Ratings')
plt.title('Distribution of Ratings per Group (Ascending)')
plt.grid(axis='y')
plt.savefig(os.path.join(RESULTS_DIR, 'ratings_per_group.png'))
plt.close()
print("Plot saved to results/ratings_per_group.png")

# 11. Select three target users
# U1 with <= 2% ratings
# U2 with ratings > 2% and <= 5%
# U3 with ratings > 5% and <= 10%
# Percentages are relative to the total number of items
n_items = df['item'].nunique()
t1 = 0.02 * n_items
t2 = 0.05 * n_items
t3 = 0.10 * n_items

print(f"\nTotal items: {n_items}")
print(f"Thresholds: T1={t1:.2f}, T2={t2:.2f}, T3={t3:.2f}")

# Filter users
u1_candidates = user_counts[user_counts <= t1]
u2_candidates = user_counts[(user_counts > t1) & (user_counts <= t2)]
u3_candidates = user_counts[(user_counts > t2) & (user_counts <= t3)]

# Select one random user from each group if available
np.random.seed(42) # For reproducibility

def get_random_user(candidates, label):
    if not candidates.empty:
        user = candidates.sample(n=1, random_state=42).index[0]
        print(f"Selected {label}: {user} (Ratings: {candidates[user]})")
        return user
    else:
        print(f"No candidates for {label}")
        return None

u1 = get_random_user(u1_candidates, "U1")
u2 = get_random_user(u2_candidates, "U2")
u3 = get_random_user(u3_candidates, "U3")

# 12. Select two target items: items with ratings between 1% and 2% of users
n_users = df['user'].nunique()
it1 = 0.01 * n_users
it2 = 0.02 * n_users

print(f"\nTotal users: {n_users}")
print(f"Item Thresholds: IT1={it1:.2f}, IT2={it2:.2f}")

# Filter items based on popularity (n_i)
# item_counts contains n_i for each item
target_item_candidates = item_counts[(item_counts > it1) & (item_counts <= it2)]

if not target_item_candidates.empty:
    # Select two random items
    if len(target_item_candidates) >= 2:
        selected_items = target_item_candidates.sample(n=2, random_state=42)
        i1 = selected_items.index[0]
        i2 = selected_items.index[1]
        print(f"\nSelected Target Items (Popularity 1-2% of users):")
        print(f"I1: {i1} (Ratings: {selected_items[i1]})")
        print(f"I2: {i2} (Ratings: {selected_items[i2]})")
    else:
        print("Not enough items in the 1-2% popularity range to select 2.")
        # Fallback or handle appropriately - for now just take what we have or None
        i1 = target_item_candidates.index[0]
        i2 = None
        print(f"I1: {i1}")
else:
    print("No items found in the 1-2% popularity range.")
    i1 = None
    i2 = None

# Save selected targets to a file for reference
with open(os.path.join(RESULTS_DIR, 'target_users.txt'), 'w') as f:
    f.write(f"{u1}\n")
    f.write(f"{u2}\n")
    f.write(f"{u3}\n")

with open(os.path.join(RESULTS_DIR, 'target_items.txt'), 'w') as f:
    f.write(f"{i1}\n")
    f.write(f"{i2}\n")

# 13. Count the number of co-rating users and co-rated items
# 14. Determine the threshold beta

print("\nStarting Steps 13 & 14...")

# Precompute user_items and item_users maps for efficiency
# Group by user and collect items into a set
user_items = df.groupby('user')['item'].apply(set).to_dict()
# Group by item and collect users into a set
item_users = df.groupby('item')['user'].apply(set).to_dict()

target_users = [u for u in [u1, u2, u3] if u is not None]
target_items_list = [i for i in [i1, i2] if i is not None]

results_13_14 = []

print("\n--- Target Users Analysis ---")
for u_target in target_users:
    target_items_set = user_items.get(u_target, set())
    n_target_ratings = len(target_items_set)
    
    no_common_users = 0
    beta_count = 0
    threshold_30 = 0.30 * n_target_ratings
    
    # Iterate over all other users
    for u_other, other_items_set in user_items.items():
        if u_other == u_target:
            continue
            
        # Intersection size
        intersection_size = len(target_items_set.intersection(other_items_set))
        
        if intersection_size > 0:
            no_common_users += 1
            
        # Step 14 check
        if intersection_size >= threshold_30:
            beta_count += 1
            
    print(f"User {u_target}: Ratings={n_target_ratings}, No_common_users={no_common_users}, Beta (>=30% overlap)={beta_count}")
    results_13_14.append({'Type': 'User', 'ID': u_target, 'Count': no_common_users, 'Beta': beta_count})

print("\n--- Target Items Analysis ---")
for i_target in target_items_list:
    target_users_set = item_users.get(i_target, set())
    
    no_corated_items = 0
    
    # Iterate over all other items
    for i_other, other_users_set in item_users.items():
        if i_other == i_target:
            continue
            
        intersection_size = len(target_users_set.intersection(other_users_set))
        
        if intersection_size > 0:
            no_corated_items += 1
            
    print(f"Item {i_target}: No_coRated_items={no_corated_items}")
    results_13_14.append({'Type': 'Item', 'ID': i_target, 'Count': no_corated_items, 'Beta': 'N/A'})

# Save results
pd.DataFrame(results_13_14).to_csv(os.path.join(RESULTS_DIR, 'results_13_14.csv'), index=False)






