# Youssef Zakaria Soubhi Abo Srewa
# 221101030
# noureldeen maher Mesbah
# 221101140
# Youssef Mohamed
# 221101573

import warnings
warnings.filterwarnings("ignore")
import math

def calculate_user_raw_cosine(user1_ratings, user2_ratings):
    """
    Calculates raw cosine similarity between two users.
    
    Args:
        user1_ratings (dict): Dictionary of {item_id: rating} for user 1.
        user2_ratings (dict): Dictionary of {item_id: rating} for user 2.
        
    Returns:
        float: Cosine similarity. Returns 0.0 if no common items or zero magnitude.
    """
    common_items = set(user1_ratings.keys()) & set(user2_ratings.keys())
    
    if not common_items:
        return 0.0
    
    numerator = sum(user1_ratings[item] * user2_ratings[item] for item in common_items)
    
    sum_sq_u1 = sum(r**2 for r in user1_ratings.values())
    sum_sq_u2 = sum(r**2 for r in user2_ratings.values())
    
    denominator = (sum_sq_u1)**0.5 * (sum_sq_u2)**0.5
    
    if denominator == 0:
        return 0.0
        
    return numerator / denominator

def calculate_user_mean_centered_cosine(user1_ratings, user2_ratings, user1_mean, user2_mean):
    """
    Calculates mean-centered cosine similarity (Pearson correlation using global means) between two users.
    
    Args:
        user1_ratings (dict): Dictionary of {item_id: rating} for user 1.
        user2_ratings (dict): Dictionary of {item_id: rating} for user 2.
        user1_mean (float): Average rating of user 1.
        user2_mean (float): Average rating of user 2.
        
    Returns:
        float: Similarity score. Returns 0.0 if no common items or zero magnitude.
    """
    common_items = set(user1_ratings.keys()) & set(user2_ratings.keys())
    
    if not common_items:
        return 0.0
        
    numerator = sum((user1_ratings[item] - user1_mean) * (user2_ratings[item] - user2_mean) for item in common_items)
    
    # Denominator sums over ALL rated items for each user (standard definition of cosine on centered vectors)
    # OR sums over common items (Pearson definition)?
    # Standard Mean-Centered Cosine usually implies centering the full vectors and then taking cosine.
    sum_sq_u1 = sum((r - user1_mean)**2 for r in user1_ratings.values())
    sum_sq_u2 = sum((r - user2_mean)**2 for r in user2_ratings.values())
    
    denominator = (sum_sq_u1)**0.5 * (sum_sq_u2)**0.5
    
    if denominator == 0:
        return 0.0
        
    return numerator / denominator

def calculate_user_pearson(user1_ratings, user2_ratings):
    """
    Calculates Pearson correlation coefficient between two users based on common items.
    
    Args:
        user1_ratings (dict): Dictionary of {item_id: rating} for user 1.
        user2_ratings (dict): Dictionary of {item_id: rating} for user 2.
        
    Returns:
        float: Pearson correlation. Returns 0.0 if insufficient common items or zero variance.
    """
    common_items = set(user1_ratings.keys()) & set(user2_ratings.keys())
    n = len(common_items)
    
    if n < 2:
        return 0.0
        
    # Calculate means based on common items only
    u1_common_ratings = [user1_ratings[item] for item in common_items]
    u2_common_ratings = [user2_ratings[item] for item in common_items]
    
    mean_u1 = sum(u1_common_ratings) / n
    mean_u2 = sum(u2_common_ratings) / n
    
    numerator = sum((r1 - mean_u1) * (r2 - mean_u2) for r1, r2 in zip(u1_common_ratings, u2_common_ratings))
    
    sum_sq_u1 = sum((r - mean_u1)**2 for r in u1_common_ratings)
    sum_sq_u2 = sum((r - mean_u2)**2 for r in u2_common_ratings)
    
    denominator = (sum_sq_u1)**0.5 * (sum_sq_u2)**0.5
    
    if denominator == 0:
        return 0.0
        
    return numerator / denominator

def calculate_item_mean_centered_cosine(item1_ratings, item2_ratings, user_means):
    """
    Calculates Adjusted Cosine Similarity between two items.
    Subtracts the USER's average rating from each rating.
    
    Args:
        item1_ratings (dict): Dictionary of {user_id: rating} for item 1.
        item2_ratings (dict): Dictionary of {user_id: rating} for item 2.
        user_means (dict): Dictionary of {user_id: mean_rating} for all users.
        
    Returns:
        float: Similarity score.
    """
    common_users = set(item1_ratings.keys()) & set(item2_ratings.keys())
    
    if not common_users:
        return 0.0
        
    numerator = 0.0
    sum_sq_i1 = 0.0
    sum_sq_i2 = 0.0
    
    for user in common_users:
        if user not in user_means:
            continue
            
        user_mean = user_means[user]
        r1_adj = item1_ratings[user] - user_mean
        r2_adj = item2_ratings[user] - user_mean
        
        numerator += r1_adj * r2_adj
        sum_sq_i1 += r1_adj**2
        sum_sq_i2 += r2_adj**2
        
    denominator = (sum_sq_i1)**0.5 * (sum_sq_i2)**0.5
    
    if denominator == 0:
        return 0.0
        
    return numerator / denominator

def calculate_item_pearson(item1_ratings, item2_ratings):
    """
    Calculates Pearson correlation between two items based on common users.
    Subtracts the ITEM's average rating (computed over common users) from each rating.
    
    Args:
        item1_ratings (dict): Dictionary of {user_id: rating} for item 1.
        item2_ratings (dict): Dictionary of {user_id: rating} for item 2.
        
    Returns:
        float: Pearson correlation.
    """
    common_users = set(item1_ratings.keys()) & set(item2_ratings.keys())
    n = len(common_users)
    
    if n < 2:
        return 0.0
        
    # Calculate means based on common users only
    i1_common_ratings = [item1_ratings[user] for user in common_users]
    i2_common_ratings = [item2_ratings[user] for user in common_users]
    
    mean_i1 = sum(i1_common_ratings) / n
    mean_i2 = sum(i2_common_ratings) / n
    
    numerator = sum((r1 - mean_i1) * (r2 - mean_i2) for r1, r2 in zip(i1_common_ratings, i2_common_ratings))
    
    sum_sq_i1 = sum((r - mean_i1)**2 for r in i1_common_ratings)
    sum_sq_i2 = sum((r - mean_i2)**2 for r in i2_common_ratings)
    
    denominator = (sum_sq_i1)**0.5 * (sum_sq_i2)**0.5
    
    if denominator == 0:
        return 0.0
        
    return numerator / denominator

def calculate_similarity_for_target_user(target_user_ratings, all_users_ratings, similarity_func, user_means=None, target_user_id=None):
    """
    Calculates similarity between a target user and all other users.
    
    Args:
        target_user_ratings (dict): Dictionary of {item_id: rating} for the target user.
        all_users_ratings (dict): Dictionary of {user_id: {item_id: rating}} for all users.
        similarity_func (function): A similarity function from this module (e.g., calculate_user_raw_cosine).
        user_means (dict, optional): Dictionary of {user_id: mean_rating}. Required for mean-centered cosine.
        target_user_id (int, optional): ID of the target user to avoid self-comparison.
        
    Returns:
        list: A list of tuples (user_id, similarity_score) sorted by similarity in descending order.
    """
    similarities = []
    
    target_mean = None
    if user_means and target_user_id is not None:
        target_mean = user_means.get(target_user_id)
        
    for user_id, other_user_ratings in all_users_ratings.items():
        if target_user_id is not None and user_id == target_user_id:
            continue
            
        try:
            # Check if the function requires means (Adjusted Cosine / Mean Centered)
            if similarity_func.__name__ in ['calculate_user_mean_centered_cosine', 'calculate_item_mean_centered_cosine']:
                 if target_mean is not None and user_means and user_id in user_means:
                     score = similarity_func(target_user_ratings, other_user_ratings, target_mean, user_means[user_id])
                 elif similarity_func.__name__ == 'calculate_item_mean_centered_cosine' and user_means:
                     # Item-based adjusted cosine takes (item1, item2, user_means)
                     # But here we are iterating users? 
                     # Wait, calculate_similarity_for_target_user is designed for USERS.
                     # If we are using it for ITEMS, the "all_users_ratings" would actually be "all_items_ratings".
                     # And "user_means" would be passed as is.
                     # The signature of calculate_item_mean_centered_cosine is (item1, item2, user_means).
                     # So we should pass user_means directly.
                     score = similarity_func(target_user_ratings, other_user_ratings, user_means)
                 else:
                     continue
            else:
                score = similarity_func(target_user_ratings, other_user_ratings)
                
            similarities.append((user_id, score))
        except Exception:
            continue
            
    # Sort by similarity score descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities
