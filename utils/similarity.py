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
    
    denominator = math.sqrt(sum_sq_u1) * math.sqrt(sum_sq_u2)
    
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
    
    denominator = math.sqrt(sum_sq_u1) * math.sqrt(sum_sq_u2)
    
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
    
    denominator = math.sqrt(sum_sq_u1) * math.sqrt(sum_sq_u2)
    
    if denominator == 0:
        return 0.0
        
    return numerator / denominator

def calculate_item_mean_centered_cosine(item1_ratings, item2_ratings, item1_mean, item2_mean):
    """
    Calculates mean-centered cosine similarity between two items.
    
    Args:
        item1_ratings (dict): Dictionary of {user_id: rating} for item 1.
        item2_ratings (dict): Dictionary of {user_id: rating} for item 2.
        item1_mean (float): Average rating of item 1.
        item2_mean (float): Average rating of item 2.
        
    Returns:
        float: Similarity score.
    """
    # Logic is identical to user-based, just interpreting keys as users instead of items
    return calculate_user_mean_centered_cosine(item1_ratings, item2_ratings, item1_mean, item2_mean)

def calculate_item_pearson(item1_ratings, item2_ratings):
    """
    Calculates Pearson correlation between two items based on common users.
    
    Args:
        item1_ratings (dict): Dictionary of {user_id: rating} for item 1.
        item2_ratings (dict): Dictionary of {user_id: rating} for item 2.
        
    Returns:
        float: Pearson correlation.
    """
    # Logic is identical to user-based
    return calculate_user_pearson(item1_ratings, item2_ratings)

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
            
        # Prepare arguments for the similarity function
        try:
            if similarity_func.__name__ == 'calculate_user_mean_centered_cosine' or similarity_func.__name__ == 'calculate_item_mean_centered_cosine':
                 if target_mean is not None and user_means:
                     score = similarity_func(target_user_ratings, other_user_ratings, target_mean, user_means[user_id])
                 else:
                     # Fallback or error if means not provided for mean-centered
                     continue 
            else:
                score = similarity_func(target_user_ratings, other_user_ratings)
                
            similarities.append((user_id, score))
        except Exception:
            # Ignore errors during calculation
            continue
            
    # Sort by similarity score descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities
