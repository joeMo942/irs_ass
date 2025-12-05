
def predict_mean_centered(user_id, item_id, neighbors_list, item_means_dict, user_item_ratings):
    """
    Predicts rating using Mean-Centered Cosine Similarity logic.
    """
    # neighbors_list is list of (item, sim)
    if not neighbors_list:
        return item_means_dict.get(item_id, 3.0)
    
    numerator = 0.0
    denominator = 0.0
    
    user_ratings = user_item_ratings.get(user_id, {})
    
    if not user_ratings:
        return item_means_dict.get(item_id, 3.0)
        
    user_mean = sum(user_ratings.values()) / len(user_ratings)
    
    for neighbor, sim in neighbors_list:
        if neighbor in user_ratings:
            rating = user_ratings[neighbor]
            numerator += sim * (rating - user_mean)
            denominator += sim
            
    if denominator == 0:
        return item_means_dict.get(item_id, 3.0)
        
    return user_mean + (numerator / denominator)

def predict_pearson(user_id, item_id, neighbors_list, item_means_dict, user_item_ratings):
    """
    Predicts rating using Pearson Correlation logic.
    """
    if not neighbors_list:
        return item_means_dict.get(item_id, 3.0)
        
    user_ratings = user_item_ratings.get(user_id, {})
    if not user_ratings:
        return item_means_dict.get(item_id, 3.0)
        
    numerator = 0.0
    denominator = 0.0
    
    target_item_mean = item_means_dict.get(item_id, 3.0)
    
    for neighbor, sim in neighbors_list:
        if neighbor in user_ratings:
            rating = user_ratings[neighbor]
            neighbor_mean = item_means_dict.get(neighbor, 3.0)
            numerator += sim * (rating - neighbor_mean)
            denominator += sim
            
    if denominator == 0:
        return target_item_mean
        
    return target_item_mean + (numerator / denominator)

def predict_user_based(target_user_id, item_id, top_neighbors, user_item_ratings, user_means):
    """
    Predicts rating for a target user and item using Mean-Centered Cosine User-Based CF.
    
    Args:
        target_user_id (int): ID of the target user.
        item_id (int): ID of the target item.
        top_neighbors (list): List of tuples (neighbor_id, similarity) sorted by similarity.
        user_item_ratings (dict): Dictionary of {user_id: {item_id: rating}}.
        user_means (dict): Dictionary of {user_id: mean_rating}.
        
    Returns:
        float: Predicted rating.
    """
    numerator = 0.0
    denominator = 0.0
    
    target_mean = user_means.get(target_user_id, 3.0)
    
    for v_id, sim in top_neighbors:
        v_ratings = user_item_ratings.get(v_id, {})
        if item_id in v_ratings:
            r_vi = v_ratings[item_id]
            v_mean = user_means.get(v_id, 3.0)
            
            numerator += sim * (r_vi - v_mean)
            denominator += sim
            
    if denominator == 0:
        return target_mean
        
    prediction = target_mean + (numerator / denominator)
    return max(1.0, min(5.0, prediction))
