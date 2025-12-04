import pandas as pd
import os

# Define base paths relative to the project root
# Assuming this script is in 'utils/' and project root is one level up
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'Dianping_SocialRec_2015', 'preprocessed_data.csv')
TARGET_USERS_PATH = os.path.join(PROJECT_ROOT, 'results', 'target_users.txt')
TARGET_ITEMS_PATH = os.path.join(PROJECT_ROOT, 'results', 'target_items.txt')

def get_preprocessed_dataset():
    """
    Loads the preprocessed dataset from CSV.
    
    Returns:
        pd.DataFrame: The dataset containing user, item, and rating.
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)

def get_target_users():
    """
    Loads the list of target users from the results directory.
    
    Returns:
        list: A list of target user IDs (as integers).
    """
    if not os.path.exists(TARGET_USERS_PATH):
        raise FileNotFoundError(f"Target users file not found at {TARGET_USERS_PATH}")
    
    with open(TARGET_USERS_PATH, 'r') as f:
        # Read lines, strip whitespace, and convert to int
        users = [int(line.strip()) for line in f if line.strip()]
    return users

def get_target_items():
    """
    Loads the list of target items from the results directory.
    
    Returns:
        list: A list of target item IDs (as integers).
    """
    if not os.path.exists(TARGET_ITEMS_PATH):
        raise FileNotFoundError(f"Target items file not found at {TARGET_ITEMS_PATH}")
    
    with open(TARGET_ITEMS_PATH, 'r') as f:
        # Read lines, strip whitespace, and convert to int
        items = [int(line.strip()) for line in f if line.strip()]
    return items
