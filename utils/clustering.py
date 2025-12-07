# Youssef Zakaria Soubhi Abo Srewa
# 221101030
# noureldeen maher Mesbah
# 221101140
# Youssef Mohamed
# 221101573

import warnings
warnings.filterwarnings("ignore")
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial


def mean_absolute_error(y_true, y_pred):
    """
    Calculate Mean Absolute Error between actual and predicted values.
    
    Args:
        y_true: array-like of actual values
        y_pred: array-like of predicted values
        
    Returns:
        float: Mean Absolute Error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    """
    Calculate Mean Squared Error between actual and predicted values.
    
    Args:
        y_true: array-like of actual values
        y_pred: array-like of predicted values
        
    Returns:
        float: Mean Squared Error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def _euclidean_distances_chunked(X, Y, chunk_size=1000):
    """
    Compute pairwise Euclidean distances using memory-efficient chunking.
    Uses the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x.y
    
    Args:
        X: ndarray of shape (n_samples_X, n_features)
        Y: ndarray of shape (n_samples_Y, n_features)
        chunk_size: Number of rows to process at once
        
    Returns:
        distances: ndarray of shape (n_samples_X, n_samples_Y)
    """
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    
    # Precompute squared norms
    X_norm_sq = np.sum(X ** 2, axis=1)
    Y_norm_sq = np.sum(Y ** 2, axis=1)
    
    # Process in chunks for memory efficiency
    distances = np.empty((n_samples_X, n_samples_Y), dtype=np.float64)
    
    for start in range(0, n_samples_X, chunk_size):
        end = min(start + chunk_size, n_samples_X)
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x.y
        # Using BLAS for matrix multiplication (np.dot)
        chunk_dist = X_norm_sq[start:end, np.newaxis] + Y_norm_sq - 2 * np.dot(X[start:end], Y.T)
        # Ensure non-negative (numerical precision issues)
        np.maximum(chunk_dist, 0, out=chunk_dist)
        distances[start:end] = chunk_dist
    
    return distances


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    Uses vectorized NumPy operations.
    
    z = (x - mean) / std
    """
    
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X):
        """Compute the mean and std to be used for later scaling."""
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0
        return self
    
    def transform(self, X):
        """Perform standardization by centering and scaling (vectorized)."""
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """Scale back the data to the original representation."""
        X = np.asarray(X, dtype=np.float64)
        return X * self.scale_ + self.mean_


class KMeans:
    """
    K-Means clustering with k-means++ initialization.
    
    Optimized with:
    - Vectorized distance calculations using NumPy/BLAS
    - Parallelization via n_jobs parameter
    - Memory-efficient chunked processing for large datasets
    
    Args:
        n_clusters: Number of clusters to form.
        random_state: Random seed for reproducibility.
        n_init: Number of times to run with different centroid seeds.
        max_iter: Maximum number of iterations per run.
        tol: Relative tolerance to declare convergence.
        n_jobs: Number of parallel jobs (-1 = all CPUs, 1 = no parallelism).
        chunk_size: Chunk size for memory-efficient distance computation.
    """
    
    def __init__(self, n_clusters=8, random_state=None, n_init=10, max_iter=300, 
                 tol=1e-4, n_jobs=1, chunk_size=5000):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.chunk_size = chunk_size
        
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
    
    def _kmeans_plusplus_init(self, X, rng):
        """
        Initialize centroids using k-means++ algorithm (vectorized).
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        centroids = np.empty((self.n_clusters, n_features), dtype=np.float64)
        
        # Choose first centroid randomly
        idx = rng.integers(n_samples)
        centroids[0] = X[idx]
        
        # Compute squared distances to first centroid (vectorized)
        min_distances = np.sum((X - centroids[0]) ** 2, axis=1)
        
        # Choose remaining centroids
        for k in range(1, self.n_clusters):
            # Choose new centroid with probability proportional to D(x)^2
            probs = min_distances / min_distances.sum()
            idx = rng.choice(n_samples, p=probs)
            centroids[k] = X[idx]
            
            # Update minimum distances (vectorized)
            new_distances = np.sum((X - centroids[k]) ** 2, axis=1)
            np.minimum(min_distances, new_distances, out=min_distances)
        
        return centroids
    
    def _compute_labels_and_inertia(self, X, centroids):
        """
        Assign samples to nearest centroids using vectorized operations
        with memory-efficient chunking.
        """
        n_samples = X.shape[0]
        labels = np.empty(n_samples, dtype=np.int32)
        inertia = 0.0
        
        # Process in chunks for memory efficiency
        for start in range(0, n_samples, self.chunk_size):
            end = min(start + self.chunk_size, n_samples)
            X_chunk = X[start:end]
            
            # Compute squared distances using vectorized operations
            # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x.c
            X_sq = np.sum(X_chunk ** 2, axis=1, keepdims=True)
            C_sq = np.sum(centroids ** 2, axis=1)
            
            # Use BLAS for matrix multiplication
            distances_sq = X_sq + C_sq - 2 * np.dot(X_chunk, centroids.T)
            
            # Find nearest centroid for each sample
            chunk_labels = np.argmin(distances_sq, axis=1)
            labels[start:end] = chunk_labels
            
            # Accumulate inertia (sum of min squared distances)
            inertia += np.sum(np.min(distances_sq, axis=1))
        
        return labels, inertia
    
    def _update_centroids(self, X, labels):
        """
        Compute new centroids as mean of assigned samples (vectorized).
        """
        n_features = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features), dtype=np.float64)
        
        for k in range(self.n_clusters):
            mask = labels == k
            count = np.sum(mask)
            if count > 0:
                new_centroids[k] = np.mean(X[mask], axis=0)
            else:
                # Handle empty cluster - keep old centroid
                new_centroids[k] = self.cluster_centers_[k] if self.cluster_centers_ is not None else X[0]
        
        return new_centroids
    
    def _single_run(self, X, rng):
        """Run a single k-means iteration."""
        # Initialize centroids
        centroids = self._kmeans_plusplus_init(X, rng)
        
        for _ in range(self.max_iter):
            # Assign labels (vectorized)
            labels, inertia = self._compute_labels_and_inertia(X, centroids)
            
            # Update centroids (vectorized)
            new_centroids = self._update_centroids(X, labels)
            
            # Check convergence
            centroid_shift = np.sum((new_centroids - centroids) ** 2)
            centroids = new_centroids
            
            if centroid_shift < self.tol:
                break
        
        # Final assignment
        labels, inertia = self._compute_labels_and_inertia(X, centroids)
        return centroids, labels, inertia
    
    def _run_single_init(self, args):
        """Helper for parallel execution."""
        X, seed = args
        rng = np.random.default_rng(seed)
        return self._single_run(X, rng)
    
    def fit(self, X):
        """
        Compute k-means clustering with optional parallelization.
        """
        X = np.asarray(X, dtype=np.float64)
        
        # Setup random number generator
        rng = np.random.default_rng(self.random_state)
        
        # Generate seeds for each initialization
        seeds = [rng.integers(0, 2**31) for _ in range(self.n_init)]
        
        best_centroids = None
        best_labels = None
        best_inertia = np.inf
        
        # Determine number of workers
        n_jobs = self.n_jobs
        if n_jobs == -1:
            import os
            n_jobs = os.cpu_count() or 1
        
        if n_jobs > 1 and self.n_init > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                args_list = [(X, seed) for seed in seeds]
                results = list(executor.map(self._run_single_init, args_list))
            
            for centroids, labels, inertia in results:
                if inertia < best_inertia:
                    best_inertia = inertia
                    best_centroids = centroids
                    best_labels = labels
        else:
            # Sequential execution
            for seed in seeds:
                init_rng = np.random.default_rng(seed)
                centroids, labels, inertia = self._single_run(X, init_rng)
                
                if inertia < best_inertia:
                    best_inertia = inertia
                    best_centroids = centroids
                    best_labels = labels
        
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        
        return self
    
    def predict(self, X):
        """Predict the closest cluster for each sample."""
        X = np.asarray(X, dtype=np.float64)
        labels, _ = self._compute_labels_and_inertia(X, self.cluster_centers_)
        return labels
    
    def fit_predict(self, X):
        """Compute cluster centers and predict cluster index for each sample."""
        self.fit(X)
        return self.labels_


def silhouette_score(X, labels, sample_size=None, random_state=None):
    """
    Compute the mean Silhouette Coefficient using vectorized operations.
    
    Optimized with chunked distance computation for memory efficiency.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    n_samples = X.shape[0]
    
    # Sample if requested
    if sample_size is not None and sample_size < n_samples:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(n_samples, size=sample_size, replace=False)
        X = X[indices]
        labels = labels[indices]
        n_samples = sample_size
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Edge case: only one cluster
    if n_clusters == 1:
        return 0.0
    
    # Precompute cluster masks and sizes
    cluster_masks = {c: labels == c for c in unique_labels}
    cluster_sizes = {c: np.sum(mask) for c, mask in cluster_masks.items()}
    
    # Compute intra-cluster mean distances (a) for each sample
    a_values = np.zeros(n_samples)
    
    # Compute inter-cluster mean distances (b) for each sample
    b_values = np.full(n_samples, np.inf)
    
    # Process each cluster
    for cluster_i in unique_labels:
        mask_i = cluster_masks[cluster_i]
        X_cluster_i = X[mask_i]
        n_i = cluster_sizes[cluster_i]
        
        if n_i > 1:
            # Compute pairwise distances within cluster (vectorized)
            # Using broadcasting: ||xi - xj||^2
            X_i_sq = np.sum(X_cluster_i ** 2, axis=1, keepdims=True)
            dist_matrix = X_i_sq + X_i_sq.T - 2 * np.dot(X_cluster_i, X_cluster_i.T)
            np.maximum(dist_matrix, 0, out=dist_matrix)
            dist_matrix = np.sqrt(dist_matrix)
            
            # Mean distance to other points in same cluster
            # Sum of distances divided by (n-1) for each point
            a_cluster = np.sum(dist_matrix, axis=1) / (n_i - 1)
            a_values[mask_i] = a_cluster
        
        # Compute mean distance to points in other clusters
        for cluster_j in unique_labels:
            if cluster_j == cluster_i:
                continue
            
            mask_j = cluster_masks[cluster_j]
            X_cluster_j = X[mask_j]
            
            # Compute distances from cluster_i points to cluster_j points (vectorized)
            X_i_sq = np.sum(X_cluster_i ** 2, axis=1, keepdims=True)
            X_j_sq = np.sum(X_cluster_j ** 2, axis=1)
            dist_matrix = X_i_sq + X_j_sq - 2 * np.dot(X_cluster_i, X_cluster_j.T)
            np.maximum(dist_matrix, 0, out=dist_matrix)
            dist_matrix = np.sqrt(dist_matrix)
            
            # Mean distance to cluster_j for each point in cluster_i
            mean_dist_to_j = np.mean(dist_matrix, axis=1)
            
            # Update b values (minimum mean distance to other clusters)
            current_b = b_values[mask_i]
            np.minimum(current_b, mean_dist_to_j, out=current_b)
            b_values[mask_i] = current_b
    
    # Compute silhouette scores
    max_ab = np.maximum(a_values, b_values)
    silhouette_values = np.where(max_ab > 0, (b_values - a_values) / max_ab, 0.0)
    
    return np.mean(silhouette_values)
