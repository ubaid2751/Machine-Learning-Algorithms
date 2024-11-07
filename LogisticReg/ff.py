# K medoids code
import numpy as np

def k_medoids(X, k, max_iterations=100):
    # Step 1: Initialize medoids randomly
    medoids_indices = np.random.choice(range(X.shape[0]), k, replace=False)
    medoids = X[medoids_indices, :]
    
    