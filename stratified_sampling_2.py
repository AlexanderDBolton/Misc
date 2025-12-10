import pandas as pd
import numpy as np
from scipy.stats import kstest, wasserstein_distance
from itertools import combinations

# Use K-S statistic (ks) or Wasserstein distance (w1)
DISTANCE_METRIC = 'ks' 

def total_distance(sample_df, population_df):
    """Calculates the sum of distances across all columns."""
    distances = []
    for col in population_df.columns:
        if DISTANCE_METRIC == 'w1':
            dist = wasserstein_distance(sample_df[col], population_df[col].values)
        else: # Default to KS
            ks_statistic, p_value = kstest(sample_df[col], population_df[col].values)
            dist = ks_statistic
        distances.append(dist)
    return sum(distances)

# Assume 'df' is your original small DataFrame with only the 5 numeric columns
# df = pd.DataFrame({'col1': np.random.rand(100), ...}) 

population_size = len(df)
sample_size = int(0.10 * population_size) # 10% sample size (e.g., 10 rows for N=100)

if sample_size < 2:
    raise ValueError("Sample size is too small for meaningful stratification/distance calculation.")

num_iterations = 10000 # Increase this number for better optimization if you have time

best_sample1 = None
best_sample2 = None
min_combined_distance = float('inf')

for i in range(num_iterations):
    # 1. Randomly draw the first sample's indices
    indices1 = np.random.choice(df.index, size=sample_size, replace=False)
    sample1_df = df.loc[indices1]
    
    # 2. Get the remaining data for the second sample
    remaining_df = df.drop(indices1)
    
    # 3. Randomly draw the second sample's indices from the *remaining* data
    indices2 = np.random.choice(remaining_df.index, size=sample_size, replace=False)
    sample2_df = remaining_df.loc[indices2]
    
    # 4. Calculate total distance for both samples
    dist1 = total_distance(sample1_df, df)
    dist2 = total_distance(sample2_df, df)
    combined_dist = dist1 + dist2
    
    # 5. Check if this pair is better than the current best
    if combined_dist < min_combined_distance:
        min_combined_distance = combined_dist
        best_sample1 = sample1_df.copy()
        best_sample2 = sample2_df.copy()

print(f"Minimum total distance found for the pair: {min_combined_distance}")

# best_sample1 and best_sample2 now contain your two non-overlapping approximately stratified samples.
