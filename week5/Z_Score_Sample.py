import pandas as pd
import numpy as np
from scipy import stats

# Example dataset
data = {'Age': [23, 25, 30, 21, 19, 29, 32, 34, 200, 22]}  # 200 is an outlier
df = pd.DataFrame(data)

print(df)

# Calculate Z-scores
df['Z_score_Age'] = stats.zscore(df['Age'])

print(df)

# Set the threshold for outliers (typically Z > 3 or Z < -3)
threshold = 3
print(threshold)

# Detect outliers
outliers = df[np.abs(df['Z_score_Age']) > threshold]

print("\nZ-scores and detected outliers:")
print(df)
print("\nDetected Outliers:")
print(outliers)
