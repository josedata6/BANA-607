import pandas as pd

# # # Load the dataset
df = pd.read_csv('Student_Grades.csv')
print("Original DataFrame:")
print(df)


# Remove Rows with Missing Values
# Remove rows where any column has a missing value

# df_dropped_rows = df.dropna()

# print("\nDataFrame after removing rows with missing values:")
# print(df_dropped_rows)

# Remove Columns with Missing Values
# #     # Remove columns that contain any missing values

# # df_dropped_columns = df.dropna(axis=1)

# # print("\nDataFrame after removing columns with missing values:")
# # print(df_dropped_columns)


# Using Mean
# Fill missing values in 'MathScore', 'EnglishScore', 'ScienceScore' and 'Attendance'

# df_filled_mean = df.fillna(df.mean(numeric_only=True))

# print("\nDataFrame after filling missing values with mean:")
# print(df_filled_mean)

# using Median

# df_filled_median = df.fillna(df.median(numeric_only=True))
# print("\nDataFrame after filling missing values with Median:")
# print(df_filled_median)

# using Mode

df_filled_mode = df.fillna(df.mode().iloc[0])
print("\nDataFrame after filling missing values with Median:")
print(df_filled_mode)

# using FWD fill
df_ffill = df.fillna(method='ffill')
print("\nDataFrame after filling missing values with Median:")
print(df_ffill)

# using BWD fill
df_bfill = df.fillna(method='bfill')
print("\nDataFrame after filling missing values with Median:")
print(df_bfill)

# detect outliers

# # # Remove outliers Age
# # df_no_outliers = df[(df['Age'] >= lower_bound) & (df['Age'] <= upper_bound)]

# # print("\nDataFrame after removing outliers:")
# # print(df_no_outliers)

# # # Detect and Remove Outliers Using IQR (Interquartile Range) for MathScore
# #     # Outliers in the MathScore column can be detected and removed using the IQR method.

# # # Calculate Q1 (25th percentile) and Q3 (75th percentile)
# # Q1 = df['MathScore'].quantile(0.25)
# # Q3 = df['MathScore'].quantile(0.75)
# # IQR = Q3 - Q1

# # # Define outlier bounds
# # lower_bound = Q1 - 1.5 * IQR
# # upper_bound = Q3 + 1.5 * IQR

# # # Detect outliers
# # outliers = df[(df['MathScore'] < lower_bound) | (df['MathScore'] > upper_bound)]

# # print(f"\nOutliers detected in MathScore column (using IQR):\n{outliers}")


