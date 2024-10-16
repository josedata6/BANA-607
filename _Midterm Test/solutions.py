## Step 1: Describe Each Column
## Write a brief description of each column and identify whether it is numerical or categorical.
#   Age: numerical
#   Gender: categorical, male & female
#   smoking_habit: categorical, Non-Smoker, Light Smoker, Heavy Smoker
#   work_out_habit: categorical, None, Occasionally, Regularly
#   heart_attack: categorical, 0 & 1
#   salary: numerical
#   education: categorical: High School, Bachelor's, Master's, & PhD


## Step 2: Load the Dataset
## Open the healthData.csv file using Python and load it into a DataFrame named healthData using pandas

import pandas as pd
import matplotlib.pyplot as plt

# # Load the dataset into a DataFrame
healthData = pd.read_csv('healthData.csv')

##Step 3: Perform Descriptive Statistics

# # Descriptive statistics for 'age'
print("Descriptive statistics for 'age':")
print(healthData['age'].describe())
print("Mode of 'age':", healthData['age'].mode()[0])
print("Skewness of 'age':", healthData['age'].skew())
print("Kurtosis of 'age':", healthData['age'].kurtosis())
# print("Percentiles for 'age':", healthData['age'].quantile([0.01, 0.10, 0.25, 0.75, 0.90, 0.99]))
# IQR_age = healthData['age'].quantile(0.75) - healthData['age'].quantile(0.25)
# print("IQR (Interquartile Range) of 'age':", IQR_age)

# # Descriptive statistics for 'salary'
# print("\nDescriptive statistics for 'salary':")
# print(healthData['salary'].describe())
# print("Mode of 'salary':", healthData['salary'].mode()[0])
# print("Skewness of 'salary':", healthData['salary'].skew())
# print("Kurtosis of 'salary':", healthData['salary'].kurtosis())
# print("Percentiles for 'salary':", healthData['salary'].quantile([0.01, 0.10, 0.25, 0.75, 0.90, 0.99]))
# IQR_salary = healthData['salary'].quantile(0.75) - healthData['salary'].quantile(0.25)
# print("IQR (Interquartile Range) of 'salary':", IQR_salary)
