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
# print("Descriptive statistics for 'age':")
# print(healthData['age'].describe())
# print("Mode of 'age':", healthData['age'].mode()[0])
# print("Skewness of 'age':", healthData['age'].skew())
# print("Kurtosis of 'age':", healthData['age'].kurtosis())
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


# # Descriptive statistics for categorical variables
# # Including mode, frequency distribution, and relative frequency

# # Descriptive statistics for 'gender'
print("\nFrequency distribution for 'gender':")
# print(healthData['gender'].value_counts())
# print("Relative frequency distribution for 'gender':")
# print(healthData['gender'].value_counts(normalize=True))

# # Step 3: Visualizations
# # ----------------------
# # Bar chart for 'gender'
healthData['gender'].value_counts().plot(kind='bar')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# # Histogram for 'age'
# plt.hist(healthData['age'].dropna(), bins=20)
# plt.title('Age Distribution')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.show()

# # Histogram for 'salary'
# plt.hist(healthData['salary'].dropna(), bins=20)
# plt.title('Salary Distribution')
# plt.xlabel('Salary')
# plt.ylabel('Frequency')
# plt.show()

# # Boxplot for 'age'
# plt.boxplot(healthData['age'].dropna())
# plt.title('Boxplot of Age')
# plt.ylabel('Age')
# plt.show()

# # Boxplot for 'salary'
# plt.boxplot(healthData['salary'].dropna())
# plt.title('Boxplot of Salary')
# plt.ylabel('Salary')
# plt.show()