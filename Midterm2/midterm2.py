import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Read the dataset
#df = pd.read_csv('ME2_Dataset.csv')
#print(df)

##### Detect Missing Values #####

# # adding missing values from the Customer_ID column with recurring number

# # Step 1: Load the CSV file into a DataFrame
# df = pd.read_csv("ME2_Dataset.csv")

# # Step 2: Convert the 'customer_ID' column to numerical values (if not already numeric)
# df['Customer_ID'] = pd.to_numeric(df['Customer_ID'], errors='coerce')

# # Step 3: Identify the range of numbers that should be in the column
# min_id, max_id = int(df['Customer_ID'].min()), int(df['Customer_ID'].max())  # Convert to integers
# expected_ids = set(range(min_id, max_id + 1))

# # Step 4: Identify the missing numbers
# actual_ids = set(df['Customer_ID'].dropna().astype(int))  # Ensure existing IDs are integers
# missing_ids = sorted(expected_ids - actual_ids)

# # Step 5: Fill in the missing rows
# # Create a DataFrame with the missing IDs
# missing_df = pd.DataFrame({'Customer_ID': missing_ids})

# # Append the missing rows and sort the DataFrame by 'customer_ID'
# df_filled = pd.concat([df, missing_df], ignore_index=True).sort_values(by='Customer_ID')

# # Reset the index after sorting
# df_filled = df_filled.reset_index(drop=True)

# # Step 6: Save the updated DataFrame back to a CSV file (optional)
# df_filled.to_csv("ME2_Dataset-v2.csv", index=False)

# print(df_filled)

# adding mode to missing numberical values of attributes Age, Purchase_Frequency, Satisfaction_Score, 
# Online_Shopping_Frequency, Store_Visits_Per_Month, Customer_rating, Dependents

# # Step 1: Load the CSV file into a DataFrame
# df = pd.read_csv("ME2_Dataset-v2.csv")

# # Step 2: Columns to fill missing values with the mode
# columns_to_fill = [
#     "Age", 
#     "Purchase_Frequency", 
#     "Satisfaction_Score", 
#     "Online_Shopping_Frequency", 
#     "Store_Visits_Per_Month", 
#     "Customer_rating", 
#     "Dependents"
# ]

# # Step 3: Fill missing values in the specified columns with their mode
# for column in columns_to_fill:
#     if column in df.columns:  # Check if the column exists in the DataFrame
#         mode_value = df[column].mode()[0]  # Get the mode (most frequent value)
#         df[column].fillna(mode_value, inplace=True)  # Fill missing values with the mode

# # Step 4: Save the updated DataFrame back to a CSV file (optional)
# df.to_csv("ME2_Dataset-v3.csv", index=False)

# # Step 5: Print the updated DataFrame (optional)
# print(df)

# adding mean to missing numbercal values of attributes Income, Spending_Score, Avg_Discount_Avail,
# Credit_Score, Days_Since_Last_Purchase

# Step 1: Load the CSV file into a DataFrame
df = pd.read_csv("ME2_Dataset-v3.csv")

# Step 2: Columns to fill missing values with the mean
columns_to_fill = [
    "Income", 
    "Spending_Score", 
    "Avg_Discount_Avail", 
    "Credit_Score", 
    "Days_Since_Last_Purchase", 
    "Dependents"
]

# Step 3: Fill missing values in the specified columns with their mean
for column in columns_to_fill:
    if column in df.columns:  # Check if the column exists in the DataFrame
        mean_value = df[column].mean()  # Calculate the mean of the column
        df[column].fillna(mean_value, inplace=True)  # Fill missing values with the mean

# Step 4: Save the updated DataFrame back to a CSV file (optional)
df.to_csv("ME2_Dataset-v4.csv", index=False)

# Step 5: Print the updated DataFrame (optional)
print(df)