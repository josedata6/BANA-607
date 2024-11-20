import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Read the dataset
#df = pd.read_csv('ME2_Dataset.csv')
#print(df)

##### Detect Missing Values #####

# adding missing values from the Customer_ID column

# Step 1: Load the CSV file into a DataFrame
df = pd.read_csv("ME2_Dataset.csv")

# Step 2: Convert the 'customer_ID' column to numerical values (if not already numeric)
df['Customer_ID'] = pd.to_numeric(df['Customer_ID'], errors='coerce')

# Step 3: Identify the range of numbers that should be in the column
min_id, max_id = int(df['Customer_ID'].min()), int(df['Customer_ID'].max())  # Convert to integers
expected_ids = set(range(min_id, max_id + 1))

# Step 4: Identify the missing numbers
actual_ids = set(df['Customer_ID'].dropna().astype(int))  # Ensure existing IDs are integers
missing_ids = sorted(expected_ids - actual_ids)

# Step 5: Fill in the missing rows
# Create a DataFrame with the missing IDs
missing_df = pd.DataFrame({'Customer_ID': missing_ids})

# Append the missing rows and sort the DataFrame by 'customer_ID'
df_filled = pd.concat([df, missing_df], ignore_index=True).sort_values(by='Customer_ID')

# Reset the index after sorting
df_filled = df_filled.reset_index(drop=True)

# Step 6: Save the updated DataFrame back to a CSV file (optional)
df_filled.to_csv("ME2_Dataset-v2.csv", index=False)

print(df_filled)
