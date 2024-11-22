
# # # turing the calumns into categorical and numerical
# import pandas as pd

# # Load the CSV file
# df = pd.read_csv("Baseball Salaries Extra.csv")

# # List of categorical columns to convert
# categorical_columns = ['Team', 'Position', 'Pitcher', 'Division', 'League', 'Yankees', 'Playoff 2017 Team']

# # Convert categorical columns to dummy/indicator variables
# df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# # Convert 'Salary' column to numerical, handling non-numeric entries
# df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')  # Coerce errors to NaN if any invalid entries

# # Optional: Fill NaN values in 'Salary' with 0 or any other method
# df['Salary'].fillna(0, inplace=True)

# # Verify changes
# print(df.head())

# #dropping name
# df = df.drop(columns=['Name'])
# df = df.drop(columns=['Name'])

######################################################
### removing dollar sign from Earnings #####
##########

# import pandas as pd

# # Load the CSV file
# df = pd.read_csv("correlationGolf.csv")

# # Remove dollar signs and convert to float (if needed)
# df["Earnings"] = df["Earnings"].replace('[\$,]', '', regex=True).astype(float)

# # Save the updated CSV file
# df.to_csv("correlationGolf.csv", index=False)

#################
###  null values be filled with the mean #####
######

# import pandas as pd

# # Load the CSV file
# df = pd.read_csv("correlationGolf.csv")

# # List of columns where you want to fill NaN with the mean
# columns = ["Age", "Yards/Drive", "Driving Accuracy", "Greens in Regulation", "Putting Average", "Sand Save Pct"]

# # Fill NaN values with the mean of each specified column
# for column in columns:
#     df[column].fillna(df[column].mean(), inplace=True)

# # Save the updated CSV file
# df.to_csv("correlationGolf.csv", index=False)

##########################
#### correcting decimal places
##########

# import pandas as pd

# # Load the CSV file
# df = pd.read_csv("correlationGolf.csv")

# # Fill NaN values in specified columns with the mean of each column
# columns = ["Age", "Yards/Drive", "Driving Accuracy", "Greens in Regulation", "Putting Average", "Sand Save Pct"]
# for column in columns:
#     df[column].fillna(df[column].mean(), inplace=True)

# # Format columns to 2 decimal places or as whole numbers
# df["Age"] = df["Age"].round(0).astype(int)  # Whole numbers for Age
# decimal_columns = ["Yards/Drive", "Driving Accuracy", "Greens in Regulation", "Putting Average", "Sand Save Pct"]
# df[decimal_columns] = df[decimal_columns].round(2)  # 2 decimal places for other columns

# # Save the updated CSV file
# df.to_csv("correlationGolf_formatted.csv", index=False)


######################

# import numpy as np
import pandas as pd





# # # Load data
# df = pd.read_csv('2016-golf-stats.csv') # importing the csv file
# # print(df.head())

# # Filling missing values with mean
# df_filled_mean = df.fillna(df.mean(numeric_only=True))
# print("\nDataFrame after filling missing values with mean:")
# print(df_filled_mean)

# # # Descriptive statistics
# df = pd.read_csv('correlationGolf_formatted.csv') # importing the csv file
# print(df.describe()) # show summary statistics for each column

# # saving output into file
# # df.to_csv('correlationGolf.csv')

# # #dropping player
# df = df.drop(columns=['Player'])

# # #Finding Correlation
# correlation_matrix = df.corr()
# print(correlation_matrix)

# # # Visualize the finding
# import seaborn as sn
# import matplotlib.pyplot as plt

# sn.heatmap(correlation_matrix, annot=True)
# plt.show()
#############################################################
################ 5. Predictive Modeling #############################
######## using "Satisfaction_Score" as the dependent variable and identify
######## appropriate independent variables
################### Select Independent Variables:
# Load dataset
df = pd.read_csv("ME2_Dataset-v5.csv")

# Select relevant columns for modeling
columns = ['Satisfaction_Score', 'Spending_Score', 'Income', 'Online_Shopping_Frequency', 
           'Store_Visits_Per_Month', 'Customer_Rating']

# Define dependent and independent variables
X = df[['Spending_Score', 'Income', 'Online_Shopping_Frequency', 'Store_Visits_Per_Month', 'Customer_Rating']]
y = df['Satisfaction_Score']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Add constant for statsmodels regression
X_train_sm = sm.add_constant(X_train)

# Fit model
ols_model = sm.OLS(y_train, X_train_sm).fit()

# Print summary
print(ols_model.summary())
