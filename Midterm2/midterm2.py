import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Read the dataset
#df = pd.read_csv('ME2_Dataset.csv')
#print(df)

#################################
##### Detect Missing Values #####
#################################

###*************** extra rows were added did not used in final submittion
# # adding missing values from the Customer_ID column with recurring number

# # # Step 1: Load the CSV file into a DataFrame
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

# # Step 6: Save the updated DataFrame back to a CSV file
# df_filled.to_csv("ME2_Dataset-v2.csv", index=False)

# print(df_filled)


##
## adding mode to missing numberical values of attributes Age, Purchase_Frequency, Satisfaction_Score, 
## Online_Shopping_Frequency, Store_Visits_Per_Month, Customer_rating, Dependents
##

# # # Step 1: Load the CSV file into a DataFrame
# df = pd.read_csv("ME2_Dataset.csv")

# # Step 2: Columns to fill missing values with the mode
# columns_to_fill = [
#     "Age", 
#     "Purchase_Frequency", 
#     "Satisfaction_Score", 
#     "Online_Shopping_Frequency", 
#     "Store_Visits_Per_Month", 
#     "Customer_Rating", 
#     "Dependents"
# ]

# # Step 3: Fill missing values in the specified columns with their mode
# for column in columns_to_fill:
#     if column in df.columns:  # Check if the column exists in the DataFrame
#         mode_value = df[column].mode()[0]  # Get the mode (most frequent value)
#         df[column].fillna(mode_value, inplace=True)  # Fill missing values with the mode

# # Step 4: Save the updated DataFrame back to a CSV file
# df.to_csv("ME2_Dataset-v3.csv", index=False)

# # Step 5: Print the updated DataFrame
# print(df)

# # adding mean to missing numbercal values of attributes Income, Spending_Score, Avg_Discount_Avail,
# # Credit_Score, Days_Since_Last_Purchase

# # # Step 1: Load the CSV file into a DataFrame
# df = pd.read_csv("ME2_Dataset-v3.csv")

# # Step 2: Columns to fill missing values with the mean
# columns_to_fill = [
#     "Income", 
#     "Spending_Score", 
#     "Avg_Discount_Avail", 
#     "Credit_Score", 
#     "Days_Since_Last_Purchase", 
#     "Dependents"
# ]

# # Step 3: Fill missing values in the specified columns with their mean
# for column in columns_to_fill:
#     if column in df.columns:  # Check if the column exists in the DataFrame
#         mean_value = df[column].mean()  # Calculate the mean of the column
#         df[column].fillna(mean_value, inplace=True)  # Fill missing values with the mean

# # Step 4: Save the updated DataFrame back to a CSV file
# df.to_csv("ME2_Dataset-v4.csv", index=False)

# # Step 5: Print the updated DataFrame
# print(df)

###### adding mode to missing categotical values for attributes Gender, Education_Level, 
###### Marital_Status, Employment_Status, Loyalty_Tier, Region, Product_Category

##### Step 1: Load the CSV file into a DataFrame ###########################################################
# df = pd.read_csv("ME2_Dataset-v4.csv")

# # Step 2: Columns to fill missing values with the mode
# columns_to_fill = [
#     "Gender", 
#     "Education_Level", 
#     "Marital_Status", 
#     "Employment_Status", 
#     "Loyalty_Tier", 
#     "Region", 
#     "Product_Category"
# ]

# # Step 3: Fill missing values in the specified columns with their mode
# for column in columns_to_fill:
#     if column in df.columns:  # Check if the column exists in the DataFrame
#         mode_value = df[column].mode()[0]  # Get the mode (most frequent value)
#         df[column].fillna(mode_value, inplace=True)  # Fill missing values with the mode

# # Step 4: Save the updated DataFrame back to a CSV file
# df.to_csv("ME2_Dataset-v5.csv", index=False)

# # Step 5: Print the updated DataFrame
# print(df)

###################################################
######## Identify and Visualize Outliers ##########

# Income - Method: Interquartile Range (IQR)
# Since "Income" typically follows a skewed or non-normal distribution, 
# IQR is more robust for detecting outliers.
# Calculate IQR
# Q1 = df['Income'].quantile(0.25)
# Q3 = df['Income'].quantile(0.75)
# IQR = Q3 - Q1

# # Define outlier boundaries
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR

# # Identify outliers
# income_outliers = df[(df['Income'] < lower_bound) | (df['Income'] > upper_bound)]

# # Visualization
# plt.figure(figsize=(12, 6))

# # Boxplot
# plt.subplot(1, 2, 1)
# plt.boxplot(df['Income'], vert=False)
# plt.title('Income - Boxplot')
# plt.legend()
# plt.show()

###### outlier for Spending_score

# import scipy.stats as stats

# # Calculate Z-scores
# df['spending_z_score'] = stats.zscore(df['Spending_Score'])

# # Define threshold for outliers
# threshold = 3
# spending_outliers = df[df['spending_z_score'].abs() > threshold]

# # Visualization
# plt.figure(figsize=(12, 6))

# # Boxplot
# plt.subplot(1, 2, 1)
# plt.boxplot(df['Spending_Score'], vert=False)
# plt.title('Spending Score - Boxplot')
# plt.legend()
# plt.show()

#### discount avg #####
# # Calculate IQR
# Q1 = df['Avg_Discount_Avail'].quantile(0.25)
# Q3 = df['Avg_Discount_Avail'].quantile(0.75)
# IQR = Q3 - Q1

# # Define outlier boundaries
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR

# # Identify outliers
# discount_outliers = df[(df['Avg_Discount_Avail'] < lower_bound) | (df['Avg_Discount_Avail'] > upper_bound)]

# # Visualization
# plt.figure(figsize=(12, 6))

# # Boxplot
# plt.subplot(1, 2, 1)
# plt.boxplot(df['Avg_Discount_Avail'], vert=False)
# plt.title('Average Discount Available - Boxplot')
# plt.legend()
# plt.show()

########################
########## Outlier for credit score ########
# from scipy.stats import median_abs_deviation

# # Calculate Median and MAD
# median = df['Credit_Score'].median()
# mad = median_abs_deviation(df['Credit_Score'])

# # Calculate Modified Z-Score
# df['credit_modified_z'] = 0.6745 * (df['Credit_Score'] - median) / mad

# # Define threshold for outliers
# threshold = 3.5
# credit_outliers = df[df['credit_modified_z'].abs() > threshold]

# # Visualization
# plt.figure(figsize=(12, 6))

# # Boxplot
# plt.subplot(1, 2, 1)
# plt.boxplot(df['Credit_Score'], vert=False)
# plt.title('Credit Score - Boxplot')
# plt.legend()
# plt.show()

# print('Credit_Score')

# #####################################
# ########### outliers for last purchase date
# # Calculate IQR
# Q1 = df['Days_Since_Last_Purchase'].quantile(0.25)
# Q3 = df['Days_Since_Last_Purchase'].quantile(0.75)
# IQR = Q3 - Q1

# # Define boundaries
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR

# # Identify outliers
# purchase_outliers = df[(df['Days_Since_Last_Purchase'] < lower_bound) | (df['Days_Since_Last_Purchase'] > upper_bound)]

# # Visualization
# plt.figure(figsize=(12, 6))

# # Boxplot
# plt.subplot(1, 2, 1)
# plt.boxplot(df['Days_Since_Last_Purchase'], vert=False)
# plt.title('Days Since Last Purchase - Boxplot')
# plt.legend()
# plt.show()

########################
############## outliers for dependents
# Define percentile thresholds
# Calculate IQR
# Q1 = df['Dependents'].quantile(0.25)
# Q3 = df['Dependents'].quantile(0.75)
# IQR = Q3 - Q1

# # Define boundaries
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR

# # Identify outliers
# purchase_outliers = df[(df['Dependents'] < lower_bound) | (df['Dependents'] > upper_bound)]

# # Visualization
# plt.figure(figsize=(12, 6))

# # Boxplot
# plt.subplot(1, 2, 1)
# plt.boxplot(df['Dependents'], vert=False)
# plt.title('Dependents - Boxplot')
# plt.legend()
# plt.show()

###############################################################
################ Descriptive Statistics
###############################################################
##### Computing Descriptive Statistics for Variables ##########

# # Function to compute descriptive statistics for all columns
# def descriptive_statistics_all(df):
#     # Use the pandas `describe` method for basic statistics
#     describe_stats = df.describe(include='all').T  # Transpose for better readability
    
#     # Add the mode to the statistics
#     describe_stats['Mode'] = df.mode().iloc[0]  # Get the first mode for each column
    
#     return describe_stats

# # Compute descriptive statistics
# all_statistics_df = descriptive_statistics_all(df)

# # Display the results
# print(all_statistics_df)


################### calculating standard diviation

# # Load the dataset
# file_path = 'ME2_Dataset-v5.csv'
# df = pd.read_csv(file_path)

# # List of columns for which to calculate the standard deviation
# columns = ["Income", "Spending_Score", "Avg_Discount_Avail", "Credit_Score", 
#            "Days_Since_Last_Purchase", "Dependents"]

# # Calculate and print the standard deviation for each column
# for column in columns:
#     if column in df.columns:
#         std_dev = df[column].std()  # Calculate standard deviation
#         print(f"Standard Deviation of {column}: {std_dev:.2f}")

#################################################################
################### Correlation Matrix ##########################
#################################################################
# Load the dataset
# file_path = 'ME2_Dataset-v5.csv'
# df = pd.read_csv(file_path)

# # Select columns for correlation analysis
# columns = ["Income", "Spending_Score", "Avg_Discount_Avail", "Credit_Score", 
#            "Days_Since_Last_Purchase", "Dependents", "Age", "Purchase_Frequency", "Satisfaction_Score", 
#             "Online_Shopping_Frequency", "Store_Visits_Per_Month", "Customer_Rating", "Dependents"]

# # Create the correlation matrix
# correlation_matrix = df[columns].corr()

# # Display the correlation matrix
# print("Correlation Matrix:")
# print(correlation_matrix)

# # visualizing full heat map to display the correlation coefficient between the variables
# plt.figure(figsize=(5,5))
# sns.heatmap(correlation_matrix, square=True, cmap='bwr', linewidth=0.8, vmin=-1, vmax=1, annot=True)
# plt.title('Correlation Coefficient among Variables')
# plt.show()

##################################################################
###################### Testing hypothesis ########################
df = pd.read_csv("ME2_Dataset-v5.csv")

### 1. Income and Spending Score
### H₀: There is no correlation between income and spending score.
### H₁: There is a significant correlation between income and spending score.

# Prepare data
# X = df['Income']  # Independent variable
# y = df['Spending_Score']  # Dependent variable

# # Add constant for the regression model
# X = sm.add_constant(X)

# # Perform regression
# model = sm.OLS(y, X).fit()

# # Print regression results
# print(model.summary())
##################################################################
##################################################################
## Avg_Discount_Avail and Credit_Score
## H₀: The average discount availability has no effect on credit score.
## H₁: The average discount availability significantly affects credit score.

# # Prepare data
# X = df['Avg_Discount_Avail']  # Independent variable
# y = df['Credit_Score']  # Dependent variable

# # Add constant for the regression model
# X = sm.add_constant(X)

# # Perform regression
# model = sm.OLS(y, X).fit()

# # Print regression results
# print(model.summary())

######################
#### Days_Since_Last_Purchase and Satisfaction_Score
### H₀: Days since the last purchase has no relationship with the satisfaction score.
### H₁: Days since the last purchase is significantly related to the satisfaction score.
#### Test: T-test or Simple Linear Regression

# from scipy.stats import ttest_ind

# # Create groups
# recent = df[df['Days_Since_Last_Purchase'] <= 30]['Satisfaction_Score']
# delayed = df[df['Days_Since_Last_Purchase'] > 30]['Satisfaction_Score']

# # Perform t-test
# t_stat, p_value = ttest_ind(recent, delayed)

# print(f"T-test results: t-statistic={t_stat:.2f}, p-value={p_value:.4f}")

###############
##### Dependents and Online_Shopping_Frequency
#### H₀: The number of dependents has no impact on online shopping frequency.
#### H₁: The number of dependents significantly impacts online shopping frequency.
##### Test: Chi-square Test
# from scipy.stats import chi2_contingency

# # Create contingency table
# contingency_table = pd.crosstab(df['Dependents'], df['Online_Shopping_Frequency'])

# # Perform Chi-square test
# chi2, p, dof, expected = chi2_contingency(contingency_table)

# # print(f"Chi-square test results: chi2={chi2:.2f}, p-value={p:.4f}, degrees of freedom={dof}")

# #######################
# ######### Age and Store_Visits_Per_Month
# #### H₀: Age has no relationship with the number of store visits per month.
# #### H₁: Age significantly affects the number of store visits per month.
# #### Test: Simple Linear Regression
# ######
# # Prepare data
# # X = df['Age']  # Independent variable
# # y = df['Store_Visits_Per_Month']  # Dependent variable

# # # Add constant
# # X = sm.add_constant(X)

# # # Perform regression
# # model = sm.OLS(y, X).fit()

# # # Print regression results
# # print(model.summary())

# #######################
# # ####### Purchase_Frequency and Customer_Rating
# # ###### H₀: Purchase frequency has no impact on customer ratings.
# # ##### H₁: Purchase frequency significantly impacts customer ratings.
# # ##### Test: T-test
# # ####### 
# # from scipy.stats import ttest_ind

# # # Create groups
# # high_freq = df[df['Purchase_Frequency'] > df['Purchase_Frequency'].median()]['Customer_Rating']
# # low_freq = df[df['Purchase_Frequency'] <= df['Purchase_Frequency'].median()]['Customer_Rating']

# # # Perform t-test
# # t_stat, p_value = ttest_ind(high_freq, low_freq)

# # print(f"T-test results: t-statistic={t_stat:.2f}, p-value={p_value:.4f}")

# ##############
# ####### Satisfaction_Score and Spending_Score
# # ###### H₀: Satisfaction score has no relationship with spending score.
# # ######m H₁: Satisfaction score is significantly related to spending score.
# # ######## Test: Simple Linear Regression
# # # Prepare data
# # X = df['Satisfaction_Score']  # Independent variable
# # y = df['Spending_Score']  # Dependent variable

# # # Add constant
# # X = sm.add_constant(X)

# # # Perform regression
# # model = sm.OLS(y, X).fit()

# # # Print regression results
# # print(model.summary())

# #################
# # ########## Credit_Score and Income
# # ######### H₀: Credit score has no correlation with income.
# # ######### H₁: Credit score is significantly correlated with income.
# # ############ Test: F-test or Simple Linear Regression
# # # Prepare data
# # X = df['Income']  # Independent variable
# # y = df['Credit_Score']  # Dependent variable

# # # Add constant
# # X = sm.add_constant(X)

# # # Perform regression
# # model = sm.OLS(y, X).fit()

# # # F-test is included in regression results
# # print(model.summary())

# ###############
# # ####### Store_Visits_Per_Month and Online_Shopping_Frequency
# # ##### H₀: The number of store visits per month is not related to online shopping frequency.
# # ##### H₁: The number of store visits per month is significantly related to online shopping frequency.
# # #### Test: Chi-square Test
# # from scipy.stats import chi2_contingency

# # # Create contingency table
# # contingency_table = pd.crosstab(df['Store_Visits_Per_Month'], df['Online_Shopping_Frequency'])

# # # Perform Chi-square test
# # chi2, p, dof, expected = chi2_contingency(contingency_table)

# # print(f"Chi-square test results: chi2={chi2:.2f}, p-value={p:.4f}, degrees of freedom={dof}")

# # ###############
# # ############## Customer_Rating and Satisfaction_Score
# # ############## H₀: Customer ratings have no relationship with satisfaction scores.
# # ############## H₁: Customer ratings are significantly related to satisfaction scores.
# # ############## Test: Simple Linear Regression
# # #############
# # # Prepare data
# # X = df['Customer_Rating']  # Independent variable
# # y = df['Satisfaction_Score']  # Dependent variable

# # # Add constant
# # X = sm.add_constant(X)

# # # Perform regression
# # model = sm.OLS(y, X).fit()

# # # Print regression results
# # print(model.summary())

# #############################################################
# ################ 5. Predictive Modeling #############################
# ######## using "Satisfaction_Score" as the dependent variable and identify
# ######## appropriate independent variables
# ################### Select Independent Variables:
# import pandas as pd
# import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# # Load dataset
# df = pd.read_csv("ME2_Dataset-v5.csv")

# # Select relevant columns for modeling
# columns = ['Satisfaction_Score', 'Spending_Score', 'Income', 'Online_Shopping_Frequency', 
#            'Store_Visits_Per_Month', 'Customer_Rating']

# # Define dependent and independent variables
X = df[['Spending_Score', 'Income', 'Online_Shopping_Frequency', 'Store_Visits_Per_Month', 'Customer_Rating']]
y = df['Satisfaction_Score']

# # Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# # Predict on the test set
y_pred = model.predict(X_test)

# # Evaluate model performance
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error: {mse:.2f}")
# print(f"R-squared: {r2:.2f}")

# # Add constant for statsmodels regression
X_train_sm = sm.add_constant(X_train)

# # Fit model
ols_model = sm.OLS(y_train, X_train_sm).fit()

# # Print summary
# print(ols_model.summary())

######### creating predictive model

# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import seaborn as sns

# # Calculate metrics
# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# # Print metrics
# print(f"Mean Absolute Error (MAE): {mae:.2f}")
# print(f"Mean Squared Error (MSE): {mse:.2f}")
# print(f"R-squared: {r2:.2f}")

# # Scatter plot
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=y_test, y=y_pred)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)  # Perfect prediction line
# plt.title("Actual vs Predicted Values")
# plt.xlabel("Actual Satisfaction Score")
# plt.ylabel("Predicted Satisfaction Score")
# plt.grid(True)
# plt.show()

# # Calculate residuals
# residuals = y_test - y_pred

# # Residuals plot
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=y_pred, y=residuals)
# plt.axhline(0, color='red', linestyle='--', linewidth=2)
# plt.title("Residuals vs Predicted Values")
# plt.xlabel("Predicted Satisfaction Score")
# plt.ylabel("Residuals")
# plt.grid(True)
# plt.show()

# # Histogram of residuals
# plt.figure(figsize=(8, 6))
# sns.histplot(residuals, kde=True, bins=30)
# plt.title("Histogram of Residuals")
# plt.xlabel("Residuals")
# plt.ylabel("Frequency")
# plt.grid(True)
# plt.show()

# # Extract coefficients
coef = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_
})

# # Bar plot of coefficients
# plt.figure(figsize=(10, 6))
# sns.barplot(data=coef, x='Coefficient', y='Feature', orient='h')
# plt.title("Feature Importance (Linear Regression Coefficients)")
# plt.xlabel("Coefficient Value")
# plt.ylabel("Feature")
# plt.grid(True)
# plt.show()

################################################################################
################### 6. Feature Selection ########################################
############## backward selection

import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# # Backward selection
# def backward_selection(X, y):
#     features = list(X.columns)
#     while len(features) > 0:
#         X_train_sm = sm.add_constant(X[features])
#         model = sm.OLS(y, X_train_sm).fit()
#         p_values = model.pvalues.iloc[1:]  # Exclude constant term
#         worst_pval = p_values.max()

#         if worst_pval > 0.05:  # Threshold for significance
#             worst_feature = p_values.idxmax()
#             features.remove(worst_feature)
#         else:
#             break

#     return features

# selected_features_backward = backward_selection(X_train, y_train)
# print("Selected features using backward selection:", selected_features_backward)

############## foward selection

import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("ME2_Dataset-v5.csv")

# Define dependent and independent variables
X = df[['Spending_Score', 'Income', 'Online_Shopping_Frequency', 'Store_Visits_Per_Month', 'Customer_Rating']]
y = df['Satisfaction_Score']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Forward selection
# def forward_selection(X, y):
#     initial_features = []
#     remaining_features = list(X.columns)
#     selected_features = []
#     current_score, best_new_score = float('inf'), float('inf')

#     while remaining_features:
#         scores_with_candidates = []
#         for candidate in remaining_features:
#             features_to_test = initial_features + [candidate]
#             X_train_sm = sm.add_constant(X[features_to_test])
#             model = sm.OLS(y, X_train_sm).fit()
#             scores_with_candidates.append((model.aic, candidate))

#         scores_with_candidates.sort()
#         best_new_score, best_candidate = scores_with_candidates[0]

#         if current_score > best_new_score:
#             remaining_features.remove(best_candidate)
#             initial_features.append(best_candidate)
#             selected_features.append(best_candidate)
#             current_score = best_new_score
#         else:
#             break

#     return selected_features

# selected_features_forward = forward_selection(X_train, y_train)
# print("Selected features using forward selection:", selected_features_forward)

# # Train model using selected features from forward selection
# X_train_fs = X_train[selected_features_forward]
# X_test_fs = X_test[selected_features_forward]

# model_fs = LinearRegression()
# model_fs.fit(X_train_fs, y_train)

# y_pred_fs = model_fs.predict(X_test_fs)
# mse_fs = mean_squared_error(y_test, y_pred_fs)
# r2_fs = r2_score(y_test, y_pred_fs)

# print(f"Forward Selection - MSE: {mse_fs:.2f}, R²: {r2_fs:.2f}")

# # Train model using selected features from backward selection
# X_train_bs = X_train[selected_features_backward]
# X_test_bs = X_test[selected_features_backward]

# model_bs = LinearRegression()
# model_bs.fit(X_train_bs, y_train)

# y_pred_bs = model_bs.predict(X_test_bs)
# mse_bs = mean_squared_error(y_test, y_pred_bs)
# r2_bs = r2_score(y_test, y_pred_bs)

# print(f"Backward Selection - MSE: {mse_bs:.2f}, R²: {r2_bs:.2f}")

#########################################################
############ Tuning the Predictive Model ################
# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset
df = pd.read_csv("ME2_Dataset-v5.csv")

# Step 2: Select features from forward/backward selection
selected_features = ['Income']  # Replace with the actual selected features
X = df[selected_features]
y = df['Satisfaction_Score']

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Tune a Linear Regression model
# Initialize and fit the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression Results:")
print(f"Mean Squared Error: {mse_lr:.2f}")
print(f"R-squared: {r2_lr:.2f}")

# Step 5: Tune a Random Forest model for comparison
# Set up hyperparameter grid for Random Forest
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf_model, rf_params, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best Random Forest model
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# Evaluate Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Results:")
print(f"Mean Squared Error: {mse_rf:.2f}")
print(f"R-squared: {r2_rf:.2f}")

# Step 6: Compare and interpret results
if r2_rf > r2_lr:
    print("\nRandom Forest performed better than Linear Regression.")
else:
    print("\nLinear Regression performed better than Random Forest.")
