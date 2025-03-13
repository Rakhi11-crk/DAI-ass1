import pandas as pd
import matplotlib.pyplot as plt
# Load the dataset
file_path = "retail_store_sales.csv"  # Update the path if needed
df = pd.read_csv(file_path)

# Display dataset info and first few rows
print(df.info())
print(df.head())

#......DATA CLEANING.....
#........................

# Handle Missing Values
df['Item'].fillna('Unknown', inplace=True)
num_cols = ['Price Per Unit', 'Quantity', 'Total Spent']

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

df['Discount Applied'].fillna('False', inplace=True)

# Remove Duplicates
df.drop_duplicates(inplace=True)

# Detect and Treat Outliers using the IQR Method
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

for col in num_cols:
    df = df[(df[col] >= lower_bound[col]) & (df[col] <= upper_bound[col])]

# Standardize Categorical Values
df['Payment Method'] = df['Payment Method'].str.strip().str.title()
df['Category'] = df['Category'].str.strip().str.title()
df['Location'] = df['Location'].str.strip().str.title()

# Display cleaned dataset info
print(df.info())


#......UNIVARIATE ANALYSIS........
#.................................

# Summary statistics
print(df[num_cols].describe())
# Plot Histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, col in enumerate(num_cols):
    axes[i].hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
# Box Plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, col in enumerate(num_cols):
    axes[i].boxplot(df[col].dropna(), patch_artist=True, boxprops=dict(facecolor="lightblue"))
    axes[i].set_title(f"Box Plot of {col}")
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.show()



#........BIVARIATE ANALYSIS..........
#....................................

# Compute Correlation Matrix
correlation_matrix = df[num_cols].corr()
print("Correlation Matrix:\n", correlation_matrix)
# Scatter plots for numerical variables
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(df['Price Per Unit'], df['Total Spent'], color='blue', alpha=0.5)
axes[0].set_title("Price Per Unit vs Total Spent")
axes[0].set_xlabel("Price Per Unit")
axes[0].set_ylabel("Total Spent")

axes[1].scatter(df['Quantity'], df['Total Spent'], color='red', alpha=0.5)
axes[1].set_title("Quantity vs Total Spent")
axes[1].set_xlabel("Quantity")
axes[1].set_ylabel("Total Spent")

axes[2].scatter(df['Price Per Unit'], df['Quantity'], color='green', alpha=0.5)
axes[2].set_title("Price Per Unit vs Quantity")
axes[2].set_xlabel("Price Per Unit")
axes[2].set_ylabel("Quantity")

plt.tight_layout()
plt.show()
# Aggregate data: Mean Total Spent per Payment Method
payment_means = df.groupby("Payment Method")["Total Spent"].mean()

# Bar plot
plt.figure(figsize=(8, 5))
plt.bar(payment_means.index, payment_means.values, color=['blue', 'red', 'green', 'orange'])
plt.title("Average Total Spent by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Average Total Spent")
plt.xticks(rotation=45)
plt.show()



#.........MULTIVARIATE ANALYSIS.......
#.....................................

from itertools import combinations

# Select numerical columns for pairwise plots
pairs = list(combinations(num_cols, 2))

# Create subplots for scatter plots
fig, axes = plt.subplots(1, len(pairs), figsize=(15, 5))

for i, (x, y) in enumerate(pairs):
    axes[i].scatter(df[x], df[y], alpha=0.5)
    axes[i].set_title(f"{x} vs {y}")
    axes[i].set_xlabel(x)
    axes[i].set_ylabel(y)

plt.tight_layout()
plt.show()
import numpy as np

# Create a heatmap manually using Matplotlib
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(correlation_matrix, cmap="coolwarm")

# Add values inside heatmap
for (i, j), val in np.ndenumerate(correlation_matrix):
    ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='white')

# Set labels
plt.xticks(range(len(num_cols)), num_cols, rotation=45)
plt.yticks(range(len(num_cols)), num_cols)
plt.colorbar(cax)
plt.title("Correlation Heatmap")
plt.show()



#.......SAVE CLEANED DATA.........
#.................................

# Save the cleaned dataset
cleaned_file_path = "cleaned_retail_store_sales.csv"  # Change filename if needed
df.to_csv(cleaned_file_path, index=False)

print(f"Cleaned dataset saved successfully as '{cleaned_file_path}'")

