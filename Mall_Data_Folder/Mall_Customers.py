# --- 1. Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from warnings import filterwarnings
filterwarnings('ignore') # Suppress unnecessary warnings for cleaner output

# --- 2. Data Loading and Preparation ---
# NOTE: Ensure 'Mall_Customers.csv' is in your working directory.
try:
    data = pd.read_csv('Mall_Data/Mall_Customers.csv.csv')
    print("Dataset loaded successfully. Showing first 5 rows:")
    print(data.head())
except FileNotFoundError:
    print("FATAL ERROR: 'Mall_Customers.csv' not found.")
    print("Please download the file and place it in the same folder as this script.")
    exit()

# Select features for clustering: Annual Income and Spending Score
# 'X' will be the feature set for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# --- 3. The Elbow Method for Optimal K ---
# Find the optimal number of clusters by calculating WCSS for K=1 to K=10
wcss = []  # Within-Cluster Sum of Squares
print("\nCalculating WCSS for Elbow Method (K=1 to K=10)...")

for i in range(1, 11):
    # Use 'i' for n_clusters and k-means++ for robust initialization
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) # inertia_ holds the WCSS value

# Plot the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='blue')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# --- 4. Final K-Means Model Training ---
# Based on the Elbow plot for this dataset, K=5 is typically the optimal choice
optimal_k = 5
print(f"\nOptimal K selected: {optimal_k}")

kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)

# Fit the model and predict the cluster for each data point
clusters = kmeans_final.fit_predict(X)

# Add cluster labels and the original features back to the main DataFrame
data['Cluster'] = clusters

# Get the coordinates of the final cluster centers
centers = kmeans_final.cluster_centers_

# Print cluster centers
print("\nCluster Centers (Annual Income vs. Spending Score):")
print(centers)

# --- 5. Cluster Visualization ---
plt.figure(figsize=(10, 7))

# Scatter plot of all data points, colored by their assigned cluster
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='Cluster', data=data, palette='viridis', s=100, legend='full')

# Plot the cluster centroids (the center of each group)
plt.scatter(centers[:, 0], centers[:, 1], s=300, c='red', marker='X', label='Centroids')

plt.title(f'Customer Segmentation using K-Means (K={optimal_k})')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()

# --- 6. Save Results ---
# Save the original data plus the new 'Cluster' column to a new CSV file
data.to_csv('Mall_Customers_with_clusters.csv', index=False)
print("\nResults saved to 'Mall_Customers_with_clusters.csv'")