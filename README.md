# Mall Customer Segmentation - Machine Learning

This repository is designed to demonstrate how to segment mall customers into different groups based on their purchasing behavior using **Machine Learning** techniques. Specifically, it uses **K-Means Clustering**, an unsupervised learning algorithm that groups data points (customers) into clusters based on similarities in their features.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Machine Learning Workflow](#machine-learning-workflow)
4. [Usage](#usage)
5. [Libraries and Tools](#libraries-and-tools)
6. [License](#license)

---

## Project Overview

This project aims to segment customers of a mall based on several features such as **Age**, **Annual Income**, and **Spending Score**. By grouping customers into segments, the mall can tailor marketing strategies, personalized offers, and other business decisions based on customer behavior.

The key objective is to use **K-Means Clustering** to find natural groupings (clusters) of customers based on their attributes.

---

## Dataset

The dataset used in this project contains the following columns:

* **CustomerID**: A unique identifier for each customer.
* **Age**: The age of the customer.
* **Annual Income**: The annual income of the customer.
* **Spending Score**: A score assigned to each customer based on their spending habits.

Example dataset:

| CustomerID | Age | Annual Income | Spending Score |
| ---------- | --- | ------------- | -------------- |
| 1          | 25  | 50,000        | 70             |
| 2          | 45  | 100,000       | 30             |
| 3          | 38  | 75,000        | 80             |

This data can be used to identify customer segments like high-income, low-spending customers, or young, high-spending customers.

---

## Machine Learning Workflow

1. **Data Preprocessing**:

   * Import the dataset using `Pandas`.
   * Clean the data (e.g., handle missing values, standardize data).
   * Select relevant features for clustering (Age, Annual Income, Spending Score).

2. **Feature Scaling**:

   * Since K-Means is sensitive to feature scales, **standardization** or **normalization** is performed on the features to bring them to the same scale.

3. **K-Means Clustering**:

   * Implement the **K-Means algorithm** from `sklearn.cluster`.
   * Use the **elbow method** to determine the optimal number of clusters (`k`).
   * Fit the model and predict clusters for each customer.

4. **Cluster Evaluation**:

   * Use metrics like **silhouette score** to evaluate the quality of the clusters.
   * Visualize the clusters using 2D plots (e.g., scatter plots).

5. **Interpretation of Results**:

   * Once clusters are created, analyze the customer segments based on their **Age**, **Income**, and **Spending Score**.
   * Identify meaningful segments (e.g., high-spending young adults, budget-conscious families).

---

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/bhousxrenz/ML-Machine-Learning-.git
cd ML-Machine-Learning-
```

### 2. Install Required Libraries

To install the required Python libraries, run the following command:

```bash
pip install -r requirements.txt
```

### 3. Load the Data

Make sure the dataset (`Mall_Customers.csv`) is located in the appropriate folder (e.g., `data/`).

```python
import pandas as pd

# Load dataset
data = pd.read_csv('data/Mall_Customers.csv')
```

### 4. Preprocess the Data

Clean and scale the data:

```python
from sklearn.preprocessing import StandardScaler

# Select relevant features
features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
```

### 5. Apply K-Means Clustering

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Choose the number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fit K-Means with optimal clusters
optimal_clusters = 5  # Based on the elbow method
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(scaled_features)

# Visualize clusters
plt.scatter(scaled_features[y_kmeans == 0, 0], scaled_features[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(scaled_features[y_kmeans == 1, 0], scaled_features[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(scaled_features[y_kmeans == 2, 0], scaled_features[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(scaled_features[y_kmeans == 3, 0], scaled_features[y_kmeans == 3, 1], s=100, c='purple', label='Cluster 4')
plt.scatter(scaled_features[y_kmeans == 4, 0], scaled_features[y_kmeans == 4, 1], s=100, c='orange', label='Cluster 5')

# Centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', label='Centroids')

plt.title('Customer Segments')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
```

---

## Libraries and Tools

* **Python 3.x**
* **Pandas**: For data manipulation and analysis.
* **Matplotlib**: For visualization of results.
* **Scikit-learn**: For applying machine learning algorithms (e.g., K-Means Clustering).
