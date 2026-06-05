import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="K-Means Clustering", layout="wide")

st.title("🛍️ Mall Customer Segmentation using K-Means Clustering")

# Load dataset directly from data folder
df = pd.read_csv("data/Mall_Customers.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Features used for clustering
features = ["Annual Income (k$)", "Spending Score (1-100)"]

X = df[features]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
st.subheader("Elbow Method")

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss, marker='o')
ax.set_title("Elbow Method")
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("WCSS")

st.pyplot(fig)

# Select K
k = st.slider("Select Number of Clusters", 2, 10, 5)

# Train Model
model = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X_scaled)

# Clustered Data
st.subheader("Clustered Dataset")
st.dataframe(df.head())

# Visualization
st.subheader("Cluster Visualization")

fig2, ax2 = plt.subplots(figsize=(8, 6))

scatter = ax2.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis",
    s=70
)

ax2.set_xlabel("Annual Income (k$)")
ax2.set_ylabel("Spending Score (1-100)")
ax2.set_title("Customer Segments")

plt.colorbar(scatter)

st.pyplot(fig2)

# Cluster Summary
st.subheader("Cluster Statistics")
st.dataframe(
    df.groupby("Cluster")[features].mean()
)

st.success("K-Means Clustering Completed Successfully!")