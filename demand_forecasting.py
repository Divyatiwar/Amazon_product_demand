import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 🔹 Step 1: Load Cleaned Dataset
df = pd.read_csv("amazon_cleaned.csv")

# 🔍 Debug sample
print("Price Nulls:", df["Price"].isnull().sum())
print("Rating Nulls:", df["Rating"].isnull().sum())
print("Reviews Nulls:", df["Reviews"].isnull().sum())
print("PopularityScore Nulls:", df["PopularityScore"].isnull().sum())

print("🔎 Sample Data:")
print(df[["Title", "Price", "Rating", "Reviews", "PopularityScore"]].head(10))

# 🛠️ STEP TO ADD HERE — Simulate reviews if 0
if df["Reviews"].sum() == 0:
    print("⚠️ All reviews are 0. Simulating fake reviews for demo.")
    df["Reviews"] = np.random.randint(50, 5000, size=len(df))

# 🔁 Recalculate PopularityScore
df["PopularityScore"] = df["Rating"] * np.log1p(df["Reviews"])

# 🔹 Step 2: Demand Bucketing
def demand_label(score):
    if score < 20:
        return "Low"
    elif score < 40:
        return "Medium"
    else:
        return "High"

df["DemandLevel"] = df["PopularityScore"].apply(demand_label)

# 🔹 Step 3: KMeans Clustering
features = df[["Price", "Rating", "Reviews", "PopularityScore"]].dropna()
print("✅ Shape of features:", features.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
df.loc[features.index, "DemandCluster"] = kmeans.fit_predict(X_scaled)

df["DemandCluster"] = df["DemandCluster"].map({0: "Medium", 1: "Low", 2: "High"})

# 🔹 Step 4: Save Final
df.to_csv("amazon_demand_forecast.csv", index=False)
print("✅ Final dataset saved with DemandLevel and DemandCluster.")
