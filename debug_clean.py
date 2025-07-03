import pandas as pd
import numpy as np

# Step 1: Load Raw Data
df = pd.read_csv("amazon_mobiles.csv")
print("🟢 Step 1 - Raw data loaded:", df.shape)
print("📄 Columns:", df.columns.tolist())
print(df.head(2))

# Step 2: Strip column names
df.columns = df.columns.str.strip()
print("\n🔍 Step 2 - Cleaned column names:", df.columns.tolist())

# Step 3: Check required columns
required_cols = ["Title", "Price", "Rating", "Reviews"]
missing = [col for col in required_cols if col not in df.columns]
if missing:
    print(f"❌ Missing required columns: {missing}")
    exit()

# Step 4: Clean Price column
df["Price"] = df["Price"].astype(str).str.replace(r"[^\d.]", "", regex=True)
print("\n💸 Step 4 - Price cleaned (first 3):", df["Price"].head(3).tolist())

# Step 5: Remove blank prices
df = df[df["Price"] != ""]
print("🧹 Step 5 - After removing blank prices:", df.shape)

# Step 6: Convert types
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")
print("🔢 Step 6 - Converted to numeric")

# Step 7: Drop rows with missing values
df.dropna(subset=["Title", "Price", "Rating", "Reviews"], inplace=True)
print("✅ Step 7 - After dropna:", df.shape)

# Step 8: Feature Engineering
df["PopularityScore"] = df["Rating"] * np.log1p(df["Reviews"])
df["PriceSegment"] = df["Price"].apply(lambda p: "Budget" if p < 10000 else "Mid" if p < 30000 else "Premium")

# Step 9: Save cleaned data
df.to_csv("amazon_cleaned.csv", index=False)
print("🎯 Step 9 - Cleaned data saved to amazon_cleaned.csv")
print(df.head(3))
