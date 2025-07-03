import pandas as pd
import numpy as np

df = pd.read_csv("amazon_mobiles.csv")
print("🟢 Loaded raw data:", df.shape)

# Clean column names
df.columns = df.columns.str.strip()

# Fix Price
df["Price"] = df["Price"].astype(str).str.replace(r"[^\d.]", "", regex=True)
df = df[df["Price"] != ""]
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# Fix Rating
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

# ⚠️ Replace missing Reviews with 0 (KEY FIX)
df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce").fillna(0)

# Drop only if essential columns are missing
df.dropna(subset=["Title", "Price", "Rating"], inplace=True)

# Feature Engineering
df["PopularityScore"] = df["Rating"] * np.log1p(df["Reviews"])

def price_segment(p):
    if p < 10000:
        return "Budget"
    elif p < 30000:
        return "Mid"
    else:
        return "Premium"

df["PriceSegment"] = df["Price"].apply(price_segment)

# Save
df.to_csv("amazon_cleaned.csv", index=False)
print("✅ Cleaned data saved with", len(df), "rows.")
print(df.head(3))
