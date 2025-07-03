import pandas as pd

# Load final dataset
df = pd.read_csv("amazon_demand_forecast.csv")

# Filter high demand products
high_demand = df[df["DemandLevel"] == "High"]

# Show top 10 high-demand products
print("🔥 Top High-Demand Products:\n")
print(high_demand[["Title", "Price", "Rating", "Reviews", "DemandLevel"]].head(10))
