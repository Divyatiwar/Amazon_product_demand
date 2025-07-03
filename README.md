
# 📦 Amazon Product Demand Forecasting

## 🧠 Objective
Predict product demand using scraped Amazon data based on features like:
- Price
- Ratings
- Reviews
- Popularity Score

## 🔧 Tools Used
- Python (Pandas, NumPy, Scikit-learn)
- Selenium for scraping
- Power BI (for dashboard visualization)
- KMeans for clustering demand segments

## 📊 Key Features
- Created `PopularityScore = Rating * log(1 + Reviews)`
- Classified demand into `Low`, `Medium`, `High`
- Clustered products using KMeans into demand buckets
- Prepared dataset for business reporting and forecasting

## 📁 Output Files
- `amazon_demand_forecast.csv`: Final dataset
- `demand_dashboard.pbix`: Interactive dashboard (coming soon)

## ✅ Sample Insights
- Top 10 high-demand products
- Category-wise or price-segment demand
- Rating vs Reviews vs Price vs Demand

