# demand_prediction_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Step 1: Load the final dataset
df = pd.read_csv("amazon_demand_forecast.csv")
print("✅ Loaded Data:", df.shape)

# Step 2: Encode DemandLevel to numbers
le = LabelEncoder()
df["DemandLabel"] = le.fit_transform(df["DemandLevel"])
print("📊 Label Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# Step 3: Features and Target
X = df[["Price", "Rating", "Reviews", "PopularityScore"]]
y = df["DemandLabel"]

# Step 4: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train RandomForest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Predict and Evaluate
y_pred = model.predict(X_test)

print("\n✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Step 7: Save the model
joblib.dump(model, "demand_predictor.pkl")
print("\n💾 Model saved as demand_predictor.pkl")
