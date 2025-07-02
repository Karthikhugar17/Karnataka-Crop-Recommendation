import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1️⃣ Load dataset
df = pd.read_csv("karnataka_smart_crop_data.csv")

# 2️⃣ Encode categorical features
le_district = LabelEncoder()
le_soil = LabelEncoder()
le_crop = LabelEncoder()

df['District_encoded'] = le_district.fit_transform(df['District'])
df['Soil_encoded'] = le_soil.fit_transform(df['Soil Type'])
df['Crop_encoded'] = le_crop.fit_transform(df['Crop'])

# 3️⃣ Features and target
X = df[['District_encoded', 'Soil_encoded', 'Temperature_C', 'Rainfall_mm', 'Humidity_%']]
y = df['Crop_encoded']

# 4️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6️⃣ Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy on test set: {accuracy:.4f}")
print("\n✅ Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le_crop.classes_))

# 7️⃣ Save model + encoders
joblib.dump(model, "crop_recommendation_model.pkl")
joblib.dump(le_district, "district_encoder.pkl")
joblib.dump(le_soil, "soil_encoder.pkl")
joblib.dump(le_crop, "crop_encoder.pkl")

print("✅ Model + encoders saved!")
