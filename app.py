import streamlit as st
import pandas as pd
import joblib
from gtts import gTTS
import requests
import json

# Load model and encoders
model = joblib.load("crop_recommendation_model.pkl")
le_district = joblib.load("district_encoder.pkl")
le_soil = joblib.load("soil_encoder.pkl")
le_crop = joblib.load("crop_encoder.pkl")

# Load district coordinates
with open("karnataka_district_coords.json", "r") as f:
    district_coords = json.load(f)

# Crop metadata (Kannada)
crop_season = {
    "Cotton": "June to October",
    "Ragi": "July to November",
    "Paddy": "June to September",
    "Wheat": "November to March",
    "Maize": "June to October",
    "Sugarcane": "December to January",
    "Sunflower": "August to December"
}

crop_reason_kn = {
    "Cotton": "ಇದು ಬ್ಲಾಕ್ ಮಣ್ಣಿನಲ್ಲಿ ಉತ್ತಮವಾಗಿ ಬೆಳೆಯುತ್ತದೆ ಮತ್ತು ಕಡಿಮೆ ತೇವಾಂಶಕ್ಕೆ ತಕ್ಕದ್ದು.",
    "Ragi": "ಇದು ಕಡಿಮೆ ನೀರಿನಲ್ಲಿ ಬೆಳೆಯುತ್ತದೆ ಮತ್ತು ಹೆಚ್ಚು ಹಾರ್ವೆಸ್ಟ್ ಲಭಿಸುತ್ತದೆ.",
    "Paddy": "ಹೆಚ್ಚು ತೇವಾಂಶ ಮತ್ತು ಮಣ್ಣಿನ ನಾರ್ಮಿಕತೆಯೊಂದಿಗೆ ಉತ್ತಮ ಬೆಳೆಯುತ್ತದೆ.",
    "Wheat": "ಇದು ಶೀತಕಾಲದಲ್ಲಿ ಬೆಳೆಯುತ್ತದೆ ಮತ್ತು ಉಷ್ಣತೆ ಕಡಿಮೆ ಇದ್ದಾಗ ಸೂಕ್ತವಾಗಿದೆ.",
    "Maize": "ಇದು ತಾತ್ಕಾಲಿಕ ಮಳೆಯ ಜೊತೆಗೆ ಉತ್ತಮವಾಗಿ ಬೆಳೆದು ನಬ್ದ ಲಾಭ ಕೊಡುತ್ತದೆ.",
    "Sugarcane": "ಇದು ಉಷ್ಣ ಮಣ್ಣು ಮತ್ತು ನೀರಿನ ಸಾಕಷ್ಟು ಲಭ್ಯತೆ ಇದ್ದಾಗ ಉತ್ತಮವಾಗಿದೆ.",
    "Sunflower": "ಇದು ಉಷ್ಣತೆಯೊಂದಿಗೆ ಬೆಳೆಯುತ್ತದೆ ಮತ್ತು ಕಡಿಮೆ ಹರವೆ ಬೇಕಾದ ಬೆಳೆಯಾಗಿದೆ."
}

# NASA climate data function with debugging
def get_nasa_climate(district, month, district_coords):
    month_map = {
        "January": "JAN", "February": "FEB", "March": "MAR", "April": "APR",
        "May": "MAY", "June": "JUN", "July": "JUL", "August": "AUG",
        "September": "SEP", "October": "OCT", "November": "NOV", "December": "DEC"
    }

    # Normalize key to match JSON
    district_key = district.strip().title()

    if district_key not in district_coords:
        print(f"❌ District '{district}' not found in coordinates.")
        return 30, 60  # fallback if missing

    lat = district_coords[district_key]["lat"]
    lon = district_coords[district_key]["lon"]

    url = f"https://power.larc.nasa.gov/api/temporal/climatology/point?parameters=T2M,RH2M&community=AG&longitude={lon}&latitude={lat}&format=JSON"

    print(f"📡 Fetching from: {url}")

    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()

        print("🌤 Available Parameters:", data["properties"]["parameter"].keys())

        temp = data["properties"]["parameter"]["T2M"][month_map[month]]
        humidity = data["properties"]["parameter"]["RH2M"][month_map[month]]
        return round(temp, 1), round(humidity, 1)

    except Exception as e:
        print("❌ NASA API error:", e)
        return 30, 60


# UI setup
st.set_page_config(page_title="Crop Recommendation", layout="centered")
st.title("🌾 ಬೆಳೆ ಶಿಫಾರಸು ವ್ಯವಸ್ಥೆ (Crop Recommendation System)")

districts = le_district.classes_
soils = le_soil.classes_
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

# User inputs
district = st.selectbox("ಜಿಲ್ಲೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ (Select District)", districts)
soil = st.selectbox("ಮಣ್ಣಿನ ಪ್ರಕಾರವನ್ನು ಆಯ್ಕೆಮಾಡಿ (Select Soil Type)", soils)
month = st.selectbox("ಬೆಳೆಯಲು ಉದ್ದೇಶಿತ ತಿಂಗಳು (Select Sowing Month)", months)
rainfall = st.slider("ಮಳೆಯ ಪ್ರಮಾಣ (Rainfall in mm)", 0, 1000, 500)

# Get climate from NASA API
avg_temp, avg_humidity = get_nasa_climate(district, month, district_coords)

st.write(f"🌡️ ಸರಾಸರಿ ಉಷ್ಣತೆ: **{avg_temp}°C**")
st.write(f"💧 ಸರಾಸರಿ ತೇವಾಂಶ: **{avg_humidity}%**")

# Predict and output
if st.button("Get Recommendation"):
    district_encoded = le_district.transform([district])[0]
    soil_encoded = le_soil.transform([soil])[0]

    input_df = pd.DataFrame(
        [[district_encoded, soil_encoded, avg_temp, rainfall, avg_humidity]],
        columns=["District_encoded", "Soil_encoded", "Temperature_C", "Rainfall_mm", "Humidity_%"]
    )

    # Predict top 3 crops
    probs = model.predict_proba(input_df)[0]
    top_indices = probs.argsort()[-3:][::-1]
    top_crops = le_crop.inverse_transform(top_indices)
    main_crop = top_crops[0]

    st.success(f"🟢 ಶಿಫಾರಸು ಮಾಡಲಾದ ಮುಖ್ಯ ಬೆಳೆ: {main_crop} (Recommended Crop: {main_crop})")

    # Text to Speech
    message = f"ನಿಮಗೆ ಶಿಫಾರಸು ಮಾಡಲಾದ ಬೆಳೆ {main_crop}"
    tts = gTTS(text=message, lang='kn')
    tts.save("output.mp3")
    st.audio("output.mp3")

    # Kannada reason and season
    st.markdown(f"🗓️ **ಬೆಳೆಯುವ ಸಮಯ:** {crop_season.get(main_crop, 'Season info not available')}")
    st.markdown(f"💬 **ಏಕೆ ಈ ಬೆಳೆ?** {crop_reason_kn.get(main_crop, 'ಈ ಬೆಳೆಗೆ ಕಾರಣ ಲಭ್ಯವಿಲ್ಲ')}")

    # Other crop suggestions
    st.write("🌿 **ಇನ್ನಷ್ಟು ಸಾಧ್ಯವಾದ ಬೆಳೆಗಳು (Other Suitable Crops):**")
    for alt_crop in top_crops[1:]:
        st.markdown(f"- {alt_crop}")




# import streamlit as st
# import numpy as np
# import joblib

# # 🚀 Load model and encoders
# model = joblib.load("crop_recommendation_model.pkl")
# le_district = joblib.load("district_encoder.pkl")
# le_soil = joblib.load("soil_encoder.pkl")
# le_crop = joblib.load("crop_encoder.pkl")

# # 🌾 App title
# st.title("🌱 Karnataka Crop Recommendation System")

# # 📌 Inputs
# district = st.selectbox("Select District", le_district.classes_)
# soil_type = st.selectbox("Select Soil Type", le_soil.classes_)
# temperature = st.number_input("Enter Temperature (°C)", min_value=0.0, max_value=60.0, value=30.0)
# rainfall = st.number_input("Enter Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
# humidity = st.number_input("Enter Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)

# # 🎯 Recommend crops
# if st.button("Recommend Crop"):
#     # Encode inputs
#     district_encoded = le_district.transform([district])[0]
#     soil_encoded = le_soil.transform([soil_type])[0]

#     input_features = np.array([[district_encoded, soil_encoded, temperature, rainfall, humidity]])

#     # Predict probabilities
#     probs = model.predict_proba(input_features)[0]
#     top3_idx = probs.argsort()[-3:][::-1]
#     top3_crops = le_crop.inverse_transform(top3_idx)
#     top3_probs = probs[top3_idx]

#     # 🌟 Display top 3 crops
#     st.success("✅ Top 3 Recommended Crops:")
#     for crop, prob in zip(top3_crops, top3_probs):
#         st.write(f"🌿 **{crop}** — Confidence: {prob:.2%}")

#     # 📝 Simple reasoning
#     st.info("🔍 **Reasoning for recommendation:**")
#     reasons = []
#     if rainfall > 200:
#         reasons.append("High rainfall suggests water-loving crops like Paddy or Sugarcane.")
#     if temperature > 35:
#         reasons.append("High temperature favors Cotton or Millets.")
#     if humidity > 70:
#         reasons.append("High humidity benefits Paddy, Coconut or Sugarcane.")
#     if soil_type.lower() in ["black soil", "red soil"]:
#         reasons.append(f"{soil_type} is suitable for Cotton, Millets or Oilseeds.")

#     if reasons:
#         for reason in reasons:
#             st.write(f"👉 {reason}")
#     else:
#         st.write("👉 Conditions are moderate; recommended crops fit general patterns in your district.")

#     # 📊 Optionally: Show raw probas in a chart
#     st.write("---")
#     st.write("📊 **Crop confidence chart:**")
#     st.bar_chart({crop: prob for crop, prob in zip(le_crop.classes_, probs)})





# import streamlit as st
# import pandas as pd
# import joblib

# # Load model + encoders
# model = joblib.load("crop_recommendation_model.pkl")
# le_district = joblib.load("district_encoder.pkl")
# le_soil = joblib.load("soil_encoder.pkl")
# le_crop = joblib.load("crop_encoder.pkl")

# # App title
# st.title("🌾 Karnataka Crop Recommendation System")

# # User inputs
# district = st.selectbox("Select District", le_district.classes_)
# soil_type = st.selectbox("Select Soil Type", le_soil.classes_)
# temperature = st.number_input("Enter Temperature (°C)", min_value=10.0, max_value=45.0, value=25.0)
# rainfall = st.number_input("Enter Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
# humidity = st.number_input("Enter Humidity (%)", min_value=10.0, max_value=100.0, value=60.0)

# # Predict button
# if st.button("Recommend Crop"):
#     # Encode inputs
#     district_enc = le_district.transform([district])[0]
#     soil_enc = le_soil.transform([soil_type])[0]
    
#     # Prepare input
#     input_data = pd.DataFrame([[district_enc, soil_enc, temperature, rainfall, humidity]],
#                               columns=['District_encoded', 'Soil_encoded', 'Temperature_C', 'Rainfall_mm', 'Humidity_%'])
    
#     # Predict
#     pred_enc = model.predict(input_data)[0]
#     pred_crop = le_crop.inverse_transform([pred_enc])[0]
    
#     st.success(f"✅ Recommended Crop: **{pred_crop}**")
