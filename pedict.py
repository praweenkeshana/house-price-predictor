import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Page Settings
st.set_page_config(page_title="House Price Predictor", layout="centered")

# Title & Description
st.title("🏡 House Price Prediction Web App")
st.write("ගෙදර විස්තර ලබා දී අනාගත මිල ගණනය කරගන්න.")

# 1. Dataset එක සෑදීම
data = {
    'SquareFeet': [1500, 2000, 1200, 1800, 2500, 3000, 1100, 2200],
    'Bedrooms': [3, 4, 2, 3, 4, 5, 2, 3],
    'AgeYears': [5, 2, 10, 8, 1, 3, 15, 6],
    'Price': [250000, 320000, 180000, 270000, 400000, 480000, 160000, 350000]
}
df = pd.DataFrame(data)

# 2. Model Training
X = df[['SquareFeet', 'Bedrooms', 'AgeYears']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

# 3. Streamlit UI Elements (Inputs)
st.subheader("ගෙදර විස්තර ඇතුළත් කරන්න:")

sqft = st.number_input("Square Feet (sqft):", min_value=500, max_value=10000, value=1800, step=100)
bedrooms = st.slider("Bedrooms ගණන:", min_value=1, max_value=10, value=3)
age = st.slider("ගෙදර වයස (අවුරුදු):", min_value=0, max_value=50, value=5)

# 4. Prediction Button
if st.button("Predict Price 💰"):
    # Convert input to DataFrame with feature names to avoid warnings
    input_data = pd.DataFrame([[sqft, bedrooms, age]], columns=['SquareFeet', 'Bedrooms', 'AgeYears'])
    prediction = model.predict(input_data)
    st.success(f"අනුමාන කළ මිල: **${prediction[0]:,.2f}**")

# 5. Dataset Preview
st.markdown("---")
st.subheader("📊 Dataset Overview")
st.dataframe(df)