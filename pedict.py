import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Page Configuration
st.set_page_config(page_title="Luxury House Predictor", page_icon="🏡", layout="wide")

# ==========================================
# CUSTOM CSS STYLING
# ==========================================
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Stylish Main Title */
    .main-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(90deg, #22d3ee, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
        text-shadow: 0px 10px 20px rgba(0,0,0,0.3);
    }
    
    .sub-title {
        color: #94a3b8;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    
    /* Custom Input Bar Labels */
    .custom-label {
        font-weight: 800;
        font-size: 1.05rem;
        color: #f8fafc;
        background-color: #334155;
        padding: 8px 14px;
        border-radius: 8px;
        border-left: 4px solid #38bdf8;
        margin-bottom: 8px;
        display: inline-block;
        width: 100%;
    }
    
    /* Green Predict Button */
    div.stButton > button:first-child {
        width: 100%;
        background: linear-gradient(90deg, #16a34a, #22c55e) !important;
        color: white !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        padding: 12px !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(34, 197, 94, 0.6) !important;
        background: linear-gradient(90deg, #15803d, #16a34a) !important;
    }
    
    /* Result Card */
    .result-card {
        background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        color: white;
        font-weight: 700;
        box-shadow: 0 10px 25px rgba(14, 165, 233, 0.3);
        margin-top: 20px;
    }
    
    /* Contact Footer */
    .footer {
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        color: #cbd5e1;
        font-size: 0.95rem;
    }
    .footer a {
        color: #38bdf8;
        text-decoration: none;
        font-weight: 600;
    }
    .footer img {
        vertical-align: middle;
        margin-right: 6px;
        margin-left: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER SECTION
# ==========================================
st.markdown('<div class="main-title">🏡 House Prediction Web App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Machine Learning Valuation & Analytics Engine</div>', unsafe_allow_html=True)

# ==========================================
# MODEL & DATASETUP
# ==========================================
@st.cache_data
def load_data():
    data = {
        'SquareFeet': [1500, 2000, 1200, 1800, 2500, 3000, 1100, 2200, 1600, 2800],
        'Bedrooms': [3, 4, 2, 3, 4, 5, 2, 3, 3, 4],
        'AgeYears': [5, 2, 10, 8, 1, 3, 15, 6, 4, 2],
        'Price': [250000, 320000, 180000, 270000, 400000, 480000, 160000, 350000, 260000, 430000]
    }
    return pd.DataFrame(data)

df = load_data()
X = df[['SquareFeet', 'Bedrooms', 'AgeYears']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

# ==========================================
# INPUT FORM SECTION
# ==========================================
st.subheader("🤖 Property Valuation Calculator")
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="custom-label">📐 Square Feet (sqft)</div>', unsafe_allow_html=True)
    sqft = st.number_input("", min_value=500, max_value=10000, value=1800, step=100, label_visibility="collapsed")
    
    st.markdown('<div class="custom-label">🛏️ Bedrooms ගණන</div>', unsafe_allow_html=True)
    bedrooms = st.slider("", min_value=1, max_value=8, value=3, label_visibility="collapsed")

with col2:
    st.markdown('<div class="custom-label">⏳ ගෙදර වයස (අවුරුදු)</div>', unsafe_allow_html=True)
    age = st.slider(" ", min_value=0, max_value=40, value=5, label_visibility="collapsed")

    st.markdown('<div class="custom-label">🔲 Currency (මුදල් වර්ගය)</div>', unsafe_allow_html=True)
    usd_to_lkr = 300.0  # Exchange Rate
    currency = st.radio("  ", ["USD ($)", "LKR (රු.)"], horizontal=True, label_visibility="collapsed")

# Predict Button (Green)
if st.button("Predict Price 💰"):
    input_df = pd.DataFrame([[sqft, bedrooms, age]], columns=['SquareFeet', 'Bedrooms', 'AgeYears'])
    pred_usd = model.predict(input_df)[0]
    
    if currency == "LKR (රු.)":
        pred_lkr = pred_usd * usd_to_lkr
        formatted_price = f"රු. {pred_lkr:,.2f} LKR"
    else:
        formatted_price = f"${pred_usd:,.2f} USD"
    
    st.markdown(f'''
        <div class="result-card">
            <div style="font-size: 1.1rem; opacity: 0.9;">Estimated House Value</div>
            <div style="font-size: 2rem; margin-top: 5px;">{formatted_price}</div>
        </div>
    ''', unsafe_allow_html=True)

st.markdown("<br><hr>", unsafe_allow_html=True)

# ==========================================
# VISUALIZATIONS & ANALYTICS (TABS)
# ==========================================
st.subheader("📈 Data Visualizations & Analytics")
tab1, tab2 = st.tabs(["📉 Price Trend (Regression Graph)", "🔥 Correlation Heatmap"])

with tab1:
    st.write("**Square Feet vs Price Relationship**")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.regplot(data=df, x='SquareFeet', y='Price', ax=ax1, color='#0284c7', marker='o')
    ax1.set_title("Price vs Square Feet Trend")
    st.pyplot(fig1)

with tab2:
    st.write("**Feature Correlation Matrix**")
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    sns.heatmap(df.corr(), annot=True, cmap="Blues", ax=ax2)
    st.pyplot(fig2)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# DATASET OVERVIEW (TOGGLE BUTTON)
# ==========================================
if 'show_data' not in st.session_state:
    st.session_state.show_data = False

def toggle_data():
    st.session_state.show_data = not st.session_state.show_data

st.button("📊 Toggle Dataset Overview", on_click=toggle_data)

if st.session_state.show_data:
    st.dataframe(df, use_container_width=True)

# ==========================================
# FOOTER / CONTACT DETAILS
# ==========================================
st.markdown('''
    <div class="footer">
        Developer Contact: <b>Praween Keshana</b> <br><br>
        <img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg" width="20" height="20">
        <b>WhatsApp:</b> <a href="https://wa.me/94740998610" target="_blank">0740998610</a>
        <img src="https://upload.wikimedia.org/wikipedia/commons/b/b8/2021_Facebook_icon.svg" width="20" height="20">
        <b>Facebook:</b> <a href="https://www.facebook.com/share/p/1CkaoTsE1v/" target="_blank">Praween Keshana</a>
    </div>
''', unsafe_allow_html=True)