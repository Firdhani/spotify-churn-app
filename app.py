import streamlit as st
import pandas as pd
import joblib

# Load aset model
model = joblib.load('spotify_churn_model.pkl')
scaler = joblib.load('spotify_scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="Spotify Churn Predictor", page_icon="🎧")

st.title("🎧 Spotify Churn Predictor")
st.write("Gunakan aplikasi ini untuk memprediksi apakah pelanggan akan berhenti berlangganan.")

# Form Input di Sidebar
st.sidebar.header("Input Data Pelanggan")

def user_input_features():
    age = st.sidebar.number_input("Usia", 15, 80, 25)
    listening_time = st.sidebar.number_input("Listening Time (Menit/Minggu)", 0, 5000, 500)
    songs_played = st.sidebar.number_input("Lagu Diputar/Hari", 0, 500, 20)
    skip_rate = st.sidebar.slider("Skip Rate", 0.0, 1.0, 0.2)
    ads_listened = st.sidebar.number_input("Iklan Didengar/Minggu", 0, 100, 5)
    offline_listening = st.sidebar.selectbox("Offline Listening?", [0, 1])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    country = st.sidebar.selectbox("Negara", ["US", "UK", "DE", "IN", "PK", "CA", "AU", "FR"])
    subscription = st.sidebar.selectbox("Tipe Langganan", ["Free", "Premium", "Student", "Family"])
    device = st.sidebar.selectbox("Perangkat", ["Mobile", "Desktop", "Web"])

    data = {
        'age': age,
        'listening_time': listening_time,
        'songs_played_per_day': songs_played,
        'skip_rate': skip_rate,
        'ads_listened_per_week': ads_listened,
        'offline_listening': offline_listening,
        'gender': gender,
        'country': country,
        'subscription_type': subscription,
        'device_type': device
    }
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()

# Tombol Prediksi
if st.button("Prediksi Sekarang"):
    # Encoding otomatis agar sesuai dengan model
    df_encoded = pd.get_dummies(df_input)
    df_final = df_encoded.reindex(columns=model_columns, fill_value=0)
    
    # Scaling
    df_scaled = scaler.transform(df_final)
    
    # Prediksi
    prob = model.predict_proba(df_scaled)[:, 1]
    
    st.subheader("Hasil Prediksi:")
    if prob >= 0.35:
        st.error(f"🚨 **RISIKO TINGGI**: Pelanggan kemungkinan akan CHURN (Peluang: {prob[0]*100:.2f}%)")
    else:
        st.success(f"✅ **AMAN**: Pelanggan kemungkinan akan TETAP BERLANGGANAN (Peluang Churn: {prob[0]*100:.2f}%)")
