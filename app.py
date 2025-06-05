import streamlit as st
import pandas as pd
import pickle

# ===== Konfigurasi Halaman Streamlit (paling atas!) =====
st.set_page_config(layout="centered", page_title="Prediksi Kelayakan Pinjaman")
st.title("📊 Prediksi Kelayakan Peminjaman")

# ===== Load Model, Scaler, Encoder =====
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# ===== Prediksi Kelayakan Pinjaman =====
st.sidebar.header("📝 Input Data Pinjaman")
gender = st.sidebar.selectbox("Jenis Kelamin", ["Male", "Female"])
married = st.sidebar.selectbox("Status Pernikahan", ["Yes", "No"])
dependents = st.sidebar.selectbox("Jumlah Tanggungan", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Pendidikan", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Wiraswasta", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Pendapatan Pemohon (1000 = 1 juta)", 0, 100000, 6000)
coapplicant_income = st.sidebar.number_input("Pendapatan Pasangan (1000 = 1 juta) ", 0, 100000, 2000)
loan_amount = st.sidebar.number_input("Jumlah Pinjaman (1000 = 1 juta)", 0, 10000, 180)
loan_term = st.sidebar.selectbox("Lama Pinjaman (dalam hari)", [360, 120, 180, 240, 300, 84, 60])
credit_history = st.sidebar.selectbox("Riwayat Kredit", [1.0, 0.0])
property_area = st.sidebar.selectbox("Area Properti", ["Urban", "Rural", "Semiurban"])

if st.sidebar.button("Prediksi Kelayakan Pinjaman"):
    import numpy as np
    # 1. Buat DataFrame
    new_applicant = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }
    df_new = pd.DataFrame([new_applicant])

    # 2. Encode fitur kategorikal
    cols_to_encode = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    for col in cols_to_encode:
        df_new[col] = encoders[col].transform(df_new[col])

    # 3. Rekayasa fitur
    df_new['Total_Income'] = df_new['ApplicantIncome'] + df_new['CoapplicantIncome']
    df_new['ApplicantIncome_to_LoanAmount'] = df_new['LoanAmount'] / (df_new['Total_Income'] + 1)
    df_new['ApplicantIncome_to_LoanAmount'] = (
        df_new['ApplicantIncome_to_LoanAmount']
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    # 4. Standarisasi fitur numerik
    numerik = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    df_new[numerik] = scaler.transform(df_new[numerik])

    # 5. Susun urutan kolom
    try:
        df_new = df_new[model.feature_names_in_]
    except Exception:
        st.error("Urutan kolom tidak sesuai dengan model. Pastikan fitur dan urutan kolom sama seperti saat training.")

    # 6. Prediksi dengan RandomForest
    rf_pred = model.predict(df_new)[0]
    rf_proba = model.predict_proba(df_new)[0]

    st.subheader("=== Hasil Prediksi RandomForest ===")
    if rf_pred == 1:
        st.success("✅ **Status: Disetujui**")
    else:
        st.error("❌ **Status: Ditolak**")

    st.markdown(
        f"""
        <div style="display: flex; gap: 20px;">
            <div style="background: #f8d7da; padding: 15px; border-radius: 8px; flex: 1;">
                <b style="color:  #721c24;">Probabilitas Ditolak (0):</b>
                <span style="font-size: 1.5em; color: #721c24;">{rf_proba[0]:.2f}</span>
            </div>
            <div style="background: #d4edda; padding: 15px; border-radius: 8px; flex: 1;">
                <b style="color:  #155724;">Probabilitas Disetujui (1):</b>
                <span style="font-size: 1.5em; color: #155724;">{rf_proba[1]:.2f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
