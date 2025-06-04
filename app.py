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

# ===== Sidebar Input =====
st.sidebar.header("📝 Input Data Nasabah")

gender = st.sidebar.selectbox("Jenis Kelamin", ["Male", "Female"])
married = st.sidebar.selectbox("Status Pernikahan", ["Yes", "No"])
dependents = st.sidebar.selectbox("Jumlah Tanggungan", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Pendidikan", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Wiraswasta", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Pendapatan Pemohon", 0, 100000, 6000)
coapplicant_income = st.sidebar.number_input("Pendapatan Pasangan", 0, 100000, 2000)
loan_amount = st.sidebar.number_input("Jumlah Pinjaman", 0, 1000, 180)
loan_term = st.sidebar.selectbox("Lama Pinjaman (dalam hari)", [360, 120, 180, 240, 300, 84, 60])
credit_history = st.sidebar.selectbox("Riwayat Kredit", [1.0, 0.0])
property_area = st.sidebar.selectbox("Area Properti", ["Urban", "Rural", "Semiurban"])

# ===== Prediksi =====
if st.sidebar.button("Prediksi Kelayakan"):
    # Buat dataframe input
    input_data = {
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

    input_df = pd.DataFrame([input_data])

    # Encode kolom kategorikal
    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[[col]])

    # Scale kolom numerik
    numerical_features = scaler.feature_names_in_
     # Tambah fitur turunan
    input_df["Total_Income"] = input_df["ApplicantIncome"] + input_df["CoapplicantIncome"]
    input_df["ApplicantIncome_to_LoanAmount"] = input_df["ApplicantIncome"] / (input_df["LoanAmount"] + 1)

    # Susun ulang kolom agar sesuai urutan saat training
    input_df = input_df[model.feature_names_in_]

    # Prediksi
    prediction = model.predict(input_df)[0]
    result = "✅ Layak" if prediction == 1 else "❌ Tidak Layak"
    st.success(f"Hasil Prediksi: **{result}**")
