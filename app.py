import streamlit as st
import pandas as pd
import pickle
import numpy as np
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

# ===== Navbar =====
selected_menu = st.selectbox(
    "Navigasi",
    [
        "Prediksi Kelayakan Pinjaman",
        "Tentang Aplikasi",
        "Petunjuk Penggunaan"
    ],
    key="navbar"
)

if selected_menu == "Prediksi Kelayakan Pinjaman":
    # ===== Prediksi Kelayakan Pinjaman =====
    st.header("📝 Input Data Pinjaman")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        married = st.selectbox("Status Pernikahan", ["Yes", "No"])
        dependents = st.selectbox("Jumlah Tanggungan", ["0", "1", "2", "3+"])
        education = st.selectbox("Pendidikan", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Wiraswasta", ["Yes", "No"])
    with col2:
        applicant_income = st.number_input("Pendapatan Pemohon (1000 = 1 juta)", 0, 100000, 6000)
        coapplicant_income = st.number_input("Pendapatan Pasangan (1000 = 1 juta) ", 0, 100000, 2000)
        loan_amount = st.number_input("Jumlah Pinjaman (1000 = 1 juta)", 0, 10000, 180)
        loan_term = st.selectbox("Lama Pinjaman (dalam hari)", [360, 120, 180, 240, 300, 84, 60])
        credit_history = st.selectbox("Riwayat Kredit", [1.0, 0.0])
        property_area = st.selectbox("Area Properti", ["Urban", "Rural", "Semiurban"])

    if st.button("Prediksi Kelayakan Pinjaman"):
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
        ApplicantIncome_to_LoanAmount = df_new['LoanAmount'] / (df_new['ApplicantIncome'] + 1)
        ApplicantIncome_to_LoanAmount = (
            ApplicantIncome_to_LoanAmount
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )
        Total_Income = df_new['ApplicantIncome'] + df_new['CoapplicantIncome']
        df_new['Total_Income'] = df_new['ApplicantIncome'] + df_new['CoapplicantIncome']
        df_new['ApplicantIncome_to_LoanAmount'] = df_new['LoanAmount'] / (Total_Income + 1)
        df_new['ApplicantIncome_to_LoanAmount'] = ApplicantIncome_to_LoanAmount

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

        # 7. Tampilkan ringkasan input dan fitur turunan
        st.subheader("Ringkasan Data Input")
        st.write("**Data yang Anda masukkan:**")

        st.table({
            'Jenis Kelamin': gender,
            'Status Pernikahan': married,
            'Jumlah Tanggungan': dependents,
            'Pendidikan': education,
            'Wiraswasta': self_employed,
            'Pendapatan Pemohon': applicant_income,
            'Pendapatan Pasangan': coapplicant_income,
            'Jumlah Pinjaman': loan_amount,
            'Lama Pinjaman (hari)': loan_term,
            'Riwayat Kredit': credit_history,
            'Area Properti': property_area,
            'Total Pendapatan': Total_Income.values[0],
            'Rasio Pinjaman terhadap Total Pendapatan': round(ApplicantIncome_to_LoanAmount.values[0], 3)
        })

        st.subheader("=== Hasil Prediksi ===")
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

        # 8. Visualisasi tambahan
        st.subheader("Visualisasi Fitur Penting")
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Rasio Pinjaman terhadap Total Pendapatan
        st.markdown("""
        **Rasio Pinjaman terhadap Total Pendapatan**
        <br>
        Grafik ini menunjukkan seberapa besar jumlah pinjaman dibandingkan dengan total pendapatan pemohon dan pasangan. Semakin tinggi rasio, semakin besar beban pinjaman terhadap pendapatan.
        """, unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(4,2))
        ax1.barh(["Rasio Pinjaman/Total Pendapatan"], [ApplicantIncome_to_LoanAmount.values[0]], color="#007bff")
        ax1.set_xlim(0, 1)
        ax1.set_xlabel("Rasio")
        ax1.set_title("Rasio Pinjaman terhadap Total Pendapatan")
        st.pyplot(fig1)

        # Visualisasi komposisi pendapatan
        st.markdown("""
        **Komposisi Pendapatan Pemohon dan Pasangan**
        <br>
        Grafik ini membandingkan besarnya pendapatan pemohon dan pasangan. Informasi ini penting untuk melihat kontribusi masing-masing terhadap total pendapatan keluarga.
        """, unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(4,2))
        ax2.bar(["Pemohon", "Pasangan"], [applicant_income, coapplicant_income], color=["#28a745", "#ffc107"])
        ax2.set_ylabel("Pendapatan (ribu)")
        ax2.set_title("Komposisi Pendapatan")
        st.pyplot(fig2)

        # Visualisasi proporsi pelunasan (Loan/Total Income per tahun)
        st.markdown("""
        **Proporsi Angsuran per Bulan terhadap Total Pendapatan**
        <br>
        Grafik ini menunjukkan seberapa besar porsi angsuran pinjaman per bulan dibandingkan dengan total pendapatan. Nilai yang tinggi dapat menandakan risiko kesulitan pembayaran.
        """, unsafe_allow_html=True)
        pelunasan_per_tahun = (loan_amount * 1000) / ((Total_Income.values[0] + 1) * 12)
        fig3, ax3 = plt.subplots(figsize=(4,2))
        ax3.bar(["Proporsi Pelunasan per Bulan"], [pelunasan_per_tahun], color="#dc3545")
        ax3.set_ylabel("Proporsi (Loan/Income)")
        ax3.set_ylim(0, 1)
        ax3.set_title("Proporsi Angsuran per Bulan terhadap Total Pendapatan")
        st.pyplot(fig3)

elif selected_menu == "Tentang Aplikasi":
    st.header("Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini digunakan untuk memprediksi kelayakan pinjaman berdasarkan data pemohon dan pasangan.
    Model yang digunakan adalah RandomForest yang telah dilatih pada data historis.
    """)

elif selected_menu == "Petunjuk Penggunaan":
    st.header("Petunjuk Penggunaan")
    st.markdown("""
    1. Pilih menu **Prediksi Kelayakan Pinjaman**.
    2. Masukkan data pemohon dan pasangan pada form yang tersedia.
    3. Klik tombol **Prediksi Kelayakan Pinjaman** untuk melihat hasil dan visualisasi.
    4. Lihat hasil prediksi dan insight visualisasi di bawah form.
    """)
