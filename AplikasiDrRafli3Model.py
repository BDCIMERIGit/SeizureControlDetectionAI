# =====================================================
# 🧠 SeizureDetect.AI — Ensemble 3 Model + Majority Voting
# Updated: Dashboard, Profile, Diagnosis flow
# =====================================================

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt

st.set_page_config(page_title="Halo Sahabat!", layout="centered")

# =====================================================
# 🎨 Custom CSS Styling (with fade-in animation)
# =====================================================
st.markdown("""
    <style>
    /* ====== GLOBAL BACKGROUND ====== */
    .stApp {
        background-color: #e4e4e4 !important;
        color: #000000 !important;
        font-family: 'Helvetica', sans-serif;
    }

    /* ====== FADE-IN ANIMATION ====== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 1.2s ease-in-out;
    }

    /* ====== BUTTON STYLE ====== */
    div.stButton > button {
        background-color: #001f3f !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6em 1.2em !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    div.stButton > button:hover {
        background-color: #003366 !important;
        transform: translateY(-2px);
    }

    /* ====== FORM FIELD ====== */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 6px !important;
        border: 1px solid #ccc !important;
        padding: 8px !important;
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    /* ====== HEADER & TITLES ====== */
    h1, h2, h3, h4 {
        color: #001f3f !important;
        font-weight: 700 !important;
    }

    /* ====== SIDEBAR ====== */
    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
    }

    /* ====== STATUS BOXES ====== */
    .stSuccess {
        background-color: #e6f7ff !important;
        border-left: 5px solid #001f3f !important;
        color: #001f3f !important;
    }
    .stWarning {
        background-color: #fff8e6 !important;
        border-left: 5px solid #ffcc00 !important;
        color: #7a6000 !important;
    }
    .stError {
        background-color: #ffe6e6 !important;
        border-left: 5px solid #cc0000 !important;
        color: #660000 !important;
    }

    /* ====== DATAFRAME STYLE ====== */
    .stDataFrame {
        border-radius: 10px !important;
        border: 1px solid #e0e0e0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* ====== LAYOUT CENTERING ====== */
    .centered-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 85vh;
        text-align: center;
    }

    .bottom-caption {
        position: fixed;
        bottom: 10px;
        left: 0;
        width: 100%;
        text-align: center;
        font-size: 0.85rem;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================
# 1️⃣ Load Model dan Metadata
# =====================================================
@st.cache_resource
def load_models_and_metadata():
    model_files = {
        "XGBoost": "bestmodel_xgb_drRafli.pkl",
        "Decision Tree": "bestmodel_dt_drRafli.pkl",
        "Random Forest": "bestmodel_rf_drRafli.pkl"
    }

    metadata_files = {
        "XGBoost": "xgb_model_metadata.pkl",
        "Decision Tree": "dt_model_metadata.pkl",
        "Random Forest": "rf_model_metadata.pkl"
    }

    models, metadatas = {}, {}
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    models[name] = pickle.load(f)
            except Exception as e:
                st.warning(f"Gagal memuat model {name}: {e}")

    for name, path in metadata_files.items():
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    metadatas[name] = pickle.load(f)
            except Exception as e:
                st.warning(f"Gagal memuat metadata {name}: {e}")

    return models, metadatas


models, metadatas = load_models_and_metadata()

ref_meta = list(metadatas.values())[0] if len(metadatas) > 0 else None
FEATURE_ORDER = ref_meta["FEATURE_ORDER"] if ref_meta else [
    'Jenis Kelamin',
    'Usia saat ini (Kategorik)',
    'Usia Terdiagnosis',
    'Jumlah OAE yang diminum',
    'Golongan Obat yang Dipakai',
    'Jenis Epilepsi',
    'Hasil Pemeriksaan EEG',
    'Hasil Pemeriksaan MRI',
    'OAE Sesuai Protokol'
]
MANUAL_ENCODING = ref_meta["MANUAL_ENCODING"] if ref_meta else {}
LABELS = {0: "Penanganan tidak terkontrol", 1: "Penanganan terkontrol"}

# =====================================================
# 2️⃣ Helper Functions
# =====================================================
def normalize_manual_encoding(manual_encoding):
    norm = {}
    for col, mapping in manual_encoding.items():
        norm[col] = {str(k).strip(): v for k, v in mapping.items()}
    return norm


MANUAL_ENCODING = normalize_manual_encoding(MANUAL_ENCODING)


def encode_input(data_dict, metadata):
    encoded = {}
    enc_map = metadata.get("MANUAL_ENCODING", {})
    for col, val in data_dict.items():
        if col in enc_map:
            encoded[col] = enc_map[col].get(str(val).strip(), 0)
        else:
            # try numeric conversion
            try:
                encoded[col] = float(val)
            except Exception:
                encoded[col] = 0
    return encoded


# =====================================================
# 3️⃣ Session Initialization
# =====================================================
if "users" not in st.session_state:
    # Default profile requested by user
    st.session_state["users"] = {
        "drachmad": {
            "name": "dr. Achmad Rafli, Sp.A(K)",
            "instansi": "RS Cipto Mangunkusumo",
            "email": "achmad.rafli@rs-cipto.go.id",
            "phone": "081234567890",
            "password": "123456",
            "jadwal": "Sabtu. 13:00 - 16:00. 16:00 - 19:00. Minggu. 08:00 - 10:30"
        }
    }

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["username"] = None

if "history" not in st.session_state:
    # history will store diagnosis records
    st.session_state["history"] = []

if "page" not in st.session_state:
    st.session_state["page"] = "home"


def go_to(page):
    st.session_state["page"] = page


# =====================================================
# 4️⃣ Pages list
# =====================================================
PAGES = ["home", "auth_choice", "register", "login", "dashboard", "profile", "diagnosis", "history"]


# =====================================================
# Utility: top nav on dashboard/profile pages
# =====================================================
def dashboard_nav():
    cols = st.columns(4)
    with cols[0]:
        if st.button("Dashboard"):
            go_to("dashboard")
    with cols[1]:
        if st.button("Profile"):
            go_to("profile")
    with cols[2]:
        if st.button("Diagnosis"):
            go_to("diagnosis")
    with cols[3]:
        if st.button("Riwayat Diagnosis"):
            go_to("history")


# =====================================================
# 5️⃣ UI Halaman
# =====================================================
if st.session_state["page"] == "home":
    st.markdown("""
        <div class="centered-container fade-in">
            <h1>Halo Sahabat!</h1>
            <h2>Selamat Datang di Aplikasi SeizureDetect.AI!</h2>
            <p><i>Experimental App untuk prediksi penanganan kejang</i></p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Mulai Aplikasi"):
            go_to("auth_choice")

    st.markdown("""
        <div class="bottom-caption fade-in">
            Developed with ❤️ by Dr. Rafli, AISeeyou, & BDC IMERI | 
            Ensemble Epilepsy Prediction Model (XGB + DT + RF)
        </div>
    """, unsafe_allow_html=True)


# =====================================================
# Auth choice / register / login
# =====================================================
elif st.session_state["page"] == "auth_choice":
    st.header("Apakah Anda sudah punya akun?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            go_to("login")
    with col2:
        if st.button("Register"):
            go_to("register")
    if st.button("Kembali ke Beranda"):
        go_to("home")

elif st.session_state["page"] == "register":
    st.header("Registrasi Akun Baru")
    with st.form("register_form"):
        name = st.text_input("Nama Lengkap")
        instansi = st.text_input("Instansi / Rumah Sakit")
        email = st.text_input("Email")
        phone = st.text_input("Nomor Telepon")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        jadwal = st.text_input("Jadwal Praktek (opsional)")
        submitted = st.form_submit_button("Daftar")

    if submitted:
        if username in st.session_state["users"]:
            st.error("Username sudah digunakan.")
        else:
            st.session_state["users"][username] = {
                "name": name,
                "instansi": instansi,
                "email": email,
                "phone": phone,
                "password": password,
                "jadwal": jadwal
            }
            st.success("Registrasi berhasil! Silakan login.")
            go_to("login")
    if st.button("Kembali"):
        go_to("auth_choice")

elif st.session_state["page"] == "login":
    st.header("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Masuk")

    if submitted:
        user = st.session_state["users"].get(username)
        if user and user["password"] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success(f"Selamat datang, {user['name']}!")
            go_to("dashboard")
        else:
            st.error("Username atau password salah.")
    if st.button("Kembali"):
        go_to("auth_choice")


# =====================================================
# Dashboard
# =====================================================
elif st.session_state["page"] == "dashboard":
    if not st.session_state["logged_in"]:
        st.warning("Silakan login terlebih dahulu.")
        go_to("login")
    else:
        user = st.session_state["users"].get(st.session_state["username"], {})
        st.title("Dashboard")
        dashboard_nav()
        st.markdown("---")

        # Profile summary
        st.subheader("Ringkasan Profil Dokter")
        col1, col2 = st.columns([2, 3])
        with col1:
            st.write("**Nama:**", user.get("name", "-"))
            st.write("**Instansi:**", user.get("instansi", "-"))
            st.write("**Jadwal Praktek:**", user.get("jadwal", "-"))
        with col2:
            # Diagnosis history summary (table)
            st.write("**Ringkasan Riwayat Diagnosis Terakhir**")
            if len(st.session_state["history"]) == 0:
                st.info("Belum ada riwayat diagnosis.")
            else:
                # show recent 5
                df_recent = pd.DataFrame(st.session_state["history"]).tail(5)
                st.dataframe(df_recent)

        st.markdown("---")

        # Trend chart (dummy data)
        st.subheader("Trend Bulanan: Jumlah Pasien yang Didiagnosis")
        months = ['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt']
        counts = [3,5,7,10,5,15,18,25,4,11]

        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(months, counts, marker='o')
        ax.set_title('Jumlah pasien per bulan')
        ax.set_ylabel('Jumlah pasien')
        ax.set_xlabel('Bulan')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)

        st.markdown("---")
        st.write("Gunakan menu di atas untuk mengakses Profile, Diagnosis, atau Riwayat Diagnosis.")


# =====================================================
# Profile page
# =====================================================
elif st.session_state["page"] == "profile":
    if not st.session_state["logged_in"]:
        st.warning("Silakan login terlebih dahulu.")
        go_to("login")
    else:
        user = st.session_state["users"].get(st.session_state["username"], {})
        st.title("Profile")
        dashboard_nav()
        st.markdown("---")
        st.write("**Nama:**", user.get('name','-'))
        st.write("**Instansi:**", user.get('instansi','-'))
        st.write("**Jadwal Praktek:**", user.get('jadwal','-'))
        st.write("**Email:**", user.get('email','-'))
        st.write("**No HP:**", user.get('phone','-'))
        st.markdown("---")
        if st.button("Kembali ke Dashboard"):
            go_to("dashboard")


# =====================================================
# Diagnosis (form) - reuse form logic
# =====================================================
elif st.session_state["page"] == "diagnosis":
    if not st.session_state["logged_in"]:
        st.warning("Silakan login terlebih dahulu.")
        go_to("login")
    else:
        st.title("Diagnosis - Masukkan Data Pasien")
        dashboard_nav()
        st.sidebar.header("🔧 Model & Metadata")
        st.sidebar.write(f"Model terdeteksi: {len(models)} / 3")

        with st.form("input_form"):
            input_data = {}
            for key in FEATURE_ORDER:
                if key in MANUAL_ENCODING:
                    choices = list(MANUAL_ENCODING[key].keys())
                    input_data[key] = st.selectbox(key, choices)
                else:
                    input_data[key] = st.text_input(f"{key} (manual)")
            submitted = st.form_submit_button("🔍 Prediksi")

        if submitted:
            if len(models) == 0 or ref_meta is None:
                st.error("Tidak ada model atau metadata ditemukan.")
            else:
                encoded = encode_input(input_data, ref_meta)
                X_input = pd.DataFrame([[encoded.get(c, 0) for c in FEATURE_ORDER]], columns=FEATURE_ORDER)

                st.subheader("📊 Hasil Prediksi Tiap Model")
                preds = {}
                for name, model in models.items():
                    try:
                        pred = int(model.predict(X_input)[0])
                        preds[name] = pred
                        st.write(f"🔹 **{name}:** {LABELS[pred]}")
                    except Exception as e:
                        st.warning(f"Gagal prediksi dengan {name}: {e}")

                if preds:
                    votes = list(preds.values())
                    vote_result = Counter(votes).most_common(1)[0][0]
                    st.markdown("---")
                    st.subheader("🗳️ Hasil Majority Voting:")
                    st.success(LABELS[vote_result])
                    st.markdown("---")

                    # Simpan ke history (tambahkan timestamp sederhana)
                    record = {
                        **input_data,
                        **{f"{k}_pred": LABELS[v] for k, v in preds.items()},
                        "Final Prediction": LABELS[vote_result]
                    }
                    st.session_state["history"].append(record)

        if st.button("Kembali ke Dashboard"):
            go_to("dashboard")


# =====================================================
# Riwayat Diagnosis (history)
# =====================================================
elif st.session_state["page"] == "history":
    if not st.session_state["logged_in"]:
        st.warning("Silakan login terlebih dahulu.")
        go_to("login")
    else:
        st.title("Riwayat Diagnosis")
        dashboard_nav()
        st.markdown("---")
        if len(st.session_state["history"]) == 0:
            st.info("Belum ada riwayat.")
        else:
            df_hist = pd.DataFrame(st.session_state["history"])
            st.dataframe(df_hist)
        if st.button("Kembali ke Dashboard"):
            go_to("dashboard")


# Footer caption
st.markdown("---")
st.caption("Developed with ❤️ by Dr. Rafli, AISeeyou, & BDC IMERI | Ensemble Epilepsy Prediction Model (XGB + DT + RF)")
