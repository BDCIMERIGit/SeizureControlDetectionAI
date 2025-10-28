# =====================================================
# 🧠 SeizureDetect.AI — Ensemble 3 Model + Majority Voting
# =====================================================

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from collections import Counter

st.set_page_config(page_title="Halo Sahabat!", layout="centered")

# =====================================================
# 🎨 Custom CSS Styling
# =====================================================
st.markdown("""
    <style>
    /* ====== GLOBAL BACKGROUND ====== */
    .stApp {
        background-color: #ffffff !important;  /* Putih */
        color: #808080 !important;              /* Teks utama hitam */
        font-family: 'Helvetica', sans-serif;
    }

    /* ====== BUTTON STYLE (navy blue dengan teks putih) ====== */
    div.stButton > button {
        background-color: #001f3f !important;   /* Navy */
        color: #ffffff !important;              /* Putih */
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6em 1.2em !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    div.stButton > button:hover {
        background-color: #003366 !important;   /* Biru lebih terang saat hover */
        transform: translateY(-2px);
    }

    /* ====== FORM FIELD ====== */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 6px !important;
        border: 1px solid #ccc !important;
        padding: 8px !important;
        background-color: #f9f9f9 !important;
        color: #000000 !important;
    }

    /* ====== HEADER & TITLES ====== */
    h1, h2, h3, h4 {
        color: #001f3f !important;
        font-weight: 700 !important;
    }

    /* ====== SIDEBAR ====== */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important; /* Putih */
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

# Pilih metadata acuan
ref_meta = None
if len(metadatas) > 0:
    ref_meta = list(metadatas.values())[0]

# Default fallback
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
            encoded[col] = 0
    return encoded


# =====================================================
# 3️⃣ Session Initialization
# =====================================================
if "users" not in st.session_state:
    st.session_state["users"] = {
        "drrafli": {
            "name": "Dr Rafli",
            "instansi": "RS Contoh",
            "email": "drrafli@example.com",
            "phone": "08123456789",
            "password": "123456"
        }
    }

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["username"] = None

if "history" not in st.session_state:
    st.session_state["history"] = []

if "page" not in st.session_state:
    st.session_state["page"] = "home"

def go_to(page): st.session_state["page"] = page


# =====================================================
# 4️⃣ Navigasi Halaman
# =====================================================
PAGES = ["home", "auth_choice", "register", "login", "form", "history"]

# =====================================================
# 5️⃣ UI Halaman
# =====================================================
if st.session_state["page"] == "home":
    st.title("Halo Sahabat!")
    st.markdown("### Selamat Datang di Aplikasi SeizureDetect.AI!")
    if st.button("Mulai Aplikasi"):
        go_to("auth_choice")

elif st.session_state["page"] == "auth_choice":
    st.header("Apakah Anda sudah punya akun?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"): go_to("login")
    with col2:
        if st.button("Register"): go_to("register")
    if st.button("Kembali ke Beranda"): go_to("home")

elif st.session_state["page"] == "register":
    st.header("Registrasi Akun Baru")
    with st.form("register_form"):
        name = st.text_input("Nama Lengkap")
        instansi = st.text_input("Instansi / Rumah Sakit")
        email = st.text_input("Email")
        phone = st.text_input("Nomor Telepon")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
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
                "password": password
            }
            st.success("Registrasi berhasil! Silakan login.")
            go_to("login")
    if st.button("Kembali"): go_to("auth_choice")

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
            go_to("form")
        else:
            st.error("Username atau password salah.")
    if st.button("Kembali"): go_to("auth_choice")

elif st.session_state["page"] == "form":
    if not st.session_state["logged_in"]:
        st.warning("Silakan login terlebih dahulu.")
        go_to("login")
    else:
        st.subheader("Masukkan Data Pasien")

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

                    st.session_state["history"].append({
                        **input_data,
                        **{f"{k}_pred": LABELS[v] for k, v in preds.items()},
                        "Final Prediction": LABELS[vote_result]
                    })

        if st.button("Lihat Riwayat"):
            go_to("history")

        if st.button("Logout"):
            st.session_state["logged_in"] = False
            go_to("login")

elif st.session_state["page"] == "history":
    st.header("Riwayat Prediksi")
    if len(st.session_state["history"]) == 0:
        st.info("Belum ada riwayat.")
    else:
        df_hist = pd.DataFrame(st.session_state["history"])
        st.dataframe(df_hist)
    if st.button("Kembali"):
        go_to("form")

st.markdown("---")
st.caption("Developed with ❤️ by Dr. Rafli, AISeeyou, & BDC IMERI | Ensemble Epilepsy Prediction Model (XGB + DT + RF)")




















