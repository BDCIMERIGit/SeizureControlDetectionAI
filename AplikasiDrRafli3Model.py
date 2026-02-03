# =========================== V3 ======================== #
# =====================================================
# 🧠 SeizureDetect.AI — Ensemble 3 Model + Majority Voting
# Updated: Dashboard, Profile, Diagnosis flow (modified)
# =====================================================

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt
import re
from datetime import datetime

st.set_page_config(page_title="Halo Sahabat!", layout="centered")

# =====================================================
# 🖼️ Header Logo Section
# =====================================================
col1, col2 = st.columns([1,1])
with col1:
    st.image("logo/logo-ui-fk-imeri.png", width=200)
with col2:
    st.image("logo/logo-RSCM.png", width=200)


#st.markdown("""
#<div class="logo-header">
#    <img src="logo/logo-ui-fk-imeri.png" height="45">
#    <img src="logo/logo-RSCM.png" height="45">
#</div>
#""", unsafe_allow_html=True)

# =====================================================
# 🎨 Custom CSS Styling (with fade-in animation) + modifications
# =====================================================
st.markdown("""
    <style>
    /* ====== GLOBAL BACKGROUND ====== */
    .stApp {
        background-color: #e7f8ff !important;
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

    /* ====== BUTTON STYLE ======
       Note: We style primary Streamlit buttons to appear navy for top nav and prediction
    */
    # div.stButton > button {
    #     background-color: #001f3f !important; /* navy */
    #     color: #ffffff !important;             /* white text before hover */
    #     border: none !important;
    #     border-radius: 6px !important;
    #     padding: 0.5em 1em !important;
    #     font-weight: 600 !important;
    #     transition: all 0.2s ease !important;
    # }

    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background-color: #ffffff !important;  /* putih */
        color: #001f3f !important;             /* teks biru navy */
        border: 2px solid #001f3f !important;  /* biar tetap tegas */
        border-radius: 8px !important;
        padding: 0.6em 1.2em !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    # div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    #     background-color: #003366 !important;
    #     transform: translateY(-2px);
    # }
    
    div.stButton > button:hover {
        background-color: #001f3f !important;  /* balik ke navy saat hover */
        color: #ffffff !important;             /* teks jadi putih */
        transform: translateY(-2px);
    }

    /* Specific: make login form submit button text white initially and slightly change on hover */
    form button {
        color: #ffffff !important;
    }

    /* ====== FORM FIELD ====== */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > input {
        border-radius: 6px !important;
        border: 1px solid #ccc !important;
        padding: 8px !important;
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    /* Ensure labels (form field labels) are black by default (special for Login) */
    label {
        color: #000000 !important;
        font-weight: 600;
    }

    /* Navy colored labels for specific diagnosis fields */
    .navy-label {
        color: #001f3f;
        font-weight: 700;
        margin-bottom: 6px;
        display:block;
        font-size: 0.95rem;
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
        /* background-color: #e6f7ff !important; */
        background-color: #d0e7ff !important;
        border-left: 5px solid #001f3f !important;
        color: #001f3f !important;
    }

    /* ====== CUSTOM ALERT (untuk st.success) ====== */
    div[data-testid="stNotification"] {
        background-color: #D0E7FF !important;   /* biru muda */
        border-left: 5px solid #001f3f !important;
        color: #001f3f !important;
    }

    /* (opsional) agar teks di dalamnya tetap kontras */
    div[data-testid="stNotification"] p {
        color: #001f3f !important;
    }

    div[data-testid="stNotification"][aria-label="Success"] {
        background-color: #D0E7FF !important;
        border-left: 5px solid #001f3f !important;
        color: #001f3f !important;
    }


    
    .stWarning {
        background-color: #fff8e6 !important; /* keknya ini */
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

    /* ====== NAV GROUP (visual grouping for top nav buttons) ====== */
    .nav-group-container .stButton > button {
        background-color: #001f3f !important;
        color: #ffffff !important;
        border-radius: 0 !important;
        padding: 0.6rem 0.8rem !important;
        margin: 0 !important;
    }
    /* create white separators by adding right border to each button except last */
    .nav-group-container .stButton > button {
        border-right: 3px solid #ffffff !important;
    }
    /* remove right border for last nav button visually by targeting nth-child via inline wrapper below is used */
    /* fallback: we will add a small spacer after last in layout */

    .nav-wrapper {
        display:flex;
        justify-content:center;
        align-items:center;
        margin-bottom: 8px;
        border-radius:8px;
        overflow:hidden;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }

    /* ====== WELCOME BOX ====== */
    .welcome-box {
        background-color: #d9f1ff;
        color: #001f3f;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #7fc9ff;
        font-weight: 600;
    }

    /* ====== CENTERED LAYOUT ====== */
    .centered-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 85vh;
        text-align: center;
    }

    /* ====== HIDE BOTTOM CAPTION ON HOME (we will conditionally not render it in Python) ====== */
    .bottom-caption {
        display: none;
    }

    /* make small adjustment for dashboard title */
    .dashboard-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #001f3f;
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
    """
    Encode input values into numeric features expected by models.
    Handles manual encodings and numeric conversions.
    """
    encoded = {}
    enc_map = metadata.get("MANUAL_ENCODING", {})
    for col, val in data_dict.items():
        if col in enc_map:
            # if val is already a category string (from selectbox) or we already mapped numeric to a category
            encoded[col] = enc_map[col].get(str(val).strip(), 0)
        else:
            # try numeric conversion
            try:
                encoded[col] = float(val)
            except Exception:
                encoded[col] = 0
    return encoded

def try_map_numeric_to_category(col, numeric_value, manual_enc_map):
    """
    Heuristic: given a numeric value and the manual encoding keys (strings),
    attempt to find which category string the numeric value belongs to.
    Support patterns:
      - 'a-b' or 'a - b'  => inclusive range
      - '<a' or '< a'
      - '>a' or '> a'
      - '>=a', '<=a'
      - exact ints like '0', '1'
    Returns the matched category string or None.
    """
    if numeric_value is None:
        return None
    try:
        x = float(numeric_value)
    except:
        return None

    for cat in manual_enc_map.keys():
        s = cat.strip()
        # range like "1-5" or "1 - 5"
        rng = re.findall(r'(-?\d+\.?\d*)\s*-\s*(-?\d+\.?\d*)', s)
        if rng:
            a = float(rng[0][0]); b = float(rng[0][1])
            if a <= x <= b:
                return cat
        # <=, >=
        m = re.match(r'^(<=|>=)\s*(-?\d+\.?\d*)$', s)
        if m:
            op = m.group(1); val = float(m.group(2))
            if op == "<=" and x <= val:
                return cat
            if op == ">=" and x >= val:
                return cat
        # < or >
        m2 = re.match(r'^(<|>)\s*(-?\d+\.?\d*)$', s)
        if m2:
            op = m2.group(1); val = float(m2.group(2))
            if op == "<" and x < val:
                return cat
            if op == ">" and x > val:
                return cat
        # patterns like '0-1 year' => try to find numbers inside
        rng2 = re.findall(r'(-?\d+\.?\d*)', s)
        if len(rng2) == 2 and ('-' in s):
            a = float(rng2[0]); b = float(rng2[1])
            if a <= x <= b:
                return cat
        # exact match numeric label
        try:
            if float(s) == x:
                return cat
        except:
            pass
    return None

# =====================================================
# 3️⃣ Session Initialization
# =====================================================
if "users" not in st.session_state:
    # Default profile requested by user
    st.session_state["users"] = {
        "drrafli": {
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
    # Visual grouping: use a horizontal container; buttons remain Streamlit buttons for functionality.
    # We will place them in 4 equal columns so visually they appear adjacent.
    cols = st.columns(4)
    labels = [("Home","home"), ("Profile","profile"), ("Diagnosis","diagnosis"), ("Riwayat Diagnosis","history")]
    for c, (label, page_name) in zip(cols, labels):
        with c:
            if st.button(label):
                go_to(page_name)

# =====================================================
# 5️⃣ UI Halaman
# =====================================================
if st.session_state["page"] == "home":
    st.markdown("""
        <div class="centered-container fade-in">
            <h1>Halo Sahabat!</h1>
            <h2>Selamat Datang di Aplikasi PediEpiAI!</h2>
            <p><i>Experimental App untuk prediksi penanganan kejang</i></p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Mulai Aplikasi"):
            go_to("auth_choice")

    # NOTE: sesuai permintaan, tulisan yang fixed tidak ditampilkan pada halaman awal.
    # (Kami juga mengatur agar caption footer hanya muncul ketika page != "home" di bagian footer)
    # If you still want it somewhere else, we can re-enable conditionally.

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
    # NOTE: ensure label colors are black (handled by CSS label rule above)
    with st.form("login_form"):
        # Use explicit labels (they will be styled black by the CSS)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Masuk")  # tombol "Masuk" diset putih oleh CSS

    if submitted:
        user = st.session_state["users"].get(username)
        if user and user["password"] == password:
            # show custom welcome box (light blue)
            st.markdown(f'<div class="welcome-box">Selamat datang, {user["name"]}!</div>', unsafe_allow_html=True)
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            # go to dashboard after showing message
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
        # change title text per req
        st.markdown('<div class="dashboard-title">Seizure Control Detection App</div>', unsafe_allow_html=True)

        # visual nav wrapper
        st.markdown('<div class="nav-wrapper nav-group-container">', unsafe_allow_html=True)
        # create nav buttons (they will be visually grouped)
        cols = st.columns([1,1,1,1])
        with cols[0]:
            if st.button("Home"):
                go_to("dashboard")  # Home stays on dashboard
        with cols[1]:
            if st.button("Profile"):
                go_to("profile")
        with cols[2]:
            if st.button("Diagnosis"):
                go_to("diagnosis")
        with cols[3]:
            if st.button("Riwayat Diagnosis"):
                go_to("history")
        st.markdown('</div>', unsafe_allow_html=True)

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

        # Logout button at bottom of dashboard
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Log Out"):
            st.session_state["logged_in"] = False
            st.session_state["username"] = None
            st.success("Anda telah keluar.")
            go_to("login")

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
        # reuse nav
        dashboard_nav()
        st.markdown("---")
        st.write("**Nama:**", user.get('name','-'))
        st.write("**Instansi:**", user.get('instansi','-'))
        st.write("**Jadwal Praktek:**", user.get('jadwal','-'))
        st.write("**Email:**", user.get('email','-'))
        st.write("**No HP:**", user.get('phone','-'))
        st.markdown("---")
        if st.button("Kembali ke Halaman Utama"):
            go_to("dashboard")

# =====================================================
# Diagnosis (form) - reuse form logic but with customization
# =====================================================

# === Custom Display Text for Golongan Obat ===
GOLONGAN_DISPLAY_MAP = {
    "Golongan 1": "Golongan 1: Asam valproat + Levetirasetam",
    "Golongan 2": "Golongan 2: Asam valproat + Topiramat",
    "Golongan 3": "Golongan 3: Asam valproat + Karbamazepin/Okskarbazepin ",
    "Golongan 4": "Golongan 4: Asam valproat + Levetirasetam + Topiramat",
    "Golongan 5": "Golongan 5: Asam valproat + Levetirasetam + Klobazam"
}

# Reverse map untuk mengembalikan pilihan user ke key asli
GOLONGAN_REVERSE_MAP = {v: k for k, v in GOLONGAN_DISPLAY_MAP.items()}

# elif st.session_state["page"] == "diagnosis":
#     if not st.session_state["logged_in"]:
#         st.warning("Silakan login terlebih dahulu.")
#         go_to("login")
#     else:
#         st.title("Diagnosis - Masukkan Data Pasien")
#         # top nav
#         dashboard_nav()
#         st.sidebar.header("🔧 Model & Metadata")
#         st.sidebar.write(f"Model terdeteksi: {len(models)} / 3")

#         with st.form("input_form"):
#             input_data = {}
#             # Build inputs: show navy-label for specific fields requested
#             for key in FEATURE_ORDER:
#                 # For the two usia fields, we present number inputs (and later map to categories)
#                 if key in ["Usia saat ini (Kategorik)", "Usia Terdiagnosis"]:
#                     st.markdown(f'<label class="navy-label">{key}</label>', unsafe_allow_html=True)
#                     # assume ages are integers >= 0
#                     val = st.number_input(f"", min_value=0, step=1, key=f"field_{key}")
#                     input_data[key] = val
#                 #else:
#                     # if key is in MANUAL_ENCODING, we present selectbox using mapping keys
#                     # if key in MANUAL_ENCODING:
#                     #     choices = list(MANUAL_ENCODING[key].keys())
#                     #     st.markdown(f'<label class="navy-label">{key}</label>', unsafe_allow_html=True)
#                     #     val = st.selectbox("", choices, key=f"field_{key}")
#                     #     input_data[key] = val

#                     elif key == "Golongan Obat yang Dipakai":
#                         st.markdown(f'<label class="navy-label">{key}</label>', unsafe_allow_html=True)
    
#                         choices = [GOLONGAN_DISPLAY_MAP[c] for c in MANUAL_ENCODING[key].keys()]  # tampilkan versi panjang
    
#                         selected_display = st.selectbox("", choices, key=f"field_{key}")
    
#                         # simpan value asli sesuai metadata
#                         input_data[key] = GOLONGAN_REVERSE_MAP[selected_display]
#                     elif:
#                         choices = list(MANUAL_ENCODING[key].keys())
#                         st.markdown(f'<label class="navy-label">{key}</label>', unsafe_allow_html=True)
#                         val = st.selectbox("", choices, key=f"field_{key}")
#                         input_data[key] = val

#                     else:
#                         # fallback text input (render navy label if requested fields)
#                         label_style = 'navy-label' if key in ['Jenis Kelamin','Jumlah OAE yang diminum','Golongan Obat yang Dipakai','Jenis Epilepsi','Hasil Pemeriksaan EEG','Hasil Pemeriksaan MRI'] else ''
#                         if label_style:
#                             st.markdown(f'<label class="navy-label">{key}</label>', unsafe_allow_html=True)
#                         else:
#                             st.markdown(f'<label>{key}</label>', unsafe_allow_html=True)
#                         val = st.text_input("", key=f"field_{key}")
#                         input_data[key] = val

#             # Prediction button text set to "Prediksi" per request
#             submitted = st.form_submit_button("Prediksi")

#         # Post-process numeric age fields to map into categorical labels (if metadata mapping exists)
#         # Use heuristics to pick matching category string from MANUAL_ENCODING (if available)
#         if submitted:
#             if len(models) == 0 or ref_meta is None:
#                 st.error("Tidak ada model atau metadata ditemukan.")
#             else:
#                 # Build a copy of input_data_for_encoding where numeric age fields are translated into category keys if possible
#                 input_for_encoding = input_data.copy()
#                 for age_key in ["Usia saat ini (Kategorik)", "Usia Terdiagnosis"]:
#                     if age_key in input_for_encoding:
#                         numval = input_for_encoding[age_key]
#                         # try to map numeric to existing MANUAL_ENCODING categories (if available)
#                         if age_key in MANUAL_ENCODING and isinstance(numval, (int, float)):
#                             mapped_cat = try_map_numeric_to_category(age_key, numval, MANUAL_ENCODING[age_key])
#                             if mapped_cat is not None:
#                                 input_for_encoding[age_key] = mapped_cat
#                             else:
#                                 # If can't map, fallback to nearest: choose first category (graceful fallback)
#                                 # (Alternatively you could create an explicit rule mapping)
#                                 input_for_encoding[age_key] = str(int(numval))
#                         else:
#                             input_for_encoding[age_key] = str(int(numval))

#                 encoded = encode_input(input_for_encoding, ref_meta)
#                 X_input = pd.DataFrame([[encoded.get(c, 0) for c in FEATURE_ORDER]], columns=FEATURE_ORDER)

#                 st.subheader("📊 Hasil Prediksi Tiap Model")
#                 preds = {}
#                 for name, model in models.items():
#                     try:
#                         # ensure model predict works with our X_input
#                         pred = int(model.predict(X_input)[0])
#                         preds[name] = pred
#                         st.write(f"🔹 **{name}:** {LABELS[pred]}")
#                     except Exception as e:
#                         st.warning(f"Gagal prediksi dengan {name}: {e}")

#                 if preds:
#                     votes = list(preds.values())
#                     vote_result = Counter(votes).most_common(1)[0][0]
#                     st.markdown("---")
#                     st.subheader("🗳️ Hasil Majority Voting:")
#                     st.success(LABELS[vote_result])
#                     st.markdown("---")

#                     # Simpan ke history (tambahkan timestamp sederhana)
#                     record = {
#                         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                         **input_data,
#                         **{f"{k}_pred": LABELS[v] for k, v in preds.items()},
#                         "Final Prediction": LABELS[vote_result]
#                     }
#                     st.session_state["history"].append(record)

#         if st.button("Kembali ke Halaman Utama"):
#             go_to("dashboard")

if st.session_state["page"] == "diagnosis":
    if not st.session_state["logged_in"]:
        st.warning("Silakan login terlebih dahulu.")
        go_to("login")
    else:
        st.title("Diagnosis - Masukkan Data Pasien")
        # top nav
        dashboard_nav()
        st.sidebar.header("🔧 Model & Metadata")
        st.sidebar.write(f"Model terdeteksi: {len(models)} / 3")

        with st.form("input_form"):
            input_data = {}
            # Build inputs: show navy-label for specific fields requested
            for key in FEATURE_ORDER:
                # 1) For the two usia fields, present number inputs (later map to categories)
                if key in ["Usia saat ini (Kategorik)", "Usia Terdiagnosis"]:
                    st.markdown(f'<label class="navy-label">{key}</label>', unsafe_allow_html=True)
                    # assume ages are integers >= 0
                    val = st.number_input("", min_value=0, step=1, key=f"field_{key}")
                    input_data[key] = val

                # 2) Specific: "Golongan Obat yang Dipakai" — show verbose labels but save original key
                elif key == "Golongan Obat yang Dipakai":
                    st.markdown(f'<label class="navy-label">{key}</label>', unsafe_allow_html=True)

                    # get raw keys from MANUAL_ENCODING if available, otherwise fallback to GOLONGAN_DISPLAY_MAP keys
                    if key in MANUAL_ENCODING:
                        raw_keys = list(MANUAL_ENCODING[key].keys())
                    else:
                        raw_keys = list(GOLONGAN_DISPLAY_MAP.keys())

                    # create choices to display (use display map if available, otherwise use the raw key itself)
                    choices = [GOLONGAN_DISPLAY_MAP.get(k, k) for k in raw_keys]

                    selected_display = st.selectbox("", choices, key=f"field_{key}")

                    # map back to original metadata key (e.g. "Golongan 1"). fallback to first raw_key.
                    input_data[key] = GOLONGAN_REVERSE_MAP.get(selected_display, raw_keys[0])

                # 3) If key exists in MANUAL_ENCODING → normal selectbox with those keys
                elif key in MANUAL_ENCODING:
                    st.markdown(f'<label class="navy-label">{key}</label>', unsafe_allow_html=True)
                    choices = list(MANUAL_ENCODING[key].keys())
                    val = st.selectbox("", choices, key=f"field_{key}")
                    input_data[key] = val

                # 4) Fallback: text input
                else:
                    label_style = 'navy-label' if key in [
                        'Jenis Kelamin','Jumlah OAE yang diminum','Golongan Obat yang Dipakai',
                        'Jenis Epilepsi','Hasil Pemeriksaan EEG','Hasil Pemeriksaan MRI'
                    ] else ''
                    if label_style:
                        st.markdown(f'<label class="{label_style}">{key}</label>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<label>{key}</label>', unsafe_allow_html=True)
                    val = st.text_input("", key=f"field_{key}")
                    input_data[key] = val

            # Prediction button text set to "Prediksi" per request
            submitted = st.form_submit_button("Prediksi")

        # Post-process numeric age fields to map into categorical labels (if metadata mapping exists)
        # Use heuristics to pick matching category string from MANUAL_ENCODING (if available)
        if submitted:
            if len(models) == 0 or ref_meta is None:
                st.error("Tidak ada model atau metadata ditemukan.")
            else:
                # Build a copy of input_data_for_encoding where numeric age fields are translated into category keys if possible
                input_for_encoding = input_data.copy()
                for age_key in ["Usia saat ini (Kategorik)", "Usia Terdiagnosis"]:
                    if age_key in input_for_encoding:
                        numval = input_for_encoding[age_key]
                        # try to map numeric to existing MANUAL_ENCODING categories (if available)
                        if age_key in MANUAL_ENCODING and isinstance(numval, (int, float)):
                            mapped_cat = try_map_numeric_to_category(age_key, numval, MANUAL_ENCODING[age_key])
                            if mapped_cat is not None:
                                input_for_encoding[age_key] = mapped_cat
                            else:
                                # If can't map, fallback to nearest: choose first category (graceful fallback)
                                input_for_encoding[age_key] = str(int(numval))
                        else:
                            input_for_encoding[age_key] = str(int(numval))

                encoded = encode_input(input_for_encoding, ref_meta)
                X_input = pd.DataFrame([[encoded.get(c, 0) for c in FEATURE_ORDER]], columns=FEATURE_ORDER)

                st.subheader("📊 Hasil Prediksi Tiap Model (Data Testing)")
                preds = {}
                for name, model in models.items():
                    try:
                        # ensure model predict works with our X_input
                        pred = int(model.predict(X_input)[0])
                        preds[name] = pred
                        st.write(f"🔹 **{name}:** {LABELS[pred]}")
                    except Exception as e:
                        st.warning(f"Gagal prediksi dengan {name}: {e}")

                if preds:
                    votes = list(preds.values())
                    vote_result = Counter(votes).most_common(1)[0][0]
                    st.markdown("---")
                    st.subheader("🗳️ Hasil Majority Voting (Data Testing):")
                    st.success(LABELS[vote_result])
                    st.markdown("---")

                    # Simpan ke history (tambahkan timestamp sederhana)
                    record = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        **input_data,
                        **{f"{k}_pred": LABELS[v] for k, v in preds.items()},
                        "Final Prediction": LABELS[vote_result]
                    }
                    st.session_state["history"].append(record)

        if st.button("Kembali ke Halaman Utama"):
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
        if st.button("Kembali ke Halaman Utama"):
            go_to("dashboard")

# Footer caption (tampilkan hanya jika bukan di halaman 'home')
if st.session_state.get("page", "") != "home":
    st.markdown("---")
    st.caption("Developed with ❤️ by Dr. Rafli, AISeeyou, & BDC IMERI | Ensemble Epilepsy Prediction Model (XGB + DT + RF)")



































