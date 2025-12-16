import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# 1. KONFIGURASI HALAMAN

st.set_page_config(
    page_title="Sistem Cerdas Harga Pangan",
    page_icon="🌾",
    layout="wide"
)


# 2. CSS (FIX TOTAL WARNA & TAMPILAN)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* 1. PAKSA BACKGROUND APLIKASI JADI TERANG */
    .stApp {
        background-color: #f4f7f6;
    }
    
    /* 2. PAKSA SEMUA TEKS MENJADI HITAM/GELAP (SOLUSI NAMBAH LABEL) */
    .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp label, .stApp span, .stApp div {
        color: #2c3e50 !important;
    }

    /* Khusus Label di atas Input (Tahun, Konsumsi, dll) */
    div[data-testid="stWidgetLabel"] p {
        color: #2c3e50 !important;
        font-weight: 600;
    }

    /* 3. PERBAIKAN INPUT BOX & DROPDOWN */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div,
    div[data-testid="stNumberInput"] div[data-baseweb="input"] > div {
        background-color: #ffffff !important;
        border: 2px solid #b0b8c4 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }

    /* Teks di dalam Input Box */
    input[type="text"], input[type="number"] {
        color: #2c3e50 !important; 
        -webkit-text-fill-color: #2c3e50 !important;
    }
    
    /* Teks Pilihan di Dropdown */
    div[data-baseweb="select"] span {
        color: #2c3e50 !important;
    }

    /* Efek Hover Input */
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover {
        border-color: #2a5298 !important;
    }

    /* 4. HEADER */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        gap: 20px;
    }
    
    /* Override warna khusus untuk Header agar tetap Putih */
    .main-header h1, .main-header p, .main-header div {
        color: white !important;
    }

    /* 5. CARD PUTIH */
    .custom-card {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #e1e4e8;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }

    /* 6. TOMBOL */
    div.stButton > button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white !important; /* Teks tombol wajib putih */
        border: none;
        padding: 12px 28px;
        font-weight: 600;
        border-radius: 8px;
        transition: transform 0.2s;
    }
    div.stButton > button p {
        color: white !important; /* Paksa teks dalam tombol putih */
    }
    div.stButton > button:hover {
        transform: scale(1.02);
    }

    /* 7. SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    .sidebar-header {
        font-weight: 700;
        color: #1e3c72 !important;
        font-size: 1.1rem;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    /* Perbaikan warna tabel */
    div[data-testid="stDataFrame"] div {
        color: #2c3e50 !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. LOAD DATA

@st.cache_data
def load_data():
    try:
        # Coba load file asli
        df = pd.read_csv("dataset_prediksi_harga_beras_final.csv")
        df = df[df['Tahun'].between(2022, 2024)]
    except Exception: # Tangkap semua error jika file gagal load
        # Data Dummy (Backup Plan)
        np.random.seed(42)
        kabupatens = [
            'Cianjur', 'Karawang', 'Indramayu', 'Subang', 'Garut', 
            'Tasikmalaya', 'Bogor', 'Sukabumi', 'Bandung', 'Cirebon'
        ]
        years = [2022, 2023, 2024]
        
        data = []
        for kab in kabupatens:
            base_price = np.random.randint(12000, 14000)
            for year in years:
                factor = (year - 2022) * 800
                prod_factor = np.random.uniform(0.9, 1.1)
                
                row = {
                    'Kabupaten': kab,
                    'Tahun': year,
                    'Luas_Lahan_Padi_(Ha)': int(np.random.randint(5000, 20000) * prod_factor),
                    'Produktivitas_Tanaman_Padi_(Ku/ha)': round(np.random.uniform(50, 65), 2),
                    'Konsumsi_Beras': int(np.random.randint(3000, 10000)),
                    'Rata_Rata_Harga_Beras': int(base_price + factor + np.random.randint(-500, 500))
                }
                row['Produksi_Padi_(Ton)'] = int((row['Luas_Lahan_Padi_(Ha)'] * row['Produktivitas_Tanaman_Padi_(Ku/ha)']) / 10)
                data.append(row)
        
        df = pd.DataFrame(data)
        
    return df

df = load_data()


# 4. TRAINING MODEL

try:
    df_encoded = pd.get_dummies(df, columns=["Kabupaten"], drop_first=True)
    X = df_encoded.drop("Rata_Rata_Harga_Beras", axis=1) 
    y = df_encoded["Rata_Rata_Harga_Beras"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
except Exception as e:
    st.error(f"Terjadi kesalahan pada pemrosesan data: {e}")
    st.stop()


# 5. SIDEBAR

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933942.png", width=70)
    st.markdown('<div class="sidebar-header">MENU UTAMA</div>', unsafe_allow_html=True)
    
    menu = st.selectbox(
        "Pilih Halaman:",
        ["📈 Visualisasi Data", "🔮 Prediksi Harga", "📂 Data Tahun 2022-2024"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if menu == "📈 Visualisasi Data":
        st.markdown('<div class="sidebar-header">🛠️ PENGATURAN GRAFIK</div>', unsafe_allow_html=True)
        jenis_chart = st.selectbox(
            "Jenis Grafik:",
            ["Line Chart (Tren Waktu)", "Bar Chart (Perbandingan)", "Pie Chart (Proporsi)", "Scatter Plot (Korelasi)"]
        )
        
        list_tahun = sorted(df['Tahun'].unique())
        if jenis_chart != "Line Chart (Tren Waktu)":
            filter_tahun_vis = st.selectbox("Pilih Tahun Data:", list_tahun, index=len(list_tahun)-1)


# 6. HEADER UTAMA

st.markdown("""
<div class="main-header">
    <div style="font-size: 3.5rem;">🌾</div>
    <div>
        <h1>Dashboard Harga Beras</h1>
        <p>Data Periode 2022 - 2024 & Prediksi Cerdas</p>
    </div>
</div>
""", unsafe_allow_html=True)


# 7. LOGIKA KONTEN


# --- MENU 1: VISUALISASI DATA ---
if menu == "📈 Visualisasi Data":
    st.markdown(f'<div class="section-title">Visualisasi: {jenis_chart}</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        if jenis_chart == "Line Chart (Tren Waktu)":
            pilih_kab = st.multiselect("Filter Kabupaten (Opsional):", df['Kabupaten'].unique(), default=df['Kabupaten'].unique()[:5])
            df_trend = df if not pilih_kab else df[df['Kabupaten'].isin(pilih_kab)]
            fig = px.line(df_trend, x="Tahun", y="Rata_Rata_Harga_Beras", color="Kabupaten", markers=True, title="Tren Harga Beras (2022-2024)")
            fig.update_xaxes(type='category')
            st.plotly_chart(fig, use_container_width=True)

        elif jenis_chart == "Bar Chart (Perbandingan)":
            df_vis = df[df['Tahun'] == filter_tahun_vis].sort_values("Rata_Rata_Harga_Beras", ascending=False)
            fig = px.bar(df_vis, x="Kabupaten", y="Rata_Rata_Harga_Beras", color="Rata_Rata_Harga_Beras", color_continuous_scale="Blues", text_auto='.2s', title=f"Harga Beras Tahun {filter_tahun_vis}")
            st.plotly_chart(fig, use_container_width=True)

        elif jenis_chart == "Pie Chart (Proporsi)":
            df_vis = df[df['Tahun'] == filter_tahun_vis]
            fig = px.pie(df_vis, values="Produksi_Padi_(Ton)", names="Kabupaten", title=f"Proporsi Produksi Padi {filter_tahun_vis}", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        elif jenis_chart == "Scatter Plot (Korelasi)":
            df_vis = df[df['Tahun'] == filter_tahun_vis]
            fig = px.scatter(df_vis, x="Produksi_Padi_(Ton)", y="Rata_Rata_Harga_Beras", size="Luas_Lahan_Padi_(Ha)", color="Kabupaten", size_max=50, title=f"Supply vs Harga - Tahun {filter_tahun_vis}")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- MENU 2: PREDIKSI HARGA ---
elif menu == "🔮 Prediksi Harga":
    st.markdown('<div class="section-title">Simulasi & Prediksi</div>', unsafe_allow_html=True)
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        kab_in = st.selectbox("📍 Pilih Kabupaten", df["Kabupaten"].unique())
        thn_in = st.selectbox("📅 Tahun Target", [2025, 2026, 2027])
        
        hist = df[df["Kabupaten"] == kab_in]
        def_luas = float(hist["Luas_Lahan_Padi_(Ha)"].mean()) if not hist.empty else 10000.0
        def_prod = float(hist["Produktivitas_Tanaman_Padi_(Ku/ha)"].mean()) if not hist.empty else 60.0
        
        luas_in = st.number_input("Luas Lahan (Ha)", value=def_luas)
        prod_in = st.number_input("Produktivitas (Ku/Ha)", value=def_prod)
        
    with col2:
        def_total = float(hist["Produksi_Padi_(Ton)"].mean()) if not hist.empty else 50000.0
        def_kons = float(hist["Konsumsi_Beras"].mean()) if not hist.empty else 5000.0
        
        total_in = st.number_input("Total Produksi (Ton)", value=def_total)
        kons_in = st.number_input("Konsumsi (Ton)", value=def_kons)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 PREDIKSI", use_container_width=True):
            input_data = {
                "Tahun": thn_in, 
                "Luas_Lahan_Padi_(Ha)": luas_in, 
                "Produktivitas_Tanaman_Padi_(Ku/ha)": prod_in, 
                "Produksi_Padi_(Ton)": total_in, 
                "Konsumsi_Beras": kons_in
            }
            for col in X.columns:
                if col.startswith("Kabupaten_"):
                    input_data[col] = 1 if col == f"Kabupaten_{kab_in}" else 0
            
            input_df = pd.DataFrame([input_data])[X.columns]
            res = model.predict(scaler.transform(input_df))[0]
            
            st.success(f"Prediksi Harga Beras di {kab_in} tahun {thn_in}:")
            st.markdown(f"<h2 style='color: #2a5298 !important; margin:0;'>Rp {res:,.2f}</h2>", unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)

# --- MENU 3: DATA TAHUN 2022-2024 ---
elif menu == "📂 Data Tahun 2022-2024":
    st.markdown('<div class="section-title">Database Harga Pangan (2022-2024)</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    st.markdown("##### 🔍 Filter Pencarian")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_kab = st.multiselect("Pilih Wilayah (Kosongkan untuk Semua):", options=df["Kabupaten"].unique(), default=[])
    with col_f2:
        filter_tahun = st.multiselect("Pilih Tahun (Kosongkan untuk Semua):", options=sorted(df["Tahun"].unique()), default=[])
    
    df_filtered = df.copy()
    if filter_kab:
        df_filtered = df_filtered[df_filtered["Kabupaten"].isin(filter_kab)]
    if filter_tahun:
        df_filtered = df_filtered[df_filtered["Tahun"].isin(filter_tahun)]
    
    st.markdown("---")
    st.write(f"Menampilkan **{len(df_filtered)}** data:")
    
    tinggi_dinamis = int((len(df_filtered) + 1) * 35 + 3)
    if tinggi_dinamis > 500: tinggi_dinamis = 500
    
    # MENAMPILKAN TABEL TANPA GRADIENT JIKA MATPLOTLIB BELUM ADA
    # (Untuk mencegah error jika Anda belum update requirements.txt)
    try:
        st.dataframe(
            df_filtered.style.format({
                "Rata_Rata_Harga_Beras": "Rp {:,.0f}",
                "Produksi_Padi_(Ton)": "{:,.0f}",
                "Luas_Lahan_Padi_(Ha)": "{:,.0f}",
                "Konsumsi_Beras": "{:,.0f}"
            }).background_gradient(cmap="Blues", subset=["Rata_Rata_Harga_Beras"]),
            use_container_width=True,
            height=tinggi_dinamis
        )
    except ImportError:
        # Jika matplotlib lupa diinstall, tabel tetap muncul (tapi polos)
        st.warning("⚠️ Tips: Tambahkan 'matplotlib' ke requirements.txt agar tabel berwarna.")
        st.dataframe(
            df_filtered.style.format({
                "Rata_Rata_Harga_Beras": "Rp {:,.0f}",
                "Produksi_Padi_(Ton)": "{:,.0f}",
                "Luas_Lahan_Padi_(Ha)": "{:,.0f}",
                "Konsumsi_Beras": "{:,.0f}"
            }),
            use_container_width=True,
            height=tinggi_dinamis
        )
    st.markdown('</div>', unsafe_allow_html=True)
