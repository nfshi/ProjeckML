import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Sistem Cerdas Harga Pangan",
    page_icon="🌾",
    layout="wide"
)

# ==========================================
# 2. CSS (BACKGROUND GRADASI & PERBAIKAN TAMPILAN)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* 1. UBAH BACKGROUND JADI GRADASI (TIDAK POLOS PUTIH) */
    .stApp {
        background: linear-gradient(to bottom right, #e0f7fa, #e1bee7);
        /* Alternatif: background-color: #e3f2fd; (Biru Muda Solid) */
    }
    
    /* 2. TEKS HITAM/GELAP AGAR KONTRAS DENGAN BACKGROUND */
    .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp label, .stApp span, .stApp div {
        color: #2c3e50 !important;
    }

    /* Khusus Label di atas Input */
    div[data-testid="stWidgetLabel"] p {
        color: #1e3c72 !important;
        font-weight: 700;
    }

    /* 3. PERBAIKAN INPUT BOX */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div,
    div[data-testid="stNumberInput"] div[data-baseweb="input"] > div {
        background-color: #ffffff !important; /* Kotak tetap putih agar bersih */
        border: 2px solid #90a4ae !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
    }

    /* Teks di dalam Input */
    input[type="text"], input[type="number"] {
        color: #2c3e50 !important; 
        -webkit-text-fill-color: #2c3e50 !important;
        font-weight: 600;
    }
    
    div[data-baseweb="select"] span {
        color: #2c3e50 !important;
    }

    /* 4. HEADER */
    .main-header {
        background: linear-gradient(135deg, #006064 0%, #00838f 100%); /* Hijau Teal Gelap */
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        gap: 20px;
        color: white !important;
    }
    
    /* Override warna teks header agar Putih */
    .main-header h1, .main-header p, .main-header div {
        color: white !important;
    }

    /* 5. CARD / KONTAINER PUTIH */
    .custom-card {
        background-color: rgba(255, 255, 255, 0.9); /* Putih sedikit transparan */
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #ffffff;
        box-shadow: 0 8px 16px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    /* 6. TOMBOL */
    div.stButton > button {
        background: linear-gradient(90deg, #006064 0%, #0097a7 100%);
        color: white !important;
        border: none;
        padding: 12px 28px;
        font-weight: 600;
        border-radius: 8px;
        transition: transform 0.2s;
        box-shadow: 0 4px 8px rgba(0,96,100, 0.3);
    }
    div.stButton > button p {
        color: white !important;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
    }

    /* 7. SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #cfd8dc;
    }
    .sidebar-header {
        font-weight: 700;
        color: #006064 !important;
        font-size: 1.1rem;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    /* Tabel */
    div[data-testid="stDataFrame"] {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    try:
        # Coba load file asli
        df = pd.read_csv("dataset_prediksi_harga_beras_final.csv")
        df = df[df['Tahun'].between(2022, 2024)]
    except Exception:
        # Data Dummy (Jika file error)
        np.random.seed(42)
        kabupatens = ['Cianjur', 'Karawang', 'Indramayu', 'Subang', 'Garut', 'Tasikmalaya', 'Bogor']
        years = [2022, 2023, 2024]
        
        data = []
        for kab in kabupatens:
            for year in years:
                row = {
                    'Kabupaten': kab,
                    'Tahun': year,
                    'Luas_Lahan_Padi_(Ha)': np.random.uniform(5000, 20000),
                    'Produktivitas_Tanaman_Padi_(Ku/ha)': np.random.uniform(50, 65),
                    'Konsumsi_Beras': np.random.uniform(1.4, 1.8), # Format Desimal Kecil
                    'Produksi_Padi_(Ton)': np.random.randint(40000, 100000),
                    'Rata_Rata_Harga_Beras': np.random.randint(11000, 14000)
                }
                data.append(row)
        df = pd.DataFrame(data)
    return df

df = load_data()

# ==========================================
# 4. TRAINING MODEL
# ==========================================
try:
    df_encoded = pd.get_dummies(df, columns=["Kabupaten"], drop_first=True)
    X = df_encoded.drop("Rata_Rata_Harga_Beras", axis=1) 
    y = df_encoded["Rata_Rata_Harga_Beras"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
except Exception:
    pass # Silent fail agar UI tetap jalan jika data error

# ==========================================
# 5. SIDEBAR
# ==========================================
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
        jenis_chart = st.selectbox("Jenis Grafik:", ["Line Chart (Tren)", "Bar Chart", "Scatter Plot"])
        
        list_tahun = sorted(df['Tahun'].unique())
        if jenis_chart != "Line Chart (Tren)":
            filter_tahun_vis = st.selectbox("Pilih Tahun:", list_tahun, index=len(list_tahun)-1)

# ==========================================
# 6. HEADER UTAMA
# ==========================================
st.markdown("""
<div class="main-header">
    <div style="font-size: 3.5rem;">🌾</div>
    <div>
        <h1>Dashboard Harga Beras</h1>
        <p>Analisis Data Pangan & Prediksi Cerdas (2022-2024)</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 7. LOGIKA KONTEN
# ==========================================

# --- MENU 1: VISUALISASI ---
if menu == "📈 Visualisasi Data":
    st.markdown(f'<div class="section-title">Visualisasi: {jenis_chart}</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        if jenis_chart == "Line Chart (Tren)":
            fig = px.line(df, x="Tahun", y="Rata_Rata_Harga_Beras", color="Kabupaten", markers=True, title="Tren Harga")
            fig.update_xaxes(type='category')
            st.plotly_chart(fig, use_container_width=True)
        elif jenis_chart == "Bar Chart":
            df_vis = df[df['Tahun'] == filter_tahun_vis].sort_values("Rata_Rata_Harga_Beras", ascending=False)
            fig = px.bar(df_vis, x="Kabupaten", y="Rata_Rata_Harga_Beras", color="Rata_Rata_Harga_Beras", color_continuous_scale="Teal")
            st.plotly_chart(fig, use_container_width=True)
        elif jenis_chart == "Scatter Plot":
            df_vis = df[df['Tahun'] == filter_tahun_vis]
            fig = px.scatter(df_vis, x="Produksi_Padi_(Ton)", y="Rata_Rata_Harga_Beras", size="Luas_Lahan_Padi_(Ha)", color="Kabupaten")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- MENU 2: PREDIKSI HARGA ---
elif menu == "🔮 Prediksi Harga":
    st.markdown('<h3 style="color:#1e3c72;">Simulasi & Prediksi Harga</h3>', unsafe_allow_html=True)
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        kab_in = st.selectbox("📍 Pilih Kabupaten", df["Kabupaten"].unique())
        thn_in = st.selectbox("📅 Tahun Target", [2025, 2026, 2027])
        
        # Ambil rata-rata history untuk default value
        hist = df[df["Kabupaten"] == kab_in]
        def_luas = float(hist["Luas_Lahan_Padi_(Ha)"].mean()) if not hist.empty else 10000.0
        def_prod = float(hist["Produktivitas_Tanaman_Padi_(Ku/ha)"].mean()) if not hist.empty else 60.0
        
        luas_in = st.number_input("Luas Lahan (Ha)", value=def_luas)
        prod_in = st.number_input("Produktivitas (Ku/Ha)", value=def_prod)
        
    with col2:
        def_total = float(hist["Produksi_Padi_(Ton)"].mean()) if not hist.empty else 50000.0
        def_kons = float(hist["Konsumsi_Beras"].mean()) if not hist.empty else 1.500
        
        total_in = st.number_input("Total Produksi (Ton)", value=def_total)
        
        # PERBAIKAN: Input Konsumsi sekarang mendukung desimal (3 angka belakang koma)
        # Label juga sudah diganti menjadi Perkapita
        kons_in = st.number_input(
            "Konsumsi Beras Perkapita (Kg/Tahun)", 
            value=def_kons, 
            format="%.3f", # Format 3 desimal (contoh: 1.613)
            step=0.001
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 HITUNG PREDIKSI", use_container_width=True):
            try:
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
                
                st.success(f"Estimasi Harga Beras di {kab_in} ({thn_in}):")
                st.markdown(f"<h1 style='color: #006064 !important;'>Rp {res:,.0f} / Liter</h1>", unsafe_allow_html=True)
            except Exception as e:
                st.error("Pastikan model sudah terlatih dengan benar.")
            
    st.markdown('</div>', unsafe_allow_html=True)

# --- MENU 3: DATA TAHUN 2022-2024 ---
elif menu == "📂 Data Tahun 2022-2024":
    st.markdown('<h3 style="color:#1e3c72;">Database Harga Pangan</h3>', unsafe_allow_html=True)
    
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_kab = st.multiselect("Filter Wilayah:", options=df["Kabupaten"].unique(), default=[])
    with col_f2:
        filter_tahun = st.multiselect("Filter Tahun:", options=sorted(df["Tahun"].unique()), default=[])
    
    df_filtered = df.copy()
    if filter_kab:
        df_filtered = df_filtered[df_filtered["Kabupaten"].isin(filter_kab)]
    if filter_tahun:
        df_filtered = df_filtered[df_filtered["Tahun"].isin(filter_tahun)]
    
    st.markdown("---")
    
    # TINGGI TABEL OTOMATIS
    tinggi = int((len(df_filtered) + 1) * 35 + 3)
    if tinggi > 500: tinggi = 500
    
    # PERBAIKAN FORMAT TABEL (AGAR TIDAK DIBULATKAN)
    # Konsumsi ditampilkan dengan 3 desimal (contoh: 1.613)
    # Produktivitas 2 desimal
    try:
        st.dataframe(
            df_filtered.style.format({
                "Rata_Rata_Harga_Beras": "Rp {:,.0f}",
                "Produksi_Padi_(Ton)": "{:,.0f}",       # Produksi biasanya bulat (Ton)
                "Luas_Lahan_Padi_(Ha)": "{:,.0f}",      # Luas biasanya bulat
                "Produktivitas_Tanaman_Padi_(Ku/ha)": "{:.2f}", # Ada koma
                "Konsumsi_Beras": "{:.3f}"              # PENTING: Menampilkan 1.613 (3 desimal)
            }).background_gradient(cmap="Teal", subset=["Rata_Rata_Harga_Beras"]),
            use_container_width=True,
            height=tinggi
        )
    except ImportError:
        # Fallback jika matplotlib belum diinstall
        st.dataframe(
            df_filtered.style.format({
                "Rata_Rata_Harga_Beras": "Rp {:,.0f}",
                "Produksi_Padi_(Ton)": "{:,.0f}",
                "Konsumsi_Beras": "{:.3f}" # Tetap menjaga format desimal
            }),
            use_container_width=True,
            height=tinggi
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
