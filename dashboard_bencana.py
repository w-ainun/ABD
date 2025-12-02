import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium import Element
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import re
import os
import glob
import pickle
try:
    import joblib
except Exception:
    joblib = None

# --------------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN & TEMA (TETAP)
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Pusat Komando Bencana",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🚨"
)

# CSS CUSTOM: COMMAND CENTER STYLE + NAVBAR
st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
    
    /* Header Glowing */
    .main-header {
        background: radial-gradient(circle at center, #001f3f 0%, #000000 100%);
        border-bottom: 1px solid #00e5ff; padding: 20px; text-align: center; margin-bottom: 20px;
        box-shadow: 0 5px 25px rgba(0, 229, 255, 0.15);
    }
    .main-header h1 { font-family: 'Courier New', monospace; color: #00e5ff; text-transform: uppercase; letter-spacing: 4px; font-weight: 900; margin: 0; }
    .main-header h3 { color: #ff0055; font-size: 0.9rem; letter-spacing: 2px; margin-top: 5px; text-transform: uppercase; }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(20,20,30,0.9), rgba(10,10,15,0.95));
        border: 1px solid #333; border-left: 3px solid #00e5ff; padding: 15px; border-radius: 6px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); border-color: #00e5ff; box-shadow: 0 0 15px rgba(0, 229, 255, 0.2); }
    .metric-card h4 { color: #8899a6; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 8px; }
    .metric-card h2 { color: #fff; font-size: 1.8rem; margin: 0; font-weight: 700; }

    /* Custom Navbar Buttons */
    .nav-btn {
        width: 100%; padding: 10px; background-color: #111; border: 1px solid #333; color: #888;
        text-align: center; cursor: pointer; border-radius: 5px; transition: 0.3s;
    }
    .nav-btn:hover { border-color: #00e5ff; color: #00e5ff; }
    .nav-active { background-color: #00e5ff !important; color: #000 !important; font-weight: bold; border: 1px solid #00e5ff; }

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 2. STATE MANAGEMENT (CORE LOGIC)
# --------------------------------------------------------------------------------
# Cukup simpan halaman saat ini, tidak perlu lagi menyimpan 'selected_project'
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Clustering' # Default Landing Page

def navigate_to(page):
    st.session_state['current_page'] = page

# --------------------------------------------------------------------------------
# 3. DATA LOADING (TETAP)
# --------------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df_c = pd.read_csv("ringkasan_klaster_gabungan.csv")
    except:
        st.error("❌ ERR: 'ringkasan_klaster_gabungan.csv' missing")
        st.stop()
        
    try:
        df_t = pd.read_csv("tabel_pdf_raw_extracted_v2.csv")
        df_t['Teks_Bersih'] = df_t['Teks_Bersih'].astype(str)
        df_t['raw_text'] = df_t['raw_text'].astype(str)
    except:
        df_t = pd.DataFrame()

    try:
        df_l = pd.read_csv("lstm_predictions_with_context.csv")
        date_cols = ['Predicted_Date_Step1', 'Predicted_Date_Step2', 'Predicted_Date_Step3']
        for col in date_cols:
            df_l[col] = pd.to_datetime(df_l[col])
    except:
        df_l = pd.DataFrame()
        
    return df_c, df_t, df_l

df_cluster, df_text, df_lstm_raw = load_data()

# Pre-processing LSTM Helper
def process_lstm_for_viz(df):
    if df.empty: return pd.DataFrame()
    df_viz = df[['Predicted_Date_Step1', 'Actual_Rumah_Terendam_Step1', 'Predicted_Rumah_Terendam_Step1', 'Input_TotalHujan_3Hari_Day7']].copy()
    df_viz.rename(columns={'Predicted_Date_Step1': 'Tanggal', 'Actual_Rumah_Terendam_Step1': 'Aktual', 'Predicted_Rumah_Terendam_Step1': 'Prediksi', 'Input_TotalHujan_3Hari_Day7': 'Hujan_Input_Terakhir'}, inplace=True)
    return df_viz.sort_values('Tanggal')

df_lstm_clean = process_lstm_for_viz(df_lstm_raw)

# Pre-processing NLP Helper
def extract_text_insights(df):
    if df.empty: return pd.DataFrame()
    results = []
    logistics_map = {"Pangan": ["makanan", "sembako", "nasi", "air", "mie"], "Shelter": ["selimut", "tenda", "terpal"], "Medis": ["obat", "vitamin", "masker"], "Evakuasi": ["perahu", "karet", "pelampung"]}
    cause_map = {"Hujan Ekstrem": ["hujan deras", "lebat"], "Sungai Meluap": ["sungai", "luapan"], "Tanggul Jebol": ["tanggul"], "Drainase": ["drainase", "gorong"]}
    
    for _, row in df.iterrows():
        txt = row['Teks_Bersih'].lower()
        raw = row['raw_text']
        # Simple extraction logic (simplified for brevity)
        house = 0
        m_rumah = re.findall(r"(\d+)\s*(?:unit|rumah)", txt)
        if m_rumah: house = max([int(x) for x in m_rumah])
        
        jiwa = 0
        m_jiwa = re.findall(r"(\d+)\s*(?:jiwa|orang|kk)", txt)
        if m_jiwa: jiwa = max([int(x) for x in m_jiwa])

        needs = [k for k, v in logistics_map.items() if any(x in txt for x in v)]
        causes = [k for k, v in cause_map.items() if any(x in txt for x in v)]
        
        match_kec = re.search(r"Kecamatan\s+([A-Z][a-z]+)", raw)
        kec = match_kec.group(1) if match_kec else "-"

        if house > 0 or jiwa > 0 or needs:
            results.append({"Tanggal": row.get('tanggal_pdf', '-'), "Kecamatan": kec, "Rumah_Rusak": house, "Jiwa_Terdampak": jiwa, "Penyebab": ",".join(causes), "Logistik": ",".join(needs)})
            
    return pd.DataFrame(results)

df_insights = extract_text_insights(df_text)

# --------------------------------------------------------------------------------
# 4. COMPONENTS VIEW (NAVBAR)
# --------------------------------------------------------------------------------

# Header (title and subtitle removed as requested)

# Custom Navbar Logic
col_nav1, col_nav2, col_nav3 = st.columns(3)

def nav_button(col, label, page_name, icon):
    active_class = "nav-active" if st.session_state['current_page'] == page_name else ""
    if col.button(f"{icon} {label}", key=f"nav_{page_name}", use_container_width=True, type="primary" if active_class else "secondary"):
        navigate_to(page_name)
        st.rerun()

nav_button(col_nav1, "CLUSTERING (RISIKO)", "Clustering", "🗺️")
nav_button(col_nav2, "PREDIKSI (LSTM)", "LSTM", "📈")
nav_button(col_nav3, "INTELIJEN (NLP)", "NLP", "🧠")

st.markdown("---")

# --------------------------------------------------------------------------------
# 5. HALAMAN 1: CLUSTERING (DEFAULT LANDING PAGE)
# --------------------------------------------------------------------------------
def render_clustering_page():
    # Top Metrics (hapus card TOTAL LOKASI dan STATUS RISIKO dari atas)
    danger_idx = df_cluster['Mengungsi'].idxmax()
    danger_row = df_cluster.loc[danger_idx]
    danger_label = int(danger_row['cluster_label'])
    max_hujan = df_cluster['RR'].max()
    
    if max_hujan > 150: status_txt, status_color = "KRITIS", "#ff0055"
    elif max_hujan > 100: status_txt, status_color = "WASPADA", "#ffcc00"
    else: status_txt, status_color = "STABIL", "#00ff99"

    # removed top metric cards (STATUS RISIKO removed)
    def card(t, v, s, c, size_h2="1.8rem"):
        return f'<div class="metric-card" style="border-left: 3px solid {c};"><h4 style="color:{c}">{t}</h4><h2 style="font-size:{size_h2};">{v}</h2><div style="font-size:11px;color:#888;">{s}</div></div>'
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Map lebih besar, dan panel NLP di samping (taruh samping dan perbesar peta)
    col_map, col_stat = st.columns([4, 1])  # map diperbesar (4:1)
    
    with col_map:
        st.markdown("")
        m = folium.Map(location=[-6.95, 107.63], zoom_start=11, tiles='CartoDB dark_matter')
        
        # Improved color mapping: jelas untuk kuning/merah/hijau
        def color_for_cluster(cluster_id):
            if cluster_id == danger_label:
                return '#ff0055'   # merah untuk zona bahaya
            elif cluster_id == 1:
                return '#ffcc00'   # kuning untuk klaster 1 (lebih terlihat)
            else:
                return '#00ff99'   # hijau untuk lainnya
        
        for _, row in df_cluster.iterrows():
            cluster_id = int(row['cluster_label'])
            count = max(1, int(row.get('Jumlah_Kejadian', 1)))
            color = color_for_cluster(cluster_id)
            
            # Scatter dummy points (kejar visual distribusi)
            np.random.seed(cluster_id + 7)
            c_lat, c_lon = -6.95, 107.60 
            if cluster_id == 0: c_lat -= 0.05
            if cluster_id == 1: c_lon += 0.05
            if cluster_id == 2: c_lat += 0.04; c_lon += 0.02
            
            lats = np.random.normal(c_lat, 0.02, count)
            lons = np.random.normal(c_lon, 0.02, count)
            
            for i in range(count):
                folium.CircleMarker(
                    [lats[i], lons[i]],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.85,  # tingkat visibilitas lebih tinggi
                    popup=f"Klaster {cluster_id}"
                ).add_to(m)
                
        st_folium(m, width="100%", height=650)

    with col_stat:
        # TARUH ZONA BAHAYA & RUMAH RUSAK DI SAMPING PETA (tanpa judul "RINGKASAN (Samping Peta)")
        total_rumah = int(df_insights['Rumah_Rusak'].sum()) if not df_insights.empty else 0
        st.markdown(card("ZONA BAHAYA", f"Klaster {danger_label}", "Risiko Tertinggi", "#ff0055"), unsafe_allow_html=True)
        st.markdown(card("RUMAH RUSAK", f"{total_rumah:,}", "Laporan NLP", "#aa00ff", size_h2="2.2rem"), unsafe_allow_html=True)
        
        st.markdown("---")
        # tetap tampilkan chart kecil di samping (tanpa header "Ringkasan Peta")
        fig = px.bar(df_cluster, x='cluster_label', y='RR', color='cluster_label', color_discrete_map={0:'#00ff99',1:'#ffcc00',2:'#ff0055'}, title=None)
        fig.update_layout(template="plotly_dark", showlegend=False, height=220, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)
        
        fig2 = px.pie(df_cluster, values='Jumlah_Kejadian', names=df_cluster['cluster_label'].apply(lambda x: f"K-{x}"), hole=0.5)
        fig2.update_traces(marker=dict(colors=[ '#00ff99', '#ffcc00', '#ff0055' ]))
        fig2.update_layout(template="plotly_dark", showlegend=False, height=220, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # Tambahkan panel samping untuk statistik NLP
    col_nlp, col_empty = st.columns([2, 1])
    
    with col_nlp:
        st.markdown("### 🧠 ANALISIS TEKS (NLP)")
        # Statistik dasar dari df_insights
        st.markdown(f"**Total Laporan Teks:** {len(df_insights):,}")
        if not df_insights.empty:
            st.markdown(f"**Total Rumah Rusak (ekstraksi teks):** {int(df_insights['Rumah_Rusak'].sum()):,}")
            st.markdown(f"**Total Jiwa Terdampak (ekstraksi teks):** {int(df_insights['Jiwa_Terdampak'].sum()):,}")
        
        st.markdown("---")
        st.markdown("#### Kebutuhan Logistik")
        all_logs = []
        for item in df_insights['Logistik']:
            if item: all_logs.extend(item.split(','))
        if all_logs:
            counts = Counter(all_logs)
            df_log = pd.DataFrame(counts.items(), columns=['Item', 'Jml'])
            fig_log = px.bar(df_log.sort_values('Jml', ascending=False), x='Item', y='Jml', title=None)
            fig_log.update_layout(template="plotly_dark", showlegend=False, height=250, margin=dict(l=0,r=0,b=0,t=0))
            st.plotly_chart(fig_log, use_container_width=True)
        else:
            st.info("Tidak ada data kebutuhan logistik yang diekstrak.")
        
    with col_empty:
        st.empty()  # Placeholder untuk kolom kosong

# --------------------------------------------------------------------------------
# 6. HALAMAN 2: LSTM (LANGSUNG DASHBOARD)
# --------------------------------------------------------------------------------
def render_lstm_page():
    st.markdown("## 📈 DASHBOARD PREDIKSI BANJIR (LSTM V.1)")
    st.caption("Model Time-Series LSTM untuk memprediksi curah hujan H+1 berdasarkan area (Kabupaten/Kota).")
    st.divider()
    
    if df_lstm_clean.empty:
        st.warning("Data LSTM tidak ditemukan.")
        return

    model_tuple = load_external_model()

    # Tentukan kolom area (cari Kabupaten / Kota), fallback ke klaster jika tidak ada
    possible = [c for c in df_cluster.columns if any(k in c.lower() for k in ['kabupaten','kota','kab_kota','nama_kab','nama_kota'])]
    if possible:
        area_col = possible[0]
        area_vals = sorted(df_cluster[area_col].dropna().astype(str).unique().tolist())
        sel_area = st.selectbox("Pilih Area (Kabupaten/Kota)", area_vals, index=0)
        mask = df_cluster[area_col].astype(str) == sel_area
        rr_area = float(df_cluster.loc[mask, 'RR'].mean()) if mask.any() else 0.0
    else:
        # fallback ke klaster jika tidak ada kolom daerah
        st.info("Kolom Kabupaten/Kota tidak ditemukan pada data. Menggunakan Klaster sebagai area.")
        cluster_options = sorted(df_cluster['cluster_label'].unique().tolist())
        sel_cluster = st.selectbox("Pilih Area (Klaster)", cluster_options, index=0)
        row = df_cluster[df_cluster['cluster_label'] == sel_cluster].iloc[0]
        rr_area = float(row.get('RR', 0.0))
        sel_area = f"Klaster {sel_cluster}"

    # gunakan nilai hujan input terakhir dari df_lstm_clean sebagai fitur tambahan
    last_hujan_input = float(df_lstm_clean['Hujan_Input_Terakhir'].iloc[-1]) if not df_lstm_clean.empty else rr_area

    st.markdown(f"**Fitur yang digunakan**: Area = {sel_area}, RR_area(mean) = {rr_area:.2f} mm, Hujan_Input_Terakhir = {last_hujan_input:.2f} mm")

    features = [rr_area, last_hujan_input]

    pred_val = predict_h_plus_1(model_tuple, features)

    # jika prediksi None -> tampilkan fallback, jika prediksi 0.0 -> beri peringatan
    if pred_val is None:
        st.warning("Model tidak ditemukan atau gagal prediksi. Menampilkan prediksi fallback (persistence).")
        st.info(f"Prediksi Curah Hujan H+1 (fallback) untuk {sel_area}: {rr_area:.2f} mm")
    else:
        if pred_val == 0.0:
            st.warning("Prediksi model = 0.00 mm — periksa apakah fitur input bernilai 0 atau model membutuhkan preprocessing (scaler).")
        st.success(f"Prediksi Curah Hujan H+1 untuk {sel_area}: {pred_val:.2f} mm")
    gauge_color = "#00ff99" # Hijau (Aman)
    status_msg = "AMAN"
    
    if pred_val is not None:
        if pred_val > 100: 
            gauge_color = "#ff0055" # Merah (Bahaya)
            status_msg = "BAHAYA BANJIR"
        elif pred_val > 50:
            gauge_color = "#ffcc00" # Kuning (Waspada)
            status_msg = "WASPADA"

        # Layout kolom untuk Teks Hasil & Grafik Gauge
        c_res1, c_res2 = st.columns([1, 2])
        
        with c_res1:
            st.success(f"Prediksi H+1: {pred_val:.2f} mm")
            st.metric(
                label="Perubahan dari Hari Ini",
                value=f"{pred_val:.2f} mm",
                delta=f"{pred_val - last_hujan_input:.2f} mm",
                delta_color="inverse" # Merah kalau naik, Hijau kalau turun
            )
            st.markdown(f"### Status: <span style='color:{gauge_color}'>{status_msg}</span>", unsafe_allow_html=True)
            
        with c_res2:
            # --- FITUR BARU: GAUGE CHART ---
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pred_val,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Intensitas Hujan ({sel_area})", 'font': {'size': 14, 'color': '#888'}},
                gauge = {
                    'axis': {'range': [0, max(150, pred_val * 1.5)], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': gauge_color},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#333",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(0, 255, 153, 0.1)'},
                        {'range': [50, 100], 'color': 'rgba(255, 204, 0, 0.1)'},
                        {'range': [100, 300], 'color': 'rgba(255, 0, 85, 0.1)'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': pred_val
                    }
                }
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=250, margin=dict(t=30, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
    else:
        st.warning("Model tidak ditemukan atau gagal prediksi. Menampilkan prediksi fallback (persistence).")
        st.info(f"Prediksi Curah Hujan H+1 (fallback) untuk {sel_area}: {rr_area:.2f} mm")
    # Chart seperti sebelumnya
    col_chart1, col_chart2 = st.columns([2, 1])
    with col_chart1:
        st.markdown("#### Validasi Time-Lag")
        fig_lag = make_subplots(specs=[[{"secondary_y": True}]])
        fig_lag.add_trace(go.Bar(x=df_lstm_clean['Tanggal'], y=df_lstm_clean['Hujan_Input_Terakhir'], name="Hujan", marker_color='rgba(0, 229, 255, 0.4)'), secondary_y=False)
        fig_lag.add_trace(go.Scatter(x=df_lstm_clean['Tanggal'], y=df_lstm_clean['Aktual'], name="Banjir", line=dict(color='#ff0055')), secondary_y=True)
        fig_lag.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=350, title=None, margin=dict(t=10, l=0, r=0, b=0))
        st.plotly_chart(fig_lag, use_container_width=True)
        
    with col_chart2:
        st.markdown("#### Scatter Plot")
        fig_sc = px.scatter(df_lstm_clean, x='Aktual', y='Prediksi', color='Hujan_Input_Terakhir', template="plotly_dark")
        fig_sc.add_shape(type="line", line=dict(dash='dash', color='white'), x0=0, y0=0, x1=df_lstm_clean['Aktual'].max(), y1=df_lstm_clean['Aktual'].max())
        fig_sc.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=10, l=0, r=0, b=0))
        st.plotly_chart(fig_sc, use_container_width=True)

# --------------------------------------------------------------------------------
# 7. HALAMAN 3: NLP (LANGSUNG DASHBOARD)
# --------------------------------------------------------------------------------
def render_nlp_page():
    st.markdown("## 🧠 DASHBOARD INTELIJEN TEKS")
    st.caption("Analisis teks dari laporan PDF harian (Regex & NER).")
    st.divider()
    
    if df_insights.empty:
        st.warning("Data Teks tidak tersedia.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Logistik Dibutuhkan")
        all_logs = []
        for item in df_insights['Logistik']:
            if item: all_logs.extend(item.split(','))
        if all_logs:
            counts = Counter(all_logs)
            df_log = pd.DataFrame(counts.items(), columns=['Item', 'Jml'])
            fig = px.pie(df_log, values='Jml', names='Item', hole=0.6, template="plotly_dark")
            fig.update_layout(height=300, margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
    with c2:
        st.markdown("#### Data Tabular Hasil Ekstraksi")
        st.dataframe(df_insights, height=300, use_container_width=True)

# --- MOVE: model loader + predictor must be defined before main router ---
@st.cache_resource
def load_external_model(model_dir=None):
    """
    Mencari model di folder model_dir.
    - bila tidak diberikan, coba beberapa lokasi relatif/absolut.
    Mengembalikan: (type, model, scaler_or_None) atau (None, None, None).
    """
    # default locations (cek beberapa kemungkinan)
    default_dirs = [
        r"C:\Users\ainun\OneDrive\Documents\ABD\pdf\dashboard-bencana\models",
        os.path.join(os.getcwd(), "models"),
        os.path.join(os.getcwd(), "dashboard-bencana", "models"),
    ]
    if model_dir:
        search_dirs = [model_dir] + default_dirs
    else:
        search_dirs = default_dirs

    found_files = []
    for d in search_dirs:
        try:
            if os.path.isdir(d):
                for f in glob.glob(os.path.join(d, "*")):
                    found_files.append(os.path.abspath(f))
        except Exception:
            continue

    if not found_files:
        # ringankan pesan ke pengguna
        st.info("Tidak menemukan file model di lokasi standar. Pastikan file .h5/.keras/.joblib/.pkl ditempatkan di folder models.")
        return (None, None, None)

    # helper untuk cari scaler di direktori model
    def try_load_scaler_from_dir(d):
        scaler = None
        try:
            cand = []
            cand += glob.glob(os.path.join(d, "*scaler*.pkl"))
            cand += glob.glob(os.path.join(d, "*scaler*.joblib"))
            cand += glob.glob(os.path.join(d, "scaler.pkl"))
            cand += glob.glob(os.path.join(d, "scaler.joblib"))
            if cand:
                sf = cand[0]
                if sf.lower().endswith(".joblib") and joblib is not None:
                    scaler = joblib.load(sf)
                else:
                    with open(sf, "rb") as fh:
                        scaler = pickle.load(fh)
            # tidak menampilkan debug message
        except Exception:
            scaler = None
        return scaler

    for f in found_files:
        lf = f.lower()
        try:
            if lf.endswith(".h5") or lf.endswith(".keras"):
                try:
                    from tensorflow.keras.models import load_model as _load_kmodel
                except Exception as e:
                    st.error(f"TensorFlow tidak tersedia: {e}")
                    return (None, None, None)
                m = _load_kmodel(f, compile=False)
                scaler = try_load_scaler_from_dir(os.path.dirname(f))
                return ("keras", m, scaler)

            if lf.endswith(".joblib") and joblib is not None:
                try:
                    m = joblib.load(f)
                    scaler = try_load_scaler_from_dir(os.path.dirname(f))
                    return ("skl", m, scaler)
                except Exception:
                    continue

            if lf.endswith(".pkl"):
                try:
                    with open(f, "rb") as fh:
                        m = pickle.load(fh)
                    scaler = try_load_scaler_from_dir(os.path.dirname(f))
                    return ("skl", m, scaler)
                except Exception:
                    continue
        except Exception:
            continue

    st.warning("Ada file di folder model tetapi tidak ada yang cocok/terbaca. Periksa format/compatibility.")
    return (None, None, None)


def predict_h_plus_1(model_tuple, feature_vector):
    """Prediksi / wrapper yang menangani bentuk input umum.
    model_tuple: (type, model, scaler_or_None)
    """
    if not model_tuple or model_tuple[1] is None:
        return None
    # unpack safely
    typ = model_tuple[0]
    model = model_tuple[1]
    scaler = model_tuple[2] if len(model_tuple) > 2 else None

    try:
        x = np.asarray(feature_vector, dtype=float)
    except Exception:
        return None

    try:
        raw_pred = None
        if typ == "keras":
            # --- LSTM expect shape (1, T, features) ---
            x_seq = np.tile(x, (7, 1))  # T=7 timesteps used as example during training
            x_input = x_seq.reshape(1, 7, len(x))
            p = model.predict(x_input, verbose=0)
            arr = np.array(p)
            arr_flat = arr.reshape(-1, 1)
            if scaler is not None:
                try:
                    inv = scaler.inverse_transform(arr_flat)
                    raw_pred = float(inv[0, 0])
                except Exception:
                    raw_pred = float(arr_flat.reshape(-1)[0])
            else:
                raw_pred = float(arr_flat.reshape(-1)[0])

        else:
            x_input = x.reshape(1, -1)
            p = model.predict(x_input)
            arr_flat = np.array(p).reshape(-1, 1)
            if scaler is not None:
                try:
                    inv = scaler.inverse_transform(arr_flat)
                    raw_pred = float(inv[0, 0])
                except Exception:
                    raw_pred = float(arr_flat.reshape(-1)[0])
            else:
                raw_pred = float(arr_flat.reshape(-1)[0])

        # fallback heuristic removed verbose debug
        if scaler is None and raw_pred is not None and raw_pred < 0 and abs(raw_pred) < 5:
            try:
                mean_rr = float(df_cluster['RR'].mean())
                std_rr = float(df_cluster['RR'].std())
                if std_rr > 0:
                    guessed = raw_pred * std_rr + mean_rr
                    raw_pred = guessed
            except Exception:
                pass

        # clip to >=0
        return max(0.0, float(raw_pred)) if raw_pred is not None else None

    except Exception as e:
        st.error(f"Gagal prediksi dengan model: {e}")
        return None
# --------------------------------------------------------------------------------
# 8. MAIN APP ROUTER
# --------------------------------------------------------------------------------

if st.session_state['current_page'] == 'Clustering':
    render_clustering_page()
    
elif st.session_state['current_page'] == 'LSTM':
    render_lstm_page()
    
elif st.session_state['current_page'] == 'NLP':
    render_nlp_page()

st.markdown("<br><br><div style='text-align: center; color: #333; font-size: 10px;'>SYSTEM DIRECT VIEW V.6.0 | NO MENU</div>", unsafe_allow_html=True)