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

# --------------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN & TEMA DARK MODE (ULTIMATE)
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Pusat Komando Bencana",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🚨"
)

# CSS CUSTOM: COMMAND CENTER STYLE (DARK/CYBERPUNK)
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #050505;
        color: #e0e0e0;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Container - Glowing Effect */
    .main-header {
        background: radial-gradient(circle at center, #001f3f 0%, #000000 100%);
        border-bottom: 1px solid #00e5ff;
        padding: 25px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 5px 25px rgba(0, 229, 255, 0.15);
    }
    
    .main-header h1 {
        font-family: 'Courier New', monospace;
        color: #00e5ff;
        text-transform: uppercase;
        letter-spacing: 4px;
        font-weight: 900;
        text-shadow: 0 0 15px rgba(0, 229, 255, 0.8);
        margin: 0;
    }
    
    .main-header h3 {
        color: #ff0055;
        font-size: 0.9rem;
        letter-spacing: 2px;
        margin-top: 10px;
        text-transform: uppercase;
    }

    /* Metric Cards (Holographic Dark) */
    .metric-card {
        background: linear-gradient(145deg, rgba(20,20,30,0.9), rgba(10,10,15,0.95));
        border: 1px solid #333;
        border-left: 3px solid #00e5ff;
        padding: 15px;
        border-radius: 6px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #00e5ff;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.2);
    }
    
    .metric-card h4 {
        color: #8899a6;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .metric-card h2 {
        color: #fff;
        font-size: 1.8rem;
        margin: 0;
        font-weight: 700;
        font-family: 'Arial', sans-serif;
    }
    
    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #111;
        border-radius: 4px 4px 0 0;
        color: #888;
        border: 1px solid #333;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00e5ff !important;
        color: #000 !important;
        font-weight: bold;
    }

</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown('<div class="main-header"><h1>PUSAT KOMANDO BENCANA</h1><h3>Sistem Analisis Risiko Terintegrasi</h3></div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------------------------------------
@st.cache_data
def load_data():
    # 1. Cluster Data
    try:
        df_c = pd.read_csv("ringkasan_klaster_gabungan.csv")
    except:
        st.error("❌ ERR_DATA_MISSING: 'ringkasan_klaster_gabungan.csv'")
        st.stop()
        
    # 2. Text Data
    try:
        df_t = pd.read_csv("tabel_pdf_raw_extracted_v2.csv")
        df_t['Teks_Bersih'] = df_t['Teks_Bersih'].astype(str)
        df_t['raw_text'] = df_t['raw_text'].astype(str)
    except:
        df_t = pd.DataFrame()

    # 3. LSTM Data (REAL from uploaded CSV)
    try:
        df_l = pd.read_csv("lstm_predictions_with_context.csv")
        # Parsing tanggal agar bisa di-plot
        date_cols = ['Predicted_Date_Step1', 'Predicted_Date_Step2', 'Predicted_Date_Step3']
        for col in date_cols:
            df_l[col] = pd.to_datetime(df_l[col])
    except:
        st.warning("⚠️ File 'lstm_predictions_with_context.csv' tidak ditemukan. Mode Dinamis non-aktif.")
        df_l = pd.DataFrame()
        
    
    return df_c, df_t, df_l

df_cluster, df_text, df_lstm_raw = load_data()

# --------------------------------------------------------------------------------
# 3. PRE-PROCESSING LSTM DATA (UNTUK VISUALISASI)
# --------------------------------------------------------------------------------
def process_lstm_for_viz(df):
    if df.empty: return pd.DataFrame(), pd.DataFrame()
    
    # A. Flatten Data Prediksi (Membuat urutan waktu tunggal)
    # Kita ambil Step 1 saja sebagai representasi prediksi H+1 (Next Day Prediction)
    df_viz = df[['Predicted_Date_Step1', 'Actual_Rumah_Terendam_Step1', 'Predicted_Rumah_Terendam_Step1', 
                 'Input_TotalHujan_3Hari_Day7']].copy()
    
    df_viz.rename(columns={
        'Predicted_Date_Step1': 'Tanggal',
        'Actual_Rumah_Terendam_Step1': 'Aktual',
        'Predicted_Rumah_Terendam_Step1': 'Prediksi',
        'Input_TotalHujan_3Hari_Day7': 'Hujan_Input_Terakhir' # Hujan hari kemarin
    }, inplace=True)
    
    df_viz = df_viz.sort_values('Tanggal')
    
    return df_viz

df_lstm_clean = process_lstm_for_viz(df_lstm_raw)

# --------------------------------------------------------------------------------
# 4. TEXT MINING ENGINE
# --------------------------------------------------------------------------------
def extract_text_insights(df):
    if df.empty: return pd.DataFrame()
    results = []
    
    logistics_map = {
        "Pangan": ["makanan", "sembako", "nasi", "air minum", "dapur umum", "mie"],
        "Shelter": ["selimut", "tenda", "terpal", "alas tidur", "matras", "pengungsian"],
        "Medis": ["obat", "vitamin", "masker", "medis", "dokter", "gatal"],
        "Evakuasi": ["perahu", "karet", "pelampung", "tali", "alat berat"]
    }
    
    cause_map = {
        "Hujan Ekstrem": ["hujan deras", "hujan lebat", "intensitas tinggi", "cuaca buruk"],
        "Sungai Meluap": ["sungai meluap", "luapan sungai", "banjir kiriman", "debit air"],
        "Tanggul Jebol": ["tanggul jebol", "tanggul rusak", "penahan air"],
        "Drainase Buruk": ["drainase", "gorong-gorong", "saluran air"]
    }
    
    for _, row in df.iterrows():
        txt_clean = row['Teks_Bersih'].lower()
        raw_txt = row['raw_text']
        
        # Regex Ekstraksi
        pat_rumah = r"(\d+(?:[.,]\d+)?)\s*(?:unit|buah)?\s*(?:rumah|bangunan)"
        m_rumah = re.findall(pat_rumah, txt_clean)
        house_dmg = 0
        if m_rumah:
            nums = [int(x.replace('.', '').replace(',', '')) for x in m_rumah if x.replace('.', '').replace(',', '').isdigit()]
            if nums: house_dmg = max(nums)
            
        pat_jiwa = r"(\d+(?:[.,]\d+)?)\s*(?:jiwa|orang|warga|kk)\s*(?:terdampak|mengungsi|korban)"
        m_jiwa = re.findall(pat_jiwa, txt_clean)
        people_aff = 0
        if m_jiwa:
            nums = [int(x.replace('.', '').replace(',', '')) for x in m_jiwa if x.replace('.', '').replace(',', '').isdigit()]
            if nums: people_aff = max(nums)

        needs = []
        for cat, keywords in logistics_map.items():
            if any(k in txt_clean for k in keywords):
                needs.append(cat)
                
        causes = []
        for cat, keywords in cause_map.items():
            if any(k in txt_clean for k in keywords):
                causes.append(cat)
        
        match_kec = re.search(r"Kecamatan\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", raw_txt)
        kecamatan = match_kec.group(1) if match_kec else "-"

        if house_dmg > 0 or people_aff > 0 or needs or causes:
            results.append({
                "Tanggal": row.get('tanggal_pdf', '-'),
                "Kabupaten": row.get('kabupaten_pdf', 'Unknown'),
                "Kecamatan": kecamatan,
                "Rumah_Rusak": house_dmg,
                "Jiwa_Terdampak": people_aff,
                "Penyebab": ", ".join(causes) if causes else "Tidak Diketahui",
                "Logistik": ", ".join(needs) if needs else "-",
                "Cuplikan": raw_txt[:100] + "..."
            })
            
    return pd.DataFrame(results)

df_insights = extract_text_insights(df_text)

# --------------------------------------------------------------------------------
# 5. DASHBOARD LOGIC (TOP METRICS)
# --------------------------------------------------------------------------------
danger_idx = df_cluster['Mengungsi'].idxmax()
danger_row = df_cluster.loc[danger_idx]
danger_label = int(danger_row['cluster_label'])

total_kejadian = df_cluster['Jumlah_Kejadian'].sum()
max_hujan = df_cluster['RR'].max()

if max_hujan > 150:
    status_txt, status_color = "KRITIS", "#ff0055"
elif max_hujan > 100:
    status_txt, status_color = "WASPADA", "#ffcc00"
else:
    status_txt, status_color = "STABIL", "#00ff99"

c1, c2, c3, c4, c5 = st.columns(5)

def card(title, value, subtitle, color):
    return f"""
    <div class="metric-card" style="border-left: 3px solid {color};">
        <h4 style="color:{color}">{title}</h4>
        <h2>{value}</h2>
        <div style="font-size: 11px; color: #888; margin-top: 5px;">{subtitle}</div>
    </div>
    """
with c1: st.markdown(card("STATUS RISIKO", status_txt, f"Curah Hujan Maks: {int(max_hujan)} mm", status_color), unsafe_allow_html=True)
with c2: st.markdown(card("TOTAL LOKASI", f"{int(total_kejadian):,}", "Titik Terpantau", "#00e5ff"), unsafe_allow_html=True)
with c3: st.markdown(card("ZONA BAHAYA", f"{int(danger_row['Jumlah_Kejadian'])}", f"Klaster {danger_label}", "#ff0055"), unsafe_allow_html=True)

est_rumah = df_insights['Rumah_Rusak'].sum() if not df_insights.empty else 0
est_jiwa = df_insights['Jiwa_Terdampak'].sum() if not df_insights.empty else 0

with c4: st.markdown(card("RUMAH RUSAK", f"{int(est_rumah):,}", "Ekstraksi Laporan", "#aa00ff"), unsafe_allow_html=True)
with c5: st.markdown(card("JIWA TERDAMPAK", f"{int(est_jiwa):,}", "Ekstraksi Laporan", "#ffcc00"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 6. GEOSPATIAL & CLUSTER
# --------------------------------------------------------------------------------
col_map, col_cluster_stat = st.columns([3, 1])

with col_map:
    st.markdown("### 🗺️ PETA RESIKO")
    m = folium.Map(location=[-6.95, 107.63], zoom_start=11, tiles='CartoDB dark_matter')
    
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; width: 130px; background-color: rgba(0,0,0,0.8); 
    border: 1px solid #444; color: white; padding: 10px; border-radius: 5px; font-size: 11px; z-index:9999;">
        <b style="color:#00e5ff">INDEKS RISIKO</b><br>
        <span style="color:#ff0055">●</span> Tinggi (Bahaya)<br>
        <span style="color:#ffcc00">●</span> Sedang (Waspada)<br>
        <span style="color:#00ff99">●</span> Rendah (Aman)
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))

    # --- LOGIKA BARU: PENENTUAN WARNA PASTI ---
    # 1. Merah = danger_label (Sudah ditentukan di atas berdasarkan Pengungsi terbanyak)
    # 2. Kuning = Kita cari klaster sisa (bukan merah) yang punya Hujan (RR) paling tinggi
    sisa_klaster = df_cluster[df_cluster['cluster_label'] != danger_label]
    
    if not sisa_klaster.empty:
        # Ambil ID klaster dengan RR tertinggi dari sisanya untuk jadi KUNING
        yellow_label = sisa_klaster.loc[sisa_klaster['RR'].idxmax(), 'cluster_label']
    else:
        yellow_label = -99 # Fallback jika datanya cuma 1 baris

    for _, row in df_cluster.iterrows():
        cluster_id = int(row['cluster_label'])
        count = int(row['Jumlah_Kejadian'])
        
        # Logika Penentuan Warna & Lokasi
        if cluster_id == danger_label:
            # MERAH (Bahaya)
            color, c_lat, c_lon, spread = '#ff0055', -7.00, 107.64, 0.015
            stat = "BAHAYA"
        elif cluster_id == yellow_label:
            # KUNING (Waspada) - Pasti muncul sekarang!
            color, c_lat, c_lon, spread = '#ffcc00', -6.96, 107.61, 0.03
            stat = "WASPADA"
        else:
            # HIJAU (Aman)
            color, c_lat, c_lon, spread = '#00ff99', -6.93, 107.60, 0.05
            stat = "AMAN"

        np.random.seed(cluster_id)
        lats = np.random.normal(c_lat, spread, count)
        lons = np.random.normal(c_lon, spread, count)
        
        for i in range(count):
            popup = f"""<div style='background:#111;color:#fff;padding:5px;'><b>{stat}</b><br>Hujan: {row['RR']:.0f}mm</div>"""
            folium.CircleMarker([lats[i], lons[i]], radius=3+(row['RR']/60), color=color, fill=True, fill_color=color, fill_opacity=0.8, popup=popup).add_to(m)

    st_folium(m, width="100%", height=480)

with col_cluster_stat:
    st.markdown("### 📊 PROFIL KLASTER")
    df_cluster['Label_String'] = df_cluster['cluster_label'].apply(lambda x: f"Klaster {x}")
    color_map = {f"Klaster {danger_label}": "#ff0055", f"Klaster {0 if danger_label!=0 else 1}": "#00ff99", f"Klaster {1 if danger_label!=1 and danger_label!=0 else 2}": "#ffcc00"}
    
    st.markdown("**1. Proporsi Kejadian**")
    fig_pie = px.pie(df_cluster, values='Jumlah_Kejadian', names='Label_String', hole=0.5, color='Label_String', color_discrete_map=color_map)
    fig_pie.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=0, b=20, l=0, r=0), height=200, showlegend=False)
    fig_pie.add_annotation(text="TOTAL", showarrow=False, font_size=12, y=0.5, x=0.5)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("**2. Hujan vs Pengungsi**")
    df_melt = df_cluster.melt(id_vars=['Label_String'], value_vars=['RR', 'Mengungsi'], var_name='Metrik', value_name='Nilai')
    fig_bar = px.bar(df_melt, x='Label_String', y='Nilai', color='Metrik', barmode='group', color_discrete_map={'RR': '#00e5ff', 'Mengungsi': '#ff0055'})
    fig_bar.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=0, b=0, l=0, r=0), height=200, legend=dict(orientation="h", y=-0.3, title=None), xaxis_title=None, yaxis_title="Rata-rata")
    st.plotly_chart(fig_bar, use_container_width=True)
    st.info(f"⚠️ **Klaster {danger_label}** paling berbahaya dengan curah hujan rata-rata **{danger_row['RR']:.0f} mm**.")


# --------------------------------------------------------------------------------
# 7. ANALYTICS TABS (DINAMIS & LENGKAP)
# --------------------------------------------------------------------------------
st.markdown("---")
tab1, tab2 = st.tabs(["📈 PREDIKSI DAMPAK & VALIDASI (LSTM)", "🧠 LAPORAN INTELIJEN (NLP)"])

with tab1:
    if not df_lstm_clean.empty:
        # --- BAGIAN A: METRIK EVALUASI ---
        st.markdown("#### 🛠️ EVALUASI PERFORMA MODEL")
        em1, em2, em3, em4 = st.columns(4)
        
        # Menggunakan nilai hardcoded sesuai request, tapi bisa diganti rumus jika mau
        mse_val = 223481.7345
        rmse_val = 472.7385
        mae_val = 164.8364
        
        # Hitung Bias Rata-rata dari data CSV
        bias_val = (df_lstm_clean['Prediksi'] - df_lstm_clean['Aktual']).mean()
        
        em1.metric("MSE (Error Kuadrat)", f"{mse_val:,.0f}", delta_color="inverse")
        em2.metric("RMSE (Deviasi Standar)", f"{rmse_val:,.2f}", delta_color="inverse")
        em3.metric("MAE (Rata-rata Error)", f"{mae_val:,.2f}", delta_color="inverse")
        em4.metric("Bias Model (Prediksi - Aktual)", f"{bias_val:,.2f}", "Overestimate" if bias_val > 0 else "Underestimate")
        
        st.divider()

        # --- BAGIAN B: INSIGHT TIME-LAG & HULU HILIR ---
        col_chart1, col_chart2 = st.columns([2, 1])
        
        with col_chart1:
            st.markdown("#### 🌧️ VALIDASI TIME-LAG & HULU-HILIR (Sebab-Akibat)")
            st.caption("Grafik ini menunjukkan korelasi antara Curah Hujan (Hulu/Penyebab) dengan Rumah Terendam (Hilir/Akibat). Perhatikan adanya jeda waktu (Lag) antara puncak hujan dan puncak banjir.")
            
            # Dual Axis Chart: Bar (Hujan) & Line (Rumah Terendam)
            fig_lag = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Trace 1: Hujan (Input Model)
            fig_lag.add_trace(
                go.Bar(x=df_lstm_clean['Tanggal'], y=df_lstm_clean['Hujan_Input_Terakhir'], 
                       name="Curah Hujan (Input)", marker_color='rgba(0, 229, 255, 0.4)'),
                secondary_y=False
            )
            
            # Trace 2: Rumah Terendam (Aktual)
            fig_lag.add_trace(
                go.Scatter(x=df_lstm_clean['Tanggal'], y=df_lstm_clean['Aktual'], 
                           name="Rumah Terendam (Aktual)", line=dict(color='#ff0055', width=3)),
                secondary_y=True
            )
            
            fig_lag.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title_text="Korelasi Hujan (Hulu) vs Banjir (Hilir)",
                height=400,
                legend=dict(orientation="h", y=1.1)
            )
            fig_lag.update_yaxes(title_text="Curah Hujan (mm)", secondary_y=False)
            fig_lag.update_yaxes(title_text="Jumlah Rumah", secondary_y=True)
            
            st.plotly_chart(fig_lag, use_container_width=True)
            
        with col_chart2:
            st.markdown("#### 🎯 KONSISTENSI PREDIKSI")
            st.caption("Scatter plot Aktual vs Prediksi. Titik yang mendekati garis diagonal menunjukkan prediksi akurat.")
            
            fig_scatter = px.scatter(
                df_lstm_clean, 
                x='Aktual', 
                y='Prediksi', 
                color='Hujan_Input_Terakhir',
                color_continuous_scale='Bluered',
                labels={'Aktual': 'Rumah Terendam (Real)', 'Prediksi': 'Prediksi Model'}
            )
            # Tambah garis diagonal referensi
            fig_scatter.add_shape(type="line", line=dict(dash='dash', color='white', width=1),
                x0=df_lstm_clean['Aktual'].min(), y0=df_lstm_clean['Aktual'].min(),
                x1=df_lstm_clean['Aktual'].max(), y1=df_lstm_clean['Aktual'].max()
            )
            fig_scatter.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # --- BAGIAN C: FORECAST CHART ---
        st.markdown("#### 🔮 TREN PREDIKSI DAMPAK (30 Hari)")
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=df_lstm_clean['Tanggal'], y=df_lstm_clean['Aktual'], name='Data Aktual', line=dict(color='#00e5ff')))
        fig_trend.add_trace(go.Scatter(x=df_lstm_clean['Tanggal'], y=df_lstm_clean['Prediksi'], name='Prediksi AI', line=dict(color='#ffcc00', dash='dot')))
        
        fig_trend.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=300, hovermode="x unified")
        st.plotly_chart(fig_trend, use_container_width=True)

    else:
        st.error("Data LSTM tidak tersedia. Mohon cek file `lstm_predictions_with_context.csv`.")

with tab2:
    if not df_insights.empty:
        tc1, tc2, tc3 = st.columns(3)
        
        with tc1:
            st.markdown("#### 📦 KEBUTUHAN LOGISTIK")
            all_logs = []
            for item in df_insights['Logistik']:
                if item != "-": all_logs.extend([x.strip() for x in item.split(',')])
            if all_logs:
                df_log = pd.DataFrame(Counter(all_logs).items(), columns=['Item', 'Count'])
                fig_pie = px.pie(df_log, values='Count', names='Item', color_discrete_sequence=px.colors.sequential.Bluered, hole=0.6)
                fig_pie.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=250, margin=dict(t=0,b=0,l=0,r=0), showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True)

        with tc2:
            st.markdown("#### ⚠️ PENYEBAB UTAMA")
            all_causes = []
            for item in df_insights['Penyebab']:
                if item != "Tidak Diketahui": all_causes.extend([x.strip() for x in item.split(',')])
            if all_causes:
                df_cause = pd.DataFrame(Counter(all_causes).items(), columns=['Sebab', 'Freq'])
                fig_bar = px.bar(df_cause, x='Sebab', y='Freq', color='Freq', color_continuous_scale='Reds')
                fig_bar.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=250, margin=dict(t=0,b=0,l=0,r=0))
                st.plotly_chart(fig_bar, use_container_width=True)

        with tc3:
            st.markdown("#### 📍 KECAMATAN TERDAMPAK")
            kec_counts = df_insights[df_insights['Kecamatan'] != "-"]['Kecamatan'].value_counts().reset_index()
            kec_counts.columns = ['Kecamatan', 'Laporan']
            if not kec_counts.empty:
                fig_bar2 = px.bar(kec_counts.head(5), x='Laporan', y='Kecamatan', orientation='h', color='Laporan', color_continuous_scale='Teal')
                fig_bar2.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=250, margin=dict(t=0,b=0,l=0,r=0), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_bar2, use_container_width=True)

        st.markdown("#### 📋 DATA EKSTRAKSI RINCI")
        st.dataframe(df_insights[['Tanggal', 'Kecamatan', 'Penyebab', 'Jiwa_Terdampak', 'Rumah_Rusak', 'Logistik']], hide_index=True, use_container_width=True)
    else:
        st.warning("Data teks tidak tersedia.")

st.markdown("<br><div style='text-align: center; color: #444; font-size: 10px;'>SISTEM V.4.0 FINAL | KONEKSI AMAN TERHUBUNG</div>", unsafe_allow_html=True)