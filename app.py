import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, Fullscreen
import random
import datetime

# --- MAKİNE ÖĞRENMESİ KÜTÜPHANELERİ ---
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. SAYFA VE GENEL AYARLAR
# =============================================================================
st.set_page_config(page_title="Shenergy Pro - Hybrid Mode", layout="wide", page_icon="⚡")

# --- CSS İLE ZORLANMIŞ KOYU MOD (DARK MODE) ---
st.markdown("""
<style>
    /* Ana Sayfa Arka Planı */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    /* Sidebar (Yan Menü) Arka Planı */
    [data-testid="stSidebar"] {
        background-color: #262730;
    }

    /* Başlıklar */
    h1, h2, h3, h4, h5, h6 {
        color: #FAFAFA !important;
    }

    /* Yazılar */
    p, label, .stMarkdown {
        color: #FAFAFA !important;
    }

    /* Metrikler */
    [data-testid="stMetricValue"] {
        color: #FAFAFA !important;
    }

    /* Inputlar */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #262730;
        color: #FAFAFA;
    }

    /* Butonlar */
    .stButton>button {
        color: #FAFAFA;
        border-color: #4B4B4B;
        background-color: #262730;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Enerjisa Shenergy - Şebeke & İklim Etkisi Paneli")
st.markdown("**Modül:** 2026 Stratejik Kesinti Tahmini | **Model:** High-Performance AI (Log-Transformed)")

# --- SESSION STATE ---
if 'analiz_basladi' not in st.session_state:
    st.session_state.analiz_basladi = False


def analizi_tetikle():
    st.session_state.analiz_basladi = True


# --- KOORDİNATLAR ---
IL_KOORDINATLARI = {"ADANA": [37.0000, 35.3213], "MERSİN": [36.8121, 34.6415], "MERSIN": [36.8121, 34.6415]}

ILCE_MERKEZLERI = {
    "SEYHAN": [37.0017, 35.3289], "YÜREĞİR": [36.9850, 35.3450], "ÇUKUROVA": [37.0500, 35.2800],
    "SARIÇAM": [37.0300, 35.4000], "CEYHAN": [37.0250, 35.8150], "KOZAN": [37.4550, 35.8150],
    "İMAMOĞLU": [37.2600, 35.6600], "KARATAŞ": [36.5700, 35.3700], "KARAİSALI": [37.2500, 35.0600],
    "POZANTI": [37.4200, 34.8700], "ALADAĞ": [37.5400, 35.3900], "FEKE": [37.8100, 35.9100],
    "YUMURTALIK": [36.7700, 35.7900], "TUFANBEYLİ": [38.2600, 36.2200],
    "AKDENİZ": [36.8121, 34.6415], "TOROSLAR": [36.8500, 34.6000], "YENİŞEHİR": [36.7800, 34.5500],
    "MEZİTLİ": [36.7600, 34.5000], "TARSUS": [36.9167, 34.8833], "ERDEMLİ": [36.6050, 34.3080],
    "SİLİFKE": [36.3778, 33.9344], "MUT": [36.6400, 33.4300], "GÜLNAR": [36.3400, 33.4000],
    "ANAMUR": [36.0700, 32.8300], "BOZYAZI": [36.1000, 32.9600], "AYDINCIK": [36.1400, 33.3200],
    "ÇAMLIYAYLA": [37.1600, 34.6000],
    "VARSAYILAN": [36.9000, 35.0000]
}

# --- İKLİM PROFİLİ (GÜNDÜZ MAKSİMUM) ---
BOLGE_IKLIM_PROFILI = {
    1: 15.0, 2: 16.5, 3: 20.0, 4: 24.5, 5: 29.0, 6: 33.5,
    7: 36.0, 8: 36.5, 9: 33.0, 10: 28.0, 11: 21.5, 12: 17.0
}


# =============================================================================
# 2. VERİ YÜKLEME VE İŞLEME
# =============================================================================

@st.cache_data
def veri_yukle(uploaded_files):
    all_data = []
    for file in uploaded_files:
        try:
            if file.name.endswith('.parquet'):
                df_temp = pd.read_parquet(file, engine='pyarrow')
            elif file.name.endswith('.csv'):
                try:
                    df_temp = pd.read_csv(file, sep=';', on_bad_lines='skip', encoding='utf-8')
                    if df_temp.shape[1] < 2: file.seek(0); df_temp = pd.read_csv(file, sep=',', encoding='utf-8')
                except:
                    file.seek(0);
                    df_temp = pd.read_csv(file, sep=';', encoding='iso-8859-9')
            else:
                df_temp = pd.read_excel(file)

            # Sütun temizliği
            df_temp.columns = df_temp.columns.str.strip()
            df_obj = df_temp.select_dtypes(['object'])
            df_temp[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
            all_data.append(df_temp)
        except Exception as e:
            st.error(f"Hata: {file.name} -> {e}")

    if all_data: return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


# Model eğitimi için yardımcı sütun bulma fonksiyonu
def sutun_bul_ai(df_columns, anahtar_kelimeler):
    text_cols = [str(c).upper() for c in df_columns]
    for anahtar in anahtar_kelimeler:
        anahtar = anahtar.upper()
        for i, col in enumerate(text_cols):
            if anahtar in col:
                return df_columns[i]
    return None


@st.cache_data(show_spinner=False)
def veri_hazirla_ve_feature_engineering(df_raw, tarih_kolonu):
    df = df_raw.copy()
    df['TARIH'] = pd.to_datetime(df[tarih_kolonu])
    df['GUN_TARIH'] = df['TARIH'].dt.date

    daily_counts = df.groupby('GUN_TARIH').size().reset_index(name='KESINTI_SAYISI')

    cols = df.columns
    col_sicaklik = sutun_bul_ai(cols, ['SICAKLIK', 'TEMP'])
    col_nem = sutun_bul_ai(cols, ['NEM', 'HUMIDITY'])
    col_risk = sutun_bul_ai(cols, ['RISK', 'SCORE'])

    agg_dict = {}
    rename_map = {}

    if col_sicaklik:
        if df[col_sicaklik].dtype == 'object': df[col_sicaklik] = pd.to_numeric(
            df[col_sicaklik].astype(str).str.replace(',', '.'), errors='coerce')
        agg_dict[col_sicaklik] = 'mean';
        rename_map[col_sicaklik] = 'SICAKLIK'
    if col_nem:
        if df[col_nem].dtype == 'object': df[col_nem] = pd.to_numeric(df[col_nem].astype(str).str.replace(',', '.'),
                                                                      errors='coerce')
        agg_dict[col_nem] = 'mean';
        rename_map[col_nem] = 'NEM'
    if col_risk:
        agg_dict[col_risk] = 'mean';
        rename_map[col_risk] = 'RISK'

    if agg_dict:
        daily_means = df.groupby('GUN_TARIH').agg(agg_dict).reset_index()
        daily_means.rename(columns=rename_map, inplace=True)
        daily_df = pd.merge(daily_counts, daily_means, on='GUN_TARIH', how='left')
    else:
        daily_df = daily_counts.copy()

    daily_df['TARIH'] = pd.to_datetime(daily_df['GUN_TARIH'])
    daily_df = daily_df.drop('GUN_TARIH', axis=1).sort_values('TARIH')

    date_range = pd.date_range(daily_df['TARIH'].min(), daily_df['TARIH'].max(), freq='D')
    full_daily_df = pd.DataFrame({'TARIH': date_range})
    full_daily_df = full_daily_df.merge(daily_df, on='TARIH', how='left')
    full_daily_df['KESINTI_SAYISI'] = full_daily_df['KESINTI_SAYISI'].fillna(0)
    full_daily_df = full_daily_df.fillna(method='ffill').fillna(method='bfill')

    daily_df = full_daily_df.copy()

    daily_df['YIL'] = daily_df['TARIH'].dt.year
    daily_df['AY'] = daily_df['TARIH'].dt.month
    daily_df['GUN'] = daily_df['TARIH'].dt.day
    daily_df['HAFTA_ICI'] = daily_df['TARIH'].dt.dayofweek
    daily_df['YILIN_GUNU'] = daily_df['TARIH'].dt.dayofyear

    daily_df['AY_SIN'] = np.sin(2 * np.pi * daily_df['AY'] / 12)
    daily_df['AY_COS'] = np.cos(2 * np.pi * daily_df['AY'] / 12)

    for lag in [1, 2, 3, 7, 14, 30, 90]: daily_df[f'LAG_{lag}'] = daily_df['KESINTI_SAYISI'].shift(lag)
    for window in [3, 7, 30, 60]:
        daily_df[f'ROLL_MEAN_{window}'] = daily_df['KESINTI_SAYISI'].rolling(window).mean()
        daily_df[f'ROLL_STD_{window}'] = daily_df['KESINTI_SAYISI'].rolling(window).std()

    daily_df['EWM_MEAN'] = daily_df['KESINTI_SAYISI'].ewm(span=7).mean()

    return daily_df.dropna()


@st.cache_resource(show_spinner="AI Modelleri Eğitiliyor...")
def model_egit(daily_df):
    feature_cols = [col for col in daily_df.columns if col not in ['TARIH', 'KESINTI_SAYISI']]
    X = daily_df[feature_cols]
    y = daily_df['KESINTI_SAYISI']

    y_log = np.log1p(y)
    train_size = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train_log, y_test_log = y_log.iloc[:train_size], y_log.iloc[train_size:]
    y_test_original = np.expm1(y_test_log)
    test_dates = daily_df.iloc[train_size:]['TARIH']

    xgb = XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8,
                       random_state=42, n_jobs=-1)
    rf = RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.02, random_state=42)

    xgb.fit(X_train, y_train_log);
    rf.fit(X_train, y_train_log);
    gb.fit(X_train, y_train_log)

    pred_log = (0.5 * xgb.predict(X_test) + 0.3 * gb.predict(X_test) + 0.2 * rf.predict(X_test))
    y_pred_final = np.expm1(pred_log)

    mae = mean_absolute_error(y_test_original, y_pred_final)
    r2 = r2_score(y_test_original, y_pred_final)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_final))

    # --- ÖZELLİK İSİMLERİNİ TÜRKÇELEŞTİRME VE BASİTLEŞTİRME ---
    isim_sozlugu = {
        'AY_SIN': 'Mevsimsel Döngü',
        'AY_COS': 'Mevsimsel Döngü (Cos)',
        'EWM_MEAN': 'Genel Trend (Eğilim)',
        'SICAKLIK': '🔥 Sıcaklık',
        'NEM': '💧 Nem Oranı',
        'RISK': '⚡ Bölgesel Risk Skoru',
        'LAG_1': 'Dünkü Arıza Sayısı',
        'LAG_2': '2 Gün Önceki Arıza',
        'LAG_3': '3 Gün Önceki Arıza',
        'LAG_7': 'Geçen Haftaki Arıza',
        'ROLL_MEAN_3': 'Son 3 Günlük Ort.',
        'ROLL_MEAN_7': 'Haftalık Ort.',
        'ROLL_MEAN_30': 'Aylık Ort.',
        'ROLL_STD_3': 'Ani Dalgalanma (3 Gün)',
        'ROLL_STD_7': 'Haftalık Kararsızlık',
        'YILIN_GUNU': 'Mevsimsel Zamanlama',
        'HAFTA_ICI': 'Hafta Günü Etkisi',
        'YIL': 'Yıl Etkisi',
        'AY': 'Ay Etkisi',
        'GUN': 'Gün Etkisi'
    }

    importance = pd.DataFrame({'feature': feature_cols, 'importance': xgb.feature_importances_})
    # Sözlükte varsa değiştir, yoksa olduğu gibi bırak
    importance['feature_tr'] = importance['feature'].map(isim_sozlugu).fillna(importance['feature'])
    importance = importance.sort_values('importance', ascending=False).head(10)

    return y_test_original, y_pred_final, test_dates, mae, r2, importance, xgb, rf, gb


def tahmin_gelecek_olustur(daily_df, xgb, rf, gb, baslangic_tarihi, bitis_tarihi):
    tarihler_gelecek = pd.date_range(start=baslangic_tarihi, end=bitis_tarihi, freq='D')

    if 'SICAKLIK' in daily_df.columns:
        sicaklik_profili = daily_df.groupby(['AY', 'GUN'])['SICAKLIK'].mean().to_dict()
    else:
        sicaklik_profili = {}

    future_data = []
    last_ewm = daily_df['EWM_MEAN'].iloc[-1]

    for date in tarihler_gelecek:
        row = {}
        row['YIL'] = date.year
        row['AY'] = date.month
        row['GUN'] = date.day
        row['HAFTA_ICI'] = date.dayofweek
        row['HAFTA_SONU'] = 1 if date.dayofweek >= 5 else 0
        row['YILIN_GUNU'] = date.dayofyear
        row['AY_SIN'] = np.sin(2 * np.pi * row['AY'] / 12)
        row['AY_COS'] = np.cos(2 * np.pi * row['AY'] / 12)

        # --- İKLİM SİMÜLASYONU ---
        hist_temp = sicaklik_profili.get((row['AY'], row['GUN']), None)
        if hist_temp is None: hist_temp = BOLGE_IKLIM_PROFILI.get(row['AY'], 20)

        if row['AY'] in [6, 7, 8, 9]:
            simulated_temp = hist_temp + random.uniform(1.5, 3.5)
        elif row['AY'] in [12, 1, 2]:
            simulated_temp = hist_temp + random.uniform(-1.0, 1.0)
        else:
            simulated_temp = hist_temp + random.uniform(0.5, 2.0)

        row['SICAKLIK'] = simulated_temp

        if 'NEM' in daily_df.columns: row['NEM'] = daily_df['NEM'].mean()
        if 'RISK' in daily_df.columns: row['RISK'] = daily_df['RISK'].mean()

        row['EWM_MEAN'] = last_ewm
        for lag in [1, 2, 3, 7, 14, 30, 90]: row[f'LAG_{lag}'] = last_ewm
        for window in [3, 7, 30, 60]: row[f'ROLL_MEAN_{window}'] = last_ewm; row[f'ROLL_STD_{window}'] = 0

        future_data.append(row)

    df_future = pd.DataFrame(future_data)

    feature_cols = [col for col in daily_df.columns if col not in ['TARIH', 'KESINTI_SAYISI']]
    for col in feature_cols:
        if col not in df_future.columns: df_future[col] = 0

    X_future = df_future[feature_cols]

    p_xgb = xgb.predict(X_future)
    p_rf = rf.predict(X_future)
    p_gb = gb.predict(X_future)

    pred_log = (0.5 * p_xgb + 0.3 * p_gb + 0.2 * p_rf)
    final_pred = np.expm1(pred_log)

    result_df = pd.DataFrame({'TARIH': tarihler_gelecek, 'TAHMIN': final_pred})
    if 'SICAKLIK' in df_future.columns: result_df['SICAKLIK'] = df_future['SICAKLIK']

    return result_df


# =============================================================================
# 3. ARAYÜZ
# =============================================================================

st.sidebar.header("📂 Veri ve Filtreler")
uploaded_files = st.sidebar.file_uploader("Veri Seti", type=["parquet", "xlsx", "csv"], accept_multiple_files=True)

if not uploaded_files:
    # HARİTA AYARI: 'CartoDB positron' (Beyaz/Açık Tema)
    m = folium.Map(location=[36.90, 35.00], zoom_start=9, control_scale=True, tiles='CartoDB positron')

    Fullscreen(position='topleft').add_to(m)
    folium.Marker([37.00, 35.32], popup="Adana", icon=folium.Icon(color='blue', icon='info-sign')).add_to(m)
    folium.Marker([36.81, 34.64], popup="Mersin", icon=folium.Icon(color='red', icon='info-sign')).add_to(m)
    st_folium(m, width="100%", height=500, returned_objects=[])

    st.info("👋 Başlamak için lütfen soldan veri dosyalarını yükleyiniz.")

else:
    df = veri_yukle(uploaded_files)
    if not df.empty:
        cols = df.columns.tolist()

        # --- DOĞRUDAN ESKİ KOD MANTIĞI İLE SÜTUN BULMA ---
        c_il = next((c for c in cols if c in ['İL (3A)', 'İl (3A)', 'Trafo_İl', 'İl', 'IL', 'Sehir', 'IL_CLEAN']), None)
        c_ilce = next((c for c in cols if c in ['İLÇE (3B)', 'İlçe (3B)', 'Trafo_İlçe', 'İlçe', 'ILCE', 'ILCE_CLEAN']),
                      None)
        c_mahalle = next((c for c in cols if c in ['Trafo_Mahalle', 'Mahalle', 'MAHALLE', 'MAH', 'MH']), None)
        c_tarih = next((c for c in cols if
                        c in ['BAŞLAMA TARİHİ VE ZAMANI (6)', 'BAŞLAMA TARİHİ', 'TARIH', 'ZAMAN', 'DATE',
                              'KESINTI_TARIH']), None)

        # Trafo sütun tespiti SADECE harita popup için kalsın
        c_trafo = next((c for c in cols if
                        c in ['ŞEBEKE UNSURU (3C)', 'ŞEBEKE UNSURU KODU (3D)', 'KOD', 'MONTAJ', 'ASSET', 'ID', 'UNSUR',
                              'ŞEBEKE']), None)

        if c_tarih:
            df['Tarih_Formatli'] = pd.to_datetime(df[c_tarih], dayfirst=True, errors='coerce')

        st.sidebar.markdown("---")
        st.sidebar.subheader("📅 Geçmiş Veri Aralığı")
        min_d = df['Tarih_Formatli'].min().date() if c_tarih else datetime.date(2022, 1, 1)
        max_d = df['Tarih_Formatli'].max().date() if c_tarih else datetime.date(2024, 12, 31)
        t_secim = st.sidebar.date_input("Analiz Aralığı", value=(min_d, max_d), min_value=min_d, max_value=max_d)

        st.sidebar.markdown("---")
        st.sidebar.subheader("🔮 Gelecek Tahmin Hedefi")
        default_start = datetime.date(2026, 6, 1)
        default_end = datetime.date(2026, 8, 31)
        future_range = st.sidebar.date_input("Tahmin Aralığı", value=(default_start, default_end),
                                             min_value=max_d + datetime.timedelta(days=1))

        # --- FİLTRE MANTIĞI ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("📍 Konum Seçimi")

        # --- 1. İL SEÇİMİ ---
        if c_il:
            iller = sorted(df[c_il].dropna().astype(str).str.strip().unique())
            secilen_il = st.sidebar.selectbox("İl", ["Tümü"] + iller)
        else:
            secilen_il = "Tümü"

        # DataFrame Kopyala
        df_temp = df.copy()
        if c_tarih and len(t_secim) == 2:
            df_temp = df_temp[(df_temp['Tarih_Formatli'].dt.date >= t_secim[0]) & (
                    df_temp['Tarih_Formatli'].dt.date <= t_secim[1])]

        # İl Filtreleme
        if secilen_il != "Tümü" and c_il:
            df_temp = df_temp[df_temp[c_il].astype(str).str.strip() == secilen_il]

        # --- 2. İLÇE SEÇİMİ ---
        if c_ilce:
            ilceler = sorted(df_temp[c_ilce].dropna().astype(str).str.strip().unique())
            secilen_ilce = st.sidebar.selectbox("İlçe", ["Tümü"] + ilceler)
            if secilen_ilce != "Tümü":
                df_temp = df_temp[df_temp[c_ilce].astype(str).str.strip() == secilen_ilce]
        else:
            secilen_ilce = "Tümü"

        # --- 3. MAHALLE SEÇİMİ ---
        if c_mahalle:
            mahalleler = sorted(df_temp[c_mahalle].dropna().astype(str).str.strip().unique())
            secilen_mahalle = st.sidebar.selectbox("Mahalle", ["Tümü"] + mahalleler)
            if secilen_mahalle != "Tümü":
                df_temp = df_temp[df_temp[c_mahalle].astype(str).str.strip() == secilen_mahalle]
        else:
            secilen_mahalle = "Tümü"

        st.sidebar.markdown("---")
        st.sidebar.button("ANALİZİ VE TAHMİNİ BAŞLAT 🚀", on_click=analizi_tetikle)

        if st.session_state.analiz_basladi:
            if df_temp.empty:
                st.warning("⚠️ Seçilen kriterlerde veri bulunamadı.")
            else:
                col_map, col_kpi = st.columns([2, 1])
                with col_map:
                    baslik = "📍 Seçili Bölge Haritası"
                    st.subheader(baslik)
                    merkez = [36.95, 35.10];
                    zoom = 9

                    if not df_temp.empty and c_ilce and c_ilce in df_temp.columns:
                        ilce_adi = str(df_temp.iloc[0][c_ilce]).upper()
                        merkez = ILCE_MERKEZLERI.get(ilce_adi, ILCE_MERKEZLERI["VARSAYILAN"])
                        zoom = 12
                    elif secilen_il != "Tümü" and secilen_il in IL_KOORDINATLARI:
                        merkez = IL_KOORDINATLARI[secilen_il];
                        zoom = 10

                    # HARİTA AYARI: 'CartoDB positron' (Beyaz)
                    m = folium.Map(location=merkez, zoom_start=zoom, tiles='CartoDB positron', control_scale=True)

                    Fullscreen(position='topleft').add_to(m)
                    marker_cluster = MarkerCluster().add_to(m)

                    limit = 1000
                    for idx, row in df_temp.head(limit).iterrows():
                        base_lat, base_lon = ILCE_MERKEZLERI.get(str(row[c_ilce]).upper(),
                                                                 ILCE_MERKEZLERI["VARSAYILAN"])
                        lat = base_lat + random.uniform(-0.02, 0.02)
                        lon = base_lon + random.uniform(-0.02, 0.02)

                        popup_txt = f"{row[c_ilce]}"
                        if c_trafo: popup_txt += f"<br>{row[c_trafo]}"

                        folium.Marker([lat, lon], popup=popup_txt,
                                      icon=folium.Icon(color="blue", icon="info-sign")).add_to(marker_cluster)
                    st_folium(m, width="100%", height=400, returned_objects=[], key=random.random())

                with col_kpi:
                    st.subheader("📊 Özet")
                    st.metric("Toplam Kayıt", len(df_temp))
                    if c_tarih:
                        st.success("✅ Model Hazır.")

                st.divider()

                if c_tarih:
                    with st.spinner('High-Performance AI Modelleri Eğitiliyor...'):
                        daily_df = veri_hazirla_ve_feature_engineering(df_temp, c_tarih)

                        t1, t2, t3, t4 = st.tabs(
                            ["🚀 GELECEK TAHMİNİ", "📉 Geçmiş Doğrulama", "🌡️ Sıcaklık Analizi", "🔍 Nedenler (Detaylı)"])

                        with t1:
                            if len(daily_df) > 5:
                                y_test, y_pred, test_dates, mae, r2, importance, xgb, rf, gb = model_egit(daily_df)

                                if len(future_range) == 2:
                                    f_start, f_end = future_range
                                    st.header(
                                        f"☀️ Kesinti Öngörüsü ({f_start.strftime('%d.%m.%Y')} - {f_end.strftime('%d.%m.%Y')})")

                                    df_future = tahmin_gelecek_olustur(daily_df, xgb, rf, gb, f_start, f_end)

                                    fig_fut = go.Figure()
                                    fig_fut.add_trace(
                                        go.Scatter(x=df_future['TARIH'], y=df_future['TAHMIN'], mode='lines',
                                                   name='AI Tahmini', line=dict(color='#E74C3C', width=3),
                                                   fill='tozeroy'))

                                    if 'SICAKLIK' in df_future.columns:
                                        fig_fut.add_trace(
                                            go.Scatter(x=df_future['TARIH'], y=df_future['SICAKLIK'], mode='lines',
                                                       name='Beklenen Sıcaklık (°C)',
                                                       line=dict(color='blue', dash='dot'), yaxis='y2'))

                                    fig_fut.update_layout(title="Gelecek Dönem Günlük Kesinti ve Sıcaklık Projeksiyonu",
                                                          template="plotly_dark",
                                                          xaxis_title="Tarih", yaxis=dict(title="Tahmini Kesinti"),
                                                          yaxis2=dict(title="Sıcaklık (°C)", overlaying='y',
                                                                      side='right'))
                                    st.plotly_chart(fig_fut, use_container_width=True)

                                    c1, c2 = st.columns(2)
                                    c1.metric("Beklenen Toplam Kesinti", f"{int(df_future['TAHMIN'].sum()):,}")
                                    pik_gun = df_future.loc[df_future['TAHMIN'].idxmax()]
                                    c2.metric("En Riskli Gün", f"{pik_gun['TARIH'].strftime('%d %B %Y')}")
                                else:
                                    st.warning("Lütfen sol menüden geçerli bir gelecek tarih aralığı seçiniz.")
                            else:
                                st.warning(
                                    "Bu bölge/trafo için yeterli geçmiş veri yok (AI için en az 5 gün veri gerekir).")

                        with t2:
                            if len(daily_df) > 5:
                                st.subheader("Geçmiş Performans")
                                m1, m2, m3 = st.columns(3)
                                m1.metric("Doğruluk Skoru (R²)", f"{r2:.2f}")
                                m2.metric("RMSE", f"{int(np.sqrt(mean_squared_error(y_test, y_pred)))}")
                                m3.metric("Ortalama Hata (MAE)", f"{int(mae)}")

                                fig_past = go.Figure()
                                fig_past.add_trace(
                                    go.Scatter(x=test_dates, y=y_test, name='Gerçekleşen', line=dict(color='blue')))
                                fig_past.add_trace(go.Scatter(x=test_dates, y=y_pred, name='AI Tahmini',
                                                              line=dict(color='red', dash='dot')))

                                fig_past.update_layout(template="plotly_dark")
                                st.plotly_chart(fig_past, use_container_width=True)

                        with t3:
                            if 'SICAKLIK' in daily_df.columns:
                                st.subheader("Sıcaklık Korelasyonu")
                                fig_temp = px.scatter(daily_df, x='SICAKLIK', y='KESINTI_SAYISI', trendline="ols")
                                fig_temp.update_layout(template="plotly_dark")
                                st.plotly_chart(fig_temp, use_container_width=True)
                            else:
                                st.info("Sıcaklık verisi yok.")

                        with t4:
                            if len(daily_df) > 5:
                                fig_imp = px.bar(importance, x='importance', y='feature_tr', orientation='h',
                                                 title="Kesintiyi Tetikleyen Ana Sebepler")
                                fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'},
                                                      yaxis_title="Etken Faktör (Türkçe)",
                                                      template="plotly_dark")
                                st.plotly_chart(fig_imp, use_container_width=True)

                                st.info("""
                                **Terimler Sözlüğü:**
                                * **Trend:** Son günlere daha çok önem veren genel kesinti eğilimi.
                                * **Mevsimsel Zamanlama:** Yılın hangi ayında/gününde olduğumuzun etkisi.
                                * **Dünkü/Geçen Haftaki Arıza:** Geçmişteki arızaların bugüne yansıması.
                                * **Dalgalanma:** Arıza sayılarının ne kadar ani değiştiği (Stabilite ölçüsü).
                                """)
                else:
                    st.error("Tarih sütunu bulunamadı.")
        else:
            st.info("👈 Analizi başlatmak için butona basınız.")