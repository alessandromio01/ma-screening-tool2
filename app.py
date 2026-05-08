import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer

# 1. IMPOSTAZIONI LAYOUT
st.set_page_config(page_title="M&A Terminal", layout="wide")

st.markdown("""
    <style>
    .main-card { background-color: #262730; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# 2. CARICAMENTO DATI
@st.cache_data
def load_data():
    df_raw = pd.read_csv("dataset_finale.csv")
    colonne_feat = ['target_age_at_acquisition', 'number_of_employees', 'total_funding_usd']
    df_raw[colonne_feat] = df_raw[colonne_feat].replace(0, np.nan)
    return df_raw

# 3. MODELLO RF
@st.cache_resource
def get_trained_rf(df_train, feature_cols):
    df_rf = df_train.copy()
    imputer_rf = KNNImputer(n_neighbors=20)
    df_rf[feature_cols] = imputer_rf.fit_transform(df_rf[feature_cols])
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(df_rf[feature_cols], df_rf['price_disclosed'])
    return rf

df = load_data()
feature_cols = ['target_age_at_acquisition', 'number_of_employees', 'total_funding_usd']

# 4. SIDEBAR
st.sidebar.title("⚙️ Configurazione")
bidder_quality = df.groupby('acquiring_company')[feature_cols].count().sum(axis=1).sort_values(ascending=False)
sorted_bidders = bidder_quality.index.tolist()
bidder = st.sidebar.selectbox("Seleziona il Bidder", sorted_bidders)
n_recs = st.sidebar.slider("Numero di Target", 1, 15, 5)

# 5. INTERFACCIA
st.title("🎯 M&A Tech Terminal")
st.markdown("---")

if st.button("Esegui Analisi"):
    with st.spinner("Pulizia dati e calcolo in corso..."):
        df_clean = df.copy()
        
        # IMPUTAZIONE NUMERICA
        imputer = KNNImputer(n_neighbors=20)
        df_clean[feature_cols] = imputer.fit_transform(df_clean[feature_cols])
        df_clean['total_funding_usd'] = df_clean['total_funding_usd'].apply(lambda x: x * np.random.uniform(0.9, 1.1))
        
        # IMPUTAZIONE CATEGORICA AGGRESSIVA (Settore e Paese)
        for col, missing_vals in zip(['target_main_category', 'country_hq'], [['unknown', 'nan', 'none', ''], ['none', 'nan', 'unknown', '']]):
            df_clean[col] = df_clean[col].astype(str).str.strip()
            # Troviamo la moda escludendo i valori mancanti
            valore_moda = df_clean[~df_clean[col].str.lower().isin(missing_vals)][col].mode()[0]
            # Sostituiamo
            df_clean[col] = df_clean[col].apply(lambda x: valore_moda if str(x).lower() in missing_vals else x)

        # STAGE 1: KNN
        storico = df_clean[df_clean['acquiring_company'] == bidder]
        profilo = storico[feature_cols].mean().values.reshape(1, -1)
        mercato = df_clean[df_clean['acquiring_company'] != bidder].reset_index(drop=True)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(mercato[feature_cols])
        profilo_scaled = scaler.transform(profilo)
        
        knn = NearestNeighbors(n_neighbors=n_recs).fit(X_scaled)
        ind = knn.kneighbors(profilo_scaled)[1]
        results = mercato.iloc[ind[0]].copy()
        
        # STAGE 2: RF
        rf_model = get_trained_rf(df_clean, feature_cols)
        prob = rf_model.predict_proba(results[feature_cols])[:, 1]
        results['Score_Num'] = np.round(prob * 100, 1)
        results['Deal Score'] = results['Score_Num'].astype(str) + "%"

        # VISUALIZZAZIONE
        display_df = results[['acquired_company', 'target_main_category', 'country_hq', 
                              'target_age_at_acquisition', 'total_funding_usd', 'Deal Score', 'Score_Num']].copy()
        display_df.columns = ['Nome Target', 'Settore', 'Paese', 'Età (Anni)', 'Capitale (USD)', 'RF Deal Score', 'Score_Num']
        
        display_df['Età (Anni)'] = display_df['Età (Anni)'].apply(lambda x: f"{x:.1f}")
        display_df['Capitale (USD)'] = display_df['Capitale (USD)'].apply(lambda x: f"${x/1e9:.2f} B" if x >= 1e9 else f"${int(x):,}")

        def color_score(val):
            s = float(str(val).replace('%', ''))
            return 'color: #00CC66; font-weight: bold' if s >= 70 else 'color: #FF4B4B; font-weight: bold' if s <= 40 else 'color: #FFA500; font-weight: bold'

        st.dataframe(display_df.drop(columns=['Score_Num']).style.map(color_score, subset=['RF Deal Score']), use_container_width=True, hide_index=True)
        st.success("✅ Analisi completata. Settori e Paesi 'Unknown' sono stati sostituiti con i valori più probabili.")
