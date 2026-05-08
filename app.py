import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer

# 1. IMPOSTAZIONI LAYOUT E STILE
st.set_page_config(page_title="M&A Terminal", layout="wide")

st.markdown("""
    <style>
    .main-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. CARICAMENTO DATI E PULIZIA (Conversione 0 in NaN per Imputer)
@st.cache_data
def load_data():
    df_raw = pd.read_csv("dataset_finale.csv")
    colonne_da_pulire = ['target_age_at_acquisition', 'number_of_employees', 'total_funding_usd']
    # Sostituiamo gli 0 con NaN per permettere al KNNImputer di stimare i valori reali
    df_raw[colonne_da_pulire] = df_raw[colonne_da_pulire].replace(0, np.nan)
    return df_raw

# 3. MODELLO PREDIZIONE (RANDOM FOREST) IN CACHE
@st.cache_resource
def get_trained_rf(df_train, feature_cols):
    df_rf = df_train.copy()
    imputer_rf = KNNImputer(n_neighbors=5)
    df_rf[feature_cols] = imputer_rf.fit_transform(df_rf[feature_cols])
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(df_rf[feature_cols], df_rf['price_disclosed'])
    return rf

# Inizializzazione
df = load_data()
feature_cols = ['target_age_at_acquisition', 'number_of_employees', 'total_funding_usd']

# 4. SIDEBAR
st.sidebar.title("⚙️ Configurazione Terminale")
bidder = st.sidebar.selectbox("Seleziona il Bidder", sorted(df['acquiring_company'].unique()))
n_recs = st.sidebar.slider("Numero di Target (Shortlist)", 1, 15, 5)

# 5. MAIN INTERFACE
st.title("🎯 M&A Tech: Two-Stage Screening")
st.markdown("*Motore Ibrido: Similarità Strategica + RF Deal Score (Market Mode)*")
st.markdown("---")

if st.button("Genera Raccomandazioni e Analizza Rischio"):
    with st.spinner("Scansione del mercato e analisi predittiva in corso..."):
        
        # --- STAGE 0: IMPUTAZIONE INTELLIGENTE ---
        df_clean = df.copy()
        imputer = KNNImputer(n_neighbors=5)
        df_clean[feature_cols] = imputer.fit_transform(df_clean[feature_cols])
        
        # --- STAGE 1: KNN (Ricerca su Mercato Libero) ---
        storico_bidder = df_clean[df_clean['acquiring_company'] == bidder]
        profilo = storico_bidder[feature_cols].mean().values.reshape(1, -1)
        
        # Escludiamo le aziende già acquisite dal bidder selezionato
        mercato_libero = df_clean[df_clean['acquiring_company'] != bidder].reset_index(drop=True)
        
        scaler = StandardScaler()
        X_mercato_scaled = scaler.fit_transform(mercato_libero[feature_cols])
        profilo_scaled = scaler.transform(profilo)
        
        knn = NearestNeighbors(n_neighbors=n_recs).fit(X_mercato_scaled)
        dist, ind = knn.kneighbors(profilo_scaled)
        
        results = mercato_libero.iloc[ind[0]].copy()
        
        # --- STAGE 2: RANDOM FOREST (Previsione Trasparenza) ---
        rf_model = get_trained_rf(df_clean, feature_cols)
        prob_trasparenza = rf_model.predict_proba(results[feature_cols])[:, 1]
        
        # Salviamo lo score numerico per i colori e quello testuale per la tabella
        results['Score_Num'] = np.round(prob_trasparenza * 100, 1)
        results['Deal Score'] = results['Score_Num'].astype(str) + "%"

        # --- VISUALIZZAZIONE ---
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Bidder", bidder)
        c2.metric("Target Trovati", n_recs)
        c3.metric("Status", "Mercato Analizzato")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("📋 Shortlist Strategica & Analisi Predittiva")
