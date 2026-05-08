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

# 2. CARICAMENTO DATI E PULIZIA RADICE
@st.cache_data
def load_data():
    df_raw = pd.read_csv("dataset_finale.csv")
    # Definiamo le feature numeriche principali
    colonne_feat = ['target_age_at_acquisition', 'number_of_employees', 'total_funding_usd']
    # Trasformiamo gli 0 in NaN per permettere al KNNImputer di lavorare correttamente
    df_raw[colonne_feat] = df_raw[colonne_feat].replace(0, np.nan)
    return df_raw

# 3. MODELLO PREDIZIONE (RANDOM FOREST) IN CACHE
@st.cache_resource
def get_trained_rf(df_train, feature_cols):
    df_rf = df_train.copy()
    # Imputazione per l'addestramento della foresta
    imputer_rf = KNNImputer(n_neighbors=5)
    df_rf[feature_cols] = imputer_rf.fit_transform(df_rf[feature_cols])
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(df_rf[feature_cols], df_rf['price_disclosed'])
    return rf

# Inizializzazione dati e feature
df = load_data()
feature_cols = ['target_age_at_acquisition', 'number_of_employees', 'total_funding_usd']

# 4. SIDEBAR - ORDINAMENTO BIDDER PER QUALITÀ DATI
st.sidebar.title("⚙️ Configurazione Terminale")

# Calcoliamo la densità dei dati: quanti valori non nulli ha ogni bidder per le feature scelte
bidder_quality = df.groupby('acquiring_company')[feature_cols].count().sum(axis=1).sort_values(ascending=False)
sorted_bidders = bidder_quality.index.tolist()

bidder = st.sidebar.selectbox(
    "Seleziona il Bidder", 
    sorted_bidders,
    help="Le aziende in alto hanno uno storico di acquisizioni con dati più completi (Identikit più affidabile)."
)

n_recs = st.sidebar.slider("Numero di Target (Shortlist)", 1, 15, 5)

# 5. MAIN INTERFACE
st.title("🎯 M&A Tech: Two-Stage Screening")
st.markdown("*Motore Ibrido: Similarità Strategica (KNN) + RF Deal Score (Market Mode)*")
st.markdown("---")

if st.button("Genera Raccomandazioni e Analizza Rischio"):
    with st.spinner("Scansione del mercato e analisi predittiva in corso..."):
        
        # --- STAGE 0: IMPUTAZIONE INTELLIGENTE ---
        # Riordiniamo il dataset basandoci su stime statistiche dei vicini invece che su zero fissi
        df_clean = df.copy()
        imputer = KNNImputer(n_neighbors=5)
        df_clean[feature_cols] = imputer.fit_transform(df_clean[feature_cols])
        
        # --- STAGE 1: KNN (Ricerca su Mercato Libero) ---
        # 1A. Calcoliamo l'Identikit basato sulla media dello storico del Bidder scelto
        storico_bidder = df_clean[df_clean['acquiring_company'] == bidder]
        profilo = storico_bidder[feature_cols].mean().values.reshape(1, -1)
        
        # 1B. Creiamo il "Mercato Libero": escludiamo ciò che il Bidder ha già in pancia
        mercato_libero = df_clean[df_clean['acquiring_company'] != bidder].reset_index(drop=True)
        
        # 1C. Scaliamo e cerchiamo le aziende più vicine al profilo
        scaler = StandardScaler()
        X_mercato_scaled = scaler.fit_transform(mercato_libero[feature_cols])
        profilo_scaled = scaler.transform(profilo)
        
        knn = NearestNeighbors(n_neighbors=n_recs).fit(X_mercato_scaled)
        dist, ind = knn.kneighbors(profilo_scaled)
        
        results = mercato_libero.iloc[ind[0]].copy()
        
        # --- STAGE 2: RANDOM FOREST (Previsione Trasparenza) ---
        rf_model = get_trained_rf(df_clean, feature_cols)
        prob_trasparenza = rf_model.predict_proba(results[feature_cols])[:, 1]
        
        # Salviamo lo score tecnico e quello testuale
        results['Score_Num'] = np.round(prob_trasparenza * 100, 1)
        results['Deal Score'] = results['Score_Num'].astype(str) + "%"

        # --- 6. VISUALIZZAZIONE RISULTATI ---
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Bidder Selezionato", bidder)
        c2.metric("Target Identificati", n_recs)
        c3.metric("Status Analisi", "Mercato Libero")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("📋 Shortlist Strategica & Analisi Predittiva")
        
        # Preparazione DataFrame per la visualizzazione pulita
        display_df = results[['acquired_company', 'target_main_category', 'country_hq', 
                              'target_age_at_acquisition', 'total_funding_usd', 'Deal Score', 'Score_Num']].copy()
        
        display_df.columns = ['Nome Target', 'Settore', 'Paese', 'Età (Anni)', 'Capitale (USD)', 'RF Deal Score', 'Score_Num']
        
        # Formattazione
