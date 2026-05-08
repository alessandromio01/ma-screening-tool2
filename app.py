import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier # Aggiunta per il Modello 2

# Impostazione layout professionale e Stile CSS (Card in stile Dark Mode)
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

# Caricamento dati
@st.cache_data
def load_data():
    return pd.read_csv("dataset_finale.csv")

df = load_data()
# Usiamo le stesse feature sia per il KNN che per la Random Forest base
feature_cols = ['target_age_at_acquisition', 'number_of_employees', 'total_funding_usd']

# Sidebar
st.sidebar.title("⚙️ Configurazione Terminale")
bidder = st.sidebar.selectbox("Seleziona il Bidder", df['acquiring_company'].unique())
n_recs = st.sidebar.slider("Numero di Target (Shortlist)", 1, 15, 5)

# Main Header
st.title("🎯 M&A Tech: Two-Stage Screening")
st.markdown("*Motore Ibrido: Similarità Euclidea (KNN) + Previsione di Successo/Trasparenza (Random Forest)*")
st.markdown("---")

if st.button("Genera Raccomandazioni e Analizza Rischio"):
    with st.spinner("Addestramento modelli e calcolo delle distanze in corso..."):
        
        # 0. Pulizia totale dei NaN
        df_clean = df.copy()
        df_clean[feature_cols] = df_clean[feature_cols].fillna(0)
        
        # --- STAGE 1: K-NEAREST NEIGHBORS (Il Talent Scout) ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean[feature_cols])
        knn = NearestNeighbors(n_neighbors=n_recs).fit(X_scaled)
        
        # Calcolo profilo
        profilo = df_clean[df_clean['acquiring_company'] == bidder][feature_cols].mean().values.reshape(1, -1)
        dist, ind = knn.kneighbors(scaler.transform(profilo))
        
        results = df_clean.iloc[ind[0]].copy()
        results.loc[results['number_of_employees'] == 0, 'number_of_employees'] = 10 # Fix visivo per il grafico
        
        # --- STAGE 2: RANDOM FOREST (Il Giudice) ---
        # Addestriamo una Random Forest veloce per prevedere 'price_disclosed' (Trasparenza del deal)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(df_clean[feature_cols], df_clean['price_disclosed'])
        
        # Facciamo prevedere alla RF la probabilità di trasparenza (classe 1) per la nostra shortlist
        probabilita_trasparenza = rf.predict_proba(results[feature_cols])[:, 1]
        
        # Aggiungiamo lo Score al dataset dei risultati
        results['Deal Score'] = np.round(probabilita_trasparenza * 100, 1).astype(str) + "%"

        # --- UI: VISUALIZZAZIONE ---
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Bidder Selezionato", bidder)
        col2.metric("Target Consigliati", n_recs)
        col3.metric("Status Pipeline", "KNN + RF Attivi")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 1. TABELLA DETTAGLIATA
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("📋 Shortlist Strategica & Analisi Predittiva")
        
        display_df = results[['acquired_company', 'target_main_category', 'country_hq', 
                              'target_age_at_acquisition', 'total_funding_usd', 'Deal Score']].copy()
        
        display_df.columns = ['Nome Target', 'Settore Industriale', 'Paese', 
                              'Età (Anni)', 'Capitale (USD)', 'RF Deal Score (Trasparenza)']
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # BOTTONE DOWNLOAD CSV
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Scarica Report (CSV)",
            data=csv,
            file_name=f'ma_report_{bidder}.csv',
            mime='text/csv',
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 2. GRAFICO INTERATTIVO
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("🔍 Mappa Visuale (Cluster e Dimensioni)")
        
        fig = px.scatter(
            results, 
            x='total_funding_usd', 
            y='target_age_at_acquisition', 
            color='target_main_category', 
            size='number_of_employees',
            hover_name='acquired_company', 
            hover_data={'Deal Score': True}, # Mostra lo score passandoci sopra col mouse!
            title="Relazione tra Età, Capitale e Settore",
            labels={
                "total_funding_usd": "Capitale Raccolto (USD)",
                "target_age_at_acquisition": "Età dell'azienda (Anni)",
                "target_main_category": "Settore Industriale",
                "number_of_employees": "Dipendenti"
            },
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.success("✅ Modelli eseguiti con successo! Pipeline ibrida completata.")
