import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Impostazione layout professionale e Stile CSS
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

# Funzione per addestrare la Random Forest una sola volta all'avvio (Ottimizzazione Senior)
@st.cache_resource
def get_trained_rf(df_train, feature_cols):
    df_clean_train = df_train.copy()
    df_clean_train[feature_cols] = df_clean_train[feature_cols].fillna(0)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(df_clean_train[feature_cols], df_clean_train['price_disclosed'])
    return rf

df = load_data()
feature_cols = ['target_age_at_acquisition', 'number_of_employees', 'total_funding_usd']

# Sidebar
st.sidebar.title("⚙️ Configurazione Terminale")
bidder = st.sidebar.selectbox("Seleziona il Bidder", df['acquiring_company'].unique())
n_recs = st.sidebar.slider("Numero di Target (Shortlist)", 1, 15, 5)

# Main Header
st.title("🎯 M&A Tech: Two-Stage Screening (Real-World Mode)")
st.markdown("*Motore Ibrido: Analisi sul Mercato Aperto + RF Deal Score*")
st.markdown("---")

if st.button("Genera Raccomandazioni e Analizza Rischio"):
    with st.spinner("Scansione del mercato e analisi predittiva in corso..."):
        
        # 0. Imputazione Intelligente dei NaN (NUOVA)
        df_clean = df.copy()
        # Usiamo il KNNImputer per stimare i dati mancanti basandoci sulle aziende simili
        imputer = KNNImputer(n_neighbors=5)
        df_clean[feature_cols] = imputer.fit_transform(df_clean[feature_cols])
        
        # --- STAGE 1: K-NEAREST NEIGHBORS (Il Talent Scout nel Mercato Aperto) ---
        
        # A. Calcoliamo l'Identikit basandoci sullo storico del Bidder
        storico_bidder = df_clean[df_clean['acquiring_company'] == bidder]
        profilo = storico_bidder[feature_cols].mean().values.reshape(1, -1)
        
        # B. Creiamo il "Mercato": ESCLUDIAMO le aziende che il Bidder ha già comprato!
        mercato_target = df_clean[df_clean['acquiring_company'] != bidder].reset_index(drop=True)
        
        # C. Scaliamo i dati e cerchiamo nel mercato
        scaler = StandardScaler()
        X_mercato_scaled = scaler.fit_transform(mercato_target[feature_cols])
        profilo_scaled = scaler.transform(profilo)
        
        knn = NearestNeighbors(n_neighbors=n_recs).fit(X_mercato_scaled)
        dist, ind = knn.kneighbors(profilo_scaled)
        
        # D. Estraiamo i risultati
        results = mercato_target.iloc[ind[0]].copy()
        results.loc[results['number_of_employees'] == 0, 'number_of_employees'] = 10 # Fix visivo per il grafico
        
        # --- STAGE 2: RANDOM FOREST (Il Giudice) ---
        rf = get_trained_rf(df_clean, feature_cols)
        
        # Previsione sulle aziende raccomandate
        probabilita_trasparenza = rf.predict_proba(results[feature_cols])[:, 1]
        
        # Creiamo due colonne: una numerica per i colori, una testuale per la tabella
        results['Deal_Score_Num'] = np.round(probabilita_trasparenza * 100, 1)
        results['Deal Score'] = results['Deal_Score_Num'].astype(str) + "%"

        # --- UI: VISUALIZZAZIONE ---
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Bidder Selezionato", bidder)
        col2.metric("Target Consigliati", n_recs)
        col3.metric("Status Pipeline", "Ricerca su Mercato Attiva")
        st.markdown('</div>', unsafe_allow_html=True)
        
    # 1. TABELLA DETTAGLIATA (Con Colori Dinamici Sicuri)
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("📋 Shortlist Strategica & Analisi Predittiva")
        
        # Prendiamo solo le colonne che ci servono (niente colonna 'Num' extra)
        display_df = results[['acquired_company', 'target_main_category', 'country_hq', 
                              'target_age_at_acquisition', 'total_funding_usd', 'Deal Score']].copy()
        
        display_df.columns = ['Nome Target', 'Settore Industriale', 'Paese', 
                              'Età (Anni)', 'Capitale (USD)', 'RF Deal Score (Trasparenza)']
        display_df.columns = ['Nome Target', 'Settore Industriale', 'Paese', 
                              'Età (Anni)', 'Capitale (USD)', 'RF Deal Score (Trasparenza)']
        
        # NUOVE RIGHE: Arrotondiamo per rendere i dati professionali
        display_df['Età (Anni)'] = display_df['Età (Anni)'].round(1)
        display_df['Capitale (USD)'] = display_df['Capitale (USD)'].apply(lambda x: f"${int(x):,}")
        
        # Funzione che legge la percentuale (es. "45.1%") e decide il colore
        def color_score(val):
            try:
                score = float(val.replace('%', ''))
                if score >= 70:
                    return 'color: #00CC66; font-weight: bold' # Verde
                elif score <= 40:
                    return 'color: #FF4B4B; font-weight: bold' # Rosso
                else:
                    return 'color: #FFA500; font-weight: bold' # Arancione
            except:
                return ''

        # Applichiamo lo stile solo alla colonna finale
        styled_df = display_df.style.map(color_score, subset=['RF Deal Score (Trasparenza)'])
        
        # hide_index=True detto direttamente a Streamlit funziona sempre!
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
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
            hover_data={'Deal Score': True, 'Deal_Score_Num': False}, # Nascondiamo il numero d'appoggio dall'hover
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
        
        st.success("✅ Ricerca nel Mercato Libero completata! Nessuna target storica suggerita.")
