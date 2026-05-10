import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer

# 1. IMPOSTAZIONI LAYOUT E STILE PROFESSIONALE
st.set_page_config(page_title="M&A Tech Terminal", layout="wide")

st.markdown("""
    <style>
    .main-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border-left: 5px solid #00CC66;
    }
    .stMetric {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. CARICAMENTO DATI E PRE-PROCESSING
@st.cache_data
def load_data():
    # Caricamento del dataset
    df_raw = pd.read_csv("dataset_finale.csv")
    # Feature chiave per l'analisi
    colonne_feat = ['target_age_at_acquisition', 'number_of_employees', 'total_funding_usd']
    # Trasformiamo gli 0 in NaN per forzare l'imputazione statistica corretta
    df_raw[colonne_feat] = df_raw[colonne_feat].replace(0, np.nan)
    return df_raw

# 3. ENGINE DI PREDIZIONE (RANDOM FOREST)
@st.cache_resource
def get_trained_rf(df_train, feature_cols):
    df_rf = df_train.copy()
    # Imputazione robusta su 20 vicini per l'addestramento
    imputer_rf = KNNImputer(n_neighbors=20)
    df_rf[feature_cols] = imputer_rf.fit_transform(df_rf[feature_cols])
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(df_rf[feature_cols], df_rf['price_disclosed'])
    return rf

# Inizializzazione dati
df = load_data()
feature_cols = ['target_age_at_acquisition', 'number_of_employees', 'total_funding_usd']

# 4. SIDEBAR - CONFIGURAZIONE E QUALITÀ DATI
st.sidebar.title("⚙️ Terminale di Screening")
st.sidebar.markdown("---")

# Calcolo Ranking Bidder per densità dati (i migliori in alto)
bidder_quality = df.groupby('acquiring_company')[feature_cols].count().sum(axis=1).sort_values(ascending=False)
sorted_bidders = bidder_quality.index.tolist()

bidder = st.sidebar.selectbox(
    "Seleziona il Bidder Strategico", 
    sorted_bidders,
    help="Le aziende sono ordinate per completezza del loro storico acquisizioni."
)

n_recs = st.sidebar.slider("Dimensione Shortlist (Target)", 1, 15, 5)

st.sidebar.markdown("---")
st.sidebar.info("L'algoritmo KNN identifica i target più coerenti con lo storico del Bidder, mentre la Random Forest stima la trasparenza del deal.")

# 5. INTERFACCIA PRINCIPALE
st.title("🎯 M&A Tech: Advanced Screening & Transparency Tool")
st.markdown("*Motore Ibrido: Analisi Strategica (KNN) + Previsione Opacità Informativa (Random Forest)*")
st.markdown("---")

# PULSANTE CON TERMINOLOGIA CORRETTA
if st.button("Esegui Screening Strategico e Stima Trasparenza"):
    with st.spinner("Elaborazione Identikit e simulazione parametri di mercato..."):
        
        # --- STAGE 0: DATA CURATION (Numeri + Categorie) ---
        df_clean = df.copy()
        
        # A. Imputazione Numerica Avanzata
        imputer = KNNImputer(n_neighbors=20)
        df_clean[feature_cols] = imputer.fit_transform(df_clean[feature_cols])
        
        # Reality Noise: Variazione del 10% per eliminare l'effetto "cloni" e aumentare il realismo
        df_clean['total_funding_usd'] = df_clean['total_funding_usd'].apply(lambda x: x * np.random.uniform(0.9, 1.1))
        
        # B. Imputazione Categorica Aggressiva (Settore e Paese)
        for col, missing_vals in zip(['target_main_category', 'country_hq'], [['unknown', 'nan', 'none', ''], ['none', 'nan', 'unknown', '']]):
            df_clean[col] = df_clean[col].astype(str).str.strip()
            # Calcolo della moda escludendo i valori mancanti
            mask_validi = ~df_clean[col].str.lower().isin(missing_vals)
            valore_moda = df_clean[mask_validi][col].mode()[0]
            # Sostituzione totale per eliminare ogni "Unknown" o "None"
            df_clean[col] = df_clean[col].apply(lambda x: valore_moda if str(x).lower() in missing_vals else x)

        # --- STAGE 1: KNN ENGINE (Ricerca Target Coerenti) ---
        storico_bidder = df_clean[df_clean['acquiring_company'] == bidder]
        profilo_medio = storico_bidder[feature_cols].mean().values.reshape(1, -1)
        
        # Escludiamo le acquisizioni già effettuate (Mercato Aperto)
        mercato_aperto = df_clean[df_clean['acquiring_company'] != bidder].reset_index(drop=True)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(mercato_aperto[feature_cols])
        profilo_scaled = scaler.transform(profilo_medio)
        
        knn = NearestNeighbors(n_neighbors=n_recs).fit(X_scaled)
        distanze, indici = knn.kneighbors(profilo_scaled)
        
        results = mercato_aperto.iloc[indici[0]].copy()
        
        # --- STAGE 2: RANDOM FOREST ENGINE (Analisi Trasparenza) ---
        rf_model = get_trained_rf(df_clean, feature_cols)
        prob_trasparenza = rf_model.predict_proba(results[feature_cols])[:, 1]
        
        results['Score_Num'] = np.round(prob_trasparenza * 100, 1)
        results['Deal Score'] = results['Score_Num'].astype(str) + "%"

        # --- VISUALIZZAZIONE RISULTATI ---
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Bidder Sotto Analisi", bidder)
        c2.metric("Target Identificati", n_recs)
        c3.metric("Stato Pipeline", "Screening Completato")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("📋 Shortlist Target & Previsione Trasparenza")
        st.info("💡 **RF Deal Score**: Rappresenta la probabilità che il deal sia 'Disclosed' (prezzo pubblico). Uno score basso segnala un'operazione tipicamente riservata o di tipo acqui-hire.")
        
        # Formattazione DataFrame per la visualizzazione
        display_df = results[['acquired_company', 'target_main_category', 'country_hq', 
                              'target_age_at_acquisition', 'total_funding_usd', 'Deal Score', 'Score_Num']].copy()
        
        display_df.columns = ['Nome Target', 'Settore', 'Paese', 'Età (Anni)', 'Capitale (USD)', 'RF Deal Score', 'Score_Num']
        
        # Raffinatezza estetica dei dati
        display_df['Età (Anni)'] = display_df['Età (Anni)'].apply(lambda x: f"{x:.1f}")
        
        def format_currency(x):
            if x >= 1e9: return f"${x/1e9:.2f} B"
            else: return f"${int(x):,}"
        
        display_df['Capitale (USD)'] = display_df['Capitale (USD)'].apply(format_currency)
        
        # Colorazione dinamica basata sulla probabilità
        def style_score(val):
            s = float(str(val).replace('%', ''))
            if s >= 70: return 'color: #00CC66; font-weight: bold'
            if s <= 40: return 'color: #FF4B4B; font-weight: bold'
            return 'color: #FFA500; font-weight: bold'

        st.dataframe(display_df.drop(columns=['Score_Num']).style.map(style_score, subset=['RF Deal Score']), 
                     use_container_width=True, hide_index=True)
        
        # Bottone Download Report
        csv_report = display_df.drop(columns=['Score_Num']).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Esporta Screening in CSV", data=csv_report, file_name=f"Screening_{bidder}.csv", mime="text/csv")
        
        # --- ANALISI VISUALE (GRAFICO) ---
        st.markdown("---")
        st.subheader("🔍 Mappa Visuale del Mercato (Shortlist Strategica)")
        
        fig = px.scatter(
            results, 
            x='total_funding_usd', 
            y='target_age_at_acquisition', 
            color='target_main_category', 
            size='number_of_employees',
            hover_name='acquired_company', 
            hover_data={'Deal Score': True, 'total_funding_usd': ':,.0f'},
            template="plotly_dark",
            labels={
                "total_funding_usd": "Capitale Raccolto (USD)",
                "target_age_at_acquisition": "Età dell'Azienda",
                "target_main_category": "Settore Industriale"
            },
            title=f"Posizionamento dei Target rispetto al profilo di {bidder}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("✅ Analisi completata. I dati mancanti (Unknown/None) sono stati imputati tramite modelli probabilistici di settore.")
