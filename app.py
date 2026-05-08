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
    colonne_feat = ['target_age_at_acquisition', 'number_of_employees', 'total_funding_usd']
    # Sostituiamo gli 0 con NaN per attivare l'imputatore intelligente
    df_raw[colonne_feat] = df_raw[colonne_feat].replace(0, np.nan)
    return df_raw

# 3. MODELLO PREDIZIONE (RANDOM FOREST) IN CACHE
@st.cache_resource
def get_trained_rf(df_train, feature_cols):
    df_rf = df_train.copy()
    # Usiamo un numero maggiore di vicini (20) per stime più bilanciate e realistiche
    imputer_rf = KNNImputer(n_neighbors=20)
    df_rf[feature_cols] = imputer_rf.fit_transform(df_rf[feature_cols])
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(df_rf[feature_cols], df_rf['price_disclosed'])
    return rf

# Inizializzazione
df = load_data()
feature_cols = ['target_age_at_acquisition', 'number_of_employees', 'total_funding_usd']

# 4. SIDEBAR - ORDINAMENTO BIDDER PER QUALITÀ DATI
st.sidebar.title("⚙️ Configurazione Terminale")
bidder_quality = df.groupby('acquiring_company')[feature_cols].count().sum(axis=1).sort_values(ascending=False)
sorted_bidders = bidder_quality.index.tolist()

bidder = st.sidebar.selectbox(
    "Seleziona il Bidder", 
    sorted_bidders,
    help="Le aziende in alto hanno uno storico di acquisizioni con dati più completi."
)
n_recs = st.sidebar.slider("Numero di Target (Shortlist)", 1, 15, 5)

# 5. MAIN INTERFACE
st.title("🎯 M&A Tech: Two-Stage Screening")
st.markdown("*Motore Ibrido: Analisi Strategica + Realismo Statistico*")
st.markdown("---")

if st.button("Genera Raccomandazioni e Analizza Rischio"):
    with st.spinner("Scansione del mercato e simulazione dati in corso..."):
        
        # --- STAGE 0: IMPUTAZIONE INTELLIGENTE REALISTICA ---
        df_clean = df.copy()
        # Aumentiamo i vicini a 20 per evitare medie distorte da singoli outlier
        imputer = KNNImputer(n_neighbors=20)
        df_clean[feature_cols] = imputer.fit_transform(df_clean[feature_cols])
        
        # AGGIUNTA DI RUMORE CASUALE (Reality Noise)
        # Impedisce che aziende diverse abbiano lo stesso identico capitale stimato
        df_clean['total_funding_usd'] = df_clean['total_funding_usd'].apply(lambda x: x * np.random.uniform(0.9, 1.1))
        
        # --- STAGE 1: KNN (Ricerca su Mercato Libero) ---
        storico_bidder = df_clean[df_clean['acquiring_company'] == bidder]
        profilo = storico_bidder[feature_cols].mean().values.reshape(1, -1)
        
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
        
        results['Score_Num'] = np.round(prob_trasparenza * 100, 1)
        results['Deal Score'] = results['Score_Num'].astype(str) + "%"

        # --- VISUALIZZAZIONE ---
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Bidder", bidder)
        c2.metric("Target Identificati", n_recs)
        c3.metric("Status", "Simulazione Realistica Attiva")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("📋 Shortlist Strategica & Analisi Predittiva")
        
        display_df = results[['acquired_company', 'target_main_category', 'country_hq', 
                              'target_age_at_acquisition', 'total_funding_usd', 'Deal Score', 'Score_Num']].copy()
        
        display_df.columns = ['Nome Target', 'Settore', 'Paese', 'Età (Anni)', 'Capitale (USD)', 'RF Deal Score', 'Score_Num']
        
        # Formattazione estetica dei numeri (Arrotondamento e Valuta Intelligente)
        display_df['Età (Anni)'] = display_df['Età (Anni)'].apply(lambda x: f"{x:.1f}")
        
        def format_currency(x):
            if x >= 1e9:
                return f"${x/1e9:.2f} B" # Formato Miliardi
            else:
                return f"${int(x):,}" # Formato standard con separatore migliaia

        display_df['Capitale (USD)'] = display_df['Capitale (USD)'].apply(format_currency)
        
        def color_score(val):
            try:
                score = float(str(val).replace('%', ''))
                if score >= 70: return 'color: #00CC66; font-weight: bold'
                if score <= 40: return 'color: #FF4B4B; font-weight: bold'
                return 'color: #FFA500; font-weight: bold'
            except: return ''

        # Tabella finale: drop fisico della colonna tecnica
        final_table = display_df.drop(columns=['Score_Num'])
        
        st.dataframe(final_table.style.map(color_score, subset=['RF Deal Score']), 
                     use_container_width=True, 
                     hide_index=True)
        
        csv_data = final_table.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Scarica Report CSV", data=csv_data, file_name=f"MA_RealMode_{bidder}.csv", mime="text/csv")
        
        # GRAFICO
        st.markdown("---")
        st.subheader("🔍 Mappa Visuale del Mercato")
        fig = px.scatter(
            results, x='total_funding_usd', y='target_age_at_acquisition', 
            color='target_main_category', size='number_of_employees',
            hover_name='acquired_company', hover_data={'Deal Score': True},
            template="plotly_dark",
            labels={"total_funding_usd": "Capitale (USD)", "target_age_at_acquisition": "Età Target"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("✅ Analisi completata. I capitali stimati riflettono la distribuzione media dei peer di settore.")
