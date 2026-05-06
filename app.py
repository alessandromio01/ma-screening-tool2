import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Impostazione layout professionale e Stile CSS (Card in stile Dark Mode)
st.set_page_config(page_title="M&A Screening Tool", layout="wide")

st.markdown("""
    <style>
    .main-card {
        background-color: #262730; /* Grigio scuro per le card, perfetto per il dark mode */
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
feature_cols = ['target_age_at_acquisition', 'number_of_employees', 'total_funding_usd']

# Sidebar
st.sidebar.title("⚙️ Configurazione")
bidder = st.sidebar.selectbox("Seleziona il Bidder", df['acquiring_company'].unique())
n_recs = st.sidebar.slider("Numero di Target", 1, 15, 5)

# Main Header
st.title("🎯 M&A Tech: Screening Strategico")
st.markdown("---")

if st.button("Genera Raccomandazioni"):
    # Pulizia totale dei NaN prima di iniziare i calcoli
    df_clean = df.copy()
    df_clean[feature_cols] = df_clean[feature_cols].fillna(0)
    
    # Modello
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[feature_cols])
    knn = NearestNeighbors(n_neighbors=n_recs).fit(X_scaled)
    
    # Calcolo profilo con i dati puliti
    profilo = df_clean[df_clean['acquiring_company'] == bidder][feature_cols].mean().values.reshape(1, -1)
    dist, ind = knn.kneighbors(scaler.transform(profilo))
    
    # 1. VISUALIZZAZIONE METRICHE
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Bidder Selezionato", bidder)
    col2.metric("Target Consigliati", n_recs)
    col3.metric("Status", "Modello Attivo")
    st.markdown('</div>', unsafe_allow_html=True)
    
    results = df.iloc[ind[0]]
    
    # 2. TABELLA DETTAGLIATA (Scritte Chiare e Pulite)
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("📋 Shortlist Dettagliata")
    
    # Rinomino le colonne per la visualizzazione
    display_df = results[['acquired_company', 'target_main_category', 'country_hq', 
                          'target_age_at_acquisition', 'total_funding_usd', 'number_of_employees']].copy()
    
    display_df.columns = ['Nome Target', 'Settore Industriale', 'Paese Sede', 
                          'Età (Anni)', 'Capitale Raccolto (USD)', 'Dipendenti']
    
    # Mostro la tabella nascondendo l'indice numerico
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption("Nota: La shortlist è generata calcolando la distanza euclidea tra le feature del bidder e quelle del target nel database.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 3. GRAFICO INTERATTIVO
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("🔍 Analisi Dimensionale Target")
    
    fig = px.scatter(
        results, 
        x='total_funding_usd', 
        y='target_age_at_acquisition', 
        color='target_main_category', 
        size='number_of_employees',
        hover_name='acquired_company', 
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
    st.success("✅ Analisi completata con successo!")
