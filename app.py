import streamlit as st
import pandas as pd

st.title("Demo M&A")
st.write("Se vedi questa scritta, l'app funziona!")
import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

st.title("🎯 M&A Tech: Screening Strategico")

# 1. Caricamento dati
@st.cache_data
def load_data():
    return pd.read_csv("dataset_finale.csv")

df = load_data()

# 2. Configurazione Feature (Devono essere le stesse del tuo modello!)
feature_cols = ['target_age_at_acquisition', 'cross_border', 'number_of_employees', 'total_funding_usd']
# Nota: se il tuo modello usa colonne diverse, aggiungile qui alla lista

# 3. Preparazione Modello (KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols].fillna(0))
knn = NearestNeighbors(n_neighbors=5).fit(X_scaled)

# 4. Interfaccia
bidder = st.sidebar.selectbox("Seleziona il Bidder", df['acquiring_company'].unique())

if st.button("Genera Raccomandazioni"):
    # Calcolo profilo
    profilo = df[df['acquiring_company'] == bidder][feature_cols].mean().values.reshape(1, -1)
    profilo_scaled = scaler.transform(profilo)
    
    # Ricerca
    dist, ind = knn.kneighbors(profilo_scaled)
    
    # Output
    st.write(f"### Target simili per: {bidder}")
    st.dataframe(df.iloc[ind[0]][['acquired_company', 'target_main_category', 'country_hq']])
    st.success("Shortlist generata con successo!")