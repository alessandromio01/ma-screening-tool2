# ============================================================
# STREAMLIT APP - TECH M&A SCREENING TOOL
# Coerente con il notebook Colab "Screening_M&A_ML"
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. CONFIGURAZIONE PAGINA
# ============================================================

st.set_page_config(
    page_title="Tech M&A Screening Tool",
    page_icon="➤",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #B0B0B0;
        margin-top: 0px;
        margin-bottom: 25px;
    }

    .metric-card {
        background-color: #1E1E1E;
        padding: 18px;
        border-radius: 12px;
        border-left: 5px solid #00CC66;
        box-shadow: 0 4px 8px rgba(0,0,0,0.25);
    }

    .section-box {
        background-color: #262730;
        padding: 18px;
        border-radius: 12px;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .small-note {
        color: #B0B0B0;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 2. CARICAMENTO DATASET
# ============================================================

@st.cache_data
def load_data(path: str = "dataset_finale.csv") -> pd.DataFrame:
    """
    Carichiamo il dataset finale creato nel notebook Colab.
    Il dataset contiene una riga per ogni operazione M&A.
    """

    df = pd.read_csv(path)

    # Convertiamo in numerico le colonne usate nei modelli.
    numeric_cols_to_convert = [
        "year_of_acquisition_announcement",
        "price_disclosed",
        "target_age_at_acquisition",
        "bidder_age_at_acquisition",
        "cross_border",
        "same_country",
        "target_has_category",
        "number_of_acquisitions",
        "number_of_employees",
        "total_funding_usd",
        "cluster"
    ]

    for col in numeric_cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rimuoviamo anni non plausibili, coerentemente con la correzione fatta nel notebook.
    # Nel dataset era presente un valore 2104 che distorceva i grafici temporali.
    if "year_of_acquisition_announcement" in df.columns:
        df = df[df["year_of_acquisition_announcement"] <= 2026].copy()

    return df


df = load_data()


# ============================================================
# 3. FEATURE SET COERENTI CON IL NOTEBOOK COLAB
# ============================================================

# Feature usate nella parte Nearest Neighbors / target recommendation.
# Sono coerenti con lo Step 6 del notebook.
nn_numeric_features = [
    "year_of_acquisition_announcement",
    "target_age_at_acquisition",
    "bidder_age_at_acquisition",
    "cross_border",
    "same_country",
    "price_disclosed",
    "target_has_category",
    "number_of_acquisitions",
    "number_of_employees",
    "total_funding_usd"
]

nn_categorical_features = [
    "target_main_category",
    "bidder_main_category",
    "country_hq",
    "country_hq_bidder"
]

# Feature usate nella parte supervised Random Forest.
# Qui price_disclosed NON deve stare tra le X, perché è la variabile target.
rf_numeric_features = [
    "year_of_acquisition_announcement",
    "target_age_at_acquisition",
    "bidder_age_at_acquisition",
    "cross_border",
    "same_country",
    "target_has_category",
    "number_of_acquisitions",
    "number_of_employees",
    "total_funding_usd",
    "cluster"
]

rf_categorical_features = [
    "target_main_category",
    "bidder_main_category",
    "country_hq",
    "country_hq_bidder"
]


# ============================================================
# 4. PREPARAZIONE MATRICE PER NEAREST NEIGHBORS
# ============================================================

@st.cache_data
def prepare_nearest_neighbors_matrix(df_input: pd.DataFrame):
    """
    Prepariamo la matrice X per Nearest Neighbors seguendo la logica del Colab:
    - selezione feature numeriche e categoriche;
    - gestione missing values;
    - dummy variables;
    - StandardScaler.
    """

    id_cols = [
        "acquisitions_id",
        "acquired_company",
        "acquiring_company",
        "cluster"
    ]

    model_data = df_input[
        id_cols + nn_numeric_features + nn_categorical_features
    ].copy()

    # Missing values numerici: mediana.
    for col in nn_numeric_features:
        model_data[col] = model_data[col].fillna(model_data[col].median())

    # Missing values categorici: Unknown.
    for col in nn_categorical_features:
        model_data[col] = model_data[col].fillna("Unknown")
        model_data[col] = model_data[col].astype(str).str.strip()

    # Dummy variables per le categoriche.
    categorical_dummies = pd.get_dummies(
        model_data[nn_categorical_features],
        drop_first=False
    )

    # Matrice numerica finale.
    X_raw = pd.concat(
        [
            model_data[nn_numeric_features],
            categorical_dummies
        ],
        axis=1
    )

    # Scaling, fondamentale perché Nearest Neighbors usa distanze.
    scaler = StandardScaler()

    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_raw),
        columns=X_raw.columns,
        index=model_data.index
    )

    return model_data, X_raw, X_scaled, scaler


model_data, X_raw_nn, X_scaled_nn, nn_scaler = prepare_nearest_neighbors_matrix(df)


# ============================================================
# 5. PREPARAZIONE RANDOM FOREST PER PRICE DISCLOSURE
# ============================================================

@st.cache_resource
def train_random_forest(df_input: pd.DataFrame):
    """
    Addestriamo una Random Forest coerente con il task supervised del notebook:
    target = price_disclosed.
    Il modello stima la probabilità che il prezzo del deal sia dichiarato.
    """

    supervised_data = df_input[
        rf_numeric_features + rf_categorical_features + ["price_disclosed"]
    ].copy()

    # Missing values numerici: mediana.
    for col in rf_numeric_features:
        supervised_data[col] = supervised_data[col].fillna(supervised_data[col].median())

    # Missing values categorici: Unknown.
    for col in rf_categorical_features:
        supervised_data[col] = supervised_data[col].fillna("Unknown")
        supervised_data[col] = supervised_data[col].astype(str).str.strip()

    # Target.
    supervised_data["price_disclosed"] = supervised_data["price_disclosed"].fillna(0).astype(int)

    # Dummy variables.
    X_num = supervised_data[rf_numeric_features]
    X_cat = pd.get_dummies(
        supervised_data[rf_categorical_features],
        drop_first=False
    )

    X_rf = pd.concat([X_num, X_cat], axis=1)
    y_rf = supervised_data["price_disclosed"]

    # Random Forest coerente con il notebook.
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=123,
        class_weight="balanced"
    )

    rf_model.fit(X_rf, y_rf)

    feature_importance = pd.DataFrame({
        "feature": X_rf.columns,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)

    return rf_model, X_rf.columns.tolist(), feature_importance


rf_model, rf_feature_columns, rf_feature_importance = train_random_forest(df)


def prepare_rf_features_for_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara le feature delle target raccomandate per la Random Forest,
    allineando le colonne dummy a quelle usate nel training.
    """

    temp = results_df[
        rf_numeric_features + rf_categorical_features
    ].copy()

    for col in rf_numeric_features:
        temp[col] = pd.to_numeric(temp[col], errors="coerce")
        temp[col] = temp[col].fillna(df[col].median())

    for col in rf_categorical_features:
        temp[col] = temp[col].fillna("Unknown")
        temp[col] = temp[col].astype(str).str.strip()

    X_num = temp[rf_numeric_features]
    X_cat = pd.get_dummies(
        temp[rf_categorical_features],
        drop_first=False
    )

    X_results = pd.concat([X_num, X_cat], axis=1)

    # Allineiamo le colonne alle feature usate dalla Random Forest in training.
    X_results = X_results.reindex(
        columns=rf_feature_columns,
        fill_value=0
    )

    return X_results


# ============================================================
# 6. FUNZIONE TARGET RECOMMENDATION
# ============================================================

def generate_recommendations(selected_bidder: str, n_recs: int) -> pd.DataFrame:
    """
    Generiamo una shortlist di target simili al profilo storico della bidder.

    Logica:
    1. prendiamo le acquisizioni storiche della bidder;
    2. calcoliamo il profilo medio sulle feature scalate;
    3. cerchiamo le osservazioni più vicine con Nearest Neighbors;
    4. escludiamo le target già acquisite dalla stessa bidder;
    5. restituiamo le migliori n_recs target.
    """

    bidder_deals = model_data[
        model_data["acquiring_company"] == selected_bidder
    ].copy()

    bidder_indices = bidder_deals.index.tolist()

    if len(bidder_indices) == 0:
        return pd.DataFrame()

    # Profilo medio della bidder, coerente con lo Step 6 del notebook.
    bidder_profile_vector = (
        X_scaled_nn
        .loc[bidder_indices]
        .mean(axis=0)
        .values
        .reshape(1, -1)
    )

    # Cerchiamo più vicini rispetto alla shortlist finale,
    # perché poi escludiamo target già acquisite e duplicati.
    n_neighbors = min(200, len(X_scaled_nn))

    nn_model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean"
    )

    nn_model.fit(X_scaled_nn)

    distances, indices = nn_model.kneighbors(bidder_profile_vector)

    recommendations = df.iloc[indices[0]].copy()
    recommendations["distance"] = distances[0]

    # Escludiamo target già acquisite dalla stessa bidder.
    already_acquired = set(bidder_deals["acquired_company"])

    recommendations = recommendations[
        ~recommendations["acquired_company"].isin(already_acquired)
    ]

    # Evitiamo duplicati sulla stessa target.
    recommendations = (
        recommendations
        .sort_values("distance")
        .drop_duplicates(subset=["acquired_company"])
        .head(n_recs)
        .copy()
    )

    if recommendations.empty:
        return recommendations

    # Random Forest: stima probabilità che il prezzo del deal sia dichiarato.
    X_results_rf = prepare_rf_features_for_results(recommendations)

    disclosure_proba = rf_model.predict_proba(X_results_rf)[:, 1]

    recommendations["rf_disclosure_probability"] = disclosure_proba
    recommendations["rf_deal_score"] = np.round(disclosure_proba * 100, 1)

    return recommendations


# ============================================================
# 7. SIDEBAR
# ============================================================

st.sidebar.title("M&A Screening Terminal")
st.sidebar.markdown("---")

# Ordiniamo le bidder per numero di acquisizioni storiche.
bidder_counts = (
    df["acquiring_company"]
    .value_counts()
    .sort_values(ascending=False)
)

available_bidders = bidder_counts.index.tolist()

selected_bidder = st.sidebar.selectbox(
    "Seleziona la bidder",
    available_bidders,
    help="Le bidder sono ordinate per numero di acquisizioni storiche nel dataset."
)

n_recs = st.sidebar.slider(
    "Numero target in shortlist",
    min_value=3,
    max_value=15,
    value=5,
    step=1
)

st.sidebar.markdown("---")

st.sidebar.info(
    "La demo usa Nearest Neighbors per identificare target simili "
    "al profilo storico della bidder e Random Forest per stimare "
    "la probabilità di price disclosure."
)

show_technical_details = st.sidebar.checkbox(
    "Mostra dettagli tecnici",
    value=False
)


# ============================================================
# 8. HEADER PRINCIPALE
# ============================================================

st.markdown(
    '<div class="main-title">➤ Tech M&A Screening Tool</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Demo interattiva: Nearest Neighbors per target screening + Random Forest per price disclosure'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# ============================================================
# 9. OVERVIEW DATASET
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Deal M&A", f"{df['acquisitions_id'].nunique():,}")
col2.metric("Bidder", f"{df['acquiring_company'].nunique():,}")
col3.metric("Target", f"{df['acquired_company'].nunique():,}")
col4.metric(
    "Periodo",
    f"{int(df['year_of_acquisition_announcement'].min())}–{int(df['year_of_acquisition_announcement'].max())}"
)

st.markdown(
    """
    <div class="section-box">
    <b>Obiettivo della demo:</b> selezionare una bidder, costruire il suo profilo storico di acquisizione
    e identificare una shortlist di target simili sulla base delle feature usate nel notebook Colab.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 10. PROFILO BIDDER
# ============================================================

st.subheader("Profilo storico della bidder selezionata")

bidder_history = df[df["acquiring_company"] == selected_bidder].copy()

profile_col1, profile_col2, profile_col3, profile_col4 = st.columns(4)

profile_col1.metric(
    "Acquisizioni storiche",
    f"{len(bidder_history):,}"
)

profile_col2.metric(
    "Età media target",
    f"{bidder_history['target_age_at_acquisition'].mean():.1f} anni"
)

profile_col3.metric(
    "Quota cross-border",
    f"{bidder_history['cross_border'].mean() * 100:.1f}%"
)

main_category = (
    bidder_history["target_main_category"]
    .dropna()
    .astype(str)
    .value_counts()
)

if len(main_category) > 0:
    profile_col4.metric("Categoria più frequente", main_category.index[0])
else:
    profile_col4.metric("Categoria più frequente", "N/A")


with st.expander("Mostra acquisizioni storiche della bidder"):
    st.dataframe(
        bidder_history[
            [
                "acquired_company",
                "target_main_category",
                "country_hq",
                "year_of_acquisition_announcement",
                "target_age_at_acquisition",
                "cross_border",
                "price_disclosed"
            ]
        ].sort_values("year_of_acquisition_announcement", ascending=False),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 11. ESECUZIONE SCREENING
# ============================================================

st.markdown("---")

if st.button("Esegui screening target", use_container_width=True):

    with st.spinner("Costruzione profilo bidder e ricerca target simili..."):

        results = generate_recommendations(
            selected_bidder=selected_bidder,
            n_recs=n_recs
        )

    if results.empty:
        st.warning(
            "Nessuna raccomandazione disponibile dopo i filtri. "
            "Prova con un'altra bidder o aumenta la dimensione della shortlist."
        )

    else:
        st.success("Screening completato.")

        # ====================================================
        # 11.1 TABELLA SHORTLIST
        # ====================================================

        st.subheader("Shortlist target consigliate")

        display_df = results[
            [
                "acquired_company",
                "target_main_category",
                "country_hq",
                "target_age_at_acquisition",
                "number_of_employees",
                "total_funding_usd",
                "cluster",
                "distance",
                "rf_deal_score"
            ]
        ].copy()

        display_df = display_df.rename(columns={
            "acquired_company": "Target consigliata",
            "target_main_category": "Categoria",
            "country_hq": "Paese",
            "target_age_at_acquisition": "Età target",
            "number_of_employees": "Dipendenti",
            "total_funding_usd": "Funding USD",
            "cluster": "Cluster",
            "distance": "Distanza NN",
            "rf_deal_score": "RF Price Disclosure Score"
        })

        display_df["Età target"] = display_df["Età target"].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
        )

        display_df["Dipendenti"] = display_df["Dipendenti"].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) else "N/A"
        )

        def format_currency(x):
            if pd.isna(x):
                return "N/A"
            if x >= 1e9:
                return f"${x / 1e9:.2f} B"
            if x >= 1e6:
                return f"${x / 1e6:.2f} M"
            return f"${x:,.0f}"

        display_df["Funding USD"] = display_df["Funding USD"].apply(format_currency)

        display_df["Distanza NN"] = display_df["Distanza NN"].apply(
            lambda x: f"{x:.3f}"
        )

        display_df["RF Price Disclosure Score"] = display_df["RF Price Disclosure Score"].apply(
            lambda x: f"{x:.1f}%"
        )

        st.info(
            "La distanza NN misura la similarità rispetto al profilo storico della bidder: "
            "valori più bassi indicano target più vicine. "
            "Il RF Price Disclosure Score indica la probabilità stimata che il prezzo del deal sia dichiarato."
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        csv = display_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Scarica shortlist in CSV",
            data=csv,
            file_name=f"screening_{selected_bidder.replace(' ', '_')}.csv",
            mime="text/csv"
        )

        # ====================================================
        # 11.2 GRAFICO DISTANZE
        # ====================================================

        st.subheader("Target più vicine al profilo storico")

        fig_distance = px.bar(
            results.sort_values("distance", ascending=True),
            x="acquired_company",
            y="distance",
            color="target_main_category",
            template="plotly_dark",
            labels={
                "acquired_company": "Target",
                "distance": "Distanza Nearest Neighbors",
                "target_main_category": "Categoria"
            },
            title=f"Target più simili al profilo storico di {selected_bidder}"
        )

        fig_distance.update_layout(
            xaxis_tickangle=-45,
            height=500
        )

        st.plotly_chart(fig_distance, use_container_width=True)

        # ====================================================
        # 11.3 RANDOM FOREST FEATURE IMPORTANCE
        # ====================================================

        st.subheader("Random Forest: variabili più importanti")

        top_importance = rf_feature_importance.head(12).copy()

        fig_importance = px.bar(
            top_importance.sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            template="plotly_dark",
            labels={
                "importance": "Importance",
                "feature": "Feature"
            },
            title="Top feature importance nella classificazione price_disclosed"
        )

        fig_importance.update_layout(height=500)

        st.plotly_chart(fig_importance, use_container_width=True)


# ============================================================
# 13. FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="small-note">
    <b>Nota metodologica:</b> questa app è una demo interattiva della pipeline sviluppata nel notebook Colab.
    Il sistema non predice con certezza la prossima acquisizione e non sostituisce una due diligence.
    L'obiettivo è supportare il primo screening M&A identificando target simili al profilo storico della bidder.
    </div>
    """,
    unsafe_allow_html=True
)
