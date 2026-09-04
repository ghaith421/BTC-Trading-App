import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from datetime import timedelta
import joblib


# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="BTC Algo Trading",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Trading BTC - Prédiction 1h")


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

@st.cache_resource
def load_model():

    try:
        model = joblib.load("btc_multioutput_rf.pkl")
        scaler = joblib.load("scaler.pkl")

        return model, scaler

    except Exception as e:

        st.error(
            "❌ Impossible de charger le modèle ou le scaler."
        )

        st.exception(e)

        return None, None


model, scaler = load_model()


# Arrêt si modèle non chargé
if model is None or scaler is None:
    st.stop()


# ============================================================
# RÉCUPÉRATION DES DONNÉES BTC
# ============================================================

@st.cache_data(ttl=300)
def fetch_data():

    try:

        df = yf.download(
            "BTC-USD",
            period="7d",
            interval="5m",
            progress=False,
            auto_adjust=False
        )

        # ----------------------------------------------------
        # Vérifier si les données existent
        # ----------------------------------------------------

        if df is None or df.empty:
            return None, None


        # ----------------------------------------------------
        # CORRECTION IMPORTANTE YFINANCE
        # ----------------------------------------------------
        # Certaines versions de yfinance retournent des
        # colonnes MultiIndex.
        #
        # Exemple :
        # ('Close', 'BTC-USD')
        #
        # On transforme cela en :
        # Close
        # ----------------------------------------------------

        if isinstance(df.columns, pd.MultiIndex):

            df.columns = df.columns.get_level_values(0)


        # ----------------------------------------------------
        # Garder uniquement les colonnes nécessaires
        # ----------------------------------------------------

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:

            raise ValueError(
                f"Colonnes manquantes dans les données : "
                f"{missing_columns}"
            )


        # ----------------------------------------------------
        # Supprimer les doublons
        # ----------------------------------------------------

        df = df[
            ~df.index.duplicated(
                keep="first"
            )
        ]


        # ----------------------------------------------------
        # Fréquence 5 minutes
        # ----------------------------------------------------

        df = df.asfreq(
            "5min",
            method="ffill"
        )


        # ====================================================
        # CALCUL DES INDICATEURS
        # ====================================================


        # ----------------------------------------------------
        # RSI
        # ----------------------------------------------------

        delta = df["Close"].diff()

        gain = (
            delta
            .where(delta > 0, 0)
            .rolling(window=14)
            .mean()
        )

        loss = (
            -delta
            .where(delta < 0, 0)
            .rolling(window=14)
            .mean()
        )

        rs = gain / loss

        df["RSI"] = (
            100
            - (
                100
                / (1 + rs)
            )
        )


        # ----------------------------------------------------
        # Moyennes mobiles
        # ----------------------------------------------------

        df["SMA_5"] = (
            df["Close"]
            .rolling(5)
            .mean()
        )

        df["SMA_20"] = (
            df["Close"]
            .rolling(20)
            .mean()
        )

        df["SMA_50"] = (
            df["Close"]
            .rolling(50)
            .mean()
        )


        # ----------------------------------------------------
        # Rendements
        # ----------------------------------------------------

        df["Returns"] = (
            df["Close"]
            .pct_change()
        )


        # ----------------------------------------------------
        # Volatilité
        # ----------------------------------------------------

        df["Volatility"] = (
            df["Returns"]
            .rolling(20)
            .std()
        )


        # ----------------------------------------------------
        # Lags du prix
        # ----------------------------------------------------

        df["Close_lag1"] = (
            df["Close"]
            .shift(1)
        )

        df["Close_lag2"] = (
            df["Close"]
            .shift(2)
        )

        df["Close_lag5"] = (
            df["Close"]
            .shift(5)
        )


        # ----------------------------------------------------
        # Ratio High / Low
        # ----------------------------------------------------

        df["High_low_ratio"] = (
            df["High"]
            / df["Low"]
        )


        # ----------------------------------------------------
        # Supprimer les NaN
        # ----------------------------------------------------

        df.dropna(inplace=True)


        # ====================================================
        # VARIABLES DU MODÈLE
        # ====================================================

        feature_columns = [

            "RSI",
            "SMA_5",
            "SMA_20",
            "SMA_50",
            "Volatility",
            "Close_lag1",
            "Close_lag2",
            "Close_lag5",
            "High_low_ratio",
            "Close"

        ]


        features = df[
            feature_columns
        ].copy()


        return df, features


    except Exception as e:

        st.error(
            "❌ Erreur lors de la récupération "
            "ou du traitement des données BTC."
        )

        st.exception(e)

        return None, None


# ============================================================
# SUPPORTS / RÉSISTANCES
# ============================================================

def detect_supports_resistances(
    df,
    order=5
):

    try:

        local_max_idx = argrelextrema(
            df["High"].values,
            np.greater,
            order=order
        )[0]

        local_min_idx = argrelextrema(
            df["Low"].values,
            np.less,
            order=order
        )[0]


        resistances = (
            df.iloc[local_max_idx]["High"]
            .tail(5)
            .astype(float)
            .tolist()
        )

        supports = (
            df.iloc[local_min_idx]["Low"]
            .tail(5)
            .astype(float)
            .tolist()
        )


        return supports, resistances


    except Exception:

        return [], []


# ============================================================
# RÉCUPÉRATION DES DONNÉES
# ============================================================

df, features = fetch_data()


# ============================================================
# AFFICHAGE
# ============================================================

if df is not None and not df.empty:


    # ========================================================
    # DERNIER PRIX
    # ========================================================

    last_price = float(
        df["Close"].iloc[-1]
    )

    last_time = df.index[-1]


    # ========================================================
    # DERNIERS INDICATEURS
    # ========================================================

    last_rsi = float(
        df["RSI"].iloc[-1]
    )

    last_volatility = float(
        df["Volatility"].iloc[-1]
    )


    # ========================================================
    # DASHBOARD
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "💰 Prix",
        f"${last_price:,.2f}"
    )


    col2.metric(
        "📊 RSI",
        f"{last_rsi:.2f}"
    )


    col3.metric(
        "📈 Volatilité",
        f"{last_volatility:.4f}"
    )


    col4.metric(
        "⏰ MAJ",
        last_time.strftime("%H:%M:%S")
    )


    # ========================================================
    # SUPPORTS / RÉSISTANCES
    # ========================================================

    supports, resistances = (
        detect_supports_resistances(df)
    )


    # ========================================================
    # PRÉPARATION DES DONNÉES POUR LE MODÈLE
    # ========================================================

    try:

        X = features.iloc[-1:].values


        # ----------------------------------------------------
        # Vérification du nombre de variables
        # ----------------------------------------------------

        expected_features = getattr(
            scaler,
            "n_features_in_",
            None
        )


        if (
            expected_features is not None
            and X.shape[1] != expected_features
        ):

            st.error(
                f"❌ Le scaler attend "
                f"{expected_features} variables, "
                f"mais le programme lui en donne "
                f"{X.shape[1]}."
            )

            st.write(
                "Variables utilisées :",
                list(features.columns)
            )

            st.stop()


        # ----------------------------------------------------
        # Standardisation
        # ----------------------------------------------------

        features_scaled = (
            scaler.transform(X)
        )


        # ====================================================
        # PRÉDICTION
        # ====================================================

        raw_prediction = model.predict(
            features_scaled
        )


        preds = np.asarray(
            raw_prediction
        )


        # ----------------------------------------------------
        # Corriger la forme
        # ----------------------------------------------------

        if preds.ndim == 2:

            preds = preds[0]

        elif preds.ndim == 1:

            pass

        else:

            raise ValueError(
                f"Format inattendu des prédictions : "
                f"{preds.shape}"
            )


        # Conversion en float
        preds = preds.astype(float)


    except Exception as e:

        st.error(
            "❌ Erreur lors de la prédiction."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # VÉRIFICATION DES PRÉDICTIONS
    # ========================================================

    number_predictions = len(preds)


    if number_predictions != 12:

        st.warning(
            f"⚠️ Le modèle retourne "
            f"{number_predictions} prédictions "
            f"au lieu de 12."
        )

        # On adapte automatiquement l'horizon
        # au nombre réel de prédictions

        future_times = [

            df.index[-1]
            + timedelta(
                minutes=5 * (i + 1)
            )

            for i in range(number_predictions)
        ]

    else:

        future_times = [

            df.index[-1]
            + timedelta(
                minutes=5 * (i + 1)
            )

            for i in range(12)
        ]


    # ========================================================
    # PRIX PRÉDIT À 1H
    # ========================================================

    future_price = float(
        preds[-1]
    )


    # ========================================================
    # VARIATION
    # ========================================================

    pct_change = (
        future_price - last_price
    ) / last_price


    # ========================================================
    # SIGNAL
    # ========================================================

    if pct_change > 0.003:

        signal = "ACHAT 🟢"

    elif pct_change < -0.003:

        signal = "VENTE 🔴"

    else:

        signal = "ATTENDRE 🟡"


    # ========================================================
    # AFFICHAGE DU SIGNAL
    # ========================================================

    st.markdown(
        f"## Signal : {signal}"
    )

    st.markdown(
        f"### Variation prévue : "
        f"{pct_change * 100:.2f}%"
    )


    # ========================================================
    # GRAPHIQUE
    # ========================================================

    fig = make_subplots(

        rows=2,
        cols=1,

        shared_xaxes=True,

        vertical_spacing=0.03,

        row_heights=[
            0.7,
            0.3
        ]

    )


    # ========================================================
    # BOUGIES BTC
    # ========================================================

    fig.add_trace(

        go.Candlestick(

            x=df.index[-50:],

            open=df["Open"].iloc[-50:],

            high=df["High"].iloc[-50:],

            low=df["Low"].iloc[-50:],

            close=df["Close"].iloc[-50:],

            name="Prix"

        ),

        row=1,
        col=1

    )


    # ========================================================
    # SUPPORTS
    # ========================================================

    for s in supports:

        fig.add_hline(

            y=float(s),

            line_dash="dash",

            line_color="green",

            opacity=0.7,

            row=1,
            col=1

        )


    # ========================================================
    # RÉSISTANCES
    # ========================================================

    for r in resistances:

        fig.add_hline(

            y=float(r),

            line_dash="dash",

            line_color="red",

            opacity=0.7,

            row=1,
            col=1

        )


    # ========================================================
    # PRÉDICTION
    # ========================================================

    fig.add_trace(

        go.Scatter(

            x=future_times,

            y=preds,

            mode="lines+markers",

            name="Prédiction 1h",

            line=dict(

                color="cyan",

                width=3,

                dash="dot"

            )

        ),

        row=1,
        col=1

    )


    # ========================================================
    # SMA 20
    # ========================================================

    fig.add_trace(

        go.Scatter(

            x=df.index[-50:],

            y=df["SMA_20"].iloc[-50:],

            mode="lines",

            name="SMA 20",

            line=dict(

                color="orange"

            )

        ),

        row=1,
        col=1

    )


    # ========================================================
    # RSI
    # ========================================================

    fig.add_trace(

        go.Scatter(

            x=df.index[-50:],

            y=df["RSI"].iloc[-50:],

            mode="lines",

            name="RSI",

            line=dict(

                color="purple"

            )

        ),

        row=2,
        col=1

    )


    # ========================================================
    # RSI 70
    # ========================================================

    fig.add_hline(

        y=70,

        line_dash="dash",

        line_color="red",

        opacity=0.5,

        row=2,
        col=1

    )


    # ========================================================
    # RSI 30
    # ========================================================

    fig.add_hline(

        y=30,

        line_dash="dash",

        line_color="green",

        opacity=0.5,

        row=2,
        col=1

    )


    # ========================================================
    # LAYOUT
    # ========================================================

    fig.update_layout(

        height=700,

        template="plotly_dark",

        xaxis_rangeslider_visible=False,

        margin=dict(
            l=20,
            r=20,
            t=40,
            b=20
        )

    )


    # ========================================================
    # AFFICHAGE GRAPHIQUE
    # ========================================================

    st.plotly_chart(

        fig,

        use_container_width=True

    )


    # ========================================================
    # TABLEAU DES PRÉDICTIONS
    # ========================================================

    with st.expander(
        "🔍 Détails des prédictions"
    ):


        prediction_table = pd.DataFrame({

            "Heure": [

                t.strftime("%H:%M")

                for t in future_times

            ],

            "Prix prédit": [

                round(
                    float(p),
                    2
                )

                for p in preds

            ]

        })


        st.dataframe(

            prediction_table,

            use_container_width=True

        )


    # ========================================================
    # INFORMATIONS MODÈLE
    # ========================================================

    with st.expander(
        "🤖 Informations du modèle"
    ):

        st.write(
            "Nombre de prédictions :",
            len(preds)
        )

        st.write(
            "Variables utilisées :"
        )

        st.write(
            list(features.columns)
        )

        st.write(
            "Dimensions des données envoyées au modèle :",
            X.shape
        )

        st.write(
            "Dimensions après scaling :",
            features_scaled.shape
        )


else:

    st.warning(
        "⏳ Attente des données BTC..."
    )


# ============================================================
# RAFRAÎCHISSEMENT AUTOMATIQUE
# ============================================================

st.markdown(
    """
    <meta http-equiv="refresh" content="300">
    """,
    unsafe_allow_html=True
)
