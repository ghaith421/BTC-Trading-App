import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from scipy.signal import argrelextrema

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import joblib


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BTC Algo Trading V2.2",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Trading BTC - Prédiction 1h")


# ============================================================
# PARAMÈTRES
# ============================================================

TUNIS_TZ = ZoneInfo("Africa/Tunis")

REFRESH_SECONDS = 60

TIMEFRAME = "5m"

PERIOD = "7d"


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

@st.cache_resource
def load_model():

    try:

        model = joblib.load(
            "btc_multioutput_rf.pkl"
        )

        scaler = joblib.load(
            "scaler.pkl"
        )

        return model, scaler

    except Exception as e:

        st.error(
            "❌ Impossible de charger "
            "btc_multioutput_rf.pkl ou scaler.pkl"
        )

        st.exception(e)

        return None, None


model, scaler = load_model()


if model is None or scaler is None:
    st.stop()


# ============================================================
# RÉCUPÉRATION DES DONNÉES
# ============================================================

@st.cache_data(ttl=60)
def fetch_data():

    try:

        df = yf.download(
            "BTC-USD",
            period=PERIOD,
            interval=TIMEFRAME,
            progress=False,
            auto_adjust=False
        )


        if df is None or df.empty:

            return None, None


        # ====================================================
        # CORRECTION MULTIINDEX YFINANCE
        # ====================================================

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = (
                df.columns
                .get_level_values(0)
            )


        # ====================================================
        # COLONNES
        # ====================================================

        required_columns = [

            "Open",
            "High",
            "Low",
            "Close",
            "Volume"

        ]


        missing_columns = [

            col
            for col in required_columns

            if col not in df.columns

        ]


        if missing_columns:

            raise ValueError(
                f"Colonnes manquantes : "
                f"{missing_columns}"
            )


        # ====================================================
        # CONVERSION NUMÉRIQUE
        # ====================================================

        for col in required_columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )


        # ====================================================
        # TIMEZONE
        # ====================================================

        if df.index.tz is not None:

            df.index = (
                df.index
                .tz_convert(
                    "Africa/Tunis"
                )
            )

        else:

            df.index = (
                df.index
                .tz_localize("UTC")
                .tz_convert(
                    "Africa/Tunis"
                )
            )


        # ====================================================
        # NETTOYAGE
        # ====================================================

        df = df[
            ~df.index.duplicated(
                keep="first"
            )
        ]

        df = df.sort_index()


        # ====================================================
        # FRÉQUENCE
        # ====================================================

        df = df.asfreq(
            "5min",
            method="ffill"
        )


        # ====================================================
        # RSI
        # ====================================================

        delta = df["Close"].diff()


        gain = (
            delta
            .where(delta > 0, 0)
            .rolling(14)
            .mean()
        )


        loss = (
            -delta
            .where(delta < 0, 0)
            .rolling(14)
            .mean()
        )


        rs = gain / loss


        df["RSI"] = (
            100
            - (
                100 / (1 + rs)
            )
        )


        # ====================================================
        # SMA
        # ====================================================

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


        # ====================================================
        # RETURNS
        # ====================================================

        df["Returns"] = (
            df["Close"]
            .pct_change()
        )


        # ====================================================
        # VOLATILITÉ
        # ====================================================

        df["Volatility"] = (
            df["Returns"]
            .rolling(20)
            .std()
        )


        # ====================================================
        # LAGS
        # ====================================================

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


        # ====================================================
        # HIGH / LOW
        # ====================================================

        df["High_low_ratio"] = (
            df["High"]
            / df["Low"]
        )


        # ====================================================
        # NETTOYAGE
        # ====================================================

        df.dropna(
            inplace=True
        )


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
            "des données BTC."
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
# CALCUL DE LA CONFLUENCE
# ============================================================

def calculate_signal(
    last_price,
    predicted_price,
    rsi,
    sma20,
    sma50,
    supports,
    resistances
):

    # ========================================================
    # SCORE
    # ========================================================

    score = 0


    # ========================================================
    # 1. MODÈLE ML
    # POIDS : 50 POINTS
    # ========================================================

    predicted_change = (

        (
            predicted_price
            - last_price
        )
        / last_price
        * 100

    )


    if predicted_change > 0.30:

        ml_signal = "HAUSSIER 🟢"

        score += 50


    elif predicted_change < -0.30:

        ml_signal = "BAISSIER 🔴"

        score -= 50


    else:

        ml_signal = "NEUTRE 🟡"


    # ========================================================
    # 2. TENDANCE SMA
    # POIDS : 25 POINTS
    # ========================================================

    if (
        last_price > sma20
        and sma20 > sma50
    ):

        trend_signal = "HAUSSIÈRE 🟢"

        score += 25


    elif (
        last_price < sma20
        and sma20 < sma50
    ):

        trend_signal = "BAISSIÈRE 🔴"

        score -= 25


    else:

        trend_signal = "NEUTRE 🟡"


    # ========================================================
    # 3. RSI
    # POIDS : 15 POINTS
    # ========================================================

    if rsi < 30:

        rsi_signal = "SURVENTE 🟢"

        score += 15


    elif rsi > 70:

        rsi_signal = "SURACHAT 🔴"

        score -= 15


    elif rsi >= 50:

        rsi_signal = "MOMENTUM HAUSSIER 🟢"

        score += 7


    else:

        rsi_signal = "MOMENTUM BAISSIER 🔴"

        score -= 7


    # ========================================================
    # 4. SUPPORT / RÉSISTANCE
    # POIDS : 10 POINTS
    # ========================================================

    support_signal = "NEUTRE 🟡"

    resistance_signal = "NEUTRE 🟡"


    # --------------------------------------------------------
    # Support le plus proche
    # --------------------------------------------------------

    if supports:

        nearest_support = max(
            [
                s
                for s in supports
                if s < last_price
            ],
            default=None
        )

        if nearest_support is not None:

            distance_support = (

                (
                    last_price
                    - nearest_support
                )
                / last_price
                * 100

            )

            if distance_support < 1.0:

                support_signal = (
                    "PROCHE SUPPORT 🟢"
                )

                score += 10


    # --------------------------------------------------------
    # Résistance la plus proche
    # --------------------------------------------------------

    if resistances:

        nearest_resistance = min(

            [
                r
                for r in resistances
                if r > last_price
            ],

            default=None

        )


        if nearest_resistance is not None:

            distance_resistance = (

                (
                    nearest_resistance
                    - last_price
                )
                / last_price
                * 100

            )


            if distance_resistance < 1.0:

                resistance_signal = (
                    "PROCHE RÉSISTANCE 🔴"
                )

                score -= 10


    # ========================================================
    # SIGNAL FINAL
    # ========================================================

    if score >= 50:

        final_signal = "ACHAT 🟢"

    elif score <= -50:

        final_signal = "VENTE 🔴"

    else:

        final_signal = "ATTENDRE 🟡"


    # ========================================================
    # NIVEAU DE CONFLUENCE
    # ========================================================

    if abs(score) >= 75:

        strength = "FORTE"

    elif abs(score) >= 50:

        strength = "MOYENNE"

    else:

        strength = "FAIBLE"


    return {

        "score": score,

        "signal": final_signal,

        "strength": strength,

        "predicted_change": predicted_change,

        "ml_signal": ml_signal,

        "trend_signal": trend_signal,

        "rsi_signal": rsi_signal,

        "support_signal": support_signal,

        "resistance_signal": resistance_signal

    }


# ============================================================
# RÉCUPÉRATION
# ============================================================

df, features = fetch_data()


# ============================================================
# APPLICATION
# ============================================================

if df is not None and not df.empty:


    # ========================================================
    # TEMPS
    # ========================================================

    current_time = datetime.now(
        TUNIS_TZ
    )

    last_time = df.index[-1]


    # ========================================================
    # PRIX
    # ========================================================

    last_price = float(
        df["Close"].iloc[-1]
    )


    # ========================================================
    # VARIATION 5 MIN
    # ========================================================

    if len(df) >= 2:

        price_5m = float(
            df["Close"].iloc[-2]
        )

        change_5m = (

            (
                last_price
                - price_5m
            )
            / price_5m
            * 100

        )

    else:

        price_5m = last_price

        change_5m = 0


    # ========================================================
    # VARIATION 1H
    # ========================================================

    if len(df) >= 13:

        price_1h = float(
            df["Close"].iloc[-13]
        )

        change_1h = (

            (
                last_price
                - price_1h
            )
            / price_1h
            * 100

        )

    else:

        price_1h = last_price

        change_1h = 0


    # ========================================================
    # VARIATION 24H
    # ========================================================

    if len(df) >= 289:

        price_24h = float(
            df["Close"].iloc[-289]
        )

        change_24h = (

            (
                last_price
                - price_24h
            )
            / price_24h
            * 100

        )

    else:

        price_24h = last_price

        change_24h = 0


    # ========================================================
    # VOLUME 24H
    # ========================================================

    if len(df) >= 288:

        volume_24h = float(

            df["Volume"]
            .iloc[-288:]
            .sum()

        )

    else:

        volume_24h = float(
            df["Volume"].sum()
        )


    # ========================================================
    # INDICATEURS
    # ========================================================

    last_rsi = float(
        df["RSI"].iloc[-1]
    )

    last_volatility = float(
        df["Volatility"].iloc[-1]
    )

    sma20 = float(
        df["SMA_20"].iloc[-1]
    )

    sma50 = float(
        df["SMA_50"].iloc[-1]
    )


    # ========================================================
    # SUPPORTS / RÉSISTANCES
    # ========================================================

    supports, resistances = (
        detect_supports_resistances(
            df
        )
    )


    # ========================================================
    # PRÉDICTION ML
    # ========================================================

    try:

        X = features.iloc[
            -1:
        ].values


        # Vérification du scaler

        expected_features = getattr(

            scaler,

            "n_features_in_",

            None

        )


        if (

            expected_features is not None

            and X.shape[1]
            != expected_features

        ):

            st.error(

                f"❌ Le scaler attend "
                f"{expected_features} variables "
                f"mais {X.shape[1]} sont fournies."

            )

            st.stop()


        # Scaling

        features_scaled = (
            scaler.transform(X)
        )


        # Prediction

        raw_prediction = (
            model.predict(
                features_scaled
            )
        )


        preds = np.asarray(
            raw_prediction
        )


        if preds.ndim == 2:

            preds = preds[0]


        preds = preds.astype(
            float
        )


    except Exception as e:

        st.error(
            "❌ Erreur pendant la prédiction."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # PRIX PRÉDIT
    # ========================================================

    future_price = float(
        preds[-1]
    )


    # ========================================================
    # CALCUL SIGNAL INTELLIGENT
    # ========================================================

    analysis = calculate_signal(

        last_price=last_price,

        predicted_price=future_price,

        rsi=last_rsi,

        sma20=sma20,

        sma50=sma50,

        supports=supports,

        resistances=resistances

    )


    score = analysis["score"]

    signal = analysis["signal"]

    strength = analysis["strength"]

    predicted_change = (
        analysis["predicted_change"]
    )


    # ========================================================
    # ÉTAT DES DONNÉES
    # ========================================================

    st.success(
        "🟢 Données BTC disponibles"
    )


    st.caption(

        "🕐 Heure actuelle Tunisie : "
        + current_time.strftime(
            "%d/%m/%Y %H:%M:%S"
        )

    )


    st.caption(

        "📡 Dernière bougie utilisée : "
        + last_time.strftime(
            "%d/%m/%Y %H:%M:%S"
        )

    )


    # ========================================================
    # DASHBOARD MARCHÉ
    # ========================================================

    st.subheader(
        "📊 Marché BTC"
    )


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    col1.metric(

        "💰 Prix BTC",

        f"${last_price:,.2f}",

        f"{change_5m:+.2f}% / 5m"

    )


    col2.metric(

        "📈 Variation 1h",

        f"{change_1h:+.2f}%"

    )


    col3.metric(

        "📅 Variation 24h",

        f"{change_24h:+.2f}%"

    )


    if volume_24h >= 1_000_000_000:

        volume_display = (
            f"${volume_24h / 1_000_000_000:.2f} Md"
        )

    elif volume_24h >= 1_000_000:

        volume_display = (
            f"${volume_24h / 1_000_000:.2f} M"
        )

    else:

        volume_display = (
            f"${volume_24h:,.0f}"
        )


    col4.metric(

        "📊 Volume 24h",

        volume_display

    )


    # ========================================================
    # INDICATEURS
    # ========================================================

    st.subheader(
        "📈 Indicateurs techniques"
    )


    ind1, ind2, ind3, ind4 = (
        st.columns(4)
    )


    ind1.metric(

        "📊 RSI",

        f"{last_rsi:.2f}"

    )


    ind2.metric(

        "📉 Volatilité",

        f"{last_volatility:.4f}"

    )


    ind3.metric(

        "📏 SMA 20",

        f"${sma20:,.2f}"

    )


    ind4.metric(

        "📏 SMA 50",

        f"${sma50:,.2f}"

    )


    # ========================================================
    # TENDANCE
    # ========================================================

    if (

        last_price > sma20
        and sma20 > sma50

    ):

        trend = "📈 HAUSSIÈRE"

    elif (

        last_price < sma20
        and sma20 < sma50

    ):

        trend = "📉 BAISSIÈRE"

    else:

        trend = "↔️ NEUTRE"


    st.info(
        f"**Tendance : {trend}**"
    )


    # ========================================================
    # SIGNAL INTELLIGENT
    # ========================================================

    st.subheader(
        "🤖 Analyse intelligente"
    )


    sig1, sig2, sig3, sig4 = (
        st.columns(4)
    )


    sig1.metric(

        "Signal",

        signal

    )


    sig2.metric(

        "Score",

        f"{score:+d} / 100"

    )


    sig3.metric(

        "Force",

        strength

    )


    sig4.metric(

        "Prévision 1h",

        f"{predicted_change:+.2f}%"

    )


    # ========================================================
    # DÉTAIL DE LA CONFLUENCE
    # ========================================================

    with st.expander(
        "🔎 Voir le détail du signal"
    ):


        detail1, detail2 = (
            st.columns(2)
        )


        with detail1:

            st.write(
                "🤖 **Modèle ML** :",
                analysis["ml_signal"]
            )

            st.write(
                "📈 **Tendance SMA** :",
                analysis["trend_signal"]
            )

            st.write(
                "📊 **RSI** :",
                analysis["rsi_signal"]
            )


        with detail2:

            st.write(
                "🟢 **Support** :",
                analysis["support_signal"]
            )

            st.write(
                "🔴 **Résistance** :",
                analysis["resistance_signal"]
            )

            st.write(
                "📐 **Variation ML 1h** :",
                f"{predicted_change:+.2f}%"
            )


        st.caption(
            "Le score mesure la confluence des indicateurs. "
            "Il ne représente pas une probabilité de réussite."
        )


    # ========================================================
    # SUPPORTS / RÉSISTANCES
    # ========================================================

    with st.expander(
        "📍 Supports et résistances"
    ):


        sr1, sr2 = (
            st.columns(2)
        )


        with sr1:

            st.write(
                "🟢 **Supports**"
            )

            if supports:

                for support in reversed(
                    supports
                ):

                    distance = (

                        (
                            last_price
                            - support
                        )
                        / last_price
                        * 100

                    )

                    st.write(

                        f"${support:,.2f}"
                        f" — "
                        f"{distance:.2f}%"

                    )

            else:

                st.write(
                    "Aucun support détecté."
                )


        with sr2:

            st.write(
                "🔴 **Résistances**"
            )

            if resistances:

                for resistance in resistances:

                    distance = (

                        (
                            resistance
                            - last_price
                        )
                        / last_price
                        * 100

                    )

                    st.write(

                        f"${resistance:,.2f}"
                        f" — "
                        f"{distance:.2f}%"

                    )

            else:

                st.write(
                    "Aucune résistance détectée."
                )


    # ========================================================
    # GRAPHIQUE
    # ========================================================

    st.subheader(
        "📊 Analyse graphique BTC"
    )


    fig = make_subplots(

        rows=3,

        cols=1,

        shared_xaxes=True,

        vertical_spacing=0.03,

        row_heights=[

            0.60,
            0.20,
            0.20

        ]

    )


    # ========================================================
    # CANDLESTICKS
    # ========================================================

    fig.add_trace(

        go.Candlestick(

            x=df.index[-100:],

            open=df["Open"].iloc[-100:],

            high=df["High"].iloc[-100:],

            low=df["Low"].iloc[-100:],

            close=df["Close"].iloc[-100:],

            name="BTC"

        ),

        row=1,

        col=1

    )


    # ========================================================
    # SMA20
    # ========================================================

    fig.add_trace(

        go.Scatter(

            x=df.index[-100:],

            y=df["SMA_20"].iloc[-100:],

            mode="lines",

            name="SMA 20",

            line=dict(

                color="orange",

                width=2

            )

        ),

        row=1,

        col=1

    )


    # ========================================================
    # SMA50
    # ========================================================

    fig.add_trace(

        go.Scatter(

            x=df.index[-100:],

            y=df["SMA_50"].iloc[-100:],

            mode="lines",

            name="SMA 50",

            line=dict(

                color="blue",

                width=2

            )

        ),

        row=1,

        col=1

    )


    # ========================================================
    # SUPPORTS
    # ========================================================

    for support in supports:

        fig.add_hline(

            y=float(support),

            line_dash="dash",

            line_color="green",

            opacity=0.6,

            row=1,

            col=1

        )


    # ========================================================
    # RÉSISTANCES
    # ========================================================

    for resistance in resistances:

        fig.add_hline(

            y=float(resistance),

            line_dash="dash",

            line_color="red",

            opacity=0.6,

            row=1,

            col=1

        )


    # ========================================================
    # PRÉDICTION
    # ========================================================

    future_times = [

        last_time
        + timedelta(
            minutes=5 * (i + 1)
        )

        for i in range(
            len(preds)
        )

    ]


    fig.add_trace(

        go.Scatter(

            x=future_times,

            y=preds,

            mode="lines+markers",

            name="Prédiction ML",

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
    # RSI
    # ========================================================

    fig.add_trace(

        go.Scatter(

            x=df.index[-100:],

            y=df["RSI"].iloc[-100:],

            mode="lines",

            name="RSI",

            line=dict(

                color="purple",

                width=2

            )

        ),

        row=2,

        col=1

    )


    fig.add_hline(

        y=70,

        line_dash="dash",

        line_color="red",

        opacity=0.5,

        row=2,

        col=1

    )


    fig.add_hline(

        y=30,

        line_dash="dash",

        line_color="green",

        opacity=0.5,

        row=2,

        col=1

    )


    # ========================================================
    # VOLUME
    # ========================================================

    fig.add_trace(

        go.Bar(

            x=df.index[-100:],

            y=df["Volume"].iloc[-100:],

            name="Volume",

            opacity=0.5

        ),

        row=3,

        col=1

    )


    # ========================================================
    # LAYOUT
    # ========================================================

    fig.update_layout(

        height=850,

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
    # AFFICHAGE
    # ========================================================

    st.plotly_chart(

        fig,

        use_container_width=True

    )


    # ========================================================
    # PRÉDICTIONS
    # ========================================================

    with st.expander(
        "🔮 Détails des prédictions"
    ):


        prediction_table = pd.DataFrame({

            "Horizon": [

                f"+{5 * (i + 1)} min"

                for i in range(
                    len(preds)
                )

            ],

            "Heure": [

                t.strftime(
                    "%H:%M"
                )

                for t in future_times

            ],

            "Prix prévu": [

                f"${float(p):,.2f}"

                for p in preds

            ],

            "Variation": [

                f"{(
                    (
                        float(p)
                        - last_price
                    )
                    / last_price
                    * 100
                ):+.2f}%"

                for p in preds

            ]

        })


        st.dataframe(

            prediction_table,

            use_container_width=True

        )


    # ========================================================
    # INFORMATIONS TECHNIQUES
    # ========================================================

    with st.expander(
        "ℹ️ Informations techniques"
    ):

        st.write(
            "Source : Yahoo Finance"
        )

        st.write(
            "Actif : BTC-USD"
        )

        st.write(
            "Timeframe : 5 minutes"
        )

        st.write(
            "Période : 7 jours"
        )

        st.write(
            "Nombre de bougies :",
            len(df)
        )

        st.write(
            "Nombre de prédictions :",
            len(preds)
        )

        st.write(
            "Variables du modèle :",
            list(features.columns)
        )

        st.write(
            "Prix actuel :",
            f"${last_price:,.2f}"
        )

        st.write(
            "Prix prévu à 1h :",
            f"${future_price:,.2f}"
        )

        st.write(
            "Score de confluence :",
            f"{score:+d} / 100"
        )


else:

    st.warning(
        "⏳ Aucune donnée BTC disponible."
    )


# ============================================================
# RAFRAÎCHISSEMENT AUTOMATIQUE
# ============================================================

st.markdown(

    f"""
    <meta
        http-equiv="refresh"
        content="{REFRESH_SECONDS}"
    >
    """,

    unsafe_allow_html=True

)
