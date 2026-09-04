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
import os


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BTC Algo Trading V2.3",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 BTC Algo Trading — V2.3")


# ============================================================
# PARAMÈTRES
# ============================================================

TUNIS_TZ = ZoneInfo("Africa/Tunis")

REFRESH_SECONDS = 60

TIMEFRAME = "5m"

PERIOD = "7d"

HISTORY_FILE = "prediction_history.csv"

PREDICTION_HORIZON_MINUTES = 60


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
        # CORRECTION MULTIINDEX
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
        # NUMÉRIQUE
        # ====================================================

        for col in required_columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )


        # ====================================================
        # TIMEZONE TUNISIE
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
        # FRÉQUENCE 5 MIN
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
        # HIGH / LOW RATIO
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
            df.iloc[
                local_max_idx
            ]["High"]
            .tail(5)
            .astype(float)
            .tolist()
        )


        supports = (
            df.iloc[
                local_min_idx
            ]["Low"]
            .tail(5)
            .astype(float)
            .tolist()
        )


        return supports, resistances


    except Exception:

        return [], []


# ============================================================
# SIGNAL INTELLIGENT V2.2
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

    score = 0


    # ========================================================
    # MODÈLE ML
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
    # TENDANCE
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
    # RSI
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
    # SUPPORT
    # ========================================================

    support_signal = "NEUTRE 🟡"

    resistance_signal = "NEUTRE 🟡"


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


            if distance_support < 1:

                support_signal = (
                    "PROCHE SUPPORT 🟢"
                )

                score += 10


    # ========================================================
    # RÉSISTANCE
    # ========================================================

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


            if distance_resistance < 1:

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
    # FORCE
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
# HISTORIQUE DES PRÉDICTIONS
# ============================================================

def save_prediction(
    prediction_time,
    target_time,
    current_price,
    predicted_price,
    signal,
    score
):

    new_row = pd.DataFrame({

        "prediction_time": [
            prediction_time.isoformat()
        ],

        "target_time": [
            target_time.isoformat()
        ],

        "current_price": [
            current_price
        ],

        "predicted_price": [
            predicted_price
        ],

        "signal": [
            signal
        ],

        "score": [
            score
        ],

        "actual_price": [
            np.nan
        ],

        "error": [
            np.nan
        ],

        "absolute_error": [
            np.nan
        ],

        "actual_change_pct": [
            np.nan
        ],

        "predicted_change_pct": [
            (
                (
                    predicted_price
                    - current_price
                )
                / current_price
                * 100
            )
        ],

        "direction_correct": [
            np.nan
        ]

    })


    # ========================================================
    # CRÉATION
    # ========================================================

    if not os.path.exists(
        HISTORY_FILE
    ):

        new_row.to_csv(
            HISTORY_FILE,
            index=False
        )

        return


    # ========================================================
    # LECTURE
    # ========================================================

    try:

        history = pd.read_csv(
            HISTORY_FILE
        )

    except Exception:

        new_row.to_csv(
            HISTORY_FILE,
            index=False
        )

        return


    # ========================================================
    # ÉVITER LES DOUBLONS
    # ========================================================

    if len(history) > 0:

        already_exists = (

            (
                history[
                    "prediction_time"
                ].astype(str)
                == prediction_time.isoformat()
            )
            &
            (
                history[
                    "target_time"
                ].astype(str)
                == target_time.isoformat()
            )

        ).any()


        if already_exists:

            return


    # ========================================================
    # AJOUT
    # ========================================================

    history = pd.concat(

        [
            history,
            new_row
        ],

        ignore_index=True

    )


    history.to_csv(
        HISTORY_FILE,
        index=False
    )


# ============================================================
# CHARGER HISTORIQUE
# ============================================================

def load_history():

    if not os.path.exists(
        HISTORY_FILE
    ):

        return pd.DataFrame()


    try:

        history = pd.read_csv(
            HISTORY_FILE
        )


        if history.empty:

            return history


        return history


    except Exception:

        return pd.DataFrame()


# ============================================================
# ÉVALUER LES PRÉDICTIONS
# ============================================================

def evaluate_predictions(
    history,
    df
):

    if history.empty:

        return history


    history = history.copy()


    # ========================================================
    # CONVERSION DATES
    # ========================================================

    history[
        "target_datetime"
    ] = pd.to_datetime(
        history["target_time"],
        utc=True
    )


    # ========================================================
    # PRIX BTC
    # ========================================================

    prices = df[
        ["Close"]
    ].copy()


    prices.index = pd.to_datetime(
        prices.index,
        utc=True
    )


    # ========================================================
    # ÉVALUATION
    # ========================================================

    for i in history.index:

        # Déjà évalué
        if pd.notna(
            history.at[
                i,
                "actual_price"
            ]
        ):

            continue


        target_time = (
            history.at[
                i,
                "target_datetime"
            ]
        )


        # ----------------------------------------------------
        # Vérifier si l'horizon est atteint
        # ----------------------------------------------------

        available = prices[
            prices.index >= target_time
        ]


        if available.empty:

            continue


        # ----------------------------------------------------
        # Prix réel
        # ----------------------------------------------------

        actual_price = float(
            available.iloc[0]["Close"]
        )


        predicted_price = float(
            history.at[
                i,
                "predicted_price"
            ]
        )


        current_price = float(
            history.at[
                i,
                "current_price"
            ]
        )


        # ----------------------------------------------------
        # Erreur
        # ----------------------------------------------------

        error = (
            predicted_price
            - actual_price
        )


        absolute_error = abs(
            error
        )


        # ----------------------------------------------------
        # Variations
        # ----------------------------------------------------

        predicted_change = (

            (
                predicted_price
                - current_price
            )
            / current_price
            * 100

        )


        actual_change = (

            (
                actual_price
                - current_price
            )
            / current_price
            * 100

        )


        # ----------------------------------------------------
        # Direction
        # ----------------------------------------------------

        predicted_direction = (

            1
            if predicted_change > 0
            else -1
            if predicted_change < 0
            else 0

        )


        actual_direction = (

            1
            if actual_change > 0
            else -1
            if actual_change < 0
            else 0

        )


        direction_correct = (

            predicted_direction
            == actual_direction

        )


        # ----------------------------------------------------
        # Sauvegarde
        # ----------------------------------------------------

        history.at[
            i,
            "actual_price"
        ] = actual_price


        history.at[
            i,
            "error"
        ] = error


        history.at[
            i,
            "absolute_error"
        ] = absolute_error


        history.at[
            i,
            "actual_change_pct"
        ] = actual_change


        history.at[
            i,
            "predicted_change_pct"
        ] = predicted_change


        history.at[
            i,
            "direction_correct"
        ] = direction_correct


    # ========================================================
    # SAUVEGARDE
    # ========================================================

    columns_to_save = [

        "prediction_time",
        "target_time",
        "current_price",
        "predicted_price",
        "signal",
        "score",
        "actual_price",
        "error",
        "absolute_error",
        "actual_change_pct",
        "predicted_change_pct",
        "direction_correct"

    ]


    history[
        columns_to_save
    ].to_csv(

        HISTORY_FILE,

        index=False

    )


    return history


# ============================================================
# CALCUL DES PERFORMANCES
# ============================================================

def calculate_metrics(history):

    if history.empty:

        return {

            "n": 0,

            "accuracy": np.nan,

            "mae": np.nan,

            "rmse": np.nan,

            "mean_error": np.nan

        }


    evaluated = history[
        history["actual_price"].notna()
    ].copy()


    if evaluated.empty:

        return {

            "n": 0,

            "accuracy": np.nan,

            "mae": np.nan,

            "rmse": np.nan,

            "mean_error": np.nan

        }


    # ========================================================
    # ACCURACY
    # ========================================================

    direction_values = (
        evaluated[
            "direction_correct"
        ]
        .dropna()
        .astype(bool)
    )


    if len(direction_values) > 0:

        accuracy = (
            direction_values.mean()
            * 100
        )

    else:

        accuracy = np.nan


    # ========================================================
    # MAE
    # ========================================================

    mae = (
        evaluated[
            "absolute_error"
        ]
        .mean()
    )


    # ========================================================
    # RMSE
    # ========================================================

    rmse = np.sqrt(

        np.mean(

            evaluated[
                "error"
            ] ** 2

        )

    )


    # ========================================================
    # ERREUR MOYENNE
    # ========================================================

    mean_error = (
        evaluated["error"]
        .mean()
    )


    return {

        "n": len(evaluated),

        "accuracy": accuracy,

        "mae": mae,

        "rmse": rmse,

        "mean_error": mean_error

    }


# ============================================================
# DONNÉES
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
    # VARIATIONS
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

        change_5m = 0


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

        change_1h = 0


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

        change_24h = 0


    # ========================================================
    # VOLUME
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
    # PRÉDICTION
    # ========================================================

    try:

        X = features.iloc[
            -1:
        ].values


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


        X_scaled = scaler.transform(
            X
        )


        raw_prediction = model.predict(
            X_scaled
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
    # HORIZON
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


    future_price = float(
        preds[-1]
    )


    # ========================================================
    # SIGNAL
    # ========================================================

    analysis = calculate_signal(

        last_price,

        future_price,

        last_rsi,

        sma20,

        sma50,

        supports,

        resistances

    )


    score = analysis["score"]

    signal = analysis["signal"]

    strength = analysis["strength"]

    predicted_change = (
        analysis["predicted_change"]
    )


    # ========================================================
    # SAUVEGARDE PRÉDICTION 1H
    # ========================================================

    target_time = (
        last_time
        + timedelta(
            minutes=PREDICTION_HORIZON_MINUTES
        )
    )


    save_prediction(

        prediction_time=last_time,

        target_time=target_time,

        current_price=last_price,

        predicted_price=future_price,

        signal=signal,

        score=score

    )


    # ========================================================
    # ÉVALUATION HISTORIQUE
    # ========================================================

    history = load_history()


    history = evaluate_predictions(
        history,
        df
    )


    metrics = calculate_metrics(
        history
    )


    # ========================================================
    # ÉTAT
    # ========================================================

    st.success(
        "🟢 Données BTC disponibles"
    )


    st.caption(

        "🕐 Heure Tunisie : "
        + current_time.strftime(
            "%d/%m/%Y %H:%M:%S"
        )

    )


    st.caption(

        "📡 Dernière bougie : "
        + last_time.strftime(
            "%d/%m/%Y %H:%M:%S"
        )

    )


    # ========================================================
    # MARCHÉ
    # ========================================================

    st.subheader(
        "📊 Marché BTC"
    )


    c1, c2, c3, c4 = (
        st.columns(4)
    )


    c1.metric(

        "💰 Prix BTC",

        f"${last_price:,.2f}",

        f"{change_5m:+.2f}% / 5m"

    )


    c2.metric(

        "📈 Variation 1h",

        f"{change_1h:+.2f}%"

    )


    c3.metric(

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


    c4.metric(

        "📊 Volume 24h",

        volume_display

    )


    # ========================================================
    # INDICATEURS
    # ========================================================

    st.subheader(
        "📈 Indicateurs techniques"
    )


    i1, i2, i3, i4 = (
        st.columns(4)
    )


    i1.metric(
        "RSI",
        f"{last_rsi:.2f}"
    )


    i2.metric(
        "Volatilité",
        f"{last_volatility:.4f}"
    )


    i3.metric(
        "SMA 20",
        f"${sma20:,.2f}"
    )


    i4.metric(
        "SMA 50",
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
    # SIGNAL
    # ========================================================

    st.subheader(
        "🤖 Signal intelligent"
    )


    s1, s2, s3, s4 = (
        st.columns(4)
    )


    s1.metric(
        "Signal",
        signal
    )


    s2.metric(
        "Score",
        f"{score:+d} / 100"
    )


    s3.metric(
        "Force",
        strength
    )


    s4.metric(
        "Prévision 1h",
        f"{predicted_change:+.2f}%"
    )


    # ========================================================
    # DÉTAIL SIGNAL
    # ========================================================

    with st.expander(
        "🔎 Détail du signal"
    ):

        d1, d2 = (
            st.columns(2)
        )


        with d1:

            st.write(
                "🤖 Modèle ML :",
                analysis["ml_signal"]
            )

            st.write(
                "📈 Tendance :",
                analysis["trend_signal"]
            )

            st.write(
                "📊 RSI :",
                analysis["rsi_signal"]
            )


        with d2:

            st.write(
                "🟢 Support :",
                analysis["support_signal"]
            )

            st.write(
                "🔴 Résistance :",
                analysis["resistance_signal"]
            )


        st.caption(
            "Le score est un indicateur de confluence "
            "et non une probabilité de réussite."
        )


    # ========================================================
    # PERFORMANCE
    # ========================================================

    st.subheader(
        "🎯 Performance réelle du modèle"
    )


    p1, p2, p3, p4 = (
        st.columns(4)
    )


    if metrics["n"] > 0:

        p1.metric(

            "🎯 Accuracy directionnelle",

            f"{metrics['accuracy']:.2f}%"

        )


        p2.metric(

            "📏 MAE",

            f"${metrics['mae']:,.2f}"

        )


        p3.metric(

            "📐 RMSE",

            f"${metrics['rmse']:,.2f}"

        )


        p4.metric(

            "📊 Prédictions évaluées",

            str(metrics["n"])

        )


    else:

        p1.metric(
            "Accuracy",
            "En attente"
        )

        p2.metric(
            "MAE",
            "En attente"
        )

        p3.metric(
            "RMSE",
            "En attente"
        )

        p4.metric(
            "Évaluées",
            "0"
        )


    st.caption(

        "⚠️ Les performances apparaissent progressivement. "
        "Une prédiction doit attendre son horizon de 1 heure "
        "avant de pouvoir être comparée au prix réel."

    )


    # ========================================================
    # GRAPHIQUE PRÉDIT VS RÉEL
    # ========================================================

    evaluated = history[
        history["actual_price"].notna()
    ].copy()


    if not evaluated.empty:

        st.subheader(
            "📉 Prix prédit vs prix réel"
        )


        evaluated[
            "prediction_time"
        ] = pd.to_datetime(
            evaluated["prediction_time"]
        )


        evaluated = evaluated.sort_values(
            "prediction_time"
        )


        fig_perf = go.Figure()


        fig_perf.add_trace(

            go.Scatter(

                x=evaluated[
                    "prediction_time"
                ],

                y=evaluated[
                    "predicted_price"
                ],

                mode="lines+markers",

                name="Prix prédit"

            )

        )


        fig_perf.add_trace(

            go.Scatter(

                x=evaluated[
                    "prediction_time"
                ],

                y=evaluated[
                    "actual_price"
                ],

                mode="lines+markers",

                name="Prix réel"

            )

        )


        fig_perf.update_layout(

            height=500,

            template="plotly_dark",

            xaxis_title="Temps",

            yaxis_title="Prix BTC",

            hovermode="x unified"

        )


        st.plotly_chart(

            fig_perf,

            use_container_width=True

        )


    else:

        st.info(

            "📊 Le graphique apparaîtra "
            "après les premières prédictions évaluées."

        )


    # ========================================================
    # HISTORIQUE DES PRÉDICTIONS
    # ========================================================

    st.subheader(
        "📝 Historique des prédictions"
    )


    if not history.empty:

        display_history = history.copy()


        display_history = (
            display_history
            .sort_values(
                "prediction_time",
                ascending=False
            )
            .head(30)
        )


        display_columns = [

            "prediction_time",

            "target_time",

            "current_price",

            "predicted_price",

            "actual_price",

            "signal",

            "score",

            "predicted_change_pct",

            "actual_change_pct",

            "absolute_error",

            "direction_correct"

        ]


        display_columns = [

            col
            for col in display_columns

            if col in display_history.columns

        ]


        st.dataframe(

            display_history[
                display_columns
            ],

            use_container_width=True

        )


    else:

        st.info(
            "Aucune prédiction enregistrée."
        )


    # ========================================================
    # GRAPHIQUE BTC
    # ========================================================

    st.subheader(
        "📊 Analyse BTC"
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
    # CANDLE
    # ========================================================

    fig.add_trace(

        go.Candlestick(

            x=df.index[-100:],

            open=df[
                "Open"
            ].iloc[-100:],

            high=df[
                "High"
            ].iloc[-100:],

            low=df[
                "Low"
            ].iloc[-100:],

            close=df[
                "Close"
            ].iloc[-100:],

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

            y=df[
                "SMA_20"
            ].iloc[-100:],

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

            y=df[
                "SMA_50"
            ].iloc[-100:],

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

            y=df[
                "RSI"
            ].iloc[-100:],

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

            y=df[
                "Volume"
            ].iloc[-100:],

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

        hovermode="x unified",

        margin=dict(

            l=20,

            r=20,

            t=40,

            b=20

        )

    )


    st.plotly_chart(

        fig,

        use_container_width=True

    )


    # ========================================================
    # PRÉDICTIONS FUTURES
    # ========================================================

    with st.expander(
        "🔮 Détails des prédictions futures"
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
    # INFORMATIONS
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
            "Horizon d'évaluation : 1 heure"
        )

        st.write(
            "Nombre de bougies :",
            len(df)
        )

        st.write(
            "Nombre de prédictions enregistrées :",
            len(history)
        )

        st.write(
            "Nombre de prédictions évaluées :",
            metrics["n"]
        )

        st.write(
            "Fichier historique :",
            HISTORY_FILE
        )


else:

    st.warning(
        "⏳ Aucune donnée BTC disponible."
    )


# ============================================================
# RAFRAÎCHISSEMENT
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
