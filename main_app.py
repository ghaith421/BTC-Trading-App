# ============================================================
# BTC ALGO TRADING - V2.4
# ML Prediction + Signals + Backtest
# ============================================================

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
import math


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BTC Algo Trading V2.4",
    page_icon="₿",
    layout="wide"
)

st.title("🤖 BTC Algo Trading — V2.4")
st.caption(
    "Machine Learning + Analyse technique + Backtest "
    "⚠️ Simulation éducative, pas une garantie de performance."
)

TUNIS_TZ = ZoneInfo("Africa/Tunis")

REFRESH_SECONDS = 60

MODEL_FILE = "btc_multioutput_rf.pkl"
SCALER_FILE = "scaler.pkl"
HISTORY_FILE = "prediction_history.csv"


# ============================================================
# AUTO REFRESH
# ============================================================

st.markdown(
    f"""
    <meta http-equiv="refresh" content="{REFRESH_SECONDS}">
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_FILE):
        st.error(f"❌ Fichier modèle introuvable : {MODEL_FILE}")
        st.stop()

    if not os.path.exists(SCALER_FILE):
        st.error(f"❌ Fichier scaler introuvable : {SCALER_FILE}")
        st.stop()

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    return model, scaler


model, scaler = load_model()


# ============================================================
# TELECHARGEMENT DES DONNEES
# ============================================================

@st.cache_data(ttl=60)
def fetch_data():

    try:

        df = yf.download(
            "BTC-USD",
            period="7d",
            interval="5m",
            progress=False,
            auto_adjust=False
        )

    except Exception as e:

        st.error(f"Erreur téléchargement BTC : {e}")
        return None, None

    if df is None or df.empty:
        return None, None


    # --------------------------------------------------------
    # Gestion MultiIndex Yahoo Finance
    # --------------------------------------------------------

    if isinstance(df.columns, pd.MultiIndex):

        df.columns = df.columns.get_level_values(0)


    # --------------------------------------------------------
    # Suppression doublons
    # --------------------------------------------------------

    df = df[~df.index.duplicated(keep="first")]


    # --------------------------------------------------------
    # Index temporel
    # --------------------------------------------------------

    if df.index.tz is None:

        df.index = df.index.tz_localize("UTC")

    else:

        df.index = df.index.tz_convert("UTC")


    # --------------------------------------------------------
    # Fréquence 5 minutes
    # --------------------------------------------------------

    df = df.asfreq("5min", method="ffill")


    # --------------------------------------------------------
    # Conversion numérique
    # --------------------------------------------------------

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for col in numeric_columns:

        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )


    # ========================================================
    # INDICATEURS
    # ========================================================

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = df["Close"].diff()

    gain = delta.where(
        delta > 0,
        0
    ).rolling(14).mean()

    loss = (
        -delta.where(
            delta < 0,
            0
        )
        .rolling(14)
        .mean()
    )

    rs = gain / loss

    df["RSI"] = 100 - (
        100 / (1 + rs)
    )


    # --------------------------------------------------------
    # Moyennes mobiles
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Rendements
    # --------------------------------------------------------

    df["Returns"] = (
        df["Close"]
        .pct_change()
    )


    # --------------------------------------------------------
    # Volatilité
    # --------------------------------------------------------

    df["Volatility"] = (
        df["Returns"]
        .rolling(20)
        .std()
    )


    # --------------------------------------------------------
    # Lags
    # --------------------------------------------------------

    df["Close_lag1"] = (
        df["Close"].shift(1)
    )

    df["Close_lag2"] = (
        df["Close"].shift(2)
    )

    df["Close_lag5"] = (
        df["Close"].shift(5)
    )


    # --------------------------------------------------------
    # High / Low ratio
    # --------------------------------------------------------

    df["High_low_ratio"] = (
        df["High"] /
        df["Low"]
    )


    # --------------------------------------------------------
    # Nettoyage
    # --------------------------------------------------------

    df.dropna(inplace=True)


    # ========================================================
    # FEATURES DU MODELE
    # ========================================================

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

    features = df[feature_columns].copy()

    return df, features


# ============================================================
# SUPPORTS / RESISTANCES
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
            .tolist()
        )

        supports = (
            df.iloc[local_min_idx]["Low"]
            .tail(5)
            .tolist()
        )

        return supports, resistances

    except Exception:

        return [], []


# ============================================================
# CALCUL DU SIGNAL
# ============================================================

def calculate_signal(
    current_price,
    predicted_price,
    rsi,
    sma20,
    sma50,
    supports,
    resistances
):

    score = 0


    # ========================================================
    # 1. MACHINE LEARNING
    # ========================================================

    predicted_change = (
        predicted_price - current_price
    ) / current_price


    if predicted_change > 0.003:

        score += 50

    elif predicted_change < -0.003:

        score -= 50


    # ========================================================
    # 2. TENDANCE
    # ========================================================

    if (
        current_price > sma20
        and sma20 > sma50
    ):

        score += 25

    elif (
        current_price < sma20
        and sma20 < sma50
    ):

        score -= 25


    # ========================================================
    # 3. RSI
    # ========================================================

    if rsi < 30:

        score += 15

    elif rsi > 70:

        score -= 15

    elif rsi >= 50:

        score += 7

    else:

        score -= 7


    # ========================================================
    # 4. SUPPORT
    # ========================================================

    if supports:

        nearest_support = min(
            supports,
            key=lambda x: abs(
                current_price - x
            )
        )

        distance_support = abs(
            current_price -
            nearest_support
        ) / current_price


        if distance_support < 0.005:

            score += 10


    # ========================================================
    # 5. RESISTANCE
    # ========================================================

    if resistances:

        nearest_resistance = min(
            resistances,
            key=lambda x: abs(
                current_price - x
            )
        )

        distance_resistance = abs(
            current_price -
            nearest_resistance
        ) / current_price


        if distance_resistance < 0.005:

            score -= 10


    # ========================================================
    # SIGNAL
    # ========================================================

    if score >= 50:

        signal = "ACHAT 🟢"

    elif score <= -50:

        signal = "VENTE 🔴"

    else:

        signal = "ATTENDRE 🟡"


    # ========================================================
    # FORCE
    # ========================================================

    abs_score = abs(score)

    if abs_score >= 75:

        strength = "FORTE"

    elif abs_score >= 50:

        strength = "MOYENNE"

    else:

        strength = "FAIBLE"


    return signal, score, strength


# ============================================================
# HISTORIQUE DES PREDICTIONS
# ============================================================

def load_prediction_history():

    if not os.path.exists(HISTORY_FILE):

        return pd.DataFrame(
            columns=[
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
        )

    try:

        return pd.read_csv(
            HISTORY_FILE
        )

    except Exception:

        return pd.DataFrame()


def save_prediction(
    current_time,
    target_time,
    current_price,
    predicted_price,
    signal,
    score
):

    history = load_prediction_history()


    # --------------------------------------------------------
    # Vérifier si cette prédiction existe déjà
    # --------------------------------------------------------

    if not history.empty:

        existing = history[
            (
                history["prediction_time"]
                .astype(str)
                ==
                str(current_time)
            )
        ]

        if not existing.empty:

            return


    predicted_change = (
        predicted_price -
        current_price
    ) / current_price


    new_row = {

        "prediction_time":
            str(current_time),

        "target_time":
            str(target_time),

        "current_price":
            float(current_price),

        "predicted_price":
            float(predicted_price),

        "signal":
            signal,

        "score":
            float(score),

        "actual_price":
            np.nan,

        "error":
            np.nan,

        "absolute_error":
            np.nan,

        "actual_change_pct":
            np.nan,

        "predicted_change_pct":
            predicted_change * 100,

        "direction_correct":
            np.nan
    }


    history = pd.concat(
        [
            history,
            pd.DataFrame([new_row])
        ],
        ignore_index=True
    )


    history.to_csv(
        HISTORY_FILE,
        index=False
    )


# ============================================================
# EVALUATION DES PREDICTIONS
# ============================================================

def evaluate_predictions(
    history,
    prices
):

    if history.empty:

        return history


    history = history.copy()


    # --------------------------------------------------------
    # Conversion dates
    # --------------------------------------------------------

    history["target_time"] = pd.to_datetime(
        history["target_time"],
        utc=True,
        errors="coerce"
    )


    prices = prices.copy()

    if prices.index.tz is None:

        prices.index = (
            prices.index
            .tz_localize("UTC")
        )

    else:

        prices.index = (
            prices.index
            .tz_convert("UTC")
        )


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    for idx, row in history.iterrows():

        if pd.notna(
            history.loc[idx, "actual_price"]
        ):

            continue


        target_time = row["target_time"]


        if pd.isna(target_time):

            continue


        available = prices[
            prices.index >= target_time
        ]


        if available.empty:

            continue


        actual_price = float(
            available["Close"].iloc[0]
        )


        current_price = float(
            row["current_price"]
        )


        predicted_price = float(
            row["predicted_price"]
        )


        error = (
            actual_price -
            predicted_price
        )


        absolute_error = abs(error)


        actual_change = (
            actual_price -
            current_price
        ) / current_price


        predicted_change = (
            predicted_price -
            current_price
        ) / current_price


        direction_correct = (
            np.sign(actual_change)
            ==
            np.sign(predicted_change)
        )


        history.loc[
            idx,
            "actual_price"
        ] = actual_price


        history.loc[
            idx,
            "error"
        ] = error


        history.loc[
            idx,
            "absolute_error"
        ] = absolute_error


        history.loc[
            idx,
            "actual_change_pct"
        ] = actual_change * 100


        history.loc[
            idx,
            "direction_correct"
        ] = direction_correct


    history.to_csv(
        HISTORY_FILE,
        index=False
    )


    return history


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    df,
    model,
    scaler,
    initial_capital=10000,
    threshold=0.003,
    fee=0.001
):

    data = df.copy()


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


    # ========================================================
    # PREDICTIONS
    # ========================================================

    X = data[
        feature_columns
    ].copy()


    X_scaled = scaler.transform(X)


    predictions = model.predict(
        X_scaled
    )


    # Dernière prédiction du modèle
    predicted_prices = predictions[:, -1]


    data["PredictedPrice"] = (
        predicted_prices
    )


    data["PredictedChange"] = (
        data["PredictedPrice"] -
        data["Close"]
    ) / data["Close"]


    # ========================================================
    # SIGNAL SIMPLE POUR LE BACKTEST
    # ========================================================

    data["Position"] = 0


    data.loc[
        data["PredictedChange"] > threshold,
        "Position"
    ] = 1


    data.loc[
        data["PredictedChange"] < -threshold,
        "Position"
    ] = -1


    # ========================================================
    # RENDEMENT DU BTC
    # ========================================================

    data["MarketReturn"] = (
        data["Close"]
        .pct_change()
        .fillna(0)
    )


    # ========================================================
    # STRATEGIE
    # ========================================================

    # On utilise le signal de la barre précédente
    # afin d'éviter d'utiliser le futur.

    data["StrategyPosition"] = (
        data["Position"].shift(1)
        .fillna(0)
    )


    data["StrategyReturn"] = (
        data["StrategyPosition"] *
        data["MarketReturn"]
    )


    # ========================================================
    # FRAIS
    # ========================================================

    position_change = (
        data["StrategyPosition"]
        .diff()
        .abs()
        .fillna(0)
    )


    data["TradingCost"] = (
        position_change * fee
    )


    data["NetStrategyReturn"] = (
        data["StrategyReturn"] -
        data["TradingCost"]
    )


    # ========================================================
    # CAPITAL
    # ========================================================

    data["StrategyEquity"] = (
        initial_capital *
        (
            1 +
            data["NetStrategyReturn"]
        ).cumprod()
    )


    # ========================================================
    # BUY & HOLD
    # ========================================================

    first_price = (
        data["Close"].iloc[0]
    )


    data["BuyHoldEquity"] = (
        initial_capital *
        data["Close"] /
        first_price
    )


    # ========================================================
    # DRAWDOWN
    # ========================================================

    rolling_max = (
        data["StrategyEquity"]
        .cummax()
    )


    data["Drawdown"] = (
        data["StrategyEquity"] /
        rolling_max
        - 1
    )


    # ========================================================
    # STATISTIQUES
    # ========================================================

    final_equity = float(
        data["StrategyEquity"].iloc[-1]
    )


    final_buyhold = float(
        data["BuyHoldEquity"].iloc[-1]
    )


    total_return = (
        final_equity /
        initial_capital
        - 1
    )


    buyhold_return = (
        final_buyhold /
        initial_capital
        - 1
    )


    max_drawdown = float(
        data["Drawdown"].min()
    )


    # ========================================================
    # TRADES
    # ========================================================

    trades = []


    current_position = 0
    entry_price = None
    entry_time = None


    for i in range(
        1,
        len(data)
    ):

        position = int(
            data["StrategyPosition"].iloc[i]
        )


        price = float(
            data["Close"].iloc[i]
        )


        timestamp = (
            data.index[i]
        )


        # ----------------------------------------------------
        # Nouvelle position
        # ----------------------------------------------------

        if (
            position != current_position
        ):

            # Fermer ancienne position
            if current_position != 0:

                if current_position == 1:

                    trade_return = (
                        price -
                        entry_price
                    ) / entry_price

                else:

                    trade_return = (
                        entry_price -
                        price
                    ) / entry_price


                trade_return -= (
                    fee * 2
                )


                trades.append({

                    "Entrée":
                        entry_time,

                    "Sortie":
                        timestamp,

                    "Position":
                        "LONG"
                        if current_position == 1
                        else "SHORT",

                    "Prix entrée":
                        entry_price,

                    "Prix sortie":
                        price,

                    "Rendement %":
                        trade_return * 100,

                    "Résultat":
                        "GAIN"
                        if trade_return > 0
                        else "PERTE"
                })


            # Ouvrir nouvelle position
            if position != 0:

                entry_price = price
                entry_time = timestamp


            current_position = position


    trades_df = pd.DataFrame(
        trades
    )


    # ========================================================
    # WIN RATE
    # ========================================================

    if not trades_df.empty:

        win_rate = (
            (
                trades_df["Rendement %"]
                > 0
            ).mean()
        )

        number_trades = len(
            trades_df
        )

    else:

        win_rate = 0
        number_trades = 0


    # ========================================================
    # SHARPE
    # ========================================================

    returns = (
        data["NetStrategyReturn"]
    )


    std_return = returns.std()


    if std_return != 0:

        sharpe = (
            returns.mean() /
            std_return
        ) * np.sqrt(
            365 * 24 * 12
        )

    else:

        sharpe = 0


    # ========================================================
    # RESULTATS
    # ========================================================

    stats = {

        "Capital initial":
            initial_capital,

        "Capital final":
            final_equity,

        "Rendement stratégie":
            total_return,

        "Capital Buy & Hold":
            final_buyhold,

        "Rendement Buy & Hold":
            buyhold_return,

        "Drawdown maximal":
            max_drawdown,

        "Win Rate":
            win_rate,

        "Nombre de trades":
            number_trades,

        "Sharpe":
            sharpe
    }


    return data, trades_df, stats


# ============================================================
# RECUPERATION DES DONNEES
# ============================================================

df, features = fetch_data()


# ============================================================
# SI DONNEES DISPONIBLES
# ============================================================

if df is not None and not df.empty:


    # ========================================================
    # PRIX ACTUEL
    # ========================================================

    last_price = float(
        df["Close"].iloc[-1]
    )


    last_data_time = (
        df.index[-1]
    )


    current_time = (
        datetime.now(
            TUNIS_TZ
        )
    )


    last_data_tunis = (
        last_data_time
        .tz_convert(TUNIS_TZ)
    )


    # ========================================================
    # INDICATEURS
    # ========================================================

    current_rsi = float(
        df["RSI"].iloc[-1]
    )


    current_sma20 = float(
        df["SMA_20"].iloc[-1]
    )


    current_sma50 = float(
        df["SMA_50"].iloc[-1]
    )


    current_volatility = float(
        df["Volatility"].iloc[-1]
    )


    # ========================================================
    # SUPPORTS / RESISTANCES
    # ========================================================

    supports, resistances = (
        detect_supports_resistances(df)
    )


    # ========================================================
    # PREDICTION ML
    # ========================================================

    features_scaled = (
        scaler.transform(
            features.iloc[-1:].values
        )
    )


    preds = model.predict(
        features_scaled
    )[0]


    future_times = [

        df.index[-1]
        +
        timedelta(
            minutes=5 * (i + 1)
        )

        for i in range(12)
    ]


    future_price = float(
        preds[-1]
    )


    pct_change = (
        future_price -
        last_price
    ) / last_price


    # ========================================================
    # SIGNAL
    # ========================================================

    signal, score, strength = (
        calculate_signal(
            last_price,
            future_price,
            current_rsi,
            current_sma20,
            current_sma50,
            supports,
            resistances
        )
    )


    # ========================================================
    # SAUVEGARDE PREDICTION
    # ========================================================

    prediction_time = (
        df.index[-1]
    )


    target_time = (
        df.index[-1]
        +
        timedelta(
            hours=1
        )
    )


    save_prediction(
        prediction_time,
        target_time,
        last_price,
        future_price,
        signal,
        score
    )


    # ========================================================
    # EVALUATION HISTORIQUE
    # ========================================================

    history = load_prediction_history()


    history = evaluate_predictions(
        history,
        df
    )


    # ========================================================
    # TITRE SIGNAL
    # ========================================================

    st.markdown(
        f"""
        ## Signal : {signal}

        ### Variation prédite à 1h :
        **{pct_change * 100:.2f}%**

        **Score : {score}/100 — Force : {strength}**
        """
    )


    # ========================================================
    # METRICS PRINCIPALES
    # ========================================================

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )


    col1.metric(
        "💰 BTC",
        f"${last_price:,.2f}"
    )


    col2.metric(
        "📊 RSI",
        f"{current_rsi:.2f}"
    )


    col3.metric(
        "📈 Volatilité",
        f"{current_volatility:.4f}"
    )


    col4.metric(
        "🤖 Prévision 1h",
        f"${future_price:,.2f}",
        f"{pct_change * 100:.2f}%"
    )


    col5.metric(
        "⏰ Heure actuelle",
        current_time.strftime(
            "%H:%M:%S"
        )
    )


    st.caption(
        "Dernière bougie BTC disponible : "
        +
        last_data_tunis.strftime(
            "%d/%m/%Y %H:%M:%S"
        )
        +
        " — Heure Tunisie"
    )


    # ========================================================
    # GRAPHIQUE PRINCIPAL
    # ========================================================

    fig = make_subplots(

        rows=2,
        cols=1,

        shared_xaxes=True,

        vertical_spacing=0.03,

        row_heights=[
            0.70,
            0.30
        ]
    )


    # --------------------------------------------------------
    # Chandeliers
    # --------------------------------------------------------

    recent = df.tail(100)


    fig.add_trace(

        go.Candlestick(

            x=recent.index,

            open=recent["Open"],

            high=recent["High"],

            low=recent["Low"],

            close=recent["Close"],

            name="BTC"
        ),

        row=1,
        col=1
    )


    # --------------------------------------------------------
    # SMA20
    # --------------------------------------------------------

    fig.add_trace(

        go.Scatter(

            x=recent.index,

            y=recent["SMA_20"],

            mode="lines",

            name="SMA 20",

            line=dict(
                width=2
            )
        ),

        row=1,
        col=1
    )


    # --------------------------------------------------------
    # SMA50
    # --------------------------------------------------------

    fig.add_trace(

        go.Scatter(

            x=recent.index,

            y=recent["SMA_50"],

            mode="lines",

            name="SMA 50",

            line=dict(
                width=2
            )
        ),

        row=1,
        col=1
    )


    # --------------------------------------------------------
    # Supports
    # --------------------------------------------------------

    for support in supports:

        fig.add_hline(

            y=support,

            line_dash="dash",

            opacity=0.6,

            annotation_text="Support",

            row=1,

            col=1
        )


    # --------------------------------------------------------
    # Resistances
    # --------------------------------------------------------

    for resistance in resistances:

        fig.add_hline(

            y=resistance,

            line_dash="dash",

            opacity=0.6,

            annotation_text="Resistance",

            row=1,

            col=1
        )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    fig.add_trace(

        go.Scatter(

            x=future_times,

            y=preds,

            mode="lines+markers",

            name="🤖 Prediction ML",

            line=dict(
                width=3,
                dash="dot"
            )
        ),

        row=1,
        col=1
    )


    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    fig.add_trace(

        go.Scatter(

            x=recent.index,

            y=recent["RSI"],

            mode="lines",

            name="RSI"
        ),

        row=2,
        col=1
    )


    fig.add_hline(

        y=70,

        line_dash="dash",

        opacity=0.5,

        row=2,

        col=1
    )


    fig.add_hline(

        y=30,

        line_dash="dash",

        opacity=0.5,

        row=2,

        col=1
    )


    fig.update_layout(

        height=750,

        template="plotly_dark",

        xaxis_rangeslider_visible=False,

        hovermode="x unified"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # PREDICTIONS 1H
    # ========================================================

    st.subheader(
        "🤖 Prédictions des 12 prochaines bougies"
    )


    prediction_table = pd.DataFrame({

        "Heure": [
            t.tz_convert(
                TUNIS_TZ
            ).strftime("%H:%M")
            for t in future_times
        ],

        "Prix prédit ($)": [
            round(
                float(p),
                2
            )
            for p in preds
        ],

        "Variation vs actuel (%)": [

            round(
                (
                    float(p) -
                    last_price
                )
                /
                last_price
                *
                100,

                3
            )

            for p in preds
        ]
    })


    st.dataframe(
        prediction_table,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # BACKTEST
    # ========================================================

    st.markdown("---")

    st.header(
        "📊 Backtest de la stratégie"
    )


    st.info(
        "Le backtest simule les signaux sur les données "
        "historiques disponibles. Il inclut des frais de "
        "transaction simulés et ne représente pas des "
        "résultats futurs garantis."
    )


    # ========================================================
    # PARAMETRES BACKTEST
    # ========================================================

    col1, col2, col3 = st.columns(3)


    initial_capital = col1.number_input(

        "💰 Capital initial",

        min_value=100.0,

        max_value=1000000.0,

        value=10000.0,

        step=1000.0
    )


    threshold_percent = col2.number_input(

        "🎯 Seuil signal (%)",

        min_value=0.1,

        max_value=10.0,

        value=0.3,

        step=0.1
    )


    fee_percent = col3.number_input(

        "💸 Frais par transaction (%)",

        min_value=0.0,

        max_value=2.0,

        value=0.1,

        step=0.05
    )


    # ========================================================
    # EXECUTION BACKTEST
    # ========================================================

    backtest_data, trades_df, stats = (
        run_backtest(

            df,

            model,

            scaler,

            initial_capital=

                initial_capital,

            threshold=

                threshold_percent / 100,

            fee=

                fee_percent / 100
        )
    )


    # ========================================================
    # RESULTATS
    # ========================================================

    st.subheader(
        "📌 Résultats"
    )


    b1, b2, b3, b4 = st.columns(4)


    b1.metric(

        "Capital final",

        f"${stats['Capital final']:,.2f}",

        f"{stats['Rendement stratégie'] * 100:.2f}%"
    )


    b2.metric(

        "Buy & Hold",

        f"${stats['Capital Buy & Hold']:,.2f}",

        f"{stats['Rendement Buy & Hold'] * 100:.2f}%"
    )


    b3.metric(

        "Win Rate",

        f"{stats['Win Rate'] * 100:.2f}%"
    )


    b4.metric(

        "Trades",

        f"{stats['Nombre de trades']}"
    )


    b5, b6, b7 = st.columns(3)


    b5.metric(

        "📉 Drawdown maximal",

        f"{stats['Drawdown maximal'] * 100:.2f}%"
    )


    b6.metric(

        "📐 Sharpe Ratio",

        f"{stats['Sharpe']:.2f}"
    )


    b7.metric(

        "💵 Capital initial",

        f"${stats['Capital initial']:,.2f}"
    )


    # ========================================================
    # GRAPHIQUE EQUITY CURVE
    # ========================================================

    st.subheader(
        "📈 Évolution du capital"
    )


    equity_fig = go.Figure()


    equity_fig.add_trace(

        go.Scatter(

            x=backtest_data.index,

            y=backtest_data[
                "StrategyEquity"
            ],

            mode="lines",

            name="🤖 Stratégie"
        )
    )


    equity_fig.add_trace(

        go.Scatter(

            x=backtest_data.index,

            y=backtest_data[
                "BuyHoldEquity"
            ],

            mode="lines",

            name="🏦 Buy & Hold"
        )
    )


    equity_fig.update_layout(

        height=500,

        template="plotly_dark",

        hovermode="x unified",

        yaxis_title="Capital ($)",

        xaxis_title="Date"
    )


    st.plotly_chart(

        equity_fig,

        use_container_width=True
    )


    # ========================================================
    # DRAWDOWN
    # ========================================================

    st.subheader(
        "📉 Drawdown"
    )


    drawdown_fig = go.Figure()


    drawdown_fig.add_trace(

        go.Scatter(

            x=backtest_data.index,

            y=backtest_data[
                "Drawdown"
            ] * 100,

            mode="lines",

            name="Drawdown"
        )
    )


    drawdown_fig.update_layout(

        height=350,

        template="plotly_dark",

        yaxis_title="Drawdown (%)",

        xaxis_title="Date"
    )


    st.plotly_chart(

        drawdown_fig,

        use_container_width=True
    )


    # ========================================================
    # HISTORIQUE DES TRADES
    # ========================================================

    st.subheader(
        "📋 Historique des trades"
    )


    if not trades_df.empty:

        display_trades = trades_df.copy()


        for col in [
            "Prix entrée",
            "Prix sortie"
        ]:

            display_trades[col] = (
                display_trades[col]
                .round(2)
            )


        display_trades[
            "Rendement %"
        ] = display_trades[
            "Rendement %"
        ].round(2)


        st.dataframe(

            display_trades,

            use_container_width=True,

            hide_index=True
        )

    else:

        st.warning(
            "Aucun trade détecté avec "
            "les paramètres actuels."
        )


    # ========================================================
    # EVALUATION DU MODELE ML
    # ========================================================

    st.markdown("---")

    st.header(
        "🧠 Évaluation du modèle ML"
    )


    if not history.empty:

        evaluated = history[
            history[
                "actual_price"
            ].notna()
        ].copy()


        if not evaluated.empty:

            # ------------------------------------------------
            # Direction Accuracy
            # ------------------------------------------------

            direction_accuracy = (
                evaluated[
                    "direction_correct"
                ]
                .astype(bool)
                .mean()
            )


            # ------------------------------------------------
            # MAE
            # ------------------------------------------------

            mae = (
                evaluated[
                    "absolute_error"
                ]
                .mean()
            )


            # ------------------------------------------------
            # RMSE
            # ------------------------------------------------

            rmse = math.sqrt(

                (
                    evaluated["error"] ** 2
                ).mean()
            )


            # ------------------------------------------------
            # Mean Error
            # ------------------------------------------------

            mean_error = (
                evaluated["error"]
                .mean()
            )


            c1, c2, c3, c4 = (
                st.columns(4)
            )


            c1.metric(

                "🎯 Direction Accuracy",

                f"{direction_accuracy * 100:.2f}%"
            )


            c2.metric(

                "MAE",

                f"${mae:,.2f}"
            )


            c3.metric(

                "RMSE",

                f"${rmse:,.2f}"
            )


            c4.metric(

                "Prédictions évaluées",

                len(evaluated)
            )


            # ------------------------------------------------
            # Prediction vs Actual
            # ------------------------------------------------

            st.subheader(
                "📊 Prédiction vs prix réel"
            )


            history_chart = go.Figure()


            history_chart.add_trace(

                go.Scatter(

                    x=evaluated[
                        "target_time"
                    ],

                    y=evaluated[
                        "predicted_price"
                    ],

                    mode="lines+markers",

                    name="Prix prédit"
                )
            )


            history_chart.add_trace(

                go.Scatter(

                    x=evaluated[
                        "target_time"
                    ],

                    y=evaluated[
                        "actual_price"
                    ],

                    mode="lines+markers",

                    name="Prix réel"
                )
            )


            history_chart.update_layout(

                height=450,

                template="plotly_dark",

                hovermode="x unified",

                yaxis_title="Prix BTC ($)",

                xaxis_title="Temps"
            )


            st.plotly_chart(

                history_chart,

                use_container_width=True
            )


        else:

            st.info(
                "⏳ Les premières prédictions "
                "doivent atteindre leur horizon "
                "de 1h avant d'être évaluées."
            )


    # ========================================================
    # DETAILS
    # ========================================================

    with st.expander(
        "🔍 Voir les supports et résistances"
    ):

        col1, col2 = st.columns(2)


        with col1:

            st.write(
                "🟢 Supports"
            )

            if supports:

                for s in supports:

                    st.write(
                        f"${s:,.2f}"
                    )

            else:

                st.write(
                    "Aucun support détecté."
                )


        with col2:

            st.write(
                "🔴 Résistances"
            )

            if resistances:

                for r in resistances:

                    st.write(
                        f"${r:,.2f}"
                    )

            else:

                st.write(
                    "Aucune résistance détectée."
                )


    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown("---")

    st.caption(
        "BTC Algo Trading V2.4 | "
        "Données Yahoo Finance | "
        "Modèle Random Forest | "
        "Backtest historique"
    )


else:

    st.warning(
        "⏳ Attente des données BTC..."
    )
